/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "Cabana_BenchmarkUtils.hpp"

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

//---------------------------------------------------------------------------//
// Performance test.
template <class Device>
void performanceTest( std::ostream& stream, const std::string& test_prefix,
                      bool sort )
{
    using exec_space = typename Device::execution_space;
    using memory_space = typename Device::memory_space;

    // Declare the neighbor list type.
    using ListTag = Cabana::FullNeighborTag;
    using LayoutTag = Cabana::VerletLayout2D;
    using BuildTag = Cabana::TeamVectorOpTag;
    using IterTag = Cabana::SerialOpTag;

    // Declare problem sizes.
    double min_dist = 1.0;
    std::vector<int> problem_sizes = { 100, 1000 };
    int num_problem_size = problem_sizes.size();
    std::vector<double> x_min( num_problem_size );
    std::vector<double> x_max( num_problem_size );

    // Declare the number of cutoff ratios (directly related to neighbors per
    // atom) to generate.
    std::vector<double> cutoff_ratios = { 2.0, 3.0 };
    int cutoff_ratios_size = cutoff_ratios.size();

    // Declare the number of cell ratios to generate.
    std::vector<double> cell_ratios = { 1.0 };
    int cell_ratios_size = cell_ratios.size();

    // Number of runs in the test loops.
    int num_run = 10;

    // Define the aosoa.
    using member_types = Cabana::MemberTypes<double[3]>;
    using aosoa_type = Cabana::AoSoA<member_types, Device>;
    using aosoa_host_type = Cabana::AoSoA<member_types, Kokkos::HostSpace>;
    std::vector<aosoa_type> aosoas( num_problem_size );

    // Create aosoas.
    for ( int p = 0; p < num_problem_size; ++p )
    {
        int num_p = problem_sizes[p];
        aosoa_host_type create_aosoa( "host_aosoa", num_p );

        // Define problem grid.
        x_min[p] = 0.0;
        x_max[p] = 1.3 * min_dist * std::pow( num_p, 1.0 / 3.0 );
        aosoas[p].resize( num_p );
        auto x_host = Cabana::slice<0>( create_aosoa, "position" );
        Cabana::Benchmark::createRandomNeighbors( x_host, x_min[p], x_max[p],
                                                  min_dist );
        Cabana::deep_copy( aosoas[p], create_aosoa );

        if ( sort )
        {
            // Sort the particles to make them more realistic, e.g. in an MD
            // simulation. They likely won't be randomly scattered about, but
            // rather will be periodically sorted for spatial locality. Bin them
            // in cells the size of the smallest cutoff distance.
            double cutoff = cutoff_ratios.front() * min_dist;
            double sort_delta[3] = { cutoff, cutoff, cutoff };
            double grid_min[3] = { x_min[p], x_min[p], x_min[p] };
            double grid_max[3] = { x_max[p], x_max[p], x_max[p] };
            auto x = Cabana::slice<0>( aosoas[p], "position" );
            Cabana::LinkedCellList<Device> linked_cell_list(
                x, sort_delta, grid_min, grid_max );
            Cabana::permute( linked_cell_list, aosoas[p] );
        }
    }

    // Loop over number of ratios (neighbors per particle).
    for ( int c0 = 0; c0 < cutoff_ratios_size; ++c0 )
    {
        // Loop over cell ratios (linked cell grid relative to cutoff)
        for ( int c1 = 0; c1 < cell_ratios_size; ++c1 )
        {

            // Create timers.
            std::stringstream create_time_name;
            create_time_name << test_prefix << "neigh_create_"
                             << cutoff_ratios[c0] << "_" << cell_ratios[c1];
            Cabana::Benchmark::Timer create_timer( create_time_name.str(),
                                                   num_problem_size );
            std::stringstream iteration_time_name;
            iteration_time_name << test_prefix << "neigh_iteration_"
                                << cutoff_ratios[c0] << "_" << cell_ratios[c1];
            Cabana::Benchmark::Timer iteration_timer( iteration_time_name.str(),
                                                      num_problem_size );

            // Loop over the problem sizes.
            int pid = 0;
            std::vector<int> psizes;
            for ( int p = 0; p < num_problem_size; ++p )
            {
                int num_p = problem_sizes[p];
                std::cout << "Running cutoff ratio " << c0 << " and cell ratio "
                          << c1 << " for " << num_p << " total particles"
                          << std::endl;

                // Track the problem size.
                psizes.push_back( problem_sizes[p] );

                // Setup for Verlet list.
                double grid_min[3] = { x_min[p], x_min[p], x_min[p] };
                double grid_max[3] = { x_max[p], x_max[p], x_max[p] };

                // Setup for neighbor iteration.
                Kokkos::View<int*, memory_space> per_particle_result( "result",
                                                                      num_p );
                auto count_op = KOKKOS_LAMBDA( const int i, const int n )
                {
                    Kokkos::atomic_add( &per_particle_result( i ), n );
                };
                Kokkos::RangePolicy<exec_space> policy( 0, num_p );

                // Run tests and time the ensemble
                for ( int t = 0; t < num_run; ++t )
                {
                    // Create the neighbor list.
                    double cutoff = cutoff_ratios[c0] * min_dist;
                    create_timer.start( pid );
                    Cabana::VerletList<memory_space, ListTag, LayoutTag,
                                       BuildTag>
                        nlist( Cabana::slice<0>( aosoas[p], "position" ), 0,
                               num_p, cutoff, cell_ratios[c1], grid_min,
                               grid_max );
                    create_timer.stop( pid );

                    // Iterate through the neighbor list.
                    iteration_timer.start( pid );
                    Cabana::neighbor_parallel_for(
                        policy, count_op, nlist, Cabana::FirstNeighborsTag(),
                        IterTag(), "test_iteration" );
                    Kokkos::fence();
                    iteration_timer.stop( pid );

                    // Print neighbor statistics once per system.
                    if ( t == 0 )
                    {
                        Kokkos::MinMaxScalar<int> min_max;
                        Kokkos::MinMax<int> reducer( min_max );
                        Kokkos::parallel_reduce(
                            "Cabana::countMinMax", policy,
                            Kokkos::Impl::min_max_functor<
                                Kokkos::View<int*, Device>>(
                                nlist._data.counts ),
                            reducer );
                        Kokkos::fence();
                        std::cout << "List min neighbors: " << min_max.min_val
                                  << std::endl;
                        std::cout << "List max neighbors: " << min_max.max_val
                                  << std::endl;
                        int total_neigh = 0;
                        Kokkos::parallel_reduce(
                            "Cabana::countSum", policy,
                            KOKKOS_LAMBDA( const int p, int& nsum ) {
                                nsum += nlist._data.counts( p );
                            },
                            total_neigh );
                        Kokkos::fence();
                        std::cout
                            << "List avg neighbors: " << total_neigh / num_p
                            << std::endl;
                        std::cout << std::endl;
                    }
                }

                // Increment the problem id.
                ++pid;
            }

            // Output results.
            outputResults( stream, "problem_size", psizes, create_timer );
            outputResults( stream, "problem_size", psizes, iteration_timer );
        }
    }
}

//---------------------------------------------------------------------------//
// main
int main( int argc, char* argv[] )
{
    // Initialize environment
    Kokkos::initialize( argc, argv );

    // Check arguments.
    if ( argc < 2 )
        throw std::runtime_error( "Incorrect number of arguments. \n \
             First argument -  file name for output \n \
             \n \
             Example: \n \
             $/: ./NeighborVerletPerformance test_results.txt\n" );

    // Get the name of the output file.
    std::string filename = argv[1];

    // Open the output file on rank 0.
    std::fstream file;
    file.open( filename, std::fstream::out );

    // Run the tests.
#ifdef KOKKOS_ENABLE_SERIAL
    using SerialDevice = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
    performanceTest<SerialDevice>( file, "serial_", true );
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    using OpenMPDevice = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
    performanceTest<OpenMPDevice>( file, "openmp_", true );
#endif

#ifdef KOKKOS_ENABLE_CUDA
    using CudaDevice = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
    performanceTest<CudaDevice>( file, "cuda_", true );
#endif

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    return 0;
}

//---------------------------------------------------------------------------//
