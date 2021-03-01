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

#include "Cabana_BenchmarkTimer.hpp"

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

//---------------------------------------------------------------------------//
// Generate random particles.
template <class Slice>
void createParticles( Slice& x, const double x_min, const double x_max,
                      const double min_dist )
{
    auto num_particles = x.size();
    double min_dist_sqr = min_dist * min_dist;
    std::default_random_engine rand_gen;
    std::uniform_real_distribution<double> rand_dist( x_min, x_max );

    // Create some coarse bins to accelerate construction.
    int nbinx = 15;
    int nbin = nbinx * nbinx * nbinx;
    double bin_dx = ( x_max - x_min ) / nbinx;
    std::vector<std::vector<int>> bins( nbin );
    auto bin_id = [=]( const int i, const int j, const int k ) {
        return k * nbinx * nbinx + j * nbinx + i;
    };

    // Seed the distribution with a particle at the origin.
    x( 0, 0 ) = ( x_max - x_min ) / 2.0;
    x( 0, 1 ) = ( x_max - x_min ) / 2.0;
    x( 0, 2 ) = ( x_max - x_min ) / 2.0;
    int p0_i = std::floor( ( x( 0, 0 ) - x_min ) / bin_dx );
    int p0_j = std::floor( ( x( 0, 1 ) - x_min ) / bin_dx );
    int p0_k = std::floor( ( x( 0, 2 ) - x_min ) / bin_dx );
    bins[bin_id( p0_i, p0_j, p0_k )].push_back( 0 );

    // Create particles. Only add particles that are outside a minimum
    // distance from other particles.
    for ( std::size_t p = 1; p < num_particles; ++p )
    {
        if ( 0 == ( p - 1 ) % ( num_particles / 4 ) )
            std::cout << "Inserting " << p << " / " << num_particles
                      << std::endl;

        bool found_neighbor = true;

        // Keep trying new random coordinates until we insert one that is not
        // within the minimum distance of any other particle.
        while ( found_neighbor )
        {
            found_neighbor = false;

            // Create particle coordinates.
            x( p, 0 ) = rand_dist( rand_gen );
            x( p, 1 ) = rand_dist( rand_gen );
            x( p, 2 ) = rand_dist( rand_gen );

            // Figure out which bin we are in.
            int bin_i = std::floor( ( x( p, 0 ) - x_min ) / bin_dx );
            int bin_j = std::floor( ( x( p, 1 ) - x_min ) / bin_dx );
            int bin_k = std::floor( ( x( p, 2 ) - x_min ) / bin_dx );
            int i_min = ( 0 < bin_i ) ? bin_i - 1 : 0;
            int j_min = ( 0 < bin_j ) ? bin_j - 1 : 0;
            int k_min = ( 0 < bin_k ) ? bin_k - 1 : 0;
            int i_max = ( nbinx > bin_i ) ? bin_i + 1 : nbinx;
            int j_max = ( nbinx > bin_j ) ? bin_j + 1 : nbinx;
            int k_max = ( nbinx > bin_k ) ? bin_k + 1 : nbinx;

            // Search the adjacent bins for neighbors.
            for ( int i = i_min; i < i_max; ++i )
            {
                for ( int j = j_min; j < j_max; ++j )
                {
                    for ( int k = k_min; k < k_max; ++k )
                    {
                        for ( auto& n : bins[bin_id( i, j, k )] )
                        {
                            double dx = x( n, 0 ) - x( p, 0 );
                            double dy = x( n, 1 ) - x( p, 1 );
                            double dz = x( n, 2 ) - x( p, 2 );
                            double dist = dx * dx + dy * dy + dz * dz;

                            if ( dist < min_dist_sqr )
                            {
                                found_neighbor = true;
                                break;
                            }
                        }
                    }
                    if ( found_neighbor )
                        break;
                }
                if ( found_neighbor )
                    break;
            }

            // Add the particle to its bin if we didn't find any neighbors.
            if ( !found_neighbor )
                bins[bin_id( bin_i, bin_j, bin_k )].push_back( p );
        }
    }
    std::cout << std::endl;
}

//---------------------------------------------------------------------------//
// Performance test.
template <class Device>
void performanceTest( std::ostream& stream, const std::string& test_prefix )
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
    std::vector<int> problem_sizes = { 1000, 10000, 100000, 1000000 };
    int num_problem_size = problem_sizes.size();
    std::vector<double> x_min( num_problem_size );
    std::vector<double> x_max( num_problem_size );

    // Declare the number of cutoff ratios (directly related to neighbors per
    // atom) to generate.
    std::vector<double> cutoff_ratios = { 4.0, 8.0 };
    int cutoff_ratios_size = cutoff_ratios.size();

    // Declare the number of cell ratios (only used for Verlet) to generate.
    std::vector<double> cell_ratios = { 1.0 };

    // Number of runs in the test loops.
    int num_run = 10;

    // Define the aosoa.
    using member_types = Cabana::MemberTypes<double[3]>;
    using aosoa_type = Cabana::AoSoA<member_types, Device>;
    std::vector<aosoa_type> aosoas( num_problem_size );

    // Create aosoas.
    for ( int p = 0; p < num_problem_size; ++p )
    {
        int num_p = problem_sizes[p];
        // Define problem grid.
        x_min[p] = 0.0;
        x_max[p] = 1.3 * min_dist * std::pow( num_p, 1.0 / 3.0 );
        aosoas[p].resize( num_p );
        auto x = Cabana::slice<0>( aosoas[p], "position" );
        createParticles( x, x_min[p], x_max[p], min_dist );

        // Sort the particles to make them more realistic, e.g. in an MD
        // simulation. They likely won't be randomly scattered about, but rather
        // will be periodically sorted for spatial locality. Bin them in cells
        // the size of the smallest cutoff distance.
        double cutoff = cutoff_ratios.front() * min_dist;
        double sort_delta[3] = { cutoff, cutoff, cutoff };
        double grid_min[3] = { x_min[p], x_min[p], x_min[p] };
        double grid_max[3] = { x_max[p], x_max[p], x_max[p] };
        Cabana::LinkedCellList<Device> linked_cell_list( x, sort_delta,
                                                         grid_min, grid_max );
        Cabana::permute( linked_cell_list, aosoas[p] );
    }

    // Loop over number of ratios (neighbors per particle).
    for ( int c = 0; c < cutoff_ratios_size; ++c )
    {
        // Will need loop over cell ratio if more than one.

        // Create timers.
        std::stringstream create_time_name;
        create_time_name << test_prefix << "neigh_create_" << cutoff_ratios[c];
        Cabana::Benchmark::Timer create_timer( create_time_name.str(),
                                               num_problem_size );
        std::stringstream iteration_time_name;
        iteration_time_name << test_prefix << "neigh_iteration_"
                            << cutoff_ratios[c];
        Cabana::Benchmark::Timer iteration_timer( iteration_time_name.str(),
                                                  num_problem_size );

        // Loop over the problem sizes.
        int pid = 0;
        std::vector<int> psizes;
        for ( int p = 0; p < num_problem_size; ++p )
        {
            int num_p = problem_sizes[p];
            std::cout << "Running cutoff ratio " << c << " for " << num_p
                      << " total particles" << std::endl;

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
                double cutoff = cutoff_ratios[c] * min_dist;
                create_timer.start( pid );
#if defined( Cabana_ENABLE_ARBORX )
                auto const nlist =
                    Cabana::Experimental::make2DNeighborList<Device>(
                        ListTag{}, Cabana::slice<0>( aosoas[p], "position" ), 0,
                        num_p, cutoff );
#else
                Cabana::VerletList<memory_space, ListTag, LayoutTag, BuildTag>
                    nlist( Cabana::slice<0>( aosoas[p], "position" ), 0, num_p,
                           cutoff, cell_ratios.back(), grid_min, grid_max );
#endif
                create_timer.stop( pid );

                // Iterate through the neighbor list.
                iteration_timer.start( pid );
                Cabana::neighbor_parallel_for( policy, count_op, nlist,
                                               Cabana::FirstNeighborsTag(),
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
                            Kokkos::View<int*, Device>>( nlist._data.counts ),
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
                    std::cout << "List avg neighbors: " << total_neigh / num_p
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
             $/: ./NeighborPerformance test_results.txt\n" );

    // Get the name of the output file.
    std::string filename = argv[1];

    // Open the output file on rank 0.
    std::fstream file;
    file.open( filename, std::fstream::out );

    // Run the tests.
#ifdef KOKKOS_ENABLE_SERIAL
    using SerialDevice = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
    performanceTest<SerialDevice>( file, "serial_" );
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    using OpenMPDevice = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
    performanceTest<OpenMPDevice>( file, "openmp_" );
#endif

#ifdef KOKKOS_ENABLE_CUDA
    using CudaDevice = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
    performanceTest<CudaDevice>( file, "cuda_" );
#endif

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    return 0;
}

//---------------------------------------------------------------------------//
