/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "../Cabana_BenchmarkUtils.hpp"

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
                      std::vector<int> problem_sizes,
                      std::vector<double> cutoff_ratios )
{
    // Declare problem sizes.
    int num_problem_size = problem_sizes.size();
    std::vector<double> x_min( num_problem_size );
    std::vector<double> x_max( num_problem_size );

    // Declare the number of cutoff ratios (directly related to neighbors per
    // atom) to generate.
    int cutoff_ratios_size = cutoff_ratios.size();

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
        x_max[p] = 1.3 * std::pow( num_p, 1.0 / 3.0 );
        double grid_min[3] = { x_min[p], x_min[p], x_min[p] };
        double grid_max[3] = { x_max[p], x_max[p], x_max[p] };
        aosoas[p].resize( num_p );
        auto x = Cabana::slice<0>( aosoas[p], "position" );
        Cabana::createParticles( Cabana::InitRandom(), x, x.size(), grid_min,
                                 grid_max );
    }

    // Loop over number of ratios (neighbors per particle).
    for ( int c = 0; c < cutoff_ratios_size; ++c )
    {
        // Will need loop over cell ratio if more than one.

        // Create timers.
        std::stringstream create_time_name;
        create_time_name << test_prefix << "linkedcell_create_"
                         << cutoff_ratios[c];
        Cabana::Benchmark::Timer create_timer( create_time_name.str(),
                                               num_problem_size );
        std::stringstream sort_time_name;
        sort_time_name << test_prefix << "linkedcell_sort_" << cutoff_ratios[c];
        Cabana::Benchmark::Timer sort_timer( sort_time_name.str(),
                                             num_problem_size );

        // Loop over the problem sizes.
        std::vector<int> psizes;
        for ( int p = 0; p < num_problem_size; ++p )
        {
            int num_p = problem_sizes[p];
            std::cout << "Running cutoff ratio " << c << " for " << num_p
                      << " total particles" << std::endl;

            // Track the problem size.
            psizes.push_back( problem_sizes[p] );

            // Create the linked cell list.
            auto x = Cabana::slice<0>( aosoas[p], "position" );
            double cutoff = cutoff_ratios[c];
            double sort_delta[3] = { cutoff, cutoff, cutoff };
            double grid_min[3] = { x_min[p], x_min[p], x_min[p] };
            double grid_max[3] = { x_max[p], x_max[p], x_max[p] };
            Cabana::LinkedCellList<Device> linked_cell_list(
                x, sort_delta, grid_min, grid_max );

            // Run tests and time the ensemble
            for ( int t = 0; t < num_run; ++t )
            {
                // Build the linked cell list.
                create_timer.start( p );
                linked_cell_list.build( x );
                create_timer.stop( p );

                // Sort the particles.
                sort_timer.start( p );
                Cabana::permute( linked_cell_list, aosoas[p] );
                sort_timer.stop( p );
            }
        }

        // Output results.
        outputResults( stream, "problem_size", psizes, create_timer );
        outputResults( stream, "problem_size", psizes, sort_timer );
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
             Optional second argument - run size (small or large) \n \
             \n \
             Example: \n \
             $/: ./LinkedCellPerformance test_results.txt\n" );

    // Get the name of the output file.
    std::string filename = argv[1];

    // Define run sizes.
    std::string run_type = "";
    if ( argc > 2 )
        run_type = argv[2];
    std::vector<int> problem_sizes = { 100, 1000 };
    std::vector<double> cutoff_ratios = { 3.0, 4.0 };
    if ( run_type == "large" )
        problem_sizes = { 1000, 10000, 100000, 1000000 };

    // Open the output file on rank 0.
    std::fstream file;
    file.open( filename, std::fstream::out );

    // Do everything on the default CPU.
    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_device_type = host_exec_space::device_type;
    // Do everything on the default device with default memory.
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;

    // Don't run twice on the CPU if only host enabled.
    if ( !std::is_same<device_type, host_device_type>{} )
    {
        performanceTest<device_type>( file, "device_", problem_sizes,
                                      cutoff_ratios );
    }
    performanceTest<host_device_type>( file, "host_", problem_sizes,
                                       cutoff_ratios );

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    return 0;
}

//---------------------------------------------------------------------------//
