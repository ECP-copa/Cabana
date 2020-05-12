/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
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
// Performance test.
template <class Device>
void performanceTest( std::ostream &stream, const std::string &test_prefix )
{
    // Declare problem sizes.
    std::vector<int> problem_sizes = { 1000, 10000, 100000, 1000000, 10000000 };
    int num_problem_size = problem_sizes.size();

    // Generate a random set of keys
    Kokkos::View<unsigned long *, Device> keys(
        Kokkos::ViewAllocateWithoutInitializing( "keys" ),
        problem_sizes.back() );
    Kokkos::View<unsigned long *, Kokkos::HostSpace> host_keys(
        Kokkos::ViewAllocateWithoutInitializing( "host_keys" ),
        problem_sizes.back() );
    std::minstd_rand0 generator( 3439203991 );
    for ( int n = 0; n < problem_sizes.back(); ++n )
        host_keys( n ) = generator();
    Kokkos::deep_copy( keys, host_keys );

    // Declare the number of bins to generate.
    std::vector<int> num_bins = { 10,     100,     1000,    10000,
                                  100000, 1000000, 10000000 };
    int num_bin_size = num_bins.size();

    // Number of runs in the test loops.
    int num_run = 10;

    // Define the aosoa.
    using member_types = Cabana::MemberTypes<double[3], double[3], double, int>;
    using aosoa_type = Cabana::AoSoA<member_types, Device>;

    // Create aosoas.
    std::vector<aosoa_type> aosoas( num_problem_size );
    for ( int i = 0; i < num_problem_size; ++i )
        aosoas[i] = aosoa_type( "aosoa", problem_sizes[i] );

    // BINNING
    // -------

    // Loop over number of bins.
    for ( int b = 0; b < num_bin_size; ++b )
    {
        // Compute the number of problems we will run with this bin size.
        int bin_num_problem = 0;
        for ( int p = 0; p < num_problem_size; ++p )
            if ( num_bins[b] <= problem_sizes[p] )
                ++bin_num_problem;

        // Create binning timers.
        std::stringstream create_time_name;
        create_time_name << test_prefix << "bin_create_" << num_bins[b];
        Cabana::Benchmark::Timer create_timer( create_time_name.str(),
                                               bin_num_problem );
        std::stringstream aosoa_permute_time_name;
        aosoa_permute_time_name << test_prefix << "bin_aosoa_permute_"
                                << num_bins[b];
        Cabana::Benchmark::Timer aosoa_permute_timer(
            aosoa_permute_time_name.str(), bin_num_problem );
        std::stringstream slice_permute_time_name;
        slice_permute_time_name << test_prefix << "bin_slice_permute_"
                                << num_bins[b];
        Cabana::Benchmark::Timer slice_permute_timer(
            slice_permute_time_name.str(), bin_num_problem );

        // Loop over the problem sizes.
        int pid = 0;
        std::vector<double> psizes;
        for ( int p = 0; p < num_problem_size; ++p )
        {
            // Only run this problem size if the number of bins does not
            // exceed the problem size.
            if ( num_bins[b] <= problem_sizes[p] )
            {
                // Track the problem size.
                psizes.push_back( problem_sizes[p] );

                // Run tests and time the ensemble
                for ( int t = 0; t < num_run; ++t )
                {
                    // Create the binning.
                    auto key_sv = Kokkos::subview(
                        keys, Kokkos::pair<int, int>( 0, problem_sizes[p] ) );
                    create_timer.start( pid );
                    auto bin_data = Cabana::binByKey( key_sv, num_bins[b] );
                    create_timer.stop( pid );

                    // Permute the aosoa
                    aosoa_permute_timer.start( pid );
                    Cabana::permute( bin_data, aosoas[p] );
                    aosoa_permute_timer.stop( pid );

                    // Permute a slice of the first member
                    auto slice = Cabana::slice<0>( aosoas[p] );
                    slice_permute_timer.start( pid );
                    Cabana::permute( bin_data, slice );
                    slice_permute_timer.stop( pid );
                }

                // Increment the problem id.
                ++pid;
            }
        }

        // Output results.
        outputResults( stream, "problem_size", psizes, create_timer );
        outputResults( stream, "problem_size", psizes, aosoa_permute_timer );
        outputResults( stream, "problem_size", psizes, slice_permute_timer );
    }

    // SORTING
    // -------

    // Create sorting timers.
    Cabana::Benchmark::Timer create_timer( test_prefix + "sort_create",
                                           num_problem_size );
    Cabana::Benchmark::Timer aosoa_permute_timer(
        test_prefix + "sort_aosoa_permute", num_problem_size );
    Cabana::Benchmark::Timer slice_permute_timer(
        test_prefix + "sort_slice_permute", num_problem_size );

    // Loop over the problem sizes.
    for ( int p = 0; p < num_problem_size; ++p )
    {
        // Run tests and time the ensemble
        for ( int t = 0; t < num_run; ++t )
        {
            // Create the binning.
            auto key_sv = Kokkos::subview(
                keys, Kokkos::pair<int, int>( 0, problem_sizes[p] ) );
            create_timer.start( p );
            auto bin_data = Cabana::sortByKey( key_sv );
            create_timer.stop( p );

            // Permute the aosoa
            aosoa_permute_timer.start( p );
            Cabana::permute( bin_data, aosoas[p] );
            aosoa_permute_timer.stop( p );

            // Permute a slice of the first member
            auto slice = Cabana::slice<0>( aosoas[p] );
            slice_permute_timer.start( p );
            Cabana::permute( bin_data, slice );
            slice_permute_timer.stop( p );
        }
    }

    // Output results.
    outputResults( stream, "problem_size", problem_sizes, create_timer );
    outputResults( stream, "problem_size", problem_sizes, aosoa_permute_timer );
    outputResults( stream, "problem_size", problem_sizes, slice_permute_timer );
}

//---------------------------------------------------------------------------//
// main
int main( int argc, char *argv[] )
{
    // Initialize environment
    Kokkos::initialize( argc, argv );

    // Check arguments.
    if ( argc < 2 )
        throw std::runtime_error( "Incorrect number of arguments. \n \
             First argument -  file name for output \n \
             \n \
             Example: \n \
             $/: ./BinSortPerformance test_results.txt\n" );

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
