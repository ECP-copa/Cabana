/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "../Cabana_BenchmarkUtils.hpp"

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

using namespace Cajita;

//---------------------------------------------------------------------------//
// Performance test.
template <class Device>
void performanceTest( std::ostream& stream,
                      std::vector<double> grid_sizes_per_dim, MPI_Comm comm,
                      const std::string& test_prefix )
{
    using exec_space = typename Device::execution_space;
    using memory_space = typename Device::memory_space;

    // Domain size setup
    std::array<double, 3> global_low_corner = { 0.0, 0.0, 0.0 };
    std::array<double, 3> global_high_corner = { 1.0, 1.0, 1.0 };
    std::array<bool, 3> is_dim_periodic = { true, true, true };

    int num_grid_size_per_dim = grid_sizes_per_dim.size();

    // number of runs in test loops
    int num_runs = 10;

    // get MPI rank
    int comm_rank;
    MPI_Comm_rank( comm, &comm_rank );

    // create timers
    Cabana::Benchmark::Timer setup_timer( test_prefix + "setup",
                                          num_grid_size_per_dim );

    Cabana::Benchmark::Timer transforms_timer( test_prefix + "transforms",
                                               num_grid_size_per_dim );
    // loop over the grid sizes
    for ( int p = 0; p < num_grid_size_per_dim; ++p )
    {
        double cell_size = 1.0 / grid_sizes_per_dim[p];
        auto global_mesh = createUniformGlobalMesh(
            global_low_corner, global_high_corner, cell_size );

        // Create the global grid
        DimBlockPartitioner<3> partitioner;
        auto global_grid =
            createGlobalGrid( comm, global_mesh, is_dim_periodic, partitioner );

        // Create a local grid
        auto local_grid = createLocalGrid( global_grid, 0 );
        auto owned_space = local_grid->indexSpace( Own(), Cell(), Local() );
        auto ghosted_space = local_grid->indexSpace( Ghost(), Cell(), Local() );

        auto vector_layout = createArrayLayout( local_grid, 2, Cell() );
        auto lhs = createArray<double, Device>( "lhs", vector_layout );
        auto lhs_view = lhs->view();
        uint64_t seed = global_grid->blockId() +
                        ( 19383747 % ( global_grid->blockId() + 1 ) );
        using rnd_type = Kokkos::Random_XorShift64_Pool<memory_space>;
        rnd_type pool;
        pool.init( seed, ghosted_space.size() );
        Kokkos::parallel_for(
            "random_complex_grid",
            createExecutionPolicy( owned_space, exec_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                auto rand = pool.get_state( i + j + k );
                lhs_view( i, j, k, 0 ) =
                    Kokkos::rand<decltype( rand ), double>::draw( rand, 0.0,
                                                                  1.0 );
                lhs_view( i, j, k, 1 ) =
                    Kokkos::rand<decltype( rand ), double>::draw( rand, 0.0,
                                                                  1.0 );
            } );

        setup_timer.start( p );

        // Create FFT options
        Experimental::FastFourierTransformParams params;

        auto fft =
            Experimental::createHeffteFastFourierTransform<double, Device>(
                *vector_layout, params );

        setup_timer.stop( p );

        transforms_timer.start( p );

        // Loop over number of runs
        for ( int t = 0; t < num_runs; ++t )
        {
            fft->forward( *lhs, Experimental::FFTScaleFull() );
            fft->reverse( *lhs, Experimental::FFTScaleNone() );
        }

        transforms_timer.stop( p );
    }

    outputResults( stream, "grid_size_per_dim", grid_sizes_per_dim, setup_timer,
                   comm );
    outputResults( stream, "grid_size_per_dim", grid_sizes_per_dim,
                   transforms_timer, comm );

    stream << std::flush;
}

//---------------------------------------------------------------------------//
// main
int main( int argc, char* argv[] )
{
    // Initialize environment
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    // Check arguments.
    if ( argc < 2 )
        throw std::runtime_error( "Incorrect number of arguments. \n \
             First argument - file name for output \n \
             Optional second argument - problem size (small or large) \n \
             \n \
             Example: \n \
             $/: ./FastFourierTransformPerformance test_results.txt\n" );

    // Define run sizes.
    std::string run_type = "";
    if ( argc > 2 )
    {
        run_type = argv[2];
    }

    // Declare the grid size per dimension
    // currently, testing 3dims+symmetric
    std::vector<double> grid_sizes_per_dim = { 16, 32 };
    if ( run_type == "large" )
    {
        grid_sizes_per_dim = { 16, 32, 64, 128 };
    }

    // Get the name of the output file.
    std::string filename = argv[1];

    // Barrier before continuing
    MPI_Barrier( MPI_COMM_WORLD );

    // Get comm rank;
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    // Open the output file on rank 0.
    std::fstream file;
    if ( 0 == comm_rank )
        file.open( filename, std::fstream::out );

    // Output file header
    if ( 0 == comm_rank )
    {
        file << "\n";
        file << "Cajita FFT Performance Benchmark"
             << "\n";
        file << "----------------------------------------------"
             << "\n";
        file << std::flush;
    }

    // Do everything on the default CPU.
    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_device_type = host_exec_space::device_type;
    // Do everything on the default device with default memory.
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;

    // Don't run twice on the CPU if only host enabled.
    if ( !std::is_same<device_type, host_device_type>{} )
    {
        performanceTest<device_type>( file, grid_sizes_per_dim, MPI_COMM_WORLD,
                                      "device_default_" );
    }
    performanceTest<host_device_type>( file, grid_sizes_per_dim, MPI_COMM_WORLD,
                                       "host_default_" );

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
