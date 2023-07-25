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

#include <Cabana_Core.hpp>
#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ratio>
#include <string>
#include <vector>

#include <mpi.h>

//---------------------------------------------------------------------------//
// Performance test.
template <class Device>
void performanceTest( std::ostream& stream,
                      const Cajita::DimBlockPartitioner<3> partitioner,
                      const std::array<int, 3> grid_sizes_per_dim,
                      const std::string& test_prefix,
                      std::vector<int> problem_halo_width, MPI_Comm comm )
{
    using exec_space = typename Device::execution_space;
    using device_type = Device;

    // the size of halo_width
    int halo_width_size = problem_halo_width.size();

    // number of runs
    int num_run = 10;

    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // Create the mesh and grid structures as usual.
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = { 2.4, -0.4, -1.5 };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, grid_sizes_per_dim );
    auto global_grid =
        createGlobalGrid( comm, global_mesh, is_dim_periodic, partitioner );

    std::cout << global_grid->ownedNumCell( 0 ) << " "
              << global_grid->ownedNumCell( 1 ) << " "
              << global_grid->ownedNumCell( 2 ) << " " << std::endl;

    // Create insertion timers
    std::stringstream halo_create_name;
    halo_create_name << test_prefix << "halo_create";
    Cabana::Benchmark::Timer halo_create_timer( halo_create_name.str(),
                                                halo_width_size );

    std::stringstream halo_gather_name;
    halo_gather_name << test_prefix << "halo_gather";
    Cabana::Benchmark::Timer halo_gather_timer( halo_gather_name.str(),
                                                halo_width_size );

    std::stringstream halo_scatter_name;
    halo_scatter_name << test_prefix << "halo_scatter";
    Cabana::Benchmark::Timer halo_scatter_timer( halo_scatter_name.str(),
                                                 halo_width_size );

    // Now we loop over halo sizes up to the size allocated to compare.
    for ( int halo_width = 0; halo_width < halo_width_size; ++halo_width )
    {
        // Create a cell array.
        auto layout = createArrayLayout( global_grid, halo_width_size, 4,
                                         Cajita::Cell() );
        auto array =
            Cajita::createArray<double, device_type>( "array", layout );

        // Assign the owned cells a value of 1 and ghosted 0.
        Cajita::ArrayOp::assign( *array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *array, 1.0, Cajita::Own() );

        // create host mirror view
        auto array_view = array->view();

        for ( int t = 0; t < num_run; ++t )
        {

            // create halo
            halo_create_timer.start( halo_width );
            auto halo =
                createHalo( Cajita::NodeHaloPattern<3>(), halo_width, *array );
            halo_create_timer.stop( halo_width );

            // gather
            halo_gather_timer.start( halo_width );
            halo->gather( exec_space(), *array );
            halo_gather_timer.stop( halo_width );

            // scatter
            halo_scatter_timer.start( halo_width );
            halo->scatter( exec_space(), Cajita::ScatterReduce::Sum(), *array );
            halo_scatter_timer.stop( halo_width );
        }
    }

    // Output results
    outputResults( stream, "halo_width", problem_halo_width, halo_create_timer,
                   comm );
    outputResults( stream, "halo_width", problem_halo_width, halo_gather_timer,
                   comm );
    outputResults( stream, "halo_width", problem_halo_width, halo_scatter_timer,
                   comm );
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
             Optional second argument - run size (small or large) \n \
             \n \
             Example: \n \
             $/: ./GridHaloPerformance test_results.txt large\n" );

    // Get the name of the output file.
    std::string filename = argv[1];

    // Define run sizes.
    std::string run_type = "";
    if ( argc > 2 )
        run_type = argv[2];
    std::vector<int> grid_sizes_per_dim_per_rank = { 16 };
    std::vector<int> problem_halo_width = { 1, 2 };
    if ( run_type == "large" )
    {
        grid_sizes_per_dim_per_rank = { 16, 32, 64, 128, 256 };
        problem_halo_width = { 1, 2, 3, 4, 5 };
    }

    // Barier before continuing.
    MPI_Barrier( MPI_COMM_WORLD );

    // Get comm rank and size;
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    // Get partitioner
    Cajita::DimBlockPartitioner<3> partitioner;
    // Get ranks per dimension
    std::array<int, 3> ranks_per_dimension =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, { 0, 0, 0 } );

    int num_grid_size = grid_sizes_per_dim_per_rank.size();
    for ( int p = 0; p < num_grid_size; ++p )
    {
        // Open the output file on rank 0.
        std::fstream file;
        if ( 0 == comm_rank )
            file.open( filename + "_" + std::to_string( comm_size ) + "_" +
                           std::to_string( grid_sizes_per_dim_per_rank[p] ),
                       std::fstream::out );

        std::array<int, 3> grid_sizes_per_dim;
        for ( int d = 0; d < 3; ++d )
        {
            grid_sizes_per_dim[d] =
                grid_sizes_per_dim_per_rank[p] * ranks_per_dimension[d];
        }

        // Output problem details.
        if ( 0 == comm_rank )
        {
            file << "\n";
            file << "Cajita Halo Performance Benchmark"
                 << "\n";
            file << "----------------------------------------------"
                 << "\n";
            file << "MPI Ranks: " << comm_size << "\n";
            file << "MPI Cartesian Dim Ranks: (" << ranks_per_dimension[0]
                 << ", " << ranks_per_dimension[1] << ", "
                 << ranks_per_dimension[2] << ")\n";
            file << "Global Cells per Dim per Rank: "
                 << grid_sizes_per_dim_per_rank[p] << "\n";
            file << "----------------------------------------------"
                 << "\n";
            file << "\n";
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
            performanceTest<device_type>( file, partitioner, grid_sizes_per_dim,
                                          "device_", problem_halo_width,
                                          MPI_COMM_WORLD );
        }
        performanceTest<host_device_type>( file, partitioner,
                                           grid_sizes_per_dim, "host_",
                                           problem_halo_width, MPI_COMM_WORLD );
    }
    // Finalize
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}

//---------------------------------------------------------------------------//
