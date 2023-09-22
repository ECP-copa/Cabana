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
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ratio>
#include <string>
#include <vector>

#include <mpi.h>

using namespace Cabana::Grid;

//---------------------------------------------------------------------------//
// Performance test.
template <class Device>
void performanceTest( std::ostream& stream,
                      const DimBlockPartitioner<3> partitioner,
                      std::vector<int> grid_sizes_per_dim_per_rank,
                      const std::string& test_prefix,
                      std::vector<int> halo_widths, MPI_Comm comm )
{
    using exec_space = typename Device::execution_space;
    using device_type = Device;

    // Total loop sizes
    int halo_width_size = halo_widths.size();
    int num_grid_size = grid_sizes_per_dim_per_rank.size();

    // number of runs
    int num_run = 10;

    // Create the mesh and grid structures as usual.
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = { 2.4, -0.4, -1.5 };
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // Now we loop over halo sizes up to the size allocated to compare.
    for ( int halo_width = 0; halo_width < halo_width_size; ++halo_width )
    {
        // Create timers
        std::stringstream halo_create_name;
        halo_create_name << test_prefix << "halo_create_" << halo_width;
        Cabana::Benchmark::Timer halo_create_timer( halo_create_name.str(),
                                                    num_grid_size );

        std::stringstream halo_gather_name;
        halo_gather_name << test_prefix << "halo_gather_" << halo_width;
        Cabana::Benchmark::Timer halo_gather_timer( halo_gather_name.str(),
                                                    num_grid_size );

        std::stringstream halo_scatter_name;
        halo_scatter_name << test_prefix << "halo_scatter_" << halo_width;
        Cabana::Benchmark::Timer halo_scatter_timer( halo_scatter_name.str(),
                                                     num_grid_size );

        // loop over the grid sizes
        int pid = 0;
        for ( int p = 0; p < num_grid_size; ++p )
        {
            auto ranks_per_dim =
                partitioner.ranksPerDimension( comm, { 0, 0, 0 } );

            std::array<int, 3> grid_sizes_per_dim;
            for ( int d = 0; d < 3; ++d )
            {
                grid_sizes_per_dim[d] =
                    grid_sizes_per_dim_per_rank[p] * ranks_per_dim[d];
            }

            auto global_mesh = createUniformGlobalMesh(
                global_low_corner, global_high_corner, grid_sizes_per_dim );
            auto global_grid = createGlobalGrid( comm, global_mesh,
                                                 is_dim_periodic, partitioner );

            // Create a cell array.
            auto layout =
                createArrayLayout( global_grid, halo_width, 4, Cell() );
            auto array = createArray<double, device_type>( "array", layout );

            // Assign the owned cells a value of 1 and ghosted 0.
            ArrayOp::assign( *array, 0.0, Ghost() );
            ArrayOp::assign( *array, 1.0, Own() );

            // create host mirror view
            auto array_view = array->view();

            for ( int t = 0; t < num_run; ++t )
            {
                // create halo
                halo_create_timer.start( pid );
                auto halo =
                    createHalo( NodeHaloPattern<3>(), halo_width, *array );
                halo_create_timer.stop( pid );

                // gather
                halo_gather_timer.start( pid );
                halo->gather( exec_space(), *array );
                halo_gather_timer.stop( pid );

                // scatter
                halo_scatter_timer.start( pid );
                halo->scatter( exec_space(), ScatterReduce::Sum(), *array );
                halo_scatter_timer.stop( pid );
            }
            // Increment the problem id.
            ++pid;
        }

        // Output results
        outputResults( stream, "grid_size_per_dim", grid_sizes_per_dim_per_rank,
                       halo_create_timer, comm );
        outputResults( stream, "grid_size_per_dim", grid_sizes_per_dim_per_rank,
                       halo_gather_timer, comm );
        outputResults( stream, "grid_size_per_dim", grid_sizes_per_dim_per_rank,
                       halo_scatter_timer, comm );
        stream << std::flush;
    }
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
    std::vector<int> grid_sizes_per_dim_per_rank = { 16, 32 };
    std::vector<int> halo_widths = { 1, 2 };
    if ( run_type == "large" )
    {
        grid_sizes_per_dim_per_rank = { 16, 32, 64, 128, 256 };
        halo_widths = { 1, 2, 3, 4, 5 };
    }

    // Barier before continuing.
    MPI_Barrier( MPI_COMM_WORLD );

    // Get comm rank and size;
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    // Get partitioner
    DimBlockPartitioner<3> partitioner;
    // Get ranks per dimension
    std::array<int, 3> ranks_per_dimension =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, { 0, 0, 0 } );

    // Open the output file on rank 0.
    std::fstream file;
    // Output problem details.
    if ( 0 == comm_rank )
    {
        file.open( filename + "_" + std::to_string( comm_size ),
                   std::fstream::out );
        file << "\n";
        file << "Cabana::Grid Halo Performance Benchmark"
             << "\n";
        file << "----------------------------------------------"
             << "\n";
        file << "MPI Ranks: " << comm_size << "\n";
        file << "MPI Cartesian Dim Ranks: (" << ranks_per_dimension[0] << ", "
             << ranks_per_dimension[1] << ", " << ranks_per_dimension[2]
             << ")\n";
        file << "----------------------------------------------"
             << "\n";
        file << "\n";
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
        performanceTest<device_type>( file, partitioner,
                                      grid_sizes_per_dim_per_rank, "device_",
                                      halo_widths, MPI_COMM_WORLD );
    }
    performanceTest<host_device_type>( file, partitioner,
                                       grid_sizes_per_dim_per_rank, "host_",
                                       halo_widths, MPI_COMM_WORLD );

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}

//---------------------------------------------------------------------------//
