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

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <array>

//---------------------------------------------------------------------------//
void gatherScatterExample( const Cajita::ManualBlockPartitioner<3>& partitioner,
                           const std::array<bool, 3>& is_dim_periodic,
                           const std::string test_name )
{
    /*
      The halo is a communication plan designed from halo exchange where some
      locally-owned elements on each rank are used as ghost data on other
      ranks. The halo supplies both forward and reverse communication
      operations. In the forward operation (the gather), data is sent from the
      uniquely-owned decomposition to the ghosted decomposition. In the
      reverse operation (the scatter), data is sent from the ghosted
      decomposition back to the uniquely-owned decomposition and collisions
      are resolved.

      In this example we will demonstrate building a halo communication
      plan and performing both scatter and gather operations.
    */

    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    if ( comm_rank == 0 )
    {
        std::cout << "=======================\n";
        std::cout << test_name << std::endl;
        std::cout << "=======================\n";
    }

    using exec_space = Kokkos::DefaultHostExecutionSpace;
    using device_type = exec_space::device_type;

    // Create the global grid.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 16, 21, 21 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Array parameters.
    unsigned array_halo_width = 3;

    // Loop over halo sizes up to the size of the array halo width.
    for ( unsigned halo_width = 1; halo_width <= array_halo_width;
          ++halo_width )
    {
        if ( comm_rank == 0 )
            std::cout << "halo width : " << halo_width << std::endl;

        // Create a cell array.
        auto layout = createArrayLayout( global_grid, array_halo_width, 4,
                                         Cajita::Cell() );
        auto array =
            Cajita::createArray<double, device_type>( "array", layout );

        // Assign the owned cells a value of 1 and the rest 0.
        Cajita::ArrayOp::assign( *array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *array, 1.0, Cajita::Own() );

        // create host mirror view
        auto host_view = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), array->view() );
        auto ghosted_space =
            array->layout()->indexSpace( Cajita::Ghost(), Cajita::Local() );

        /*
           print out cell values along x-axis. Owned cell are 1 and ghosted cell
           are 0
        */
        if ( comm_rank == 0 )
        {
            std::cout << "Array Values\n";
            for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
                std::cout << host_view( i, 13, 13, 0 ) << " ";
            std::cout << std::endl;
        }

        // Create a halo.
        auto halo = createHalo( *array, Cajita::FullHaloPattern(), halo_width );

        // Gather into the ghosts.
        halo->gather( exec_space(), *array );

        /*
           print out cell values along x-axis. After gather, ghotsted cell
           values are re-assigned with owned cell values of neighbors.
        */
        if ( comm_rank == 0 )
        {
            std::cout << "Array Valueas after gather\n";
            for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
                std::cout << host_view( i, 13, 13, 0 ) << " ";
            std::cout << std::endl;
        }

        // Scatter from the ghosts back to owned.
        halo->scatter( exec_space(), Cajita::ScatterReduce::Sum(), *array );

        /*
           print out cell values along x-axis. After scatter, owned cell values
           are reduced with ghost cell values of neighbors.
        */
        if ( comm_rank == 0 )
        {
            std::cout << "Array Valueas after scatter\n";
            for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
                std::cout << host_view( i, 13, 13, 0 ) << " ";
            std::cout << std::endl << std::endl;
        }
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        // Let MPI compute the partitioning for this test.
        int comm_size;
        MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
        std::array<int, 3> ranks_per_dim = { comm_size, 1, 1 };
        MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
        Cajita::ManualBlockPartitioner<3> partitioner( ranks_per_dim );

        // Boundaries are not periodic. Thus, halos at boundary does not
        // commuicates
        std::array<bool, 3> is_dim_periodic = { false, false, false };
        gatherScatterExample( partitioner, is_dim_periodic,
                              "non periodic test" );

        // Boundaries are periodic. Thus, halos at boundarys communicate
        // with its preriodic neighbors
        is_dim_periodic = { true, true, true };
        gatherScatterExample( partitioner, is_dim_periodic, "periodic test" );
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
