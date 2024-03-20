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

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <array>

//---------------------------------------------------------------------------//
// Grid Halo example.
//---------------------------------------------------------------------------//
void gridHaloExample()
{
    /*
      The halo is a communication plan designed from halo exchange where some
      locally-owned elements on each rank are used as ghost data on other
      ranks. The halo supplies both forward and reverse communication
      operations. In the forward operation (the gather), data is sent from the
      uniquely-owned decomposition to the ghosted decomposition. In the
      reverse operation (the scatter), data is sent from the ghosted
      decomposition back to the uniquely-owned decomposition and collisions
      are resolved. Grid halos in Cabana::Grid are relvatively simple because
      all grids are logically rectilinear.

      In this example we will demonstrate building a halo communication
      plan and performing both scatter and gather operations.
    */

    using exec_space = Kokkos::DefaultHostExecutionSpace;
    using device_type = exec_space::device_type;

    // Use linear MPI partitioning to make this example simpler.
    Cabana::Grid::DimBlockPartitioner<3> partitioner( Cabana::Grid::Dim::J,
                                                      Cabana::Grid::Dim::K );

    /*
      Boundaries are not periodic in this example, thus ranks at system
      boundaries do not communicate. Because of this the details printed from
      rank 0 on the left side will never change as it is on the system boundary
      (without an MPI neighbor).

      Changing values here to true will show what happens for periodic grid
      communication instead, where the left and right sides of the local domain
      should then match.
    */
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // Create the mesh and grid structures as usual.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 26, 21, 21 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Get the current rank for printing output.
    int comm_rank = global_grid->blockId();
    if ( comm_rank == 0 )
    {
        std::cout << "Cabana::Grid Grid Halo Example" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    /*
      Here the halo width allocated for the system is not necessarily always
      fully communicated - we can use any integer value from zero to the value
      allocated.
    */
    unsigned allocated_halo_width = 3;

    // Now we loop over halo sizes up to the size allocated to compare.
    for ( unsigned halo_width = 1; halo_width <= allocated_halo_width;
          ++halo_width )
    {
        if ( comm_rank == 0 )
            std::cout << "halo width: " << halo_width << std::endl;

        // Create a cell array.
        auto layout = createArrayLayout( global_grid, allocated_halo_width, 4,
                                         Cabana::Grid::Cell() );
        auto array =
            Cabana::Grid::createArray<double, device_type>( "array", layout );

        // Assign the owned cells a value of 1 and ghosted 0.
        Cabana::Grid::ArrayOp::assign( *array, 0.0, Cabana::Grid::Ghost() );
        Cabana::Grid::ArrayOp::assign( *array, 1.0, Cabana::Grid::Own() );

        // create host mirror view
        auto array_view = array->view();
        auto ghosted_space = array->layout()->indexSpace(
            Cabana::Grid::Ghost(), Cabana::Grid::Local() );

        /*
           Print out cell values along a single slice of the x-axis (recalling
           that the MPI decomposition is only along x). Owned cells should be 1
           and ghosted cells 0.
        */
        if ( comm_rank == 0 )
        {
            std::cout << "Array Values before communication\n";
            for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
                std::cout << array_view( i, 13, 13, 0 ) << " ";
            std::cout << std::endl;
        }

        /*
          Create a halo. Note that this is done with the array data and that no
          communication has occurred yet. The halo width (within the value
          allocated) is also passed, together with options for the type of
          communication:
           - Node pattern communicates with 26 MPI neighbors in 3D (8 in 2D)
           - Face pattern communicates with 6 MPI neighbors in 3D (4 in 2D)
        */
        auto halo = createHalo( Cabana::Grid::NodeHaloPattern<3>(), halo_width,
                                *array );

        /*
          Gather into the ghosts. This performs the grid communication from the
          owning rank to surrounding neighbors for the field data selected. Note
          that this can be done with any execution space that is compatible with
          the array/halo memory space being used.
        */
        halo->gather( exec_space(), *array );

        /*
           Print out cell values again along the x-axis after the gather.
           Ghosted cell values are re-assigned with owned cell values of their
           neighbors.
        */
        if ( comm_rank == 0 )
        {
            std::cout << "Array Values after gather\n";
            for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
                std::cout << array_view( i, 13, 13, 0 ) << " ";
            std::cout << std::endl;
        }

        /*
          Now scatter back from the ghosts to the rank on which it is owned.
          This is a similar interface as the gather operation, but here we can
          specify different update operations: sum (used here), min, max, or
          replace the value.
        */
        halo->scatter( exec_space(), Cabana::Grid::ScatterReduce::Sum(),
                       *array );

        /*
          Print out cell values one more time along the x-axis after the
          scatter. Now owned cell values near the boundary should be 2 (updated
          from their ghost cells on MPI neighbors.
        */
        if ( comm_rank == 0 )
        {
            std::cout << "Array Values after scatter\n";
            for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
                std::cout << array_view( i, 13, 13, 0 ) << " ";
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

        gridHaloExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
