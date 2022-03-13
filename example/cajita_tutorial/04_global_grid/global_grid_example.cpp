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

#include <iostream>

//---------------------------------------------------------------------------//
// Global Grid example.
//---------------------------------------------------------------------------//
void globalGridExample()
{
    /*
      The Cajita grid defines the indexing of the grid, separate from the
      physical size and characteristics of the mesh. The global grid accordingly
      defines indexing throughout the entire mesh domain.
    */
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    if ( comm_rank == 0 )
    {
        std::cout << "Cajita Global Grid Example" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    /*
      Both the global mesh and partitioning information are necessary to create
      the global grid indexing. In addition, information about periodicity is
      required.

      First, create the partitioner, in this case 2D.
    */
    Cajita::DimBlockPartitioner<2> partitioner;

    // And create the global mesh.
    double cell_size = 0.23;
    std::array<int, 2> global_num_cell = { 27, 15 };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    /*
      Define the periodicity of the system. This is needed for the grid to
      determine how to index at the boundary, i.e. whether there are ghost mesh
      cells at the edge or not.
    */
    std::array<bool, 2> is_dim_periodic = { true, true };

    // Create the global grid.
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    /*
      Now extract grid details that are the same globally (on each MPI rank):
      periodicity, number of blocks (MPI decomposition) in total and per
      dimension, and number of mesh entities per dimension.

      Note that the mesh is returned as a shared pointer.
    */
    if ( comm_rank == 0 )
    {
        std::cout << "Global global grid information:" << std::endl;
        bool periodic_x = global_grid->isPeriodic( Cajita::Dim::I );
        std::cout << "Periodicity in X: " << periodic_x << std::endl;

        int num_blocks_y = global_grid->dimNumBlock( Cajita::Dim::J );
        std::cout << "Number of blocks in Y: " << num_blocks_y << std::endl;

        int num_blocks = global_grid->totalNumBlock();
        std::cout << "Number of blocks total: " << num_blocks << std::endl;

        int num_cells_x =
            global_grid->globalNumEntity( Cajita::Cell(), Cajita::Dim::I );
        std::cout << "Number of cells in X: " << num_cells_x << std::endl;

        int num_faces_y = global_grid->globalNumEntity(
            Cajita::Face<Cajita::Dim::I>(), Cajita::Dim::J );
        std::cout << "Number of X Faces in Y: " << num_faces_y << std::endl;

        std::cout << "\nPer rank global grid information:" << std::endl;
    }

    /*
      The global grid also stores information to describe each separate block
      (MPI rank): whether it sits on a global system boundary, it's position
      ("ID") within the MPI block decomposition, and the number of mesh cells
      "owned" (uniquely managed by this rank).
    */
    bool on_lo_x = global_grid->onLowBoundary( Cajita::Dim::I );
    std::cout << "Rank-" << comm_rank << " on low X boundary: " << on_lo_x
              << std::endl;

    bool on_hi_y = global_grid->onHighBoundary( Cajita::Dim::J );
    std::cout << "Rank-" << comm_rank << " on high Y boundary: " << on_hi_y
              << std::endl;

    bool block_id = global_grid->blockId();
    std::cout << "Rank-" << comm_rank << " block ID: " << block_id << std::endl;

    bool block_id_x = global_grid->dimBlockId( Cajita::Dim::I );
    std::cout << "Rank-" << comm_rank << " block ID in X: " << block_id_x
              << std::endl;

    int num_cells_y = global_grid->ownedNumCell( Cajita::Dim::J );
    std::cout << "Rank-" << comm_rank
              << " owned mesh cells in Y: " << num_cells_y << std::endl;

    // Barrier for cleaner printing.
    MPI_Barrier( MPI_COMM_WORLD );

    /*
      Other information can be extracted which is somewhat lower level. First,
      the MPI rank of a given block ID can be obtained; this returns -1 if it
      an invalid ID for the current decomposition.
     */
    if ( comm_rank == 0 )
    {
        std::cout << std::endl;
        // In this case, if the block ID is passed as an array with length equal
        // to the spatial dimension.
        bool block_rank = global_grid->blockRank( { 0, 0 } );
        std::cout << "MPI rank of the first block: " << block_rank << std::endl;
    }

    /*
      Second, the offset (the index from the bottom left corner in a given
      dimension) of one block can be extracted.
    */
    int offset_x = global_grid->globalOffset( Cajita::Dim::I );
    std::cout << "Rank-" << comm_rank << " offset in X: " << offset_x
              << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        globalGridExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
