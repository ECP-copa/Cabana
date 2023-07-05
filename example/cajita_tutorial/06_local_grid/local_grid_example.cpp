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

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

#include <mpi.h>

//---------------------------------------------------------------------------//
// Local Grid example.
//---------------------------------------------------------------------------//
void localGridExample()
{
    /*
      The local grid is a subset of the global grid, determined by the
      partitioning of the total system. Just as the global grid defines
      indexing for the entire domain, the local grid defines indexing for the
      local MPI rank only, as well as indexing of ghosted portions of the grid
      for MPI communication. The local grid is the "main" data class in the
      Cajita subpackage as application users will likely interact with it the
      most and it includes interfaces to all other grid/mesh classes.
    */

    // Here we partition only in x to simplify the example below.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> input_ranks_per_dim = { comm_size, 1, 1 };
    Cajita::ManualBlockPartitioner<3> partitioner( input_ranks_per_dim );

    // Create the global mesh.
    std::array<int, 3> global_num_cell = { 20, 10, 10 };
    std::array<double, 3> global_low_corner = { -2.0, -1.0, 1.0 };
    std::array<double, 3> global_high_corner = { 2.0, 0.0, 2.0 };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    // Get the current rank for printing output.
    int comm_rank = global_grid->blockId();
    if ( comm_rank == 0 )
    {
        std::cout << "Cajita Local Grid Example" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    /*
      We create the local grid from the global grid and a halo width -
      the number of cells communicated with MPI neighbor ranks. The halo width
      can be queried later if needed.
    */
    int halo_width = 2;
    auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );
    std::cout << "Minimum halo cell width: " << local_grid->haloCellWidth()
              << std::endl;

    /*
      The local grid also stores which MPI ranks are neighbors in the grid (26
      in 3D, 8 in 2D), using indexing relative to the current local domain. A
      self neighbor is possible, depending on the MPI decomposition and system
      periodicity.
    */
    std::cout << "Neighbor ranks for rank 0: ";
    if ( comm_rank == 0 )
    {
        for ( int i = -1; i < 2; ++i )
            for ( int j = -1; j < 2; ++j )
                for ( int k = -1; k < 2; ++k )
                {
                    std::cout << local_grid->neighborRank( i, j, k ) << " ";
                }
        std::cout << std::endl;
    }

    /*
      The local grid holds a number of index spaces (previous example) to
      facilitate operations on the owned and owned+ghosted grid.

      All options are set by the type tags described in the Cajita Types
      example:
       - Own vs Ghost here determines whether ghosted cells from neighbor ranks
         are included in the indexing. Note that Ghost returns BOTH owned and
         ghosted cells for a contiguous index space.
       - All geometric mesh entities are usable in the interface (Cell, Node,
         Face, Edge)
       - Local vs Global determines whether the indexing is relative (starting
        from zero in the current domain) or absolute (the current subset of the
        global domain). Local indices are useful for directly iterating over
        local arrays, while global indices are useful for things like boundary
        conditions.
    */
    auto own_local_cell_space = local_grid->indexSpace(
        Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    std::cout << "Index space (Own, Cell, Local):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << own_local_cell_space.min( d ) << " ";
    std::cout << std::endl << "Max: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << own_local_cell_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      Next we extract the owned and ghosted edges. Note here that for edges (as
      well as for faces) there is a template dimension parameter - there is a
      separate index space for each spatial dimension.
    */
    auto ghost_local_edge_space = local_grid->indexSpace(
        Cajita::Own(), Cajita::Edge<Cajita::Dim::I>(), Cajita::Local() );
    std::cout << "Index space  (Own, I-Edge, Local):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << ghost_local_edge_space.min( d ) << " ";
    std::cout << std::endl << "Max: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << ghost_local_edge_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      We finally create an index space across edges without ghosts with global
      indexing. Note that it is not possible to create a ghosted index space for
      the local grid with global indexing (because ghosts are neither unique nor
      considered in the global grid).
    */
    auto own_global_node_space = local_grid->indexSpace(
        Cajita::Own(), Cajita::Cell(), Cajita::Global() );
    std::cout << "Index space  (Own, I-Edge, Global):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << own_global_node_space.min( d ) << " ";
    std::cout << std::endl << "Max: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << own_global_node_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      It is possible to convert between local and global indicies if needed in
      the IndexConversion namespace.
    */

    /*
      The local grid can also create index spaces describing the overlapped
      region between two subdomains (MPI neighbors) due to ghosted regions,
      shared index spaces. These have similar options to the previous, but
      require an offset to a specific neighbor rank. Because it involves ghosted
      entities, shared index spaces always use local indexing.
    */
    auto owned_shared_cell_space =
        local_grid->sharedIndexSpace( Cajita::Own(), Cajita::Cell(), -1, 0, 1 );
    std::cout << "Shared index space (Own, Cell):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << owned_shared_cell_space.min( d ) << " ";
    std::cout << std::endl << "Max: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << owned_shared_cell_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      A halo width can be optionally passed (between zero and the halo width
      used to build the local grid) to reduce the shared space.
    */
    owned_shared_cell_space = local_grid->sharedIndexSpace(
        Cajita::Own(), Cajita::Cell(), -1, 0, 1, 1 );
    std::cout << "Shared index space (Own, Cell, halo_width=1) :\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << owned_shared_cell_space.min( d ) << " ";
    std::cout << std::endl << "Max: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << owned_shared_cell_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      Next, note the difference between a shared index space with owned entities
      only (above) and owned + ghosted.
    */
    auto ghost_shared_cell_space = local_grid->sharedIndexSpace(
        Cajita::Ghost(), Cajita::Cell(), -1, 0, 1 );
    std::cout << "Shared index space (Ghost, Cell):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << ghost_shared_cell_space.min( d ) << " ";
    std::cout << std::endl << "Max: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << ghost_shared_cell_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      Similarly, boundary index spaces enable iterating over only the boundary
      in a given direction. If the local subdomain is not on the boundary or the
      system is periodic (as in the example), the boundary index space is empty.
      In other words, the boundary index space contains the entities on the
      system boundaries that do not participate in halo communication.
    */
    auto boundary_cell_space = local_grid->boundaryIndexSpace(
        Cajita::Ghost(), Cajita::Cell(), -1, 0, 1 );
    std::cout << "Boundary index space (Ghost, Cell) size: "
              << boundary_cell_space.size() << std::endl;

    /*
      As a useful exercise, the periodicity of the global grid can be set to
      false in one or more dimensions and the example re-run to compare the
      differences in the local grid index spaces.
    */

    /*
      Also note that the local grid stores the global grid it was built from
      directly.
    */
    const auto global_grid_copy = local_grid->globalGrid();
    int num_blocks = global_grid_copy.totalNumBlock();
    std::cout << "Global grid copied (total blocks still " << num_blocks << ")"
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

        localGridExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
