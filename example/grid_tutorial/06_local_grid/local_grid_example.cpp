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
      Cabana::Grid subpackage as application users will likely interact with it
      the most and it includes interfaces to all other grid/mesh classes.
    */

    // Here we partition only in z to simplify the example below.
    Cabana::Grid::DimBlockPartitioner<3> partitioner( Cabana::Grid::Dim::I,
                                                      Cabana::Grid::Dim::J );

    // Create the global mesh.
    std::array<int, 3> global_num_cell = { 20, 10, 10 };
    std::array<double, 3> global_low_corner = { -2.0, -1.0, 1.0 };
    std::array<double, 3> global_high_corner = { 2.0, 0.0, 2.0 };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Get the current rank for printing output.
    int comm_rank = global_grid->blockId();
    if ( comm_rank == 0 )
    {
        std::cout << "Cabana::Grid Local Grid Example" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    /*
      We create the local grid from the global grid and a halo width -
      the number of cells communicated with MPI neighbor ranks. The halo width
      can be queried later if needed.
    */
    int halo_width = 2;
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );
    std::cout << "Minimum halo cell width: " << local_grid->haloCellWidth()
              << "\n"
              << std::endl;

    /*
      Also note that the local grid stores the global grid it was built from.
    */
    const auto global_grid_copy = local_grid->globalGrid();
    int num_blocks = global_grid_copy.totalNumBlock();
    std::cout << "Global grid copied (total blocks still " << num_blocks
              << ")\n"
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
        std::cout << "\n" << std::endl;
    }

    /*
      The local grid holds a number of index spaces (previous example) to
      facilitate operations on the owned and owned+ghosted grid.

      All options are set by the type tags described in the Cabana::Grid Types
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
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

    /*
      The index spaces include the upper and lower bounds, as well as the total
      size (across all dimensions).
    */
    std::cout << "Index space (Own, Cell, Local):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << own_local_cell_space.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << own_local_cell_space.max( d ) << " ";
    std::cout << "\nSize: ";
    std::cout << own_local_cell_space.size() << " ";
    std::cout << "\n" << std::endl;

    /*
      Next we extract the owned and ghosted edges. Note here that for edges (as
      well as for faces) there is a template dimension parameter - there is a
      separate index space for each spatial dimension.
    */
    auto ghost_local_edge_space = local_grid->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Edge<Cabana::Grid::Dim::I>(),
        Cabana::Grid::Local() );
    std::cout << "Index space  (Own, I-Edge, Local):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << ghost_local_edge_space.min( d ) << " ";
    std::cout << "\nMax: ";
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
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Global() );
    std::cout << "Index space  (Own, I-Edge, Global):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << own_global_node_space.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << own_global_node_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      It is possible to convert between local and global indices if needed in
      the IndexConversion namespace.
    */

    /*
      The local grid can also create index spaces describing the overlapped
      region between two subdomains (MPI neighbors) due to ghosted regions,
      shared index spaces. These have similar options to the previous, but
      require an offset to a specific neighbor rank. Because it involves ghosted
      entities, shared index spaces always use local indexing.
    */
    auto owned_shared_cell_space = local_grid->sharedIndexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), -1, 0, 1 );
    std::cout << "Shared index space (Own, Cell):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << owned_shared_cell_space.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << owned_shared_cell_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      A halo width can be optionally passed (between zero and the halo width
      used to build the local grid) to reduce the shared space.
    */
    owned_shared_cell_space = local_grid->sharedIndexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), -1, 0, 1, 1 );
    std::cout << "Shared index space (Own, Cell, halo_width=1):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << owned_shared_cell_space.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << owned_shared_cell_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      Next, note the difference between a shared index space with owned entities
      only (above) and owned + ghosted.
    */
    auto ghost_shared_cell_space = local_grid->sharedIndexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Cell(), -1, 0, 1 );
    std::cout << "Shared index space (Ghost, Cell):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << ghost_shared_cell_space.min( d ) << " ";
    std::cout << "\nMax: ";
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
        Cabana::Grid::Ghost(), Cabana::Grid::Cell(), -1, 0, 1 );
    std::cout << "Boundary index space (Ghost, Cell) size: "
              << boundary_cell_space.size() << "\n"
              << std::endl;

    /*
      We now create a partially non-periodic global grid to highlight some
      details of the local grid boundary index spaces.
    */
    is_dim_periodic = { true, true, false };
    auto non_periodic_global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );
    auto non_periodic_local_grid =
        Cabana::Grid::createLocalGrid( non_periodic_global_grid, halo_width );

    /*
      For the non-periodic dimension with Cell, the ghosted boundaryIndexSpace,
      is equivalent to the sharedIndexSpace: indexing over a boundary is the
      same as an MPI neighbor in this case (and uses the same halo width).
    */
    auto non_periodic_boundary = non_periodic_local_grid->boundaryIndexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Cell(), -1, 0, 1 );
    std::cout << "Non-periodic Boundary index space (Ghost, Cell):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << non_periodic_boundary.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << non_periodic_boundary.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      The owned boundaryIndexSpace is the opposite - the cells within the halo
      width, but within the owned boundary rather than outside of it.
    */
    non_periodic_boundary = non_periodic_local_grid->boundaryIndexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), -1, 0, 1 );
    std::cout << "Non-periodic Boundary index space (Own, Cell):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << non_periodic_boundary.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << non_periodic_boundary.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      For Nodes, Faces, and Edges, the ghosted boundaryIndexSpace is still equal
      to the halo width, but shifted as compared to Cells (which sit on the cell
      centers).
    */
    non_periodic_boundary = non_periodic_local_grid->boundaryIndexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Node(), -1, 0, 1 );
    std::cout << "Non-periodic Boundary index space (Ghost, Node):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << non_periodic_boundary.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << non_periodic_boundary.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      For Nodes, Faces, and Edges, the ghosted boundaryIndexSpace can be subtly
      different than the sharedIndexSpace. This is again because an entity is
      always uniquely owned, but Nodes, Faces, and Edges sit directly between
      MPI ranks (where there is symmetry between which rank owns and ghosts), as
      well as on the (non-periodic) system boundaries where there is only an
      owning MPI rank.
    */
    non_periodic_boundary = non_periodic_local_grid->boundaryIndexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Node(), -1, 0, 1 );
    std::cout << "Non-periodic Boundary index space (Own, Node):\nMin: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << non_periodic_boundary.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << non_periodic_boundary.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      As an exercise, try creating index spaces with Faces or Edges and compare
      to Cells and Nodes. Note in particular that for these entities the
      index spaces will be different in dimensions that do or do not match the
      direction of the entity (e.g. Dim::I vs Dim::J index spaces for
      Face<Dim::I> ).
    */
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
