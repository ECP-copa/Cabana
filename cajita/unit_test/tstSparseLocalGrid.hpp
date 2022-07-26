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
#include <Cajita_DynamicPartitioner.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_SparseLocalGrid.hpp>
#include <Cajita_Types.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <numeric>

using namespace Cajita;
using namespace Cajita::Experimental;

namespace Test
{
template <typename EntityType>
void sparseLocalGridTest( EntityType t2 )
{
    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 16, 32, 64 };
    int cell_num_per_tile_dim = 4;
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh_ptr = Cajita::createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create and initialize sparse partitioner
    std::array<bool, 3> periodic = { false, false, false };
    DynamicPartitioner<TEST_DEVICE, 4> partitioner( MPI_COMM_WORLD,
                                                    global_num_cell, 10 );

    // Create global grid
    auto global_grid_ptr = Cajita::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh_ptr, periodic, partitioner );

    // Create a local grid.
    int halo_width = 3;
    auto local_grid_ptr = Cajita::Experimental::createSparseLocalGrid(
        global_grid_ptr, halo_width, cell_num_per_tile_dim );

    // Check sizes - constructor should correct the halo_width to
    // k*cell_num_per_tile_dim
    EXPECT_EQ( local_grid_ptr->haloCellWidth(), 4 );
    EXPECT_EQ( local_grid_ptr->haloTileWidth(), 1 );
    halo_width = 4;

    // Get the local number of cells.
    auto owned_cell_space_local =
        local_grid_ptr->indexSpace( Own(), t2, Local() );
    std::vector<int> local_num_cells( 3 );
    for ( int d = 0; d < 3; ++d )
        local_num_cells[d] = owned_cell_space_local.extent( d );

    // Compute a global set of local cell size arrays.
    auto grid_comm = global_grid_ptr->comm();
    std::vector<int> cart_dims( 3 );
    std::vector<int> cart_period( 3 );
    std::vector<int> cart_rank( 3 );
    MPI_Cart_get( grid_comm, 3, cart_dims.data(), cart_period.data(),
                  cart_rank.data() );

    std::vector<int> local_num_cell_i( cart_dims[Dim::I], 0 );
    std::vector<int> local_num_cell_j( cart_dims[Dim::J], 0 );
    std::vector<int> local_num_cell_k( cart_dims[Dim::K], 0 );

    local_num_cell_i[cart_rank[Dim::I]] = local_num_cells[Dim::I];
    local_num_cell_j[cart_rank[Dim::J]] = local_num_cells[Dim::J];
    local_num_cell_k[cart_rank[Dim::K]] = local_num_cells[Dim::K];

    MPI_Allreduce( MPI_IN_PLACE, local_num_cell_i.data(), cart_dims[Dim::I],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_cell_j.data(), cart_dims[Dim::J],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_cell_k.data(), cart_dims[Dim::K],
                   MPI_INT, MPI_MAX, grid_comm );

    // Check the neighbor rank
    for ( int i = -1; i < 2; ++i )
        for ( int j = -1; j < 2; ++j )
            for ( int k = -1; k < 2; ++k )
            {
                std::vector<int> nr = { cart_rank[Dim::I] + i,
                                        cart_rank[Dim::J] + j,
                                        cart_rank[Dim::K] + k };
                if ( nr[Dim::I] < 0 || nr[Dim::I] >= cart_dims[Dim::I] ||
                     nr[Dim::J] < 0 || nr[Dim::J] >= cart_dims[Dim::J] ||
                     nr[Dim::K] < 0 || nr[Dim::K] >= cart_dims[Dim::K] )
                    continue;
                int nrank;
                MPI_Cart_rank( grid_comm, nr.data(), &nrank );
                EXPECT_EQ( local_grid_ptr->neighborRank( i, j, k ), nrank );
            }

    // Check to make sure we got the right number of total cells in each
    // dimension.
    EXPECT_EQ( global_num_cell[Dim::I],
               std::accumulate( local_num_cell_i.begin(),
                                local_num_cell_i.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::J],
               std::accumulate( local_num_cell_j.begin(),
                                local_num_cell_j.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::K],
               std::accumulate( local_num_cell_k.begin(),
                                local_num_cell_k.end(), 0 ) );

    // Check the local cell bounds.
    EXPECT_EQ( owned_cell_space_local.min( Dim::I ), halo_width );
    EXPECT_EQ( owned_cell_space_local.max( Dim::I ),
               local_num_cells[Dim::I] + halo_width );
    EXPECT_EQ( owned_cell_space_local.min( Dim::J ), halo_width );
    EXPECT_EQ( owned_cell_space_local.max( Dim::J ),
               local_num_cells[Dim::J] + halo_width );
    EXPECT_EQ( owned_cell_space_local.min( Dim::K ), halo_width );
    EXPECT_EQ( owned_cell_space_local.max( Dim::K ),
               local_num_cells[Dim::K] + halo_width );

    // Check the global owned cell bounds.
    auto global_owned_cell_space =
        local_grid_ptr->indexSpace( Own(), t2, Global() );
    EXPECT_EQ( global_owned_cell_space.min( Dim::I ),
               global_grid_ptr->globalOffset( Dim::I ) );
    EXPECT_EQ( global_owned_cell_space.max( Dim::I ),
               global_grid_ptr->globalOffset( Dim::I ) +
                   local_num_cells[Dim::I] );
    EXPECT_EQ( global_owned_cell_space.min( Dim::J ),
               global_grid_ptr->globalOffset( Dim::J ) );
    EXPECT_EQ( global_owned_cell_space.max( Dim::J ),
               global_grid_ptr->globalOffset( Dim::J ) +
                   local_num_cells[Dim::J] );
    EXPECT_EQ( global_owned_cell_space.min( Dim::K ),
               global_grid_ptr->globalOffset( Dim::K ) );
    EXPECT_EQ( global_owned_cell_space.max( Dim::K ),
               global_grid_ptr->globalOffset( Dim::K ) +
                   local_num_cells[Dim::K] );

    // Check the ghosted cell bounds.
    auto ghosted_cell_space =
        local_grid_ptr->indexSpace( Ghost(), t2, Local() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( ghosted_cell_space.extent( d ),
                   owned_cell_space_local.extent( d ) + 2 * halo_width );
    }

    // Check the cells we own that we will share with our neighbors. Cover
    // enough of the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto owned_cell_space = local_grid_ptr->indexSpace( Own(), t2, Global() );
    if ( cart_rank[Dim::I] > 0 )
    {
        auto owned_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Own(), t2, -1, 0, 0 );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) + halo_width );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) );
    }
    if ( cart_rank[Dim::J] > 0 )
    {

        auto owned_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Own(), t2, 0, -1, 0 );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) + halo_width );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) );
    }
    if ( cart_rank[Dim::K] > 0 )
    {
        auto owned_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Own(), t2, 0, 0, -1 );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) + halo_width );
    }

    if ( cart_rank[Dim::I] + 1 < cart_dims[Dim::I] )
    {
        auto owned_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Own(), t2, 1, 0, 0 );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) - halo_width );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) );
    }
    if ( cart_rank[Dim::J] + 1 < cart_dims[Dim::J] )
    {

        auto owned_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Own(), t2, 0, 1, 0 );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) - halo_width );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) );
    }
    if ( cart_rank[Dim::K] + 1 < cart_dims[Dim::K] )
    {
        auto owned_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Own(), t2, 0, 0, 1 );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) );
        EXPECT_EQ( owned_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) - halo_width );
        EXPECT_EQ( owned_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) );
    }

    // Check the tiles are ghosts that our neighbors own. Cover enough of the
    // neighbors that we know the bounds are correct in each dimension. The
    // three variations here cover all of the cases.
    if ( cart_rank[Dim::I] > 0 )
    {
        auto ghosted_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Ghost(), t2, -1, 0, 0 );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) - halo_width );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) );
    }
    if ( cart_rank[Dim::J] > 0 )
    {
        auto ghosted_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Ghost(), t2, 0, -1, 0 );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) - halo_width );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) );
    }
    if ( cart_rank[Dim::K] > 0 )
    {
        auto ghosted_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Ghost(), t2, 0, 0, -1 );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) - halo_width );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) );
    }

    if ( cart_rank[Dim::I] + 1 < cart_dims[Dim::I] )
    {
        auto ghosted_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Ghost(), t2, 1, 0, 0 );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) + halo_width );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) );
    }
    if ( cart_rank[Dim::J] + 1 < cart_dims[Dim::J] )
    {
        auto ghosted_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Ghost(), t2, 0, 1, 0 );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) + halo_width );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::K ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) );
    }
    if ( cart_rank[Dim::K] + 1 < cart_dims[Dim::K] )
    {
        auto ghosted_shared_tile_space =
            local_grid_ptr->sharedTileIndexSpace<2>( Ghost(), t2, 0, 0, 1 );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::I ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::I ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::I ) );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.min( Dim::J ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::J ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::J ) );
        EXPECT_EQ( ghosted_shared_tile_space.min( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) );
        EXPECT_EQ( ghosted_shared_tile_space.max( Dim::K ) *
                       cell_num_per_tile_dim,
                   owned_cell_space.max( Dim::K ) + halo_width );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( sparse_local_grid, 3d_sprase_local_grid )
{
    // Currently, periodic BC is not supported in Sparse Grid
    sparseLocalGridTest( Cell() );
    sparseLocalGridTest( Node() );
}

//---------------------------------------------------------------------------//
} // namespace Test