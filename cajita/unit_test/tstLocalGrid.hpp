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

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <numeric>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void periodicTest()
{
    // Let MPI compute the partitioning for this test.
    UniformDimPartitioner partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 101, 85, 99 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid.
    int halo_width = 2;
    auto local_grid = createLocalGrid( global_grid, halo_width );

    // Check sizes
    EXPECT_EQ( local_grid->haloCellWidth(), halo_width );

    //////////////////
    // CELL SPACES
    //////////////////

    // Get the local number of cells.
    auto owned_cell_space = local_grid->indexSpace( Own(), Cell(), Local() );
    std::vector<int> local_num_cells( 3 );
    for ( int d = 0; d < 3; ++d )
        local_num_cells[d] = owned_cell_space.extent( d );

    // Compute a global set of local cell size arrays.
    auto grid_comm = global_grid->comm();
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
                int nrank;
                MPI_Cart_rank( grid_comm, nr.data(), &nrank );
                EXPECT_EQ( local_grid->neighborRank( i, j, k ), nrank );
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
    EXPECT_EQ( owned_cell_space.min( Dim::I ), halo_width );
    EXPECT_EQ( owned_cell_space.max( Dim::I ),
               local_num_cells[Dim::I] + halo_width );
    EXPECT_EQ( owned_cell_space.min( Dim::J ), halo_width );
    EXPECT_EQ( owned_cell_space.max( Dim::J ),
               local_num_cells[Dim::J] + halo_width );
    EXPECT_EQ( owned_cell_space.min( Dim::K ), halo_width );
    EXPECT_EQ( owned_cell_space.max( Dim::K ),
               local_num_cells[Dim::K] + halo_width );

    // Check the global owned cell bounds.
    auto global_owned_cell_space =
        local_grid->indexSpace( Own(), Cell(), Global() );
    EXPECT_EQ( global_owned_cell_space.min( Dim::I ),
               global_grid->globalOffset( Dim::I ) );
    EXPECT_EQ( global_owned_cell_space.max( Dim::I ),
               global_grid->globalOffset( Dim::I ) + local_num_cells[Dim::I] );
    EXPECT_EQ( global_owned_cell_space.min( Dim::J ),
               global_grid->globalOffset( Dim::J ) );
    EXPECT_EQ( global_owned_cell_space.max( Dim::J ),
               global_grid->globalOffset( Dim::J ) + local_num_cells[Dim::J] );
    EXPECT_EQ( global_owned_cell_space.min( Dim::K ),
               global_grid->globalOffset( Dim::K ) );
    EXPECT_EQ( global_owned_cell_space.max( Dim::K ),
               global_grid->globalOffset( Dim::K ) + local_num_cells[Dim::K] );

    // Check the ghosted cell bounds.
    auto ghosted_cell_space =
        local_grid->indexSpace( Ghost(), Cell(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( ghosted_cell_space.extent( d ),
                   owned_cell_space.extent( d ) + 2 * halo_width );
    }

    // Check the cells we own that we will share with our neighbors. Cover
    // enough of the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto owned_shared_cell_space =
        local_grid->sharedIndexSpace( Own(), Cell(), -1, 0, 1 );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::I ),
               owned_cell_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::I ),
               owned_cell_space.min( Dim::I ) + halo_width );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::J ),
               owned_cell_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::J ),
               owned_cell_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::K ),
               owned_cell_space.max( Dim::K ) - halo_width );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::K ),
               owned_cell_space.max( Dim::K ) );

    owned_shared_cell_space =
        local_grid->sharedIndexSpace( Own(), Cell(), 1, -1, 0 );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::I ),
               owned_cell_space.max( Dim::I ) - halo_width );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::I ),
               owned_cell_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::J ),
               owned_cell_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::J ),
               owned_cell_space.min( Dim::J ) + halo_width );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::K ),
               owned_cell_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::K ),
               owned_cell_space.max( Dim::K ) );

    owned_shared_cell_space =
        local_grid->sharedIndexSpace( Own(), Cell(), 0, 1, -1 );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::I ),
               owned_cell_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::I ),
               owned_cell_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::J ),
               owned_cell_space.max( Dim::J ) - halo_width );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::J ),
               owned_cell_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::K ),
               owned_cell_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::K ),
               owned_cell_space.min( Dim::K ) + halo_width );

    // Check owned shared cell spaces again but this time with a specified
    // halo width.
    owned_shared_cell_space =
        local_grid->sharedIndexSpace( Own(), Cell(), -1, 0, 1, 1 );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::I ),
               owned_cell_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::I ),
               owned_cell_space.min( Dim::I ) + 1 );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::J ),
               owned_cell_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::J ),
               owned_cell_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::K ),
               owned_cell_space.max( Dim::K ) - 1 );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::K ),
               owned_cell_space.max( Dim::K ) );

    owned_shared_cell_space =
        local_grid->sharedIndexSpace( Own(), Cell(), 1, -1, 0, 1 );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::I ),
               owned_cell_space.max( Dim::I ) - 1 );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::I ),
               owned_cell_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::J ),
               owned_cell_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::J ),
               owned_cell_space.min( Dim::J ) + 1 );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::K ),
               owned_cell_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::K ),
               owned_cell_space.max( Dim::K ) );

    owned_shared_cell_space =
        local_grid->sharedIndexSpace( Own(), Cell(), 0, 1, -1, 1 );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::I ),
               owned_cell_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::I ),
               owned_cell_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::J ),
               owned_cell_space.max( Dim::J ) - 1 );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::J ),
               owned_cell_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_cell_space.min( Dim::K ),
               owned_cell_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_cell_space.max( Dim::K ),
               owned_cell_space.min( Dim::K ) + 1 );

    // Check the cells are ghosts that our neighbors own. Cover enough of the
    // neighbors that we know the bounds are correct in each dimension. The
    // three variations here cover all of the cases.
    auto ghosted_shared_cell_space =
        local_grid->sharedIndexSpace( Ghost(), Cell(), -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::I ), 0 );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::I ), halo_width );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::J ),
               owned_cell_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::J ),
               owned_cell_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::K ),
               owned_cell_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::K ),
               owned_cell_space.max( Dim::K ) + halo_width );

    ghosted_shared_cell_space =
        local_grid->sharedIndexSpace( Ghost(), Cell(), 1, -1, 0 );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::I ),
               owned_cell_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::I ),
               owned_cell_space.max( Dim::I ) + halo_width );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::J ), 0 );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::J ), halo_width );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::K ),
               owned_cell_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::K ),
               owned_cell_space.max( Dim::K ) );

    ghosted_shared_cell_space =
        local_grid->sharedIndexSpace( Ghost(), Cell(), 0, 1, -1 );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::I ),
               owned_cell_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::I ),
               owned_cell_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::J ),
               owned_cell_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::J ),
               owned_cell_space.max( Dim::J ) + halo_width );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::K ), 0 );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::K ), halo_width );

    // Check the ghosted shared cell spaces again but this time with a
    // specified halo width.
    ghosted_shared_cell_space =
        local_grid->sharedIndexSpace( Ghost(), Cell(), -1, 0, 1, 1 );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::I ), halo_width - 1 );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::I ), halo_width );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::J ),
               owned_cell_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::J ),
               owned_cell_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::K ),
               owned_cell_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::K ),
               owned_cell_space.max( Dim::K ) + 1 );

    ghosted_shared_cell_space =
        local_grid->sharedIndexSpace( Ghost(), Cell(), 1, -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::I ),
               owned_cell_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::I ),
               owned_cell_space.max( Dim::I ) + 1 );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::J ), halo_width - 1 );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::J ), halo_width );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::K ),
               owned_cell_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::K ),
               owned_cell_space.max( Dim::K ) );

    ghosted_shared_cell_space =
        local_grid->sharedIndexSpace( Ghost(), Cell(), 0, 1, -1, 1 );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::I ),
               owned_cell_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::I ),
               owned_cell_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::J ),
               owned_cell_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::J ),
               owned_cell_space.max( Dim::J ) + 1 );
    EXPECT_EQ( ghosted_shared_cell_space.min( Dim::K ), halo_width - 1 );
    EXPECT_EQ( ghosted_shared_cell_space.max( Dim::K ), halo_width );

    //////////////////
    // NODE SPACES
    //////////////////

    // Get the local number of nodes.
    auto owned_node_space = local_grid->indexSpace( Own(), Node(), Local() );
    std::vector<int> local_num_nodes( 3 );
    for ( int d = 0; d < 3; ++d )
        local_num_nodes[d] = owned_node_space.extent( d );

    // Compute a global set of local node size arrays.
    std::vector<int> local_num_node_i( cart_dims[Dim::I], 0 );
    std::vector<int> local_num_node_j( cart_dims[Dim::J], 0 );
    std::vector<int> local_num_node_k( cart_dims[Dim::K], 0 );
    local_num_node_i[cart_rank[Dim::I]] = local_num_nodes[Dim::I];
    local_num_node_j[cart_rank[Dim::J]] = local_num_nodes[Dim::J];
    local_num_node_k[cart_rank[Dim::K]] = local_num_nodes[Dim::K];
    MPI_Allreduce( MPI_IN_PLACE, local_num_node_i.data(), cart_dims[Dim::I],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_node_j.data(), cart_dims[Dim::J],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_node_k.data(), cart_dims[Dim::K],
                   MPI_INT, MPI_MAX, grid_comm );

    // Check to make sure we got the right number of total nodes in each
    // dimension.
    EXPECT_EQ( global_num_cell[Dim::I],
               std::accumulate( local_num_node_i.begin(),
                                local_num_node_i.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::J],
               std::accumulate( local_num_node_j.begin(),
                                local_num_node_j.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::K],
               std::accumulate( local_num_node_k.begin(),
                                local_num_node_k.end(), 0 ) );

    // Check the local node bounds.
    EXPECT_EQ( owned_node_space.min( Dim::I ), halo_width );
    EXPECT_EQ( owned_node_space.max( Dim::I ),
               local_num_nodes[Dim::I] + halo_width );
    EXPECT_EQ( owned_node_space.min( Dim::J ), halo_width );
    EXPECT_EQ( owned_node_space.max( Dim::J ),
               local_num_nodes[Dim::J] + halo_width );
    EXPECT_EQ( owned_node_space.min( Dim::K ), halo_width );
    EXPECT_EQ( owned_node_space.max( Dim::K ),
               local_num_nodes[Dim::K] + halo_width );

    // Check the global node bounds.
    auto global_owned_node_space =
        local_grid->indexSpace( Own(), Node(), Global() );
    EXPECT_EQ( global_owned_node_space.min( Dim::I ),
               global_grid->globalOffset( Dim::I ) );
    EXPECT_EQ( global_owned_node_space.max( Dim::I ),
               global_grid->globalOffset( Dim::I ) + local_num_nodes[Dim::I] );
    EXPECT_EQ( global_owned_node_space.min( Dim::J ),
               global_grid->globalOffset( Dim::J ) );
    EXPECT_EQ( global_owned_node_space.max( Dim::J ),
               global_grid->globalOffset( Dim::J ) + local_num_nodes[Dim::J] );
    EXPECT_EQ( global_owned_node_space.min( Dim::K ),
               global_grid->globalOffset( Dim::K ) );
    EXPECT_EQ( global_owned_node_space.max( Dim::K ),
               global_grid->globalOffset( Dim::K ) + local_num_nodes[Dim::K] );

    // Check the ghosted node bounds.
    auto ghosted_node_space =
        local_grid->indexSpace( Ghost(), Node(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( ghosted_node_space.extent( d ),
                   owned_node_space.extent( d ) + 2 * halo_width + 1 );
    }

    // Check the nodes we own that we will share with our neighbors. Cover
    // enough of the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto owned_shared_node_space =
        local_grid->sharedIndexSpace( Own(), Node(), -1, 0, 1 );
    EXPECT_EQ( owned_shared_node_space.min( Dim::I ),
               owned_node_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::I ),
               owned_node_space.min( Dim::I ) + halo_width + 1 );
    EXPECT_EQ( owned_shared_node_space.min( Dim::J ),
               owned_node_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::J ),
               owned_node_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_node_space.min( Dim::K ),
               owned_node_space.max( Dim::K ) - halo_width );
    EXPECT_EQ( owned_shared_node_space.max( Dim::K ),
               owned_node_space.max( Dim::K ) );

    owned_shared_node_space =
        local_grid->sharedIndexSpace( Own(), Node(), 1, -1, 0 );
    EXPECT_EQ( owned_shared_node_space.min( Dim::I ),
               owned_node_space.max( Dim::I ) - halo_width );
    EXPECT_EQ( owned_shared_node_space.max( Dim::I ),
               owned_node_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_node_space.min( Dim::J ),
               owned_node_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::J ),
               owned_node_space.min( Dim::J ) + halo_width + 1 );
    EXPECT_EQ( owned_shared_node_space.min( Dim::K ),
               owned_node_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::K ),
               owned_node_space.max( Dim::K ) );

    owned_shared_node_space =
        local_grid->sharedIndexSpace( Own(), Node(), 0, 1, -1 );
    EXPECT_EQ( owned_shared_node_space.min( Dim::I ),
               owned_node_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::I ),
               owned_node_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_node_space.min( Dim::J ),
               owned_node_space.max( Dim::J ) - halo_width );
    EXPECT_EQ( owned_shared_node_space.max( Dim::J ),
               owned_node_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_node_space.min( Dim::K ),
               owned_node_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::K ),
               owned_node_space.min( Dim::K ) + halo_width + 1 );

    // Check the owned shared node spaces again but this time with a specified
    // halo width.
    owned_shared_node_space =
        local_grid->sharedIndexSpace( Own(), Node(), -1, 0, 1, 1 );
    EXPECT_EQ( owned_shared_node_space.min( Dim::I ),
               owned_node_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::I ),
               owned_node_space.min( Dim::I ) + 2 );
    EXPECT_EQ( owned_shared_node_space.min( Dim::J ),
               owned_node_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::J ),
               owned_node_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_node_space.min( Dim::K ),
               owned_node_space.max( Dim::K ) - 1 );
    EXPECT_EQ( owned_shared_node_space.max( Dim::K ),
               owned_node_space.max( Dim::K ) );

    owned_shared_node_space =
        local_grid->sharedIndexSpace( Own(), Node(), 1, -1, 0, 1 );
    EXPECT_EQ( owned_shared_node_space.min( Dim::I ),
               owned_node_space.max( Dim::I ) - 1 );
    EXPECT_EQ( owned_shared_node_space.max( Dim::I ),
               owned_node_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_node_space.min( Dim::J ),
               owned_node_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::J ),
               owned_node_space.min( Dim::J ) + 2 );
    EXPECT_EQ( owned_shared_node_space.min( Dim::K ),
               owned_node_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::K ),
               owned_node_space.max( Dim::K ) );

    owned_shared_node_space =
        local_grid->sharedIndexSpace( Own(), Node(), 0, 1, -1, 1 );
    EXPECT_EQ( owned_shared_node_space.min( Dim::I ),
               owned_node_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::I ),
               owned_node_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_node_space.min( Dim::J ),
               owned_node_space.max( Dim::J ) - 1 );
    EXPECT_EQ( owned_shared_node_space.max( Dim::J ),
               owned_node_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_node_space.min( Dim::K ),
               owned_node_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_node_space.max( Dim::K ),
               owned_node_space.min( Dim::K ) + 2 );

    // Check the nodes are ghosts that our neighbors own. Cover enough of the
    // neighbors that we know the bounds are correct in each dimension. The
    // three variations here cover all of the cases.
    auto ghosted_shared_node_space =
        local_grid->sharedIndexSpace( Ghost(), Node(), -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::I ), 0 );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::I ), halo_width );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::J ),
               owned_node_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::J ),
               owned_node_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::K ),
               owned_node_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::K ),
               owned_node_space.max( Dim::K ) + halo_width + 1 );

    ghosted_shared_node_space =
        local_grid->sharedIndexSpace( Ghost(), Node(), 1, -1, 0 );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::I ),
               owned_node_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::I ),
               owned_node_space.max( Dim::I ) + halo_width + 1 );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::J ), 0 );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::J ), halo_width );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::K ),
               owned_node_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::K ),
               owned_node_space.max( Dim::K ) );

    ghosted_shared_node_space =
        local_grid->sharedIndexSpace( Ghost(), Node(), 0, 1, -1 );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::I ),
               owned_node_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::I ),
               owned_node_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::J ),
               owned_node_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::J ),
               owned_node_space.max( Dim::J ) + halo_width + 1 );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::K ), 0 );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::K ), halo_width );

    // Check the ghosted shared node spaces again - this time with a specified
    // halo width.
    ghosted_shared_node_space =
        local_grid->sharedIndexSpace( Ghost(), Node(), -1, 0, 1, 1 );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::I ),
               owned_node_space.min( Dim::I ) - 1 );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::I ),
               owned_node_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::J ),
               owned_node_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::J ),
               owned_node_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::K ),
               owned_node_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::K ),
               owned_node_space.max( Dim::K ) + 2 );

    ghosted_shared_node_space =
        local_grid->sharedIndexSpace( Ghost(), Node(), 1, -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::I ),
               owned_node_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::I ),
               owned_node_space.max( Dim::I ) + 2 );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::J ),
               owned_node_space.min( Dim::J ) - 1 );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::J ),
               owned_node_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::K ),
               owned_node_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::K ),
               owned_node_space.max( Dim::K ) );

    ghosted_shared_node_space =
        local_grid->sharedIndexSpace( Ghost(), Node(), 0, 1, -1, 1 );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::I ),
               owned_node_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::I ),
               owned_node_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::J ),
               owned_node_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::J ),
               owned_node_space.max( Dim::J ) + 2 );
    EXPECT_EQ( ghosted_shared_node_space.min( Dim::K ),
               owned_node_space.min( Dim::K ) - 1 );
    EXPECT_EQ( ghosted_shared_node_space.max( Dim::K ),
               owned_node_space.min( Dim::K ) );

    //////////////////
    // I-FACE SPACES
    //////////////////

    // Get the local number of I-faces.
    auto owned_i_face_space =
        local_grid->indexSpace( Own(), Face<Dim::I>(), Local() );
    std::vector<int> local_num_i_faces( 3 );
    for ( int d = 0; d < 3; ++d )
        local_num_i_faces[d] = owned_i_face_space.extent( d );

    // Compute a global set of local I-face size arrays.
    std::vector<int> local_num_i_face_i( cart_dims[Dim::I], 0 );
    std::vector<int> local_num_i_face_j( cart_dims[Dim::J], 0 );
    std::vector<int> local_num_i_face_k( cart_dims[Dim::K], 0 );
    local_num_i_face_i[cart_rank[Dim::I]] = local_num_i_faces[Dim::I];
    local_num_i_face_j[cart_rank[Dim::J]] = local_num_i_faces[Dim::J];
    local_num_i_face_k[cart_rank[Dim::K]] = local_num_i_faces[Dim::K];
    MPI_Allreduce( MPI_IN_PLACE, local_num_i_face_i.data(), cart_dims[Dim::I],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_i_face_j.data(), cart_dims[Dim::J],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_i_face_k.data(), cart_dims[Dim::K],
                   MPI_INT, MPI_MAX, grid_comm );

    // Check to make sure we got the right number of total I-faces in each
    // dimension.
    EXPECT_EQ( global_num_cell[Dim::I],
               std::accumulate( local_num_i_face_i.begin(),
                                local_num_i_face_i.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::J],
               std::accumulate( local_num_i_face_j.begin(),
                                local_num_i_face_j.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::K],
               std::accumulate( local_num_i_face_k.begin(),
                                local_num_i_face_k.end(), 0 ) );

    // Check the global bounds.
    auto global_owned_i_face_space =
        local_grid->indexSpace( Own(), Face<Dim::I>(), Global() );
    EXPECT_EQ( global_owned_i_face_space.min( Dim::I ),
               global_grid->globalOffset( Dim::I ) );
    EXPECT_EQ( global_owned_i_face_space.max( Dim::I ),
               global_grid->globalOffset( Dim::I ) + local_num_nodes[Dim::I] );
    EXPECT_EQ( global_owned_i_face_space.min( Dim::J ),
               global_grid->globalOffset( Dim::J ) );
    EXPECT_EQ( global_owned_i_face_space.max( Dim::J ),
               global_grid->globalOffset( Dim::J ) + local_num_cells[Dim::J] );
    EXPECT_EQ( global_owned_i_face_space.min( Dim::K ),
               global_grid->globalOffset( Dim::K ) );
    EXPECT_EQ( global_owned_i_face_space.max( Dim::K ),
               global_grid->globalOffset( Dim::K ) + local_num_cells[Dim::K] );

    // Check the I-faces we own that we will share with our neighbors. Cover
    // enough of the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto owned_shared_i_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::I>(), -1, 0, 1 );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.min( Dim::I ) + halo_width + 1 );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.max( Dim::K ) - halo_width );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.max( Dim::K ) );

    owned_shared_i_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::I>(), 1, -1, 0 );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.max( Dim::I ) - halo_width );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.min( Dim::J ) + halo_width );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.max( Dim::K ) );

    owned_shared_i_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::I>(), 0, 1, -1 );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.max( Dim::J ) - halo_width );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.min( Dim::K ) + halo_width );

    // Check the I-face owned shared spaces again, this time with a specified
    // halo width.
    owned_shared_i_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::I>(), -1, 0, 1, 1 );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.min( Dim::I ) + 2 );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.max( Dim::K ) - 1 );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.max( Dim::K ) );

    owned_shared_i_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::I>(), 1, -1, 0, 1 );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.max( Dim::I ) - 1 );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.min( Dim::J ) + 1 );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.max( Dim::K ) );

    owned_shared_i_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::I>(), 0, 1, -1, 1 );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.max( Dim::J ) - 1 );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.min( Dim::K ) + 1 );

    // Check the I-faces are ghosts that our neighbors own. Cover enough of
    // the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto ghosted_shared_i_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::I>(), -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::I ), 0 );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::I ), halo_width );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.max( Dim::K ) + halo_width );

    ghosted_shared_i_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::I>(), 1, -1, 0 );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.max( Dim::I ) + halo_width + 1 );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::J ), 0 );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::J ), halo_width );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.max( Dim::K ) );

    ghosted_shared_i_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::I>(), 0, 1, -1 );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.max( Dim::J ) + halo_width );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::K ), 0 );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::K ), halo_width );

    // Check the I-face ghosted shared spaces again, this time with a
    // specified halo width.
    ghosted_shared_i_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::I>(), -1, 0, 1, 1 );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.min( Dim::I ) - 1 );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.max( Dim::K ) + 1 );

    ghosted_shared_i_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::I>(), 1, -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.max( Dim::I ) + 2 );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.min( Dim::J ) - 1 );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.max( Dim::K ) );

    ghosted_shared_i_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::I>(), 0, 1, -1, 1 );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::I ),
               owned_i_face_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::I ),
               owned_i_face_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::J ),
               owned_i_face_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::J ),
               owned_i_face_space.max( Dim::J ) + 1 );
    EXPECT_EQ( ghosted_shared_i_face_space.min( Dim::K ),
               owned_i_face_space.min( Dim::K ) - 1 );
    EXPECT_EQ( ghosted_shared_i_face_space.max( Dim::K ),
               owned_i_face_space.min( Dim::K ) );

    //////////////////
    // J-FACE SPACES
    //////////////////

    // Get the local number of j-faces.
    auto owned_j_face_space =
        local_grid->indexSpace( Own(), Face<Dim::J>(), Local() );
    std::vector<int> local_num_j_faces( 3 );
    for ( int d = 0; d < 3; ++d )
        local_num_j_faces[d] = owned_j_face_space.extent( d );

    // Compute a global set of local j-face size arrays.
    std::vector<int> local_num_j_face_i( cart_dims[Dim::I], 0 );
    std::vector<int> local_num_j_face_j( cart_dims[Dim::J], 0 );
    std::vector<int> local_num_j_face_k( cart_dims[Dim::K], 0 );
    local_num_j_face_i[cart_rank[Dim::I]] = local_num_j_faces[Dim::I];
    local_num_j_face_j[cart_rank[Dim::J]] = local_num_j_faces[Dim::J];
    local_num_j_face_k[cart_rank[Dim::K]] = local_num_j_faces[Dim::K];
    MPI_Allreduce( MPI_IN_PLACE, local_num_j_face_i.data(), cart_dims[Dim::I],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_j_face_j.data(), cart_dims[Dim::J],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_j_face_k.data(), cart_dims[Dim::K],
                   MPI_INT, MPI_MAX, grid_comm );

    // Check to make sure we got the right number of total j-faces in each
    // dimension.
    EXPECT_EQ( global_num_cell[Dim::I],
               std::accumulate( local_num_j_face_i.begin(),
                                local_num_j_face_i.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::J],
               std::accumulate( local_num_j_face_j.begin(),
                                local_num_j_face_j.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::K],
               std::accumulate( local_num_j_face_k.begin(),
                                local_num_j_face_k.end(), 0 ) );

    // Check the global bounds.
    auto global_owned_j_face_space =
        local_grid->indexSpace( Own(), Face<Dim::J>(), Global() );
    EXPECT_EQ( global_owned_j_face_space.min( Dim::I ),
               global_grid->globalOffset( Dim::I ) );
    EXPECT_EQ( global_owned_j_face_space.max( Dim::I ),
               global_grid->globalOffset( Dim::I ) + local_num_cells[Dim::I] );
    EXPECT_EQ( global_owned_j_face_space.min( Dim::J ),
               global_grid->globalOffset( Dim::J ) );
    EXPECT_EQ( global_owned_j_face_space.max( Dim::J ),
               global_grid->globalOffset( Dim::J ) + local_num_nodes[Dim::J] );
    EXPECT_EQ( global_owned_j_face_space.min( Dim::K ),
               global_grid->globalOffset( Dim::K ) );
    EXPECT_EQ( global_owned_j_face_space.max( Dim::K ),
               global_grid->globalOffset( Dim::K ) + local_num_cells[Dim::K] );

    // Check the j-faces we own that we will share with our neighbors. Cover
    // enough of the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto owned_shared_j_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::J>(), -1, 0, 1 );
    EXPECT_EQ( owned_shared_j_face_space.min( Dim::I ),
               owned_j_face_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_j_face_space.max( Dim::I ),
               owned_j_face_space.min( Dim::I ) + halo_width );
    EXPECT_EQ( owned_shared_j_face_space.min( Dim::J ),
               owned_j_face_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_j_face_space.max( Dim::J ),
               owned_j_face_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_j_face_space.min( Dim::K ),
               owned_j_face_space.max( Dim::K ) - halo_width );
    EXPECT_EQ( owned_shared_j_face_space.max( Dim::K ),
               owned_j_face_space.max( Dim::K ) );

    owned_shared_j_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::J>(), 1, -1, 0 );
    EXPECT_EQ( owned_shared_j_face_space.min( Dim::I ),
               owned_j_face_space.max( Dim::I ) - halo_width );
    EXPECT_EQ( owned_shared_j_face_space.max( Dim::I ),
               owned_j_face_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_j_face_space.min( Dim::J ),
               owned_j_face_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_j_face_space.max( Dim::J ),
               owned_j_face_space.min( Dim::J ) + halo_width + 1 );
    EXPECT_EQ( owned_shared_j_face_space.min( Dim::K ),
               owned_j_face_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_j_face_space.max( Dim::K ),
               owned_j_face_space.max( Dim::K ) );

    owned_shared_j_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::J>(), 0, 1, -1 );
    EXPECT_EQ( owned_shared_j_face_space.min( Dim::I ),
               owned_j_face_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_j_face_space.max( Dim::I ),
               owned_j_face_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_j_face_space.min( Dim::J ),
               owned_j_face_space.max( Dim::J ) - halo_width );
    EXPECT_EQ( owned_shared_j_face_space.max( Dim::J ),
               owned_j_face_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_j_face_space.min( Dim::K ),
               owned_j_face_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_j_face_space.max( Dim::K ),
               owned_j_face_space.min( Dim::K ) + halo_width );

    // Check the j-faces are ghosts that our neighbors own. Cover enough of
    // the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto ghosted_shared_j_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::J>(), -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_j_face_space.min( Dim::I ), 0 );
    EXPECT_EQ( ghosted_shared_j_face_space.max( Dim::I ), halo_width );
    EXPECT_EQ( ghosted_shared_j_face_space.min( Dim::J ),
               owned_j_face_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_j_face_space.max( Dim::J ),
               owned_j_face_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_j_face_space.min( Dim::K ),
               owned_j_face_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_j_face_space.max( Dim::K ),
               owned_j_face_space.max( Dim::K ) + halo_width );

    ghosted_shared_j_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::J>(), 1, -1, 0 );
    EXPECT_EQ( ghosted_shared_j_face_space.min( Dim::I ),
               owned_j_face_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_j_face_space.max( Dim::I ),
               owned_j_face_space.max( Dim::I ) + halo_width );
    EXPECT_EQ( ghosted_shared_j_face_space.min( Dim::J ), 0 );
    EXPECT_EQ( ghosted_shared_j_face_space.max( Dim::J ), halo_width );
    EXPECT_EQ( ghosted_shared_j_face_space.min( Dim::K ),
               owned_j_face_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_j_face_space.max( Dim::K ),
               owned_j_face_space.max( Dim::K ) );

    ghosted_shared_j_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::J>(), 0, 1, -1 );
    EXPECT_EQ( ghosted_shared_j_face_space.min( Dim::I ),
               owned_j_face_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_j_face_space.max( Dim::I ),
               owned_j_face_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_j_face_space.min( Dim::J ),
               owned_j_face_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_j_face_space.max( Dim::J ),
               owned_j_face_space.max( Dim::J ) + halo_width + 1 );
    EXPECT_EQ( ghosted_shared_j_face_space.min( Dim::K ), 0 );
    EXPECT_EQ( ghosted_shared_j_face_space.max( Dim::K ), halo_width );

    //////////////////
    // K-FACE SPACES
    //////////////////

    // Get the local number of k-faces.
    auto owned_k_face_space =
        local_grid->indexSpace( Own(), Face<Dim::K>(), Local() );
    std::vector<int> local_num_k_faces( 3 );
    for ( int d = 0; d < 3; ++d )
        local_num_k_faces[d] = owned_k_face_space.extent( d );

    // Compute a global set of local k-face size arrays.
    std::vector<int> local_num_k_face_i( cart_dims[Dim::I], 0 );
    std::vector<int> local_num_k_face_j( cart_dims[Dim::J], 0 );
    std::vector<int> local_num_k_face_k( cart_dims[Dim::K], 0 );
    local_num_k_face_i[cart_rank[Dim::I]] = local_num_k_faces[Dim::I];
    local_num_k_face_j[cart_rank[Dim::J]] = local_num_k_faces[Dim::J];
    local_num_k_face_k[cart_rank[Dim::K]] = local_num_k_faces[Dim::K];
    MPI_Allreduce( MPI_IN_PLACE, local_num_k_face_i.data(), cart_dims[Dim::I],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_k_face_j.data(), cart_dims[Dim::J],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_k_face_k.data(), cart_dims[Dim::K],
                   MPI_INT, MPI_MAX, grid_comm );

    // Check to make sure we got the right number of total k-faces in each
    // dimension.
    EXPECT_EQ( global_num_cell[Dim::I],
               std::accumulate( local_num_k_face_i.begin(),
                                local_num_k_face_i.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::J],
               std::accumulate( local_num_k_face_j.begin(),
                                local_num_k_face_j.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::K],
               std::accumulate( local_num_k_face_k.begin(),
                                local_num_k_face_k.end(), 0 ) );

    // Check the global bounds.
    auto global_owned_k_face_space =
        local_grid->indexSpace( Own(), Face<Dim::K>(), Global() );
    EXPECT_EQ( global_owned_k_face_space.min( Dim::I ),
               global_grid->globalOffset( Dim::I ) );
    EXPECT_EQ( global_owned_k_face_space.max( Dim::I ),
               global_grid->globalOffset( Dim::I ) + local_num_cells[Dim::I] );
    EXPECT_EQ( global_owned_k_face_space.min( Dim::J ),
               global_grid->globalOffset( Dim::J ) );
    EXPECT_EQ( global_owned_k_face_space.max( Dim::J ),
               global_grid->globalOffset( Dim::J ) + local_num_cells[Dim::J] );
    EXPECT_EQ( global_owned_k_face_space.min( Dim::K ),
               global_grid->globalOffset( Dim::K ) );
    EXPECT_EQ( global_owned_k_face_space.max( Dim::K ),
               global_grid->globalOffset( Dim::K ) + local_num_nodes[Dim::K] );

    // Check the k-faces we own that we will share with our neighbors. Cover
    // enough of the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto owned_shared_k_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::K>(), -1, 0, 1 );
    EXPECT_EQ( owned_shared_k_face_space.min( Dim::I ),
               owned_k_face_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_k_face_space.max( Dim::I ),
               owned_k_face_space.min( Dim::I ) + halo_width );
    EXPECT_EQ( owned_shared_k_face_space.min( Dim::J ),
               owned_k_face_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_k_face_space.max( Dim::J ),
               owned_k_face_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_k_face_space.min( Dim::K ),
               owned_k_face_space.max( Dim::K ) - halo_width );
    EXPECT_EQ( owned_shared_k_face_space.max( Dim::K ),
               owned_k_face_space.max( Dim::K ) );

    owned_shared_k_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::K>(), 1, -1, 0 );
    EXPECT_EQ( owned_shared_k_face_space.min( Dim::I ),
               owned_k_face_space.max( Dim::I ) - halo_width );
    EXPECT_EQ( owned_shared_k_face_space.max( Dim::I ),
               owned_k_face_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_k_face_space.min( Dim::J ),
               owned_k_face_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_k_face_space.max( Dim::J ),
               owned_k_face_space.min( Dim::J ) + halo_width );
    EXPECT_EQ( owned_shared_k_face_space.min( Dim::K ),
               owned_k_face_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_k_face_space.max( Dim::K ),
               owned_k_face_space.max( Dim::K ) );

    owned_shared_k_face_space =
        local_grid->sharedIndexSpace( Own(), Face<Dim::K>(), 0, 1, -1 );
    EXPECT_EQ( owned_shared_k_face_space.min( Dim::I ),
               owned_k_face_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_k_face_space.max( Dim::I ),
               owned_k_face_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_k_face_space.min( Dim::J ),
               owned_k_face_space.max( Dim::J ) - halo_width );
    EXPECT_EQ( owned_shared_k_face_space.max( Dim::J ),
               owned_k_face_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_k_face_space.min( Dim::K ),
               owned_k_face_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_k_face_space.max( Dim::K ),
               owned_k_face_space.min( Dim::K ) + halo_width + 1 );

    // Check the k-faces are ghosts that our neighbors own. Cover enough of
    // the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto ghosted_shared_k_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::K>(), -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_k_face_space.min( Dim::I ), 0 );
    EXPECT_EQ( ghosted_shared_k_face_space.max( Dim::I ), halo_width );
    EXPECT_EQ( ghosted_shared_k_face_space.min( Dim::J ),
               owned_k_face_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_k_face_space.max( Dim::J ),
               owned_k_face_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_k_face_space.min( Dim::K ),
               owned_k_face_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_k_face_space.max( Dim::K ),
               owned_k_face_space.max( Dim::K ) + halo_width + 1 );

    ghosted_shared_k_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::K>(), 1, -1, 0 );
    EXPECT_EQ( ghosted_shared_k_face_space.min( Dim::I ),
               owned_k_face_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_k_face_space.max( Dim::I ),
               owned_k_face_space.max( Dim::I ) + halo_width );
    EXPECT_EQ( ghosted_shared_k_face_space.min( Dim::J ), 0 );
    EXPECT_EQ( ghosted_shared_k_face_space.max( Dim::J ), halo_width );
    EXPECT_EQ( ghosted_shared_k_face_space.min( Dim::K ),
               owned_k_face_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_k_face_space.max( Dim::K ),
               owned_k_face_space.max( Dim::K ) );

    ghosted_shared_k_face_space =
        local_grid->sharedIndexSpace( Ghost(), Face<Dim::K>(), 0, 1, -1 );
    EXPECT_EQ( ghosted_shared_k_face_space.min( Dim::I ),
               owned_k_face_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_k_face_space.max( Dim::I ),
               owned_k_face_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_k_face_space.min( Dim::J ),
               owned_k_face_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_k_face_space.max( Dim::J ),
               owned_k_face_space.max( Dim::J ) + halo_width );
    EXPECT_EQ( ghosted_shared_k_face_space.min( Dim::K ), 0 );
    EXPECT_EQ( ghosted_shared_k_face_space.max( Dim::K ), halo_width );

    //////////////////
    // I-EDGE SPACES
    //////////////////

    // Get the local number of I-edges.
    auto owned_i_edge_space =
        local_grid->indexSpace( Own(), Edge<Dim::I>(), Local() );
    std::vector<int> local_num_i_edges( 3 );
    for ( int d = 0; d < 3; ++d )
        local_num_i_edges[d] = owned_i_edge_space.extent( d );

    // Compute a global set of local I-edge size arrays.
    std::vector<int> local_num_i_edge_i( cart_dims[Dim::I], 0 );
    std::vector<int> local_num_i_edge_j( cart_dims[Dim::J], 0 );
    std::vector<int> local_num_i_edge_k( cart_dims[Dim::K], 0 );
    local_num_i_edge_i[cart_rank[Dim::I]] = local_num_i_edges[Dim::I];
    local_num_i_edge_j[cart_rank[Dim::J]] = local_num_i_edges[Dim::J];
    local_num_i_edge_k[cart_rank[Dim::K]] = local_num_i_edges[Dim::K];
    MPI_Allreduce( MPI_IN_PLACE, local_num_i_edge_i.data(), cart_dims[Dim::I],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_i_edge_j.data(), cart_dims[Dim::J],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_i_edge_k.data(), cart_dims[Dim::K],
                   MPI_INT, MPI_MAX, grid_comm );

    // Check to make sure we got the right number of total I-edges in each
    // dimension.
    EXPECT_EQ( global_num_cell[Dim::I],
               std::accumulate( local_num_i_edge_i.begin(),
                                local_num_i_edge_i.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::J],
               std::accumulate( local_num_i_edge_j.begin(),
                                local_num_i_edge_j.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::K],
               std::accumulate( local_num_i_edge_k.begin(),
                                local_num_i_edge_k.end(), 0 ) );

    // Check the global bounds.
    auto global_owned_i_edge_space =
        local_grid->indexSpace( Own(), Edge<Dim::I>(), Global() );
    EXPECT_EQ( global_owned_i_edge_space.min( Dim::I ),
               global_grid->globalOffset( Dim::I ) );
    EXPECT_EQ( global_owned_i_edge_space.max( Dim::I ),
               global_grid->globalOffset( Dim::I ) + local_num_nodes[Dim::I] );
    EXPECT_EQ( global_owned_i_edge_space.min( Dim::J ),
               global_grid->globalOffset( Dim::J ) );
    EXPECT_EQ( global_owned_i_edge_space.max( Dim::J ),
               global_grid->globalOffset( Dim::J ) + local_num_cells[Dim::J] );
    EXPECT_EQ( global_owned_i_edge_space.min( Dim::K ),
               global_grid->globalOffset( Dim::K ) );
    EXPECT_EQ( global_owned_i_edge_space.max( Dim::K ),
               global_grid->globalOffset( Dim::K ) + local_num_cells[Dim::K] );

    // Check the I-edges we own that we will share with our neighbors. Cover
    // enough of the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto owned_shared_i_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::I>(), -1, 0, 1 );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.min( Dim::I ) + halo_width );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.max( Dim::K ) - halo_width );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.max( Dim::K ) );

    owned_shared_i_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::I>(), 1, -1, 0 );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.max( Dim::I ) - halo_width );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.min( Dim::J ) + halo_width + 1 );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.max( Dim::K ) );

    owned_shared_i_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::I>(), 0, 1, -1 );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.max( Dim::J ) - halo_width );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.min( Dim::K ) + halo_width + 1 );

    // Check the I-edge owned shared spaces again, this time with a specified
    // halo width.
    owned_shared_i_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::I>(), -1, 0, 1, 1 );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.min( Dim::I ) + 1 );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.max( Dim::K ) - 1 );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.max( Dim::K ) );

    owned_shared_i_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::I>(), 1, -1, 0, 1 );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.max( Dim::I ) - 1 );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.min( Dim::J ) + 2 );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.max( Dim::K ) );

    owned_shared_i_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::I>(), 0, 1, -1, 1 );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.max( Dim::J ) - 1 );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.min( Dim::K ) + 2 );

    // Check the I-edges are ghosts that our neighbors own. Cover enough of
    // the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto ghosted_shared_i_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::I>(), -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::I ), 0 );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::I ), halo_width );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.max( Dim::K ) + halo_width + 1 );

    ghosted_shared_i_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::I>(), 1, -1, 0 );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.max( Dim::I ) + halo_width );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::J ), 0 );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::J ), halo_width );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.max( Dim::K ) );

    ghosted_shared_i_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::I>(), 0, 1, -1 );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.max( Dim::J ) + halo_width + 1 );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::K ), 0 );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::K ), halo_width );

    // Check the I-edge ghosted shared spaces again, this time with a
    // specified halo width.
    ghosted_shared_i_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::I>(), -1, 0, 1, 1 );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.min( Dim::I ) - 1 );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.max( Dim::K ) + 2 );

    ghosted_shared_i_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::I>(), 1, -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.max( Dim::I ) + 1 );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.min( Dim::J ) - 1 );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.max( Dim::K ) );

    ghosted_shared_i_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::I>(), 0, 1, -1, 1 );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::I ),
               owned_i_edge_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::I ),
               owned_i_edge_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::J ),
               owned_i_edge_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::J ),
               owned_i_edge_space.max( Dim::J ) + 2 );
    EXPECT_EQ( ghosted_shared_i_edge_space.min( Dim::K ),
               owned_i_edge_space.min( Dim::K ) - 1 );
    EXPECT_EQ( ghosted_shared_i_edge_space.max( Dim::K ),
               owned_i_edge_space.min( Dim::K ) );

    //////////////////
    // J-EDGE SPACES
    //////////////////

    // Get the local number of j-edges.
    auto owned_j_edge_space =
        local_grid->indexSpace( Own(), Edge<Dim::J>(), Local() );
    std::vector<int> local_num_j_edges( 3 );
    for ( int d = 0; d < 3; ++d )
        local_num_j_edges[d] = owned_j_edge_space.extent( d );

    // Compute a global set of local j-edge size arrays.
    std::vector<int> local_num_j_edge_i( cart_dims[Dim::I], 0 );
    std::vector<int> local_num_j_edge_j( cart_dims[Dim::J], 0 );
    std::vector<int> local_num_j_edge_k( cart_dims[Dim::K], 0 );
    local_num_j_edge_i[cart_rank[Dim::I]] = local_num_j_edges[Dim::I];
    local_num_j_edge_j[cart_rank[Dim::J]] = local_num_j_edges[Dim::J];
    local_num_j_edge_k[cart_rank[Dim::K]] = local_num_j_edges[Dim::K];
    MPI_Allreduce( MPI_IN_PLACE, local_num_j_edge_i.data(), cart_dims[Dim::I],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_j_edge_j.data(), cart_dims[Dim::J],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_j_edge_k.data(), cart_dims[Dim::K],
                   MPI_INT, MPI_MAX, grid_comm );

    // Check to make sure we got the right number of total j-edges in each
    // dimension.
    EXPECT_EQ( global_num_cell[Dim::I],
               std::accumulate( local_num_j_edge_i.begin(),
                                local_num_j_edge_i.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::J],
               std::accumulate( local_num_j_edge_j.begin(),
                                local_num_j_edge_j.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::K],
               std::accumulate( local_num_j_edge_k.begin(),
                                local_num_j_edge_k.end(), 0 ) );

    // Check the global bounds.
    auto global_owned_j_edge_space =
        local_grid->indexSpace( Own(), Edge<Dim::J>(), Global() );
    EXPECT_EQ( global_owned_j_edge_space.min( Dim::I ),
               global_grid->globalOffset( Dim::I ) );
    EXPECT_EQ( global_owned_j_edge_space.max( Dim::I ),
               global_grid->globalOffset( Dim::I ) + local_num_cells[Dim::I] );
    EXPECT_EQ( global_owned_j_edge_space.min( Dim::J ),
               global_grid->globalOffset( Dim::J ) );
    EXPECT_EQ( global_owned_j_edge_space.max( Dim::J ),
               global_grid->globalOffset( Dim::J ) + local_num_nodes[Dim::J] );
    EXPECT_EQ( global_owned_j_edge_space.min( Dim::K ),
               global_grid->globalOffset( Dim::K ) );
    EXPECT_EQ( global_owned_j_edge_space.max( Dim::K ),
               global_grid->globalOffset( Dim::K ) + local_num_cells[Dim::K] );

    // Check the j-edges we own that we will share with our neighbors. Cover
    // enough of the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto owned_shared_j_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::J>(), -1, 0, 1 );
    EXPECT_EQ( owned_shared_j_edge_space.min( Dim::I ),
               owned_j_edge_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_j_edge_space.max( Dim::I ),
               owned_j_edge_space.min( Dim::I ) + halo_width + 1 );
    EXPECT_EQ( owned_shared_j_edge_space.min( Dim::J ),
               owned_j_edge_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_j_edge_space.max( Dim::J ),
               owned_j_edge_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_j_edge_space.min( Dim::K ),
               owned_j_edge_space.max( Dim::K ) - halo_width );
    EXPECT_EQ( owned_shared_j_edge_space.max( Dim::K ),
               owned_j_edge_space.max( Dim::K ) );

    owned_shared_j_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::J>(), 1, -1, 0 );
    EXPECT_EQ( owned_shared_j_edge_space.min( Dim::I ),
               owned_j_edge_space.max( Dim::I ) - halo_width );
    EXPECT_EQ( owned_shared_j_edge_space.max( Dim::I ),
               owned_j_edge_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_j_edge_space.min( Dim::J ),
               owned_j_edge_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_j_edge_space.max( Dim::J ),
               owned_j_edge_space.min( Dim::J ) + halo_width );
    EXPECT_EQ( owned_shared_j_edge_space.min( Dim::K ),
               owned_j_edge_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_j_edge_space.max( Dim::K ),
               owned_j_edge_space.max( Dim::K ) );

    owned_shared_j_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::J>(), 0, 1, -1 );
    EXPECT_EQ( owned_shared_j_edge_space.min( Dim::I ),
               owned_j_edge_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_j_edge_space.max( Dim::I ),
               owned_j_edge_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_j_edge_space.min( Dim::J ),
               owned_j_edge_space.max( Dim::J ) - halo_width );
    EXPECT_EQ( owned_shared_j_edge_space.max( Dim::J ),
               owned_j_edge_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_j_edge_space.min( Dim::K ),
               owned_j_edge_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_j_edge_space.max( Dim::K ),
               owned_j_edge_space.min( Dim::K ) + halo_width + 1 );

    // Check the j-edges are ghosts that our neighbors own. Cover enough of
    // the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto ghosted_shared_j_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::J>(), -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_j_edge_space.min( Dim::I ), 0 );
    EXPECT_EQ( ghosted_shared_j_edge_space.max( Dim::I ), halo_width );
    EXPECT_EQ( ghosted_shared_j_edge_space.min( Dim::J ),
               owned_j_edge_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_j_edge_space.max( Dim::J ),
               owned_j_edge_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_j_edge_space.min( Dim::K ),
               owned_j_edge_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_j_edge_space.max( Dim::K ),
               owned_j_edge_space.max( Dim::K ) + halo_width + 1 );

    ghosted_shared_j_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::J>(), 1, -1, 0 );
    EXPECT_EQ( ghosted_shared_j_edge_space.min( Dim::I ),
               owned_j_edge_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_j_edge_space.max( Dim::I ),
               owned_j_edge_space.max( Dim::I ) + halo_width + 1 );
    EXPECT_EQ( ghosted_shared_j_edge_space.min( Dim::J ), 0 );
    EXPECT_EQ( ghosted_shared_j_edge_space.max( Dim::J ), halo_width );
    EXPECT_EQ( ghosted_shared_j_edge_space.min( Dim::K ),
               owned_j_edge_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_j_edge_space.max( Dim::K ),
               owned_j_edge_space.max( Dim::K ) );

    ghosted_shared_j_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::J>(), 0, 1, -1 );
    EXPECT_EQ( ghosted_shared_j_edge_space.min( Dim::I ),
               owned_j_edge_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_j_edge_space.max( Dim::I ),
               owned_j_edge_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_j_edge_space.min( Dim::J ),
               owned_j_edge_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_j_edge_space.max( Dim::J ),
               owned_j_edge_space.max( Dim::J ) + halo_width );
    EXPECT_EQ( ghosted_shared_j_edge_space.min( Dim::K ), 0 );
    EXPECT_EQ( ghosted_shared_j_edge_space.max( Dim::K ), halo_width );

    //////////////////
    // K-EDGE SPACES
    //////////////////

    // Get the local number of k-edges.
    auto owned_k_edge_space =
        local_grid->indexSpace( Own(), Edge<Dim::K>(), Local() );
    std::vector<int> local_num_k_edges( 3 );
    for ( int d = 0; d < 3; ++d )
        local_num_k_edges[d] = owned_k_edge_space.extent( d );

    // Compute a global set of local k-edge size arrays.
    std::vector<int> local_num_k_edge_i( cart_dims[Dim::I], 0 );
    std::vector<int> local_num_k_edge_j( cart_dims[Dim::J], 0 );
    std::vector<int> local_num_k_edge_k( cart_dims[Dim::K], 0 );
    local_num_k_edge_i[cart_rank[Dim::I]] = local_num_k_edges[Dim::I];
    local_num_k_edge_j[cart_rank[Dim::J]] = local_num_k_edges[Dim::J];
    local_num_k_edge_k[cart_rank[Dim::K]] = local_num_k_edges[Dim::K];
    MPI_Allreduce( MPI_IN_PLACE, local_num_k_edge_i.data(), cart_dims[Dim::I],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_k_edge_j.data(), cart_dims[Dim::J],
                   MPI_INT, MPI_MAX, grid_comm );
    MPI_Allreduce( MPI_IN_PLACE, local_num_k_edge_k.data(), cart_dims[Dim::K],
                   MPI_INT, MPI_MAX, grid_comm );

    // Check to make sure we got the right number of total k-edges in each
    // dimension.
    EXPECT_EQ( global_num_cell[Dim::I],
               std::accumulate( local_num_k_edge_i.begin(),
                                local_num_k_edge_i.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::J],
               std::accumulate( local_num_k_edge_j.begin(),
                                local_num_k_edge_j.end(), 0 ) );
    EXPECT_EQ( global_num_cell[Dim::K],
               std::accumulate( local_num_k_edge_k.begin(),
                                local_num_k_edge_k.end(), 0 ) );

    // Check the global bounds.
    auto global_owned_k_edge_space =
        local_grid->indexSpace( Own(), Edge<Dim::K>(), Global() );
    EXPECT_EQ( global_owned_k_edge_space.min( Dim::I ),
               global_grid->globalOffset( Dim::I ) );
    EXPECT_EQ( global_owned_k_edge_space.max( Dim::I ),
               global_grid->globalOffset( Dim::I ) + local_num_cells[Dim::I] );
    EXPECT_EQ( global_owned_k_edge_space.min( Dim::J ),
               global_grid->globalOffset( Dim::J ) );
    EXPECT_EQ( global_owned_k_edge_space.max( Dim::J ),
               global_grid->globalOffset( Dim::J ) + local_num_cells[Dim::J] );
    EXPECT_EQ( global_owned_k_edge_space.min( Dim::K ),
               global_grid->globalOffset( Dim::K ) );
    EXPECT_EQ( global_owned_k_edge_space.max( Dim::K ),
               global_grid->globalOffset( Dim::K ) + local_num_nodes[Dim::K] );

    // Check the k-edges we own that we will share with our neighbors. Cover
    // enough of the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto owned_shared_k_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::K>(), -1, 0, 1 );
    EXPECT_EQ( owned_shared_k_edge_space.min( Dim::I ),
               owned_k_edge_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_k_edge_space.max( Dim::I ),
               owned_k_edge_space.min( Dim::I ) + halo_width + 1 );
    EXPECT_EQ( owned_shared_k_edge_space.min( Dim::J ),
               owned_k_edge_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_k_edge_space.max( Dim::J ),
               owned_k_edge_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_k_edge_space.min( Dim::K ),
               owned_k_edge_space.max( Dim::K ) - halo_width );
    EXPECT_EQ( owned_shared_k_edge_space.max( Dim::K ),
               owned_k_edge_space.max( Dim::K ) );

    owned_shared_k_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::K>(), 1, -1, 0 );
    EXPECT_EQ( owned_shared_k_edge_space.min( Dim::I ),
               owned_k_edge_space.max( Dim::I ) - halo_width );
    EXPECT_EQ( owned_shared_k_edge_space.max( Dim::I ),
               owned_k_edge_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_k_edge_space.min( Dim::J ),
               owned_k_edge_space.min( Dim::J ) );
    EXPECT_EQ( owned_shared_k_edge_space.max( Dim::J ),
               owned_k_edge_space.min( Dim::J ) + halo_width + 1 );
    EXPECT_EQ( owned_shared_k_edge_space.min( Dim::K ),
               owned_k_edge_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_k_edge_space.max( Dim::K ),
               owned_k_edge_space.max( Dim::K ) );

    owned_shared_k_edge_space =
        local_grid->sharedIndexSpace( Own(), Edge<Dim::K>(), 0, 1, -1 );
    EXPECT_EQ( owned_shared_k_edge_space.min( Dim::I ),
               owned_k_edge_space.min( Dim::I ) );
    EXPECT_EQ( owned_shared_k_edge_space.max( Dim::I ),
               owned_k_edge_space.max( Dim::I ) );
    EXPECT_EQ( owned_shared_k_edge_space.min( Dim::J ),
               owned_k_edge_space.max( Dim::J ) - halo_width );
    EXPECT_EQ( owned_shared_k_edge_space.max( Dim::J ),
               owned_k_edge_space.max( Dim::J ) );
    EXPECT_EQ( owned_shared_k_edge_space.min( Dim::K ),
               owned_k_edge_space.min( Dim::K ) );
    EXPECT_EQ( owned_shared_k_edge_space.max( Dim::K ),
               owned_k_edge_space.min( Dim::K ) + halo_width );

    // Check the k-edges are ghosts that our neighbors own. Cover enough of
    // the neighbors that we know the bounds are correct in each
    // dimension. The three variations here cover all of the cases.
    auto ghosted_shared_k_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::K>(), -1, 0, 1 );
    EXPECT_EQ( ghosted_shared_k_edge_space.min( Dim::I ), 0 );
    EXPECT_EQ( ghosted_shared_k_edge_space.max( Dim::I ), halo_width );
    EXPECT_EQ( ghosted_shared_k_edge_space.min( Dim::J ),
               owned_k_edge_space.min( Dim::J ) );
    EXPECT_EQ( ghosted_shared_k_edge_space.max( Dim::J ),
               owned_k_edge_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_k_edge_space.min( Dim::K ),
               owned_k_edge_space.max( Dim::K ) );
    EXPECT_EQ( ghosted_shared_k_edge_space.max( Dim::K ),
               owned_k_edge_space.max( Dim::K ) + halo_width );

    ghosted_shared_k_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::K>(), 1, -1, 0 );
    EXPECT_EQ( ghosted_shared_k_edge_space.min( Dim::I ),
               owned_k_edge_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_k_edge_space.max( Dim::I ),
               owned_k_edge_space.max( Dim::I ) + halo_width + 1 );
    EXPECT_EQ( ghosted_shared_k_edge_space.min( Dim::J ), 0 );
    EXPECT_EQ( ghosted_shared_k_edge_space.max( Dim::J ), halo_width );
    EXPECT_EQ( ghosted_shared_k_edge_space.min( Dim::K ),
               owned_k_edge_space.min( Dim::K ) );
    EXPECT_EQ( ghosted_shared_k_edge_space.max( Dim::K ),
               owned_k_edge_space.max( Dim::K ) );

    ghosted_shared_k_edge_space =
        local_grid->sharedIndexSpace( Ghost(), Edge<Dim::K>(), 0, 1, -1 );
    EXPECT_EQ( ghosted_shared_k_edge_space.min( Dim::I ),
               owned_k_edge_space.min( Dim::I ) );
    EXPECT_EQ( ghosted_shared_k_edge_space.max( Dim::I ),
               owned_k_edge_space.max( Dim::I ) );
    EXPECT_EQ( ghosted_shared_k_edge_space.min( Dim::J ),
               owned_k_edge_space.max( Dim::J ) );
    EXPECT_EQ( ghosted_shared_k_edge_space.max( Dim::J ),
               owned_k_edge_space.max( Dim::J ) + halo_width + 1 );
    EXPECT_EQ( ghosted_shared_k_edge_space.min( Dim::K ), 0 );
    EXPECT_EQ( ghosted_shared_k_edge_space.max( Dim::K ), halo_width );
}

//---------------------------------------------------------------------------//
void notPeriodicTest()
{
    // Create a different MPI communication on every rank, effectively making
    // it serial.
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    MPI_Comm serial_comm;
    MPI_Comm_split( MPI_COMM_WORLD, comm_rank, 0, &serial_comm );

    // Let MPI compute the partitioning for this test.
    UniformDimPartitioner partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 101, 85, 99 };
    std::array<bool, 3> is_dim_periodic = { false, false, false };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = createGlobalGrid( serial_comm, global_mesh,
                                         is_dim_periodic, partitioner );
    auto grid_comm = global_grid->comm();

    // Create a local mesh.
    int halo_width = 2;
    auto local_grid = createLocalGrid( global_grid, halo_width );

    // Check sizes
    EXPECT_EQ( local_grid->haloCellWidth(), halo_width );

    // Get the owned number of cells.
    auto owned_cell_space = local_grid->indexSpace( Own(), Cell(), Local() );
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( owned_cell_space.extent( d ), global_num_cell[d] );

    // Get the ghosted number of cells.
    auto ghosted_cell_space =
        local_grid->indexSpace( Ghost(), Cell(), Local() );
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( ghosted_cell_space.extent( d ), global_num_cell[d] );

    // Get the owned number of nodes.
    auto owned_node_space = local_grid->indexSpace( Own(), Node(), Local() );
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( owned_node_space.extent( d ), global_num_cell[d] + 1 );

    // Get the ghosted number of nodes.
    auto ghosted_node_space =
        local_grid->indexSpace( Ghost(), Node(), Local() );
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( ghosted_node_space.extent( d ), global_num_cell[d] + 1 );

    // Get the owned number of I-faces.
    auto owned_i_face_space =
        local_grid->indexSpace( Own(), Face<Dim::I>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dim::I == d )
            EXPECT_EQ( owned_i_face_space.extent( d ), global_num_cell[d] + 1 );
        else
            EXPECT_EQ( owned_i_face_space.extent( d ), global_num_cell[d] );
    }

    // Get the ghosted number of I-faces.
    auto ghosted_i_face_space =
        local_grid->indexSpace( Ghost(), Face<Dim::I>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dim::I == d )
            EXPECT_EQ( ghosted_i_face_space.extent( d ),
                       global_num_cell[d] + 1 );
        else
            EXPECT_EQ( ghosted_i_face_space.extent( d ), global_num_cell[d] );
    }

    // Get the owned number of J-faces.
    auto owned_j_face_space =
        local_grid->indexSpace( Own(), Face<Dim::J>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dim::J == d )
            EXPECT_EQ( owned_j_face_space.extent( d ), global_num_cell[d] + 1 );
        else
            EXPECT_EQ( owned_j_face_space.extent( d ), global_num_cell[d] );
    }

    // Get the ghosted number of J-faces.
    auto ghosted_j_face_space =
        local_grid->indexSpace( Ghost(), Face<Dim::J>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dim::J == d )
            EXPECT_EQ( ghosted_j_face_space.extent( d ),
                       global_num_cell[d] + 1 );
        else
            EXPECT_EQ( ghosted_j_face_space.extent( d ), global_num_cell[d] );
    }

    // Get the owned number of K-faces.
    auto owned_k_face_space =
        local_grid->indexSpace( Own(), Face<Dim::K>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dim::K == d )
            EXPECT_EQ( owned_k_face_space.extent( d ), global_num_cell[d] + 1 );
        else
            EXPECT_EQ( owned_k_face_space.extent( d ), global_num_cell[d] );
    }

    // Get the ghosted number of K-faces.
    auto ghosted_k_face_space =
        local_grid->indexSpace( Ghost(), Face<Dim::K>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dim::K == d )
            EXPECT_EQ( ghosted_k_face_space.extent( d ),
                       global_num_cell[d] + 1 );
        else
            EXPECT_EQ( ghosted_k_face_space.extent( d ), global_num_cell[d] );
    }

    // Check neighbor ranks and shared spaces.
    for ( int i = -1; i < 2; ++i )
        for ( int j = -1; j < 2; ++j )
            for ( int k = -1; k < 2; ++k )
            {
                if ( i == 0 && j == 0 && k == 0 )
                {
                    std::vector<int> nr = {
                        global_grid->dimBlockId( Dim::I ) + i,
                        global_grid->dimBlockId( Dim::J ) + j,
                        global_grid->dimBlockId( Dim::K ) + k };
                    int nrank;
                    MPI_Cart_rank( grid_comm, nr.data(), &nrank );
                    EXPECT_EQ( local_grid->neighborRank( i, j, k ), nrank );
                }
                else
                {
                    EXPECT_EQ( local_grid->neighborRank( i, j, k ), -1 );

                    auto owned_shared_cell_space =
                        local_grid->sharedIndexSpace( Own(), Cell(), i, j, k );
                    EXPECT_EQ( owned_shared_cell_space.size(), 0 );

                    auto ghosted_shared_cell_space =
                        local_grid->sharedIndexSpace( Ghost(), Cell(), i, j,
                                                      k );
                    EXPECT_EQ( ghosted_shared_cell_space.size(), 0 );

                    auto owned_shared_node_space =
                        local_grid->sharedIndexSpace( Own(), Node(), i, j, k );
                    EXPECT_EQ( owned_shared_node_space.size(), 0 );

                    auto ghosted_shared_node_space =
                        local_grid->sharedIndexSpace( Ghost(), Node(), i, j,
                                                      k );
                    EXPECT_EQ( ghosted_shared_node_space.size(), 0 );

                    auto owned_shared_i_face_space =
                        local_grid->sharedIndexSpace( Own(), Face<Dim::I>(), i,
                                                      j, k );
                    EXPECT_EQ( owned_shared_i_face_space.size(), 0 );

                    auto ghosted_shared_i_face_space =
                        local_grid->sharedIndexSpace( Ghost(), Face<Dim::I>(),
                                                      i, j, k );
                    EXPECT_EQ( ghosted_shared_i_face_space.size(), 0 );

                    auto owned_shared_j_face_space =
                        local_grid->sharedIndexSpace( Own(), Face<Dim::J>(), i,
                                                      j, k );
                    EXPECT_EQ( owned_shared_j_face_space.size(), 0 );

                    auto ghosted_shared_j_face_space =
                        local_grid->sharedIndexSpace( Ghost(), Face<Dim::J>(),
                                                      i, j, k );
                    EXPECT_EQ( ghosted_shared_j_face_space.size(), 0 );

                    auto owned_shared_k_face_space =
                        local_grid->sharedIndexSpace( Own(), Face<Dim::K>(), i,
                                                      j, k );
                    EXPECT_EQ( owned_shared_k_face_space.size(), 0 );

                    auto ghosted_shared_k_face_space =
                        local_grid->sharedIndexSpace( Ghost(), Face<Dim::K>(),
                                                      i, j, k );
                    EXPECT_EQ( ghosted_shared_k_face_space.size(), 0 );
                }
            }

    // Free the serial communicator we made
    MPI_Comm_free( &serial_comm );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( local_grid, api_test )
{
    periodicTest();
    notPeriodicTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
