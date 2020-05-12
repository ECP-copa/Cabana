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
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void gridTest( const std::array<bool, 3> &is_dim_periodic )
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
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Check the number of entities.
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( global_num_cell[d],
                   global_grid->globalNumEntity( Cell(), d ) );
        if ( is_dim_periodic[d] )
            EXPECT_EQ( global_num_cell[d],
                       global_grid->globalNumEntity( Node(), d ) );
        else
            EXPECT_EQ( global_num_cell[d] + 1,
                       global_grid->globalNumEntity( Node(), d ) );
    }

    // Number of I faces
    if ( is_dim_periodic[Dim::I] )
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::I>(), Dim::I ),
                   global_num_cell[Dim::I] );
    else
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::I>(), Dim::I ),
                   global_num_cell[Dim::I] + 1 );
    EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::I>(), Dim::J ),
               global_num_cell[Dim::J] );
    EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::I>(), Dim::K ),
               global_num_cell[Dim::K] );

    // Number of J faces.
    EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::J>(), Dim::I ),
               global_num_cell[Dim::I] );
    if ( is_dim_periodic[Dim::I] )
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::J>(), Dim::J ),
                   global_num_cell[Dim::J] );
    else
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::J>(), Dim::J ),
                   global_num_cell[Dim::J] + 1 );
    EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::J>(), Dim::K ),
               global_num_cell[Dim::K] );

    // Number of K faces.
    EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::K>(), Dim::I ),
               global_num_cell[Dim::I] );
    EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::K>(), Dim::J ),
               global_num_cell[Dim::J] );
    if ( is_dim_periodic[Dim::I] )
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::K>(), Dim::K ),
                   global_num_cell[Dim::K] );
    else
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::K>(), Dim::K ),
                   global_num_cell[Dim::K] + 1 );

    // Number of I edges
    EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::I>(), Dim::I ),
               global_num_cell[Dim::I] );
    if ( is_dim_periodic[Dim::J] )
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::I>(), Dim::J ),
                   global_num_cell[Dim::J] );
    else
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::I>(), Dim::J ),
                   global_num_cell[Dim::J] + 1 );
    if ( is_dim_periodic[Dim::K] )
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::I>(), Dim::K ),
                   global_num_cell[Dim::K] );
    else
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::I>(), Dim::K ),
                   global_num_cell[Dim::K] + 1 );

    // Number of J edges
    if ( is_dim_periodic[Dim::I] )
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::J>(), Dim::I ),
                   global_num_cell[Dim::I] );
    else
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::J>(), Dim::I ),
                   global_num_cell[Dim::I] + 1 );
    EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::J>(), Dim::J ),
               global_num_cell[Dim::J] );
    if ( is_dim_periodic[Dim::K] )
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::J>(), Dim::K ),
                   global_num_cell[Dim::K] );
    else
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::J>(), Dim::K ),
                   global_num_cell[Dim::K] + 1 );

    // Number of K edges
    if ( is_dim_periodic[Dim::I] )
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::K>(), Dim::I ),
                   global_num_cell[Dim::I] );
    else
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::K>(), Dim::I ),
                   global_num_cell[Dim::I] + 1 );
    if ( is_dim_periodic[Dim::J] )
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::K>(), Dim::J ),
                   global_num_cell[Dim::J] );
    else
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::K>(), Dim::J ),
                   global_num_cell[Dim::J] + 1 );
    EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::K>(), Dim::K ),
               global_num_cell[Dim::K] );

    // Check the partitioning. The grid communicator has a Cartesian topology.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    auto grid_comm = global_grid->comm();
    int grid_comm_size;
    MPI_Comm_size( grid_comm, &grid_comm_size );
    int grid_comm_rank;
    MPI_Comm_rank( grid_comm, &grid_comm_rank );
    EXPECT_EQ( grid_comm_size, comm_size );
    EXPECT_EQ( global_grid->totalNumBlock(), grid_comm_size );
    EXPECT_EQ( global_grid->blockId(), grid_comm_rank );

    auto ranks_per_dim =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, global_num_cell );
    std::vector<int> cart_dims( 3 );
    std::vector<int> cart_period( 3 );
    std::vector<int> cart_rank( 3 );
    MPI_Cart_get( grid_comm, 3, cart_dims.data(), cart_period.data(),
                  cart_rank.data() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( cart_period[d], is_dim_periodic[d] );
        EXPECT_EQ( global_grid->dimBlockId( d ), cart_rank[d] );
        EXPECT_EQ( global_grid->dimNumBlock( d ), ranks_per_dim[d] );
    }

    for ( int d = 0; d < 3; ++d )
    {
        std::vector<int> dim_cells_per_rank( global_grid->dimNumBlock( d ), 0 );
        dim_cells_per_rank[global_grid->dimBlockId( d )] =
            global_grid->ownedNumCell( d );
        MPI_Allreduce( MPI_IN_PLACE, dim_cells_per_rank.data(),
                       dim_cells_per_rank.size(), MPI_INT, MPI_MAX,
                       MPI_COMM_WORLD );
        int dim_offset = 0;
        for ( int n = 0; n < global_grid->dimBlockId( d ); ++n )
            dim_offset += dim_cells_per_rank[n];
        int dim_sum = 0;
        for ( int n = 0; n < global_grid->dimNumBlock( d ); ++n )
            dim_sum += dim_cells_per_rank[n];
        EXPECT_EQ( global_grid->globalOffset( d ), dim_offset );
        EXPECT_EQ( global_grid->globalNumEntity( Cell(), d ), dim_sum );
    }

    // Check block ranks
    if ( is_dim_periodic[Dim::I] )
        EXPECT_EQ( global_grid->blockRank( -1, 0, 0 ),
                   global_grid->blockRank(
                       global_grid->dimNumBlock( Dim::I ) - 1, 0, 0 ) );
    else
        EXPECT_EQ( global_grid->blockRank( -1, 0, 0 ), -1 );

    if ( is_dim_periodic[Dim::I] )
        EXPECT_EQ(
            global_grid->blockRank( global_grid->dimNumBlock( Dim::I ), 0, 0 ),
            global_grid->blockRank( 0, 0, 0 ) );
    else
        EXPECT_EQ(
            global_grid->blockRank( global_grid->dimNumBlock( Dim::I ), 0, 0 ),
            -1 );

    if ( is_dim_periodic[Dim::J] )
        EXPECT_EQ( global_grid->blockRank( 0, -1, 0 ),
                   global_grid->blockRank(
                       0, global_grid->dimNumBlock( Dim::J ) - 1, 0 ) );
    else
        EXPECT_EQ( global_grid->blockRank( 0, -1, 0 ), -1 );

    if ( is_dim_periodic[Dim::J] )
        EXPECT_EQ(
            global_grid->blockRank( 0, global_grid->dimNumBlock( Dim::J ), 0 ),
            global_grid->blockRank( 0, 0, 0 ) );
    else
        EXPECT_EQ(
            global_grid->blockRank( 0, global_grid->dimNumBlock( Dim::J ), 0 ),
            -1 );

    if ( is_dim_periodic[Dim::K] )
        EXPECT_EQ( global_grid->blockRank( 0, 0, -1 ),
                   global_grid->blockRank(
                       0, 0, global_grid->dimNumBlock( Dim::K ) - 1 ) );
    else
        EXPECT_EQ( global_grid->blockRank( 0, 0, -1 ), -1 );

    if ( is_dim_periodic[Dim::K] )
        EXPECT_EQ(
            global_grid->blockRank( 0, 0, global_grid->dimNumBlock( Dim::K ) ),
            global_grid->blockRank( 0, 0, 0 ) );
    else
        EXPECT_EQ(
            global_grid->blockRank( 0, 0, global_grid->dimNumBlock( Dim::K ) ),
            -1 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( global_grid, grid_test )
{
    std::array<bool, 3> periodic = { true, true, true };
    gridTest( periodic );
    std::array<bool, 3> not_periodic = { false, false, false };
    gridTest( not_periodic );
}

//---------------------------------------------------------------------------//

} // end namespace Test
