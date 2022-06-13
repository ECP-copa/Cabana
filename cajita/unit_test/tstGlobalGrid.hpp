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

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_Partitioner.hpp>
#include <Cajita_DynamicPartitioner.hpp>
#include <Cajita_Types.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void gridTest3d( const std::array<bool, 3>& is_dim_periodic )
{
    // Let MPI compute the partitioning for this test.
    DimBlockPartitioner<3> partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 47, 38, 53 };
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

    auto owned_cells_partitioner = partitioner.ownedCellsPerDimension(
        global_grid->comm(), global_num_cell );
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( global_grid->ownedNumCell( d ), owned_cells_partitioner[d] );

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

    // Check setNumCellAndOffset
    // todo(sschulz): Need to have a more realistic change, since there might
    // be some checking involved within the function.
    std::array<int, 3> num_cell = { 314, 314, 314 };
    global_grid->setNumCellAndOffset( num_cell, num_cell );
    for ( std::size_t i = 0; i < 3; ++i )
    {
        EXPECT_EQ( global_grid->ownedNumCell( i ), num_cell[i] );
        EXPECT_EQ( global_grid->globalOffset( i ), num_cell[i] );
    }
}

//---------------------------------------------------------------------------//
void gridTest2d( const std::array<bool, 2>& is_dim_periodic )
{
    // Let MPI compute the partitioning for this test.
    DimBlockPartitioner<2> partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 2> global_num_cell = { 47, 38 };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Check the number of entities.
    for ( int d = 0; d < 2; ++d )
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

    // Number of J faces.
    EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::J>(), Dim::I ),
               global_num_cell[Dim::I] );
    if ( is_dim_periodic[Dim::I] )
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::J>(), Dim::J ),
                   global_num_cell[Dim::J] );
    else
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::J>(), Dim::J ),
                   global_num_cell[Dim::J] + 1 );

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
    std::vector<int> cart_dims( 2 );
    std::vector<int> cart_period( 2 );
    std::vector<int> cart_rank( 2 );
    MPI_Cart_get( grid_comm, 2, cart_dims.data(), cart_period.data(),
                  cart_rank.data() );
    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_EQ( cart_period[d], is_dim_periodic[d] );
        EXPECT_EQ( global_grid->dimBlockId( d ), cart_rank[d] );
        EXPECT_EQ( global_grid->dimNumBlock( d ), ranks_per_dim[d] );
    }

    auto owned_cells_partitioner = partitioner.ownedCellsPerDimension(
        global_grid->comm(), global_num_cell );
    for ( int d = 0; d < 2; ++d )
        EXPECT_EQ( global_grid->ownedNumCell( d ), owned_cells_partitioner[d] );

    for ( int d = 0; d < 2; ++d )
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
        EXPECT_EQ( global_grid->blockRank( -1, 0 ),
                   global_grid->blockRank(
                       global_grid->dimNumBlock( Dim::I ) - 1, 0 ) );
    else
        EXPECT_EQ( global_grid->blockRank( -1, 0 ), -1 );

    if ( is_dim_periodic[Dim::I] )
        EXPECT_EQ(
            global_grid->blockRank( global_grid->dimNumBlock( Dim::I ), 0 ),
            global_grid->blockRank( 0, 0 ) );
    else
        EXPECT_EQ(
            global_grid->blockRank( global_grid->dimNumBlock( Dim::I ), 0 ),
            -1 );

    if ( is_dim_periodic[Dim::J] )
        EXPECT_EQ( global_grid->blockRank( 0, -1 ),
                   global_grid->blockRank(
                       0, global_grid->dimNumBlock( Dim::J ) - 1 ) );
    else
        EXPECT_EQ( global_grid->blockRank( 0, -1 ), -1 );

    if ( is_dim_periodic[Dim::J] )
        EXPECT_EQ(
            global_grid->blockRank( 0, global_grid->dimNumBlock( Dim::J ) ),
            global_grid->blockRank( 0, 0 ) );
    else
        EXPECT_EQ(
            global_grid->blockRank( 0, global_grid->dimNumBlock( Dim::J ) ),
            -1 );

    // Check setNumCellAndOffset
    // todo(sschulz): Need to have a more realistic change, since there might
    // be some checking involved within the function.
    std::array<int, 2> num_cell = { 314, 314 };
    global_grid->setNumCellAndOffset( num_cell, num_cell );
    for ( std::size_t i = 0; i < 2; ++i )
    {
        EXPECT_EQ( global_grid->ownedNumCell( i ), num_cell[i] );
        EXPECT_EQ( global_grid->globalOffset( i ), num_cell[i] );
    }
}

void sparseGridTest3d()
{
    // Spares grid related settings
    constexpr int cell_per_tile_dim = 4;
    const std::array<bool, 3> is_dim_periodic = { false, false, false };

    // Create Sparse global mesh
    std::array<int, 3> global_num_tile = { 16, 8, 4 };
    std::array<int, 3> global_num_cell = {
        global_num_tile[0] * cell_per_tile_dim,
        global_num_tile[1] * cell_per_tile_dim,
        global_num_tile[2] * cell_per_tile_dim };

    double cell_size = 0.1;
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };

    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Sparse paritioner
    float max_workload_coeff = 1.5;
    int workload_num =
        global_num_cell[0] * global_num_cell[1] * global_num_cell[2];
    int num_step_rebalance = 100;
    int max_optimize_iteration = 10;

    DynamicPartitioner<TEST_DEVICE, cell_per_tile_dim> partitioner(
        MPI_COMM_WORLD, max_workload_coeff, workload_num, num_step_rebalance,
        global_num_cell, max_optimize_iteration );

    // test ranks per dim
    auto ranks_per_dim =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, global_num_cell );
    // initialize partitions (averagely divide the whole domain)
    std::array<std::vector<int>, 3> rec_partitions;
    for ( int d = 0; d < 3; ++d )
    {
        int ele = global_num_tile[d] / ranks_per_dim[d];
        int part = 0;
        for ( int i = 0; i < ranks_per_dim[d]; ++i )
        {
            rec_partitions[d].push_back( part );
            part += ele;
        }
        rec_partitions[d].push_back( global_num_tile[d] );
    }
    partitioner.initializeRecPartition( rec_partitions[0], rec_partitions[1],
                                        rec_partitions[2] );

    // Create spares global grid
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

    // Check the number of faces entries
    {
        // Number of I faces
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::I>(), Dim::I ),
                   global_num_cell[Dim::I] + 1 );
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::I>(), Dim::J ),
                   global_num_cell[Dim::J] );
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::I>(), Dim::K ),
                   global_num_cell[Dim::K] );

        // Number of J faces.
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::J>(), Dim::I ),
                   global_num_cell[Dim::I] );

        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::J>(), Dim::J ),
                   global_num_cell[Dim::J] + 1 );
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::J>(), Dim::K ),
                   global_num_cell[Dim::K] );

        // Number of K faces.
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::K>(), Dim::I ),
                   global_num_cell[Dim::I] );
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::K>(), Dim::J ),
                   global_num_cell[Dim::J] );
        EXPECT_EQ( global_grid->globalNumEntity( Face<Dim::K>(), Dim::K ),
                   global_num_cell[Dim::K] + 1 );
    }

    // Check the number of edges entires
    {
        // Number of I edges
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::I>(), Dim::I ),
                   global_num_cell[Dim::I] );

        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::I>(), Dim::J ),
                   global_num_cell[Dim::J] + 1 );

        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::I>(), Dim::K ),
                   global_num_cell[Dim::K] + 1 );

        // Number of J edges

        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::J>(), Dim::I ),
                   global_num_cell[Dim::I] + 1 );
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::J>(), Dim::J ),
                   global_num_cell[Dim::J] );

        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::J>(), Dim::K ),
                   global_num_cell[Dim::K] + 1 );

        // Number of K edges

        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::K>(), Dim::I ),
                   global_num_cell[Dim::I] + 1 );

        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::K>(), Dim::J ),
                   global_num_cell[Dim::J] + 1 );
        EXPECT_EQ( global_grid->globalNumEntity( Edge<Dim::K>(), Dim::K ),
                   global_num_cell[Dim::K] );
    }

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

    auto owned_cells_partitioner =
        partitioner.ownedCellsPerDimension( global_grid->comm() );
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( global_grid->ownedNumCell( d ), owned_cells_partitioner[d] );

    // Update partitioner and check num cell and global offset
    auto part = partitioner.getCurrentPartition();
    for ( int d = 0; d < 3; ++d )
        for ( int id = 1; id < ranks_per_dim[d]; id++ )
            part[d][id] += 1;

    partitioner.initializeRecPartition( part[0], part[1], part[2] );

    std::array<int, 3> new_owned_num_cell;
    std::array<int, 3> new_global_cell_offset;
    partitioner.ownedCellInfo( global_grid->comm(), global_num_cell,
                               new_owned_num_cell, new_global_cell_offset );

    // Check setNumCellAndOffset
    // todo(sschulz): Need to have a more realistic change, since there might
    // be some checking involved within the function.
    global_grid->setNumCellAndOffset( new_owned_num_cell,
                                      new_global_cell_offset );
    for ( std::size_t i = 0; i < 3; ++i )
        EXPECT_EQ( global_grid->globalOffset( i ), new_global_cell_offset[i] );

    owned_cells_partitioner =
        partitioner.ownedCellsPerDimension( global_grid->comm() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( owned_cells_partitioner[d], new_owned_num_cell[d] );
        EXPECT_EQ( global_grid->ownedNumCell( d ), owned_cells_partitioner[d] );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( global_grid, 3d_grid_test )
{
    std::array<bool, 3> periodic = { true, true, true };
    gridTest3d( periodic );
    std::array<bool, 3> not_periodic = { false, false, false };
    gridTest3d( not_periodic );
}

TEST( global_grid, 2d_grid_test )
{
    std::array<bool, 2> periodic = { true, true };
    gridTest2d( periodic );
    std::array<bool, 2> not_periodic = { false, false };
    gridTest2d( not_periodic );
}

TEST( global_grid, 3d_sparse_grid_test ) { sparseGridTest3d(); }

//---------------------------------------------------------------------------//

} // end namespace Test
