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

#include <Cajita_SparseMapDynamicPartitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Kokkos_Core.hpp>

#include <ctime>
#include <gtest/gtest.h>
#include <set>

#include <mpi.h>

using namespace Cajita;

namespace Test
{

/*!
  \brief In this test, every cell in the whole domain is registered, so the
  ground truth partition should be the average partition
*/
void uniform_distribution_automatic_rank()
{
    // define the domain size
    constexpr int size_tile_per_dim = 16;
    constexpr int cell_per_tile_dim = 4;
    constexpr int size_per_dim = size_tile_per_dim * cell_per_tile_dim;
    constexpr int total_size = size_per_dim * size_per_dim * size_per_dim;

    // some settings for partitioner
    float max_workload_coeff = 1.5;
    int workload_num = total_size;
    int num_step_rebalance = 100;
    int max_optimize_iteration = 10;
    std::array<int, 3> global_cells_per_dim = {
        size_tile_per_dim * cell_per_tile_dim,
        size_tile_per_dim * cell_per_tile_dim,
        size_tile_per_dim * cell_per_tile_dim };

    // partitioner
    SparseMapDynamicPartitioner<TEST_DEVICE, cell_per_tile_dim> partitioner(
        MPI_COMM_WORLD, max_workload_coeff, workload_num, num_step_rebalance,
        global_cells_per_dim, max_optimize_iteration );

    // check the value of some pre-computed constants
    auto cbptd = partitioner.cell_bits_per_tile_dim;
    EXPECT_EQ( cbptd, 2 );

    auto cnptd = partitioner.cell_num_per_tile_dim;
    EXPECT_EQ( cnptd, 4 );

    // test ranks per dim
    auto ranks_per_dim =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, global_cells_per_dim );

    EXPECT_EQ( ranks_per_dim[0] >= 1, true );
    EXPECT_EQ( ranks_per_dim[1] >= 1, true );
    EXPECT_EQ( ranks_per_dim[2] >= 1, true );

    // initialize partitions (averagely divide the whole domain)
    std::array<std::vector<int>, 3> rec_partitions;
    for ( int d = 0; d < 3; ++d )
    {
        int ele = size_tile_per_dim / ranks_per_dim[d];
        int part = 0;
        for ( int i = 0; i < ranks_per_dim[d]; ++i )
        {
            rec_partitions[d].push_back( part );
            part += ele;
        }
        rec_partitions[d].push_back( size_tile_per_dim );
    }
    partitioner.initializeRecPartition( rec_partitions[0], rec_partitions[1],
                                        rec_partitions[2] );

    // test getCurrentPartition function
    {
        auto part = partitioner.getCurrentPartition();
        for ( int d = 0; d < 3; ++d )
            for ( int id = 0; id < ranks_per_dim[d] + 1; id++ )
                EXPECT_EQ( part[d][id], rec_partitions[d][id] );
    }

    // test ownedCellsPerDimension function
    // Ground truth should be the average cell num in ranks (based on the inital
    // partition)
    std::array<int, 3> cart_rank;
    std::array<int, 3> periodic_dims = { 0, 0, 0 };
    int reordered_cart_ranks = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create( MPI_COMM_WORLD, 3, ranks_per_dim.data(),
                     periodic_dims.data(), reordered_cart_ranks, &cart_comm );
    int linear_rank;
    MPI_Comm_rank( cart_comm, &linear_rank );
    // make a new communicater with MPI_Cart_create
    MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

    auto owned_cells_per_dim = partitioner.ownedCellsPerDimension( cart_comm );
    auto owned_tiles_per_dim = partitioner.ownedTilesPerDimension( cart_comm );
    float gt_imbalance_factor = 1.0f;
    for ( int d = 0; d < 3; ++d )
    {
        auto gt_tile = rec_partitions[d][cart_rank[d] + 1] -
                       rec_partitions[d][cart_rank[d]];
        EXPECT_EQ( owned_tiles_per_dim[d], gt_tile );
        EXPECT_EQ( owned_cells_per_dim[d], gt_tile * cell_per_tile_dim );
        gt_imbalance_factor *= gt_tile;
    }
    gt_imbalance_factor /=
        static_cast<float>( size_tile_per_dim * size_tile_per_dim *
                            size_tile_per_dim ) /
        ranks_per_dim[0] / ranks_per_dim[1] / ranks_per_dim[2];

    // initialize sparseMap, register every tile on every MPI rank
    // basic settings for sparseMap
    double cell_size = 0.1;
    int pre_alloc_size = size_per_dim * size_per_dim;
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_cells_per_dim[0],
        global_low_corner[1] + cell_size * global_cells_per_dim[1],
        global_low_corner[2] + cell_size * global_cells_per_dim[2] };
    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_cells_per_dim );
    // create a new sparseMap
    auto sis = createSparseMap<TEST_EXECSPACE>( global_mesh, pre_alloc_size );
    // register tiles to the sparseMap
    Kokkos::parallel_for(
        "insert_cell_to_sparse_map",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, size_per_dim ),
        KOKKOS_LAMBDA( int i ) {
            for ( int j = 0; j < size_per_dim; j++ )
                for ( int k = 0; k < size_per_dim; k++ )
                {
                    sis.insertCell( i, j, k );
                }
        } );
    Kokkos::fence();

    // compute workload and do partition optimization
    partitioner.setLocalWorkloadBySparseMap( sis, MPI_COMM_WORLD );
    partitioner.optimizePartition( MPI_COMM_WORLD );

    // check results (should be the same as the average partition)
    owned_cells_per_dim = partitioner.ownedCellsPerDimension( cart_comm );
    for ( int d = 0; d < 3; ++d )
    {
        auto gt_tile = rec_partitions[d][cart_rank[d] + 1] -
                       rec_partitions[d][cart_rank[d]];

        EXPECT_EQ( owned_cells_per_dim[d], gt_tile * cell_per_tile_dim );
    }

    auto imbalance_factor = partitioner.computeImbalanceFactor( cart_comm );
    EXPECT_FLOAT_EQ( imbalance_factor, gt_imbalance_factor );
}

auto generate_random_tiles( const std::array<std::vector<int>, 3>& gt_partition,
                            const Kokkos::Array<int, 3>& cart_rank,
                            const int size_tile_per_dim,
                            int occupy_tile_num_per_rank )
    -> Kokkos::View<int* [3], TEST_MEMSPACE>
{
    // register valid tiles in each MPI rank
    // compute the sub-domain size (divided by the ground truth partition)
    const int area_size = size_tile_per_dim * size_tile_per_dim;
    occupy_tile_num_per_rank = occupy_tile_num_per_rank >= area_size
                                   ? area_size
                                   : occupy_tile_num_per_rank;
    std::set<std::array<int, 3>> tiles_set;

    int start[3], size[3];
    for ( int d = 0; d < 3; ++d )
    {
        start[d] = gt_partition[d][cart_rank[d]];
        size[d] = gt_partition[d][cart_rank[d] + 1] - start[d];
    }

    // insert the corner tiles to the set, to ensure the uniqueness of the
    // ground truth partition
    tiles_set.insert( { start[0], start[1], start[2] } );
    tiles_set.insert( { start[0] + size[0] - 1, start[1] + size[1] - 1,
                        start[2] + size[2] - 1 } );

    // insert random tiles to the set
    while ( static_cast<int>( tiles_set.size() ) < occupy_tile_num_per_rank )
    {
        int rand_offset[3];
        for ( int d = 0; d < 3; ++d )
            rand_offset[d] = std::rand() % size[d];
        tiles_set.insert( { start[0] + rand_offset[0],
                            start[1] + rand_offset[1],
                            start[2] + rand_offset[2] } );
    }

    // tiles_set => tiles_view (host)
    typedef typename TEST_EXECSPACE::array_layout layout;
    Kokkos::View<int* [3], layout, Kokkos::HostSpace> tiles_view_host(
        "tiles_view_host", tiles_set.size() );
    int i = 0;
    for ( auto it = tiles_set.begin(); it != tiles_set.end(); ++it )
    {
        for ( int d = 0; d < 3; ++d )
            tiles_view_host( i, d ) = ( *it )[d];
        i++;
    }

    // create tiles view on device
    Kokkos::View<int* [3], TEST_MEMSPACE> tiles_view =
        Kokkos::create_mirror_view_and_copy( TEST_MEMSPACE(), tiles_view_host );
    return tiles_view;
}

/*!
  \brief In this test, the ground truth partition is first randomly chosen, then
  a given number of tiles are regiestered on each rank (the most bottom-left and
  top-right tiles are always registered to ensure the uniqueness of the ground
  truth partition )
  \param occupy_num_per_rank the tile number that will be registered on each MPI
  rank
*/
void random_distribution_automatic_rank( int occupy_num_per_rank )
{
    // define the domain size
    constexpr int size_tile_per_dim = 32;
    constexpr int cell_per_tile_dim = 4;
    constexpr int size_per_dim = size_tile_per_dim * cell_per_tile_dim;
    constexpr int total_size = size_per_dim * size_per_dim * size_per_dim;
    srand( time( 0 ) );

    // some settings for partitioner
    float max_workload_coeff = 1.5;
    int particle_num = total_size;
    int num_step_rebalance = 100;
    int max_optimize_iteration = 10;

    std::array<int, 3> global_cells_per_dim = { size_per_dim, size_per_dim,
                                                size_per_dim };

    // partitioner
    SparseMapDynamicPartitioner<TEST_DEVICE, cell_per_tile_dim> partitioner(
        MPI_COMM_WORLD, max_workload_coeff, particle_num, num_step_rebalance,
        global_cells_per_dim, max_optimize_iteration );

    // check the value of some pre-computed constants
    auto cbptd = partitioner.cell_bits_per_tile_dim;
    EXPECT_EQ( cbptd, 2 );

    auto cnptd = partitioner.cell_num_per_tile_dim;
    EXPECT_EQ( cnptd, 4 );

    // ranks per dim test
    auto ranks_per_dim =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, global_cells_per_dim );

    EXPECT_EQ( ranks_per_dim[0] >= 1, true );
    EXPECT_EQ( ranks_per_dim[1] >= 1, true );
    EXPECT_EQ( ranks_per_dim[2] >= 1, true );

    // compute the rank ID
    Kokkos::Array<int, 3> cart_rank;
    std::array<int, 3> periodic_dims = { 0, 0, 0 };
    int reordered_cart_ranks = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create( MPI_COMM_WORLD, 3, ranks_per_dim.data(),
                     periodic_dims.data(), reordered_cart_ranks, &cart_comm );
    int linear_rank;
    MPI_Comm_rank( cart_comm, &linear_rank );
    MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

    // generate random ground truth partition on the root rank
    std::array<std::set<int>, 3> gt_partition_set;
    std::array<std::vector<int>, 3> gt_partition;
    int world_rank, world_size;
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    for ( int d = 0; d < 3; ++d )
    {
        gt_partition[d].resize( ranks_per_dim[d] + 1 );
    }

    if ( world_rank == 0 )
    {
        for ( int d = 0; d < 3; ++d )
        {
            gt_partition_set[d].insert( 0 );
            while ( static_cast<int>( gt_partition_set[d].size() ) <
                    ranks_per_dim[d] )
            {
                int rand_num = std::rand() % size_tile_per_dim;
                gt_partition_set[d].insert( rand_num );
            }
            gt_partition_set[d].insert( size_tile_per_dim );
            int i = 0;
            for ( auto it = gt_partition_set[d].begin();
                  it != gt_partition_set[d].end(); ++it )
            {
                gt_partition[d][i++] = *it;
            }
        }
    }

    // broadcast the ground truth partition to all ranks
    for ( int d = 0; d < 3; ++d )
    {
        MPI_Barrier( MPI_COMM_WORLD );
        MPI_Bcast( gt_partition[d].data(), gt_partition[d].size(), MPI_INT, 0,
                   MPI_COMM_WORLD );
        MPI_Barrier( MPI_COMM_WORLD );
    }

    // init partitions (average partition)
    std::array<std::vector<int>, 3> rec_partitions;
    for ( int d = 0; d < 3; ++d )
    {
        int ele = size_tile_per_dim / ranks_per_dim[d];
        int part = 0;
        for ( int i = 0; i < ranks_per_dim[d]; ++i )
        {
            rec_partitions[d].push_back( part );
            part += ele;
        }
        rec_partitions[d].push_back( size_tile_per_dim );
    }

    partitioner.initializeRecPartition( rec_partitions[0], rec_partitions[1],
                                        rec_partitions[2] );

    // basic settings for domain size and position
    double cell_size = 0.1;
    int pre_alloc_size = size_per_dim * size_per_dim;
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_cells_per_dim[0],
        global_low_corner[1] + cell_size * global_cells_per_dim[1],
        global_low_corner[2] + cell_size * global_cells_per_dim[2] };

    // randomly generate a fixed number of tiles on every MPI rank
    auto tiles_view = generate_random_tiles(
        gt_partition, cart_rank, size_tile_per_dim, occupy_num_per_rank );
    // create a new sparseMap
    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_cells_per_dim );
    auto sis = createSparseMap<TEST_EXECSPACE>( global_mesh, pre_alloc_size );
    // register selected tiles to the sparseMap
    Kokkos::parallel_for(
        "insert_tile_to_sparse_map",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, tiles_view.extent( 0 ) ),
        KOKKOS_LAMBDA( int id ) {
            sis.insertTile( tiles_view( id, 0 ), tiles_view( id, 1 ),
                            tiles_view( id, 2 ) );
        } );
    Kokkos::fence();

    // compute workload from a sparseMap and do partition optimization
    dynamic_cast<SparseMapDynamicPartitioner<TEST_DEVICE, cell_per_tile_dim>*>(
        &partitioner )
        ->setLocalWorkloadBySparseMap( sis, MPI_COMM_WORLD );
    partitioner.optimizePartition( MPI_COMM_WORLD );

    // check results (should be the same as the gt_partition)
    auto part = partitioner.getCurrentPartition();
    for ( int d = 0; d < 3; ++d )
    {
        for ( int id = 0; id < ranks_per_dim[d] + 1; id++ )
            EXPECT_EQ( part[d][id], gt_partition[d][id] );
    }

    auto imbalance_factor = partitioner.computeImbalanceFactor( cart_comm );
    EXPECT_FLOAT_EQ( imbalance_factor, 1.0f );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( sparse_dim_partitioner, sparse_dim_partitioner_uniform_test )
{
    uniform_distribution_automatic_rank();
}
TEST( sparse_dim_partitioner, sparse_dim_partitioner_random_tile_test )
{
    random_distribution_automatic_rank( 32 );
}
//---------------------------------------------------------------------------//
} // end namespace Test
