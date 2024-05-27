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

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_SparseDimPartitioner.hpp>
#include <Cabana_Grid_SparseHalo.hpp>
#include <Cabana_Grid_SparseIndexSpace.hpp>
#include <Cabana_Grid_SparseLocalGrid.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>

#include <Cabana_DeepCopy.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

using namespace Cabana::Grid;
using namespace Cabana::Grid::Experimental;

namespace Test
{
// Test data type.
struct TestData
{
    double ds[3];
    float f;
};

// ---------------------------------------------------------------------------
// generate a random partition, to mimic a random simulation status
std::array<std::vector<int>, 3>
generate_random_partition( const std::array<int, 3> ranks_per_dim,
                           const int size_tile_per_dim, const int world_rank )
{
    std::array<std::set<int>, 3> gt_partition_set;
    std::array<std::vector<int>, 3> gt_partition;
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

    return gt_partition;
}

// ---------------------------------------------------------------------------
bool is_ghosted_by_neighbor( const std::array<int, 3> tile_id,
                             const std::array<int, 3> neighbor_id,
                             const std::array<int, 3> low_corner,
                             const std::array<int, 3> high_corner,
                             const int halo_tile_width )
{
    if ( neighbor_id[0] == 0 && neighbor_id[1] == 0 && neighbor_id[2] == 0 )
        return false;
    std::array<int, 3> valid_low;
    std::array<int, 3> valid_high;
    for ( int d = 0; d < 3; ++d )
    {
        if ( neighbor_id[d] == 0 )
        {
            valid_low[d] = low_corner[d];
            valid_high[d] = high_corner[d];
        }
        else if ( neighbor_id[d] == 1 )
        {
            valid_low[d] = high_corner[d] - halo_tile_width;
            valid_high[d] = high_corner[d];
        }
        else // -1
        {
            valid_low[d] = low_corner[d];
            valid_high[d] = low_corner[d] + halo_tile_width;
        }
    }

    bool result = true;
    for ( int d = 0; d < 3; ++d )
    {
        result =
            result && tile_id[d] >= valid_low[d] && tile_id[d] < valid_high[d];
    }

    return result;
}

// sample output: tile_set, tiles, tile_owned_rank, tile_ghosted_ranks
void sample_halo_on_single_rank(
    std::set<std::array<int, 3>>& tile_set,
    std::vector<std::array<int, 3>>& tiles, std::vector<int>& tile_owned_rank,
    std::vector<std::set<std::array<int, 3>>>& tile_ghosted_ranks,
    const int rank_id, const std::array<int, 3> rank_cart_id,
    const std::array<int, 3> low_corner, const std::array<int, 3> high_corner,
    const std::array<int, 3> ranks_per_dim, const float activate_percent,
    const int halo_tile_width )
{
    // compute total halo number
    std::array<int, 3> domain_size = { high_corner[0] - low_corner[0],
                                       high_corner[1] - low_corner[1],
                                       high_corner[2] - low_corner[2] };
    int halo_tile_num = ( domain_size[0] * domain_size[1] * domain_size[2] ) -
                        ( ( domain_size[0] - 2 * halo_tile_width ) *
                          ( domain_size[1] - 2 * halo_tile_width ) *
                          ( domain_size[2] - 2 * halo_tile_width ) );
    int sample_num = static_cast<int>( halo_tile_num * activate_percent );
    // sample_num = 4;

    // start sampling
    int sid = 0;
    int base_index = tiles.size();
    tiles.resize( base_index + sample_num );
    tile_owned_rank.resize( base_index + sample_num, rank_id );
    tile_ghosted_ranks.resize( base_index + sample_num );

    while ( sid < sample_num )
    {
        // sample to determine the range of the halo grid
        int d0 = std::rand() % 3; // which dimension to fix
        int d1 = ( d0 + 1 ) % 3;
        int d2 = ( d0 + 2 ) % 3;
        int lh = std::rand() % 2; // which side, low or high
        std::array<int, 3> new_sample;
        if ( lh == 0 )
        {
            new_sample[d0] = std::rand() % halo_tile_width + low_corner[d0];
            new_sample[d1] = std::rand() % domain_size[d1] + low_corner[d1];
            new_sample[d2] = std::rand() % domain_size[d2] + low_corner[d2];
        }
        else
        {
            new_sample[d0] =
                high_corner[d0] - 1 - std::rand() % halo_tile_width;
            new_sample[d1] = std::rand() % domain_size[d1] + low_corner[d1];
            new_sample[d2] = std::rand() % domain_size[d2] + low_corner[d2];
        }

        // insert sample if it's not in the tile set
        if ( tile_set.find( new_sample ) == tile_set.end() )
        {
            tile_set.insert( new_sample );
            tiles[sid + base_index] = new_sample;
            std::set<std::array<int, 3>> neighbors;
            // check all neighbors and decide whether this tile is in their
            // ghosted space
            for ( int i = -1; i < 2; ++i )
                for ( int j = -1; j < 2; ++j )
                    for ( int k = -1; k < 2; ++k )
                    {
                        if ( i == 0 && j == 0 && k == 0 )
                            continue;
                        std::array<int, 3> n = { rank_cart_id[0] + i,
                                                 rank_cart_id[1] + j,
                                                 rank_cart_id[2] + k };
                        // check if the neighbor is valid
                        if ( n[0] >= 0 && n[0] < ranks_per_dim[0] &&
                             n[1] >= 0 && n[1] < ranks_per_dim[1] &&
                             n[2] >= 0 && n[2] < ranks_per_dim[2] )
                        {
                            if ( is_ghosted_by_neighbor(
                                     new_sample, { i, j, k }, low_corner,
                                     high_corner, halo_tile_width ) )
                            {
                                neighbors.insert( n );
                            }
                        }
                    }
            tile_ghosted_ranks[sid + base_index] = neighbors;
            sid++;
        }
    }
}

// TODO add some doc
void generate_random_halo_tiles(
    std::vector<std::array<int, 3>>& tiles, std::vector<int>& tile_owned_rank,
    std::vector<int>& tile_ghosted_num,
    const std::array<std::vector<int>, 3>& gt_partition,
    const MPI_Comm& cart_comm, const std::array<int, 3> ranks_per_dim,
    const int world_rank, const float activate_percent = 0.1f,
    const int halo_tile_width = 1 )
{
    int comm_size;
    MPI_Comm_size( cart_comm, &comm_size );

    // set to ensure uniqueness of each sampler
    std::set<std::array<int, 3>> tile_set;
    std::vector<std::set<std::array<int, 3>>> tile_ghosted_ranks;

    // only sample on one rank and broadcast to other ranks
    if ( world_rank == 0 )
    {
        for ( int rid = 0; rid < comm_size; ++rid )
        {
            std::array<int, 3> cart_rank;
            MPI_Cart_coords( cart_comm, rid, 3, cart_rank.data() );

            std::array<int, 3> low_corner = { gt_partition[0][cart_rank[0]],
                                              gt_partition[1][cart_rank[1]],
                                              gt_partition[2][cart_rank[2]] };
            std::array<int, 3> high_corner = {
                gt_partition[0][cart_rank[0] + 1],
                gt_partition[1][cart_rank[1] + 1],
                gt_partition[2][cart_rank[2] + 1] };

            sample_halo_on_single_rank( tile_set, tiles, tile_owned_rank,
                                        tile_ghosted_ranks, rid, cart_rank,
                                        low_corner, high_corner, ranks_per_dim,
                                        activate_percent, halo_tile_width );
        }
    }

    // broadcast the sampled results to all ranks
    /// prepare data
    int tile_set_size = static_cast<int>( tiles.size() );
    MPI_Barrier( MPI_COMM_WORLD );
    MPI_Bcast( &tile_set_size, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD );

    std::array<std::vector<int>, 3> tiles_tmp;
    for ( int d = 0; d < 3; ++d )
        tiles_tmp[d].resize( tile_set_size );
    tile_ghosted_num.resize( tile_set_size );
    tile_owned_rank.resize( tile_set_size );
    if ( world_rank == 0 )
    {
        for ( int_least64_t i = 0; i < tile_set_size; ++i )
        {
            tile_ghosted_num[i] =
                static_cast<int>( tile_ghosted_ranks[i].size() );
            for ( int d = 0; d < 3; ++d )
                tiles_tmp[d][i] = tiles[i][d];
        }
    }

    MPI_Barrier( MPI_COMM_WORLD );

    /// broadcast
    MPI_Bcast( tile_owned_rank.data(), tile_set_size, MPI_INT, 0,
               MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD );

    MPI_Bcast( tile_ghosted_num.data(), tile_set_size, MPI_INT, 0,
               MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD );

    for ( int d = 0; d < 3; ++d )
    {
        MPI_Bcast( tiles_tmp[d].data(), tile_set_size, MPI_INT, 0,
                   MPI_COMM_WORLD );
        MPI_Barrier( MPI_COMM_WORLD );
    }

    /// post-process for ranks
    if ( world_rank != 0 )
    {
        tiles.resize( tile_set_size );
        for ( int d = 0; d < 3; ++d )
            for ( int i = 0; i < tile_set_size; ++i )
            {
                tiles[i][d] = tiles_tmp[d][i];
            }
    }
}

// ---------------------------------------------------------------------------
void generate_ground_truth(
    ScatterReduce::Sum, std::unordered_map<std::string, TestData>& ground_truth,
    const TestData base_values, const std::vector<std::array<int, 3>>& tiles,
    const std::vector<int>& tile_ghosted_num )
{
    for ( std::size_t i = 0; i < tiles.size(); ++i )
    {
        std::string key = std::to_string( tiles[i][0] ) + "-" +
                          std::to_string( tiles[i][1] ) + "-" +
                          std::to_string( tiles[i][2] );
        // sum
        for ( int d = 0; d < 3; ++d )
            ground_truth[key].ds[d] =
                base_values.ds[d] * ( 1.0 + tile_ghosted_num[i] * 0.1 );

        ground_truth[key].f =
            base_values.f * ( 1.0f + tile_ghosted_num[i] * 0.1f );
    }
}

void generate_ground_truth(
    ScatterReduce::Max, std::unordered_map<std::string, TestData>& ground_truth,
    const TestData base_values, const std::vector<std::array<int, 3>>& tiles,
    const std::vector<int>& )
{
    for ( std::size_t i = 0; i < tiles.size(); ++i )
    {
        std::string key = std::to_string( tiles[i][0] ) + "-" +
                          std::to_string( tiles[i][1] ) + "-" +
                          std::to_string( tiles[i][2] );
        // max(base values, ghosted values)
        for ( int d = 0; d < 3; ++d )
            ground_truth[key].ds[d] = base_values.ds[d];

        ground_truth[key].f = base_values.f;
    }
}

void generate_ground_truth(
    ScatterReduce::Min, std::unordered_map<std::string, TestData>& ground_truth,
    const TestData base_values, const std::vector<std::array<int, 3>>& tiles,
    const std::vector<int>& )
{
    for ( std::size_t i = 0; i < tiles.size(); ++i )
    {
        std::string key = std::to_string( tiles[i][0] ) + "-" +
                          std::to_string( tiles[i][1] ) + "-" +
                          std::to_string( tiles[i][2] );
        // min(base values, ghosted values)
        for ( int d = 0; d < 3; ++d )
            ground_truth[key].ds[d] = base_values.ds[d] * 0.1;

        ground_truth[key].f = base_values.f * 0.1f;
    }
}

// ---------------------------------------------------------------------------
// convert std::set to device-side view
template <typename T>
auto vec2view( const std::vector<std::array<T, 3>>& in_vec )
    -> Kokkos::View<T* [3], TEST_MEMSPACE>
{
    // set => view (host)
    typedef typename TEST_EXECSPACE::array_layout layout;
    Kokkos::View<T* [3], layout, Kokkos::HostSpace> host_view( "view_host",
                                                               in_vec.size() );
    int i = 0;
    for ( auto it = in_vec.begin(); it != in_vec.end(); ++it )
    {
        for ( int d = 0; d < 3; ++d )
            host_view( i, d ) = ( *it )[d];
        ++i;
    }

    // create tiles view on device
    Kokkos::View<T* [3], TEST_MEMSPACE> dev_view =
        Kokkos::create_mirror_view_and_copy( TEST_MEMSPACE(), host_view );
    return dev_view;
}

// ---------------------------------------------------------------------------
template <typename ReduceOp, typename EntityType>
void haloScatterAndGatherTest( ReduceOp reduce_op, EntityType entity )
{
    using T = float;
    // general scenario and grid information
    constexpr int size_tile_per_dim = 16;
    constexpr int cell_per_tile_dim = 4;
    constexpr int cell_bits_per_tile_dim = 2;
    constexpr int cell_per_tile =
        cell_per_tile_dim * cell_per_tile_dim * cell_per_tile_dim;
    constexpr int size_cell_per_dim = size_tile_per_dim * cell_per_tile_dim;
    int pre_alloc_size = size_cell_per_dim * size_cell_per_dim;

    T cell_size = 0.1f;
    std::array<int, 3> global_num_cell(
        { size_cell_per_dim, size_cell_per_dim, size_cell_per_dim } );
    std::array<T, 3> global_low_corner = { 1.2f, -2.3f, 0.0f };
    std::array<T, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // sparse partitioner
    T max_workload_coeff = 1.5;
    int workload_num =
        size_cell_per_dim * size_cell_per_dim * size_cell_per_dim;
    int num_step_rebalance = 200;
    int max_optimize_iteration = 10;
    SparseDimPartitioner<TEST_MEMSPACE, cell_per_tile_dim> partitioner(
        MPI_COMM_WORLD, max_workload_coeff, workload_num, num_step_rebalance,
        global_num_cell, max_optimize_iteration );

    // rank-related information
    Kokkos::Array<int, 3> cart_rank;
    std::array<int, 3> periodic_dims = { 0, 0, 0 };
    int reordered_cart_ranks = 1;
    MPI_Comm cart_comm;
    int linear_rank;

    // MPI rank topo and rank ID
    auto ranks_per_dim =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, global_num_cell );
    MPI_Cart_create( MPI_COMM_WORLD, 3, ranks_per_dim.data(),
                     periodic_dims.data(), reordered_cart_ranks, &cart_comm );
    MPI_Comm_rank( cart_comm, &linear_rank );
    MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

    // sample sparse partitions
    auto gt_partitions = generate_random_partition(
        ranks_per_dim, size_tile_per_dim, linear_rank );
    partitioner.initializeRecPartition( gt_partitions[0], gt_partitions[1],
                                        gt_partitions[2] );

    // create global mesh+grid, local grid, sparse map and other related things
    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    int halo_width = 2;
    int halo_tile_width = 1;
    auto local_grid =
        createSparseLocalGrid( global_grid, halo_width, cell_per_tile_dim );

    auto sparse_map =
        createSparseMap<TEST_EXECSPACE>( global_mesh, pre_alloc_size );

    // create sparse array
    using DataTypes = Cabana::MemberTypes<double[3], float>;
    auto sparse_layout =
        createSparseArrayLayout<DataTypes>( local_grid, sparse_map, entity );
    auto sparse_array = createSparseArray<TEST_MEMSPACE>(
        std::string( "test_sparse_grid" ), *sparse_layout );

    auto halo = createSparseHalo<TEST_MEMSPACE, cell_bits_per_tile_dim>(
        NodeHaloPattern<3>(), sparse_array );

    // sample valid halos on rank 0 and broadcast to other ranks
    // Kokkos::View<T* [3], TEST_MEMSPACE> tile_view;
    std::vector<std::array<int, 3>> tiles;
    std::vector<int> tile_owned_rank;
    std::vector<int> tile_ghosted_num;
    generate_random_halo_tiles( tiles, tile_owned_rank, tile_ghosted_num,
                                gt_partitions, cart_comm, ranks_per_dim,
                                linear_rank );

    // compute ground truth of each halo grids according to reduce-op type
    // the grid should have base value x assigned by the owner
    // and every other ghosters assign 0.1x to the grid
    // x is a factor multiplied to different data members
    std::unordered_map<std::string, TestData> ground_truth;
    TestData base_values{ { 1.0, 10.0, 100.0 }, 0.1f };

    generate_ground_truth( reduce_op, ground_truth, base_values, tiles,
                           tile_ghosted_num );

    // register owned and ghosted halos in sparse map
    auto tiles_view = vec2view( tiles );
    {
        Kokkos::Array<int, 3> all_low = {
            gt_partitions[0][cart_rank[0]] - halo_tile_width,
            gt_partitions[1][cart_rank[1]] - halo_tile_width,
            gt_partitions[2][cart_rank[2]] - halo_tile_width };
        Kokkos::Array<int, 3> all_high = {
            gt_partitions[0][cart_rank[0] + 1] + halo_tile_width,
            gt_partitions[1][cart_rank[1] + 1] + halo_tile_width,
            gt_partitions[2][cart_rank[2] + 1] + halo_tile_width };

        Kokkos::parallel_for(
            "register sparse map",
            Kokkos::RangePolicy<TEST_EXECSPACE>( 0, tiles.size() ),
            KOKKOS_LAMBDA( const int id ) {
                if ( tiles_view( id, 0 ) >= all_low[0] &&
                     tiles_view( id, 0 ) < all_high[0] &&
                     tiles_view( id, 1 ) >= all_low[1] &&
                     tiles_view( id, 1 ) < all_high[1] &&
                     tiles_view( id, 2 ) >= all_low[2] &&
                     tiles_view( id, 2 ) < all_high[2] )
                {
                    sparse_map.insertTile( tiles_view( id, 0 ),
                                           tiles_view( id, 1 ),
                                           tiles_view( id, 2 ) );
                }
            } );

        sparse_array->resize( sparse_map.sizeCell() );
        halo->template register_halo<TEST_EXECSPACE>( sparse_map );
        MPI_Barrier( MPI_COMM_WORLD );
    }

    // assign values on sparse array
    /// every valid owned halo would have value x
    /// every valid ghosted halo would have value 0.1x
    /// no other grids will be registered

    Kokkos::View<int* [3], TEST_MEMSPACE> info(
        Kokkos::ViewAllocateWithoutInitializing( "tile_cell_info" ),
        sparse_array->size() );
    {
        Kokkos::Array<int, 3> low = { gt_partitions[0][cart_rank[0]],
                                      gt_partitions[1][cart_rank[1]],
                                      gt_partitions[2][cart_rank[2]] };
        Kokkos::Array<int, 3> high = { gt_partitions[0][cart_rank[0] + 1],
                                       gt_partitions[1][cart_rank[1] + 1],
                                       gt_partitions[2][cart_rank[2] + 1] };
        double base_0[3] = { base_values.ds[0], base_values.ds[1],
                             base_values.ds[2] };
        float base_1 = base_values.f;
        auto& map = sparse_map;
        auto array = *sparse_array;
        Kokkos::parallel_for(
            "assign values to sparse array",
            Kokkos::RangePolicy<TEST_EXECSPACE>( 0, map.capacity() ),
            KOKKOS_LAMBDA( const int id ) {
                if ( map.valid_at( id ) )
                {
                    auto tid = map.value_at( id );
                    auto tkey = map.key_at( id );
                    int ti, tj, tk;
                    map.key2ijk( tkey, ti, tj, tk );
                    // owned tiles
                    if ( ti >= low[0] && ti < high[0] && tj >= low[1] &&
                         tj < high[1] && tk >= low[2] && tk < high[2] )
                    {
                        for ( int ci = 0; ci < cell_per_tile_dim; ci++ )
                            for ( int cj = 0; cj < cell_per_tile_dim; cj++ )
                                for ( int ck = 0; ck < cell_per_tile_dim; ck++ )
                                {
                                    int cid = map.cell_local_id( ci, cj, ck );

                                    array.template get<0>( tid, cid, 0 ) =
                                        base_0[0];
                                    array.template get<0>( tid, cid, 1 ) =
                                        base_0[1];
                                    array.template get<0>( tid, cid, 2 ) =
                                        base_0[2];
                                    array.template get<1>( tid, cid ) = base_1;
                                }
                    }
                    // ghosted tiles
                    else
                    {
                        for ( int ci = 0; ci < cell_per_tile_dim; ci++ )
                            for ( int cj = 0; cj < cell_per_tile_dim; cj++ )
                                for ( int ck = 0; ck < cell_per_tile_dim; ck++ )
                                {
                                    int cid = map.cell_local_id( ci, cj, ck );
                                    array.template get<0>( tid, cid, 0 ) =
                                        base_0[0] * 0.1;
                                    array.template get<0>( tid, cid, 1 ) =
                                        base_0[1] * 0.1;
                                    array.template get<0>( tid, cid, 2 ) =
                                        base_0[2] * 0.1;
                                    array.template get<1>( tid, cid ) =
                                        base_1 * 0.1f;
                                }
                    }

                    for ( int ci = 0; ci < cell_per_tile_dim; ci++ )
                        for ( int cj = 0; cj < cell_per_tile_dim; cj++ )
                            for ( int ck = 0; ck < cell_per_tile_dim; ck++ )
                            {
                                int cid = map.cell_local_id( ci, cj, ck );
                                info( tid * cell_per_tile + cid, 0 ) = ti;
                                info( tid * cell_per_tile + cid, 1 ) = tj;
                                info( tid * cell_per_tile + cid, 2 ) = tk;
                            }
                }
            } );
    }
    MPI_Barrier( MPI_COMM_WORLD );
    // halo scatter and gather
    /// false means the heighbors' halo counting information is not
    /// collected
    halo->scatter( TEST_EXECSPACE(), reduce_op, *sparse_array, false );
    /// halo counting info already collected in the previous scatter, thus true
    /// and no need to recount again
    halo->gather( TEST_EXECSPACE(), *sparse_array, true );
    MPI_Barrier( MPI_COMM_WORLD );

    // check results
    auto mirror = Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       sparse_array->aosoa() );
    auto info_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), info );
    auto slice_0 = Cabana::slice<0>( mirror );
    auto slice_1 = Cabana::slice<1>( mirror );

    for ( unsigned long cid = 0; cid < mirror.size(); ++cid )
    {
        std::string tijk = std::to_string( info_mirror( cid, 0 ) ) + "-" +
                           std::to_string( info_mirror( cid, 1 ) ) + "-" +
                           std::to_string( info_mirror( cid, 2 ) );

        if ( ground_truth.find( tijk ) == ground_truth.end() )
            throw std::runtime_error(
                std::string( "[ERROR] didn't find tile [[" ) + tijk +
                std::string( "]]" ) );
        else
        {
            const auto& gt_values = ground_truth[tijk];
            EXPECT_DOUBLE_EQ( slice_0( cid, 0 ), gt_values.ds[0] );
            EXPECT_DOUBLE_EQ( slice_0( cid, 1 ), gt_values.ds[1] );
            EXPECT_DOUBLE_EQ( slice_0( cid, 2 ), gt_values.ds[2] );

            EXPECT_FLOAT_EQ( slice_1( cid ), gt_values.f );
        }
    }
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, sparse_halo_scatter_and_gather_sum )
{
    haloScatterAndGatherTest( ScatterReduce::Sum(), Node() );
}

// TODO: test min/max
// no need to check replace op since it is already called and tested inside
// SparseHalo::gather(...)"
// TEST( TEST_CATEGORY, sparse_halo_scatter_and_gather_max ) {}
// TEST( TEST_CATEGORY, sparse_halo_scatter_and_gather_min ) {}

}; // end namespace Test
