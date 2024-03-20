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

#include <Cabana_Grid_SparseArray.hpp>
#include <Cabana_Grid_SparseDimPartitioner.hpp>
#include <Cabana_Grid_SparseLocalGrid.hpp>

#include <Cabana_DeepCopy.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

using namespace Cabana::Grid;
using namespace Cabana::Grid::Experimental;

namespace Test
{

// generate a random partition, to mimic a random simulation status
std::array<std::vector<int>, 3>
generate_random_partition( std::array<int, 3> ranks_per_dim,
                           int size_tile_per_dim )
{
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

    return gt_partition;
}

// convert std::set to device-side view
template <typename T>
auto set2view( const std::set<std::array<T, 3>>& in_set )
    -> Kokkos::View<T* [3], TEST_MEMSPACE>
{
    // set => view (host)
    typedef typename TEST_EXECSPACE::array_layout layout;
    Kokkos::View<T* [3], layout, Kokkos::HostSpace> host_view( "view_host",
                                                               in_set.size() );
    int i = 0;
    for ( auto it = in_set.begin(); it != in_set.end(); ++it )
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

// return random generated particles and occupied tile numbers (last two params)
template <typename T>
void generate_random_particles( const int particle_number,
                                const std::array<int, 3>& part_start,
                                const std::array<int, 3>& part_end,
                                const int cell_per_tile_dim,
                                const std::array<T, 3> global_low_corner,
                                const T cell_size,
                                std::set<std::array<int, 3>>& tile_set,
                                std::set<std::array<T, 3>>& par_pos_set )
{
    // range of particle positions
    T start[3], size[3];
    for ( int d = 0; d < 3; ++d )
    {
        // because each particle will activate three around tiles, we apply
        // 1.01 cell_size offset compared to the real partition to ensure
        // all the activated tiles sit inside the valid partition range
        start[d] = global_low_corner[d] +
                   cell_size * ( 2.01f + cell_per_tile_dim * (T)part_start[d] );
        size[d] =
            cell_size *
            ( cell_per_tile_dim * (T)( part_end[d] - part_start[d] ) - 4.02f );
    }

    // insert random particles to the set
    while ( static_cast<int>( par_pos_set.size() ) < particle_number )
    {
        T rand_offset[3];
        for ( int d = 0; d < 3; ++d )
            rand_offset[d] = (T)std::rand() / (T)RAND_MAX * size[d];
        std::array<T, 3> new_pos = { start[0] + rand_offset[0],
                                     start[1] + rand_offset[1],
                                     start[2] + rand_offset[2] };
        par_pos_set.insert( new_pos );

        std::array<int, 3> grid_base;
        for ( int d = 0; d < 3; ++d )
        {
            grid_base[d] =
                int( std::lround( ( new_pos[d] - global_low_corner[d] ) /
                                  cell_size ) ) -
                1;
        }

        for ( int i = 0; i <= 2; i++ )
            for ( int j = 0; j <= 2; j++ )
                for ( int k = 0; k <= 2; k++ )
                {
                    tile_set.insert( {
                        ( grid_base[0] + i ) / cell_per_tile_dim,
                        ( grid_base[1] + j ) / cell_per_tile_dim,
                        ( grid_base[2] + k ) / cell_per_tile_dim,
                    } );
                }
    }
}
template <typename EntityType>
void sparse_array_test( int par_num, EntityType entity )
{
    // basic scenario information
    constexpr int size_tile_per_dim = 32;
    constexpr int cell_per_tile_dim = 4;
    constexpr int cell_per_tile =
        cell_per_tile_dim * cell_per_tile_dim * cell_per_tile_dim;
    constexpr int size_per_dim = size_tile_per_dim * cell_per_tile_dim;

    int pre_alloc_size = size_per_dim * size_per_dim;

    using T = float;
    // Create global mesh
    T cell_size = 0.1f;
    std::array<int, 3> global_num_cell(
        { size_per_dim, size_per_dim, size_per_dim } );
    // global low corners: random numbuers
    std::array<T, 3> global_low_corner = { 1.0f, -1.0f, 0.0f };
    std::array<T, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // sparse partitioner
    T max_workload_coeff = 1.5;
    int workload_num = size_per_dim * size_per_dim * size_per_dim;
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

    // scene initialization
    auto gt_partitions =
        generate_random_partition( ranks_per_dim, size_tile_per_dim );
    partitioner.initializeRecPartition( gt_partitions[0], gt_partitions[1],
                                        gt_partitions[2] );

    std::set<std::array<int, 3>> tile_set;
    std::set<std::array<T, 3>> par_pos_set;
    generate_random_particles( par_num,
                               { gt_partitions[0][cart_rank[0]],
                                 gt_partitions[1][cart_rank[1]],
                                 gt_partitions[2][cart_rank[2]] },
                               { gt_partitions[0][cart_rank[0] + 1],
                                 gt_partitions[1][cart_rank[1] + 1],
                                 gt_partitions[2][cart_rank[2] + 1] },
                               cell_per_tile_dim, global_low_corner, cell_size,
                               tile_set, par_pos_set );
    auto tile_view = set2view( tile_set );
    auto par_view = set2view( par_pos_set );

    // mesh/grid related initialization
    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    int halo_width = 2;
    auto local_grid =
        createSparseLocalGrid( global_grid, halo_width, cell_per_tile_dim );

    auto sparse_map =
        createSparseMap<TEST_EXECSPACE>( global_mesh, pre_alloc_size );

    // def test sparse array
    // first int[3] tile i,j,k; second float cid+tid; third double[3] 0.1*cell
    // i,j,k; forth: tilekey, tid
    using DataTypes = Cabana::MemberTypes<int[3], float, double[3], int[2]>;
    auto test_layout =
        createSparseArrayLayout<DataTypes>( local_grid, sparse_map, entity );
    auto test_array = createSparseArray<TEST_MEMSPACE>(
        std::string( "test_sparse_grid" ), *test_layout );

    // insert particles
    test_array->registerSparseGrid( par_view, par_num );
    test_array->reserveFromMap( 1.2 );

    // size-related tests
    EXPECT_EQ( test_array->size(), sparse_map.sizeCell() );
    EXPECT_EQ( test_array->capacity() >= sparse_map.reservedCellSize( 1.2 ),
               true );
    EXPECT_EQ( test_array->empty(), false );
    EXPECT_EQ( test_array->numSoA(), sparse_map.sizeTile() );
    for ( std::size_t i = 0; i < test_array->numSoA(); ++i )
        EXPECT_EQ( test_array->arraySize( i ), cell_per_tile );

    // check particle insertion results
    int cell_num = par_view.extent( 0 ) * 27;
    Kokkos::View<int*, TEST_MEMSPACE> qtid_res(
        Kokkos::ViewAllocateWithoutInitializing( "query_tile_id" ), cell_num );

    T dx_inv = 1.0f / cell_size;
    Kokkos::Array<T, 3> low_corner = {
        global_low_corner[0], global_low_corner[1], global_low_corner[2] };
    Kokkos::parallel_for(
        "insert_check",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, par_view.extent( 0 ) ),
        KOKKOS_LAMBDA( const int id ) {
            T pos[3] = { par_view( id, 0 ) - low_corner[0],
                         par_view( id, 1 ) - low_corner[1],
                         par_view( id, 2 ) - low_corner[2] };
            int grid_base[3] = {
                static_cast<int>( std::lround( pos[0] * dx_inv ) - 1 ),
                static_cast<int>( std::lround( pos[1] * dx_inv ) - 1 ),
                static_cast<int>( std::lround( pos[2] * dx_inv ) - 1 ) };
            int offset = 0;
            for ( int i = 0; i <= 2; ++i )
                for ( int j = 0; j <= 2; ++j )
                    for ( int k = 0; k <= 2; ++k )
                    {
                        int cell_id[3] = { grid_base[0] + i, grid_base[1] + j,
                                           grid_base[2] + k };
                        auto tid = sparse_map.queryTile( cell_id[0], cell_id[1],
                                                         cell_id[2] );
                        qtid_res( id * 27 + offset ) = tid;
                        offset++;
                    }
        } );
    // check if all required cell are registered
    auto qtid_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), qtid_res );
    for ( int i = 0; i < cell_num; ++i )
    {
        EXPECT_EQ( qtid_mirror( i ) < (int)test_array->numSoA(), true );
        EXPECT_EQ( qtid_mirror( i ) >= 0, true );
    }

    // assign value
    auto& map = test_layout->sparseMap();
    // tile ijk, cell ijk, tile key
    Kokkos::View<int* [7], TEST_MEMSPACE> info(
        Kokkos::ViewAllocateWithoutInitializing( "tile_cell_info" ),
        test_array->size() );
    auto array = *test_array;
    Kokkos::parallel_for(
        "assign_value_to_sparse_cells",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, map.capacity() ),
        KOKKOS_LAMBDA( const int index ) {
            if ( map.valid_at( index ) )
            {
                auto tid = map.value_at( index );
                auto tkey = map.key_at( index );
                int ti, tj, tk;
                map.key2ijk( tkey, ti, tj, tk );

                for ( int ci = 0; ci < cell_per_tile_dim; ci++ )
                    for ( int cj = 0; cj < cell_per_tile_dim; cj++ )
                        for ( int ck = 0; ck < cell_per_tile_dim; ck++ )
                        {
                            // indices
                            int cid = map.cell_local_id( ci, cj, ck );
                            Kokkos::Array<int, 3> cell_ijk(
                                { ti * cell_per_tile_dim + ci,
                                  tj * cell_per_tile_dim + cj,
                                  tk * cell_per_tile_dim + ck } );
                            Kokkos::Array<int, 3> tile_ijk( { ti, tj, tk } );
                            Kokkos::Array<int, 3> local_cell_ijk(
                                { ci, cj, ck } );

                            // access: cell ijk (- channel id)
                            array.template get<0>( cell_ijk, 0 ) = ti;
                            array.template get<0>( cell_ijk, 1 ) = tj;
                            array.template get<0>( cell_ijk, 2 ) = tk;

                            // access: tile ijk - cell ijk
                            auto& second = array.template get<1>(
                                tile_ijk, local_cell_ijk );
                            second = tid + cid;

                            // access: tile id - cell ijk (- channel id)
                            array.template get<2>( tid, local_cell_ijk, 0 ) =
                                ci * 0.1;
                            array.template get<2>( tid, local_cell_ijk, 1 ) =
                                cj * 0.1;
                            array.template get<2>( tid, local_cell_ijk, 2 ) =
                                ck * 0.1;

                            // access: tile id - cell id (- channel id)
                            array.template get<3>( tid, cid, 0 ) = (int)tkey;
                            array.template get<3>( tid, cid, 1 ) = (int)tid;

                            // record info
                            info( tid * cell_per_tile + cid, 0 ) = ti;
                            info( tid * cell_per_tile + cid, 1 ) = tj;
                            info( tid * cell_per_tile + cid, 2 ) = tk;
                            info( tid * cell_per_tile + cid, 3 ) = ci;
                            info( tid * cell_per_tile + cid, 4 ) = cj;
                            info( tid * cell_per_tile + cid, 5 ) = ck;
                            info( tid * cell_per_tile + cid, 6 ) = (int)tkey;
                        }
            }
        } );
    // insert end

    // check value
    auto mirror = Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       test_array->aosoa() );
    auto info_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), info );
    auto slice_0 = Cabana::slice<0>( mirror );
    auto slice_1 = Cabana::slice<1>( mirror );
    auto slice_2 = Cabana::slice<2>( mirror );
    auto slice_3 = Cabana::slice<3>( mirror );
    for ( unsigned long cid = 0; cid < mirror.size(); ++cid )
    {
        EXPECT_EQ( slice_0( cid, 0 ), info_mirror( cid, 0 ) );
        EXPECT_EQ( slice_0( cid, 1 ), info_mirror( cid, 1 ) );
        EXPECT_EQ( slice_0( cid, 2 ), info_mirror( cid, 2 ) );

        int tid = cid / cell_per_tile;
        int local_cid = cid % cell_per_tile;
        EXPECT_FLOAT_EQ( slice_1( cid ), (float)( local_cid + tid ) );

        EXPECT_DOUBLE_EQ( slice_2( cid, 0 ), info_mirror( cid, 3 ) * 0.1 );
        EXPECT_DOUBLE_EQ( slice_2( cid, 1 ), info_mirror( cid, 4 ) * 0.1 );
        EXPECT_DOUBLE_EQ( slice_2( cid, 2 ), info_mirror( cid, 5 ) * 0.1 );

        EXPECT_EQ( slice_3( cid, 0 ), info_mirror( cid, 6 ) );
        EXPECT_EQ( slice_3( cid, 1 ), tid );
    }
    // check value end

    test_array->clear();
    EXPECT_EQ( test_array->size(), 0 );
    EXPECT_EQ( test_layout->sparseMap().sizeTile(), 0 );
}

template <typename EntityType>
void full_occupy_test( EntityType entity )
{
    // basic scenario information
    constexpr int size_tile_per_dim = 8;
    constexpr int cell_per_tile_dim = 4;
    constexpr int cell_per_tile =
        cell_per_tile_dim * cell_per_tile_dim * cell_per_tile_dim;
    constexpr int size_per_dim = size_tile_per_dim * cell_per_tile_dim;

    int pre_alloc_size = size_per_dim * size_per_dim;

    using T = float;
    // Create global mesh
    T cell_size = 0.1f;
    std::array<int, 3> global_num_cell(
        { size_per_dim, size_per_dim, size_per_dim } );
    // global low corners: random numbuers
    std::array<T, 3> global_low_corner = { .15f, -1.7f, 3.1f };
    std::array<T, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // sparse partitioner
    T max_workload_coeff = 1.5;
    int workload_num = size_per_dim * size_per_dim * size_per_dim;
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

    // scene initialization
    auto gt_partitions =
        generate_random_partition( ranks_per_dim, size_tile_per_dim );
    partitioner.initializeRecPartition( gt_partitions[0], gt_partitions[1],
                                        gt_partitions[2] );

    // mesh/grid related initialization
    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    int halo_width = 2;
    auto local_grid =
        createSparseLocalGrid( global_grid, halo_width, cell_per_tile_dim );

    auto sparse_map =
        createSparseMap<TEST_EXECSPACE>( global_mesh, pre_alloc_size );

    // def test sparse array
    // first int[3] tile i,j,k; second float[3] 0.1*cell i,j,k; third: tilekey,
    // tid
    using DataTypes = Cabana::MemberTypes<int[3], float[3], int[2]>;
    auto test_layout =
        createSparseArrayLayout<DataTypes>( local_grid, sparse_map, entity );
    auto test_array = createSparseArray<TEST_MEMSPACE>(
        std::string( "test_sparse_grid" ), *test_layout );

    // valid tile range
    Kokkos::Array<int, 3> start = {
        gt_partitions[0][cart_rank[0]] * cell_per_tile_dim,
        gt_partitions[1][cart_rank[1]] * cell_per_tile_dim,
        gt_partitions[2][cart_rank[2]] * cell_per_tile_dim };
    Kokkos::Array<int, 3> end = {
        gt_partitions[0][cart_rank[0] + 1] * cell_per_tile_dim,
        gt_partitions[1][cart_rank[1] + 1] * cell_per_tile_dim,
        gt_partitions[2][cart_rank[2] + 1] * cell_per_tile_dim };

    // fully insert
    auto& map = test_layout->sparseMap();
    Kokkos::parallel_for(
        "sparse_grid_fully_insert",
        Kokkos::MDRangePolicy<TEST_EXECSPACE, Kokkos::Rank<3>>( start, end ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            map.insertCell( i, j, k );
        } );
    test_array->resize( map.sizeCell() );

    // assign value
    Kokkos::View<int*** [2], TEST_MEMSPACE> info(
        Kokkos::ViewAllocateWithoutInitializing( "tile_cell_info" ),
        size_tile_per_dim, size_tile_per_dim, size_tile_per_dim );

    auto array = *test_array;
    Kokkos::parallel_for(
        "assign_value_fully_occupy_grid",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, map.capacity() ),
        KOKKOS_LAMBDA( const int index ) {
            if ( map.valid_at( index ) )
            {
                auto tid = map.value_at( index );
                auto tkey = map.key_at( index );
                int ti, tj, tk;
                map.key2ijk( tkey, ti, tj, tk );

                for ( int ci = 0; ci < cell_per_tile_dim; ci++ )
                    for ( int cj = 0; cj < cell_per_tile_dim; cj++ )
                        for ( int ck = 0; ck < cell_per_tile_dim; ck++ )
                        {
                            // indices
                            int cid = map.cell_local_id( ci, cj, ck );
                            Kokkos::Array<int, 3> cell_ijk(
                                { ti * cell_per_tile_dim + ci,
                                  tj * cell_per_tile_dim + cj,
                                  tk * cell_per_tile_dim + ck } );
                            Kokkos::Array<int, 3> tile_ijk( { ti, tj, tk } );
                            Kokkos::Array<int, 3> local_cell_ijk(
                                { ci, cj, ck } );

                            // access: cell ijk (- channel id)
                            array.template get<0>( cell_ijk, 1 ) = tj;
                            array.template get<0>( cell_ijk, 0 ) = ti;
                            array.template get<0>( cell_ijk, 2 ) = tk;

                            // access: tile ijk - cell ijk (- channel id)
                            array.template get<1>( tile_ijk, local_cell_ijk,
                                                   0 ) = ci * 0.1;
                            array.template get<1>( tile_ijk, local_cell_ijk,
                                                   1 ) = cj * 0.1;
                            // access: tile id - cell ijk (- channel id)
                            array.template get<1>( tid, local_cell_ijk, 2 ) =
                                ck * 0.1;

                            // access: tile id - cell id (- channel id)
                            array.template get<2>( tid, cid, 0 ) = (int)tkey;
                            array.template get<2>( tid, cid, 1 ) = (int)tid;

                            // record info
                            info( ti, tj, tk, 0 ) = (int)tid;
                            info( ti, tj, tk, 1 ) = (int)tkey;
                        }
            }
        } );

    // insert end

    // check value
    auto mirror = Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                       test_array->aosoa() );
    auto info_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), info );
    auto slice_0 = Cabana::slice<0>( mirror );
    auto slice_1 = Cabana::slice<1>( mirror );
    auto slice_2 = Cabana::slice<2>( mirror );

    for ( int ci = start[0]; ci < end[0]; ci++ )
        for ( int cj = start[1]; cj < end[1]; cj++ )
            for ( int ck = start[2]; ck < end[2]; ck++ )
            {
                int ti = ci / cell_per_tile_dim;
                int local_ci = ci % cell_per_tile_dim;

                int tj = cj / cell_per_tile_dim;
                int local_cj = cj % cell_per_tile_dim;

                int tk = ck / cell_per_tile_dim;
                int local_ck = ck % cell_per_tile_dim;

                // compute the real array_id of the current cell
                int cid = info_mirror( ti, tj, tk, 0 ) * cell_per_tile +
                          local_ck * cell_per_tile_dim * cell_per_tile_dim +
                          local_cj * cell_per_tile_dim + local_ci;

                EXPECT_EQ( slice_0( cid, 0 ), ti );
                EXPECT_EQ( slice_0( cid, 1 ), tj );
                EXPECT_EQ( slice_0( cid, 2 ), tk );

                EXPECT_FLOAT_EQ( slice_1( cid, 0 ), local_ci * 0.1 );
                EXPECT_FLOAT_EQ( slice_1( cid, 1 ), local_cj * 0.1 );
                EXPECT_FLOAT_EQ( slice_1( cid, 2 ), local_ck * 0.1 );

                EXPECT_EQ( slice_2( cid, 0 ), info_mirror( ti, tj, tk, 1 ) );
                EXPECT_EQ( slice_2( cid, 1 ), info_mirror( ti, tj, tk, 0 ) );

                cid++;
            }

    test_array->clear();
    EXPECT_EQ( test_array->size(), 0 );
    EXPECT_EQ( test_layout->sparseMap().sizeTile(), 0 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( sparse_array, 3d_sparse_array_sparse_occupy )
{
    sparse_array_test( 100, Node() );
    sparse_array_test( 20, Cell() );
}

TEST( sparse_array, 3d_sparse_array_full_occupy )
{
    full_occupy_test( Node() );
    full_occupy_test( Cell() );
}

//---------------------------------------------------------------------------//
} // namespace Test
