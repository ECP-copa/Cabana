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

#include <Cajita_IndexSpace.hpp>
#include <Cajita_SparseIndexSpace.hpp>

#include <Kokkos_Core.hpp>
#include <map>

#include <gtest/gtest.h>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void testAtomicOr()
{
    constexpr int size = 16;

    Kokkos::View<int*** [2], TEST_MEMSPACE> tile_insert_record(
        "hash_record", size, size, size );

    auto tile_insert_record_mirror =
        Kokkos::create_mirror_view( Kokkos::HostSpace(), tile_insert_record );
    for ( int i = 0; i < size; i++ )
        for ( int j = 0; j < size; j++ )
            for ( int k = 0; k < size; k++ )
                for ( int d = 0; d < 2; d++ )
                {
                    tile_insert_record_mirror( i, j, k, d ) = false;
                }
    Kokkos::deep_copy( tile_insert_record, tile_insert_record_mirror );

    Kokkos::View<bool[size][size][size], TEST_DEVICE> tile_label( "label" );

    TEST_EXECSPACE().fence();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, size ),
        KOKKOS_LAMBDA( int tile_i ) {
            for ( int tile_j = 0; tile_j < size; tile_j++ )
                for ( int tile_k = 0; tile_k < size; tile_k++ )
                {
                    if ( Kokkos::atomic_fetch_or(
                             &( tile_insert_record( tile_i, tile_j, tile_k,
                                                    0 ) ),
                             1 ) == 0 )
                    {
                        tile_label( tile_i, tile_j, tile_k ) = true;
                    }
                }
        } );
    TEST_EXECSPACE().fence();

    auto tile_label_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), tile_label );

    for ( int i = 0; i < size; i++ )
        for ( int j = 0; j < size; j++ )
            for ( int k = 0; k < size; k++ )
            {
                EXPECT_EQ( tile_label_mirror( i, j, k ), true );
            }
}

void testAtomicOrPro()
{
    constexpr int size = 64;

    Kokkos::View<int*** [2], TEST_MEMSPACE> tile_insert_record(
        "hash_record", size, size, size );

    auto tile_insert_record_mirror =
        Kokkos::create_mirror_view( Kokkos::HostSpace(), tile_insert_record );
    for ( int i = 0; i < size; i++ )
        for ( int j = 0; j < size; j++ )
            for ( int k = 0; k < size; k++ )
                for ( int d = 0; d < 2; d++ )
                {
                    tile_insert_record_mirror( i, j, k, d ) = false;
                }
    Kokkos::deep_copy( tile_insert_record, tile_insert_record_mirror );

    Kokkos::View<bool[size][size][size], TEST_DEVICE> tile_label( "label" );

    TEST_EXECSPACE().fence();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, size ),
        KOKKOS_LAMBDA( int tile_i ) {
            for ( int tile_j = 0; tile_j < size; tile_j++ )
                for ( int tile_k = 0; tile_k < size; tile_k++ )
                {
                    if ( Kokkos::atomic_fetch_or(
                             &( tile_insert_record( 0, 0, 0, 0 ) ), 1 ) == 0 )
                    {
                        tile_label( tile_i, tile_j, tile_k ) = true;
                    }
                }
        } );
    TEST_EXECSPACE().fence();

    auto tile_label_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), tile_label );
    int count = 0;
    for ( int i = 0; i < size; i++ )
        for ( int j = 0; j < size; j++ )
            for ( int k = 0; k < size; k++ )
            {
                if ( tile_label_mirror( i, j, k ) == 1 )
                    count++;
            }

    EXPECT_EQ( count, 1 );
}

template <int SizeBit = 2, int Size = 4>
void testTileSpace()
{
    // size 4x4x4
    constexpr int size_bit = SizeBit;
    constexpr int size = Size;
    using TIS = TileMap<size_bit, size, size * size>;
    Kokkos::View<int[size][size][size], TEST_DEVICE> offset_res( "offset" );
    Kokkos::View<int[size][size][size], TEST_DEVICE> i_res( "i" );
    Kokkos::View<int[size][size][size], TEST_DEVICE> j_res( "j" );
    Kokkos::View<int[size][size][size], TEST_DEVICE> k_res( "k" );
    TIS tis;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, size ),
        KOKKOS_LAMBDA( const int k ) {
            for ( int j = 0; j < size; j++ )
                for ( int i = 0; i < size; i++ )
                {
                    auto idx = tis.coordToOffset( i, j, k );
                    offset_res( i, j, k ) = idx;
                    int ci, cj, ck;
                    tis.offsetToCoord( idx, ci, cj, ck );
                    i_res( i, j, k ) = ci;
                    j_res( i, j, k ) = cj;
                    k_res( i, j, k ) = ck;
                }
        } );
    auto offset_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), offset_res );
    auto i_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), i_res );
    auto j_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), j_res );
    auto k_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), k_res );
    for ( int k = 0; k < size; k++ )
        for ( int j = 0; j < size; j++ )
            for ( int i = 0; i < size; i++ )
            {
                EXPECT_EQ( offset_mirror( i, j, k ),
                           k * size * size + j * size + i );
                EXPECT_EQ( i_mirror( i, j, k ), i );
                EXPECT_EQ( j_mirror( i, j, k ), j );
                EXPECT_EQ( k_mirror( i, j, k ), k );
            }
}

template <HashTypes HashType>
void testBlockSpace()
{
    constexpr unsigned long long cell_bits_per_tile_dim = 2;
    constexpr unsigned long long cell_num_per_tile_dim = 4;
    constexpr unsigned long long cell_num_per_tile =
        cell_num_per_tile_dim * cell_num_per_tile_dim * cell_num_per_tile_dim;
    using key_type = uint64_t;
    using value_type = uint32_t;
    constexpr int size = 8;
    int pre_alloc_size = size * size;
    BlockMap<TEST_MEMSPACE, cell_bits_per_tile_dim, cell_num_per_tile_dim,
             cell_num_per_tile, HashType, key_type, value_type>
        bis( size, size, size, pre_alloc_size );

    Kokkos::View<int[size][size][size], TEST_DEVICE> i_res( "i" );
    Kokkos::View<int[size][size][size], TEST_DEVICE> j_res( "j" );
    Kokkos::View<int[size][size][size], TEST_DEVICE> k_res( "k" );
    TEST_EXECSPACE().fence();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, size ), KOKKOS_LAMBDA( int i ) {
            for ( int j = 0; j < size; j++ )
                for ( int k = 0; k < size; k++ )
                {
                    bis.insert( i, j, k );
                    auto tile_key = bis.ijk2key( i, j, k );
                    int ti, tj, tk;
                    bis.key2ijk( tile_key, ti, tj, tk );
                    i_res( i, j, k ) = ti;
                    j_res( i, j, k ) = tj;
                    k_res( i, j, k ) = tk;
                }
        } );
    TEST_EXECSPACE().fence();
    auto i_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), i_res );
    auto j_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), j_res );
    auto k_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), k_res );
    for ( int i = 0; i < size; i++ )
        for ( int j = 0; j < size; j++ )
            for ( int k = 0; k < size; k++ )
            {
                EXPECT_EQ( i_mirror( i, j, k ), i );
                EXPECT_EQ( j_mirror( i, j, k ), j );
                EXPECT_EQ( k_mirror( i, j, k ), k );
            }
    bool test[size * size * size];

    Kokkos::View<int[size * size * size], TEST_DEVICE> id_find_res( "id_find" );
    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, size ), KOKKOS_LAMBDA( int i ) {
            for ( int j = 0; j < size; j++ )
                for ( int k = 0; k < size; k++ )
                    id_find_res( i * size * size + j * size + k ) =
                        bis.find( i, j, k );
        } );
    ;
    auto id_find_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), id_find_res );
    for ( int i = 0; i < size; i++ )
        for ( int j = 0; j < size; j++ )
            for ( int k = 0; k < size; k++ )
                test[i * size * size + j * size + k] = false;

    for ( int i = 0; i < size; i++ )
        for ( int j = 0; j < size; j++ )
            for ( int k = 0; k < size; k++ )
                test[id_find_mirror( i * size * size + j * size + k )] = true;

    for ( int i = 0; i < size; i++ )
        for ( int j = 0; j < size; j++ )
            for ( int k = 0; k < size; k++ )
            {
                auto id = i * size * size + j * size + k;
                EXPECT_EQ( test[id], true );
            }
}

void testSparseMapFullInsert()
{
    constexpr int dim_n = 3;
    constexpr int size_tile_per_dim = 4;
    constexpr int size_per_dim = size_tile_per_dim * 4;

    std::array<int, dim_n> size( { size_per_dim, size_per_dim, size_per_dim } );
    int pre_alloc_size = size_per_dim * size_per_dim;
    // Create the global mesh
    double cell_size = 0.1;
    std::array<int, 3> global_num_cell = size;
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create Sparse Map
    // SparseMap<TEST_EXECSPACE> sis( size, pre_alloc_size );
    auto sis = createSparseMap<TEST_EXECSPACE>( global_mesh, pre_alloc_size );

    auto cbd = sis.cell_bits_per_tile_dim;
    EXPECT_EQ( cbd, 2 );

    auto cnd = sis.cell_num_per_tile_dim;
    EXPECT_EQ( cnd, 4 );

    auto cmd = sis.cell_mask_per_tile_dim;
    EXPECT_EQ( cmd, 3 );

    auto cbt = sis.cell_bits_per_tile;
    EXPECT_EQ( cbt, 6 );

    auto cnt = sis.cell_num_per_tile;
    EXPECT_EQ( cnt, 64 );

    Kokkos::View<int***, TEST_DEVICE> qid_res( "query_id", size[0], size[1],
                                               size[2] );

    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, size_per_dim ),
        KOKKOS_LAMBDA( int i ) {
            for ( int j = 0; j < size_per_dim; j++ )
                for ( int k = 0; k < size_per_dim; k++ )
                {
                    sis.insertCell( i, j, k );
                }
        } );

    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, size_per_dim ),
        KOKKOS_LAMBDA( int i ) {
            for ( int j = 0; j < size_per_dim; j++ )
                for ( int k = 0; k < size_per_dim; k++ )
                {
                    qid_res( i, j, k ) = sis.queryCell( i, j, k );
                }
        } );

    bool test[size_per_dim * size_per_dim * size_per_dim];
    auto qid_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), qid_res );

    for ( int i = 0; i < size[0]; i++ )
        for ( int j = 0; j < size[1]; j++ )
            for ( int k = 0; k < size[2]; k++ )
                test[i * size[1] * size[2] + j * size[2] + k] = false;

    for ( int i = 0; i < size[0]; i++ )
        for ( int j = 0; j < size[1]; j++ )
            for ( int k = 0; k < size[2]; k++ )
                test[qid_mirror( i, j, k )] = true;

    for ( int i = 0; i < size[0]; i++ )
        for ( int j = 0; j < size[1]; j++ )
            for ( int k = 0; k < size[2]; k++ )
            {
                EXPECT_EQ( test[i * size[2] * size[1] + j * size[2] + k],
                           true );
            }

    constexpr int total_tile_num =
        size_tile_per_dim * size_tile_per_dim * size_tile_per_dim;

    uint32_t cap = sis.capacity();
    EXPECT_EQ( cap >= total_tile_num, true );

    auto s = sis.size();
    EXPECT_EQ( s, total_tile_num );

    uint32_t new_set_cap = total_tile_num * 10;
    sis.reserve( new_set_cap );
    uint32_t new_cap = sis.capacity();
    EXPECT_EQ( new_cap >= new_set_cap, true );

    auto new_s = sis.size();
    EXPECT_EQ( new_s, total_tile_num );
}

void testSparseMapSparseInsert()
{
    constexpr int dim_n = 3;
    constexpr int size_tile_per_dim = 8;
    constexpr int size_per_dim = size_tile_per_dim * 4;
    constexpr int total_size = size_per_dim * size_per_dim * size_per_dim;

    std::array<int, dim_n> size( { size_per_dim, size_per_dim, size_per_dim } );
    int pre_alloc_size = size_per_dim * size_per_dim;
    // Create the global mesh
    double cell_size = 0.1;
    std::array<int, 3> global_num_cell = size;
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    // Create Sparse Map
    auto sis = createSparseMap<TEST_EXECSPACE>( global_mesh, pre_alloc_size );

    auto cbd = sis.cell_bits_per_tile_dim;
    EXPECT_EQ( cbd, 2 );

    auto cnd = sis.cell_num_per_tile_dim;
    EXPECT_EQ( cnd, 4 );

    auto cmd = sis.cell_mask_per_tile_dim;
    EXPECT_EQ( cmd, 3 );

    auto cbt = sis.cell_bits_per_tile;
    EXPECT_EQ( cbt, 6 );

    auto cnt = sis.cell_num_per_tile;
    EXPECT_EQ( cnt, 64 );

    constexpr int insert_cell_num = 20;
    Kokkos::View<int*, Kokkos::HostSpace> host_cell_1did( "cell_ids_1d",
                                                          insert_cell_num );
    std::map<int, int> cell_register;
    for ( int i = 0; i < insert_cell_num; ++i )
    {
        host_cell_1did( i ) = ( std::rand() % total_size );
        int tile_k = ( host_cell_1did( i ) % size_per_dim ) >> 2;
        int tile_j =
            ( ( host_cell_1did( i ) / size_per_dim ) % size_per_dim ) >> 2;
        int tile_i = ( ( host_cell_1did( i ) / size_per_dim / size_per_dim ) %
                       size_per_dim ) >>
                     2;
        cell_register[tile_i * size_tile_per_dim * size_tile_per_dim +
                      tile_j * size_tile_per_dim + tile_k] = 1;
    }
    int valid_cell_num = cell_register.size();
    Kokkos::View<int*, TEST_DEVICE> dev_cell_1did( "cell_ids_1d_dev",
                                                   insert_cell_num );
    Kokkos::deep_copy( dev_cell_1did, host_cell_1did );

    Kokkos::View<int*, TEST_DEVICE> qtkey_res( "query_tile_key",
                                               insert_cell_num );
    Kokkos::View<int*, TEST_DEVICE> qtid_res( "query_tile_id",
                                              insert_cell_num );

    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, insert_cell_num ),
        KOKKOS_LAMBDA( int cell_idx ) {
            int id = dev_cell_1did( cell_idx );
            int k = id % size_per_dim;
            int j = ( id / size_per_dim ) % size_per_dim;
            int i = ( id / size_per_dim / size_per_dim ) % size_per_dim;
            sis.insertCell( i, j, k );
        } );

    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, insert_cell_num ),
        KOKKOS_LAMBDA( int cell_idx ) {
            int id = dev_cell_1did( cell_idx );
            int k = id % size_per_dim;
            int j = ( id / size_per_dim ) % size_per_dim;
            int i = ( id / size_per_dim / size_per_dim ) % size_per_dim;
            auto qid = sis.queryTile( i, j, k );
            qtkey_res( cell_idx ) = qid;
        } );

    Kokkos::parallel_for(
        "test_value_at",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, sis.capacity() ),
        KOKKOS_LAMBDA( const int index ) {
            if ( sis.valid_at( index ) )
            {
                auto tileKey = sis.key_at( index );
                auto tileId = sis.value_at( index );
                int ti, tj, tk;
                sis.key2ijk( tileKey, ti, tj, tk );
                int cell_idx = ti + tj + tk;
                qtid_res( cell_idx ) = tileId;
            }
        } );

    bool test[size_per_dim * size_per_dim * size_per_dim];

    auto qtkey_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), qtkey_res );
    auto qtid_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), qtid_res );
    for ( int i = 0; i < insert_cell_num; ++i )
        EXPECT_EQ( qtid_mirror( i ) < insert_cell_num, true );

    for ( int i = 0; i < total_size; i++ )
        test[i] = false;

    for ( int i = 0; i < insert_cell_num; ++i )
        test[qtkey_mirror( i )] = true;

    int idx = 0;
    while ( idx < valid_cell_num )
    {
        EXPECT_EQ( test[idx], true );
        ++idx;
    }
    while ( idx < total_size )
    {
        EXPECT_EQ( test[idx], false );
        ++idx;
    }

    auto s = sis.size();
    EXPECT_EQ( s, valid_cell_num );

    uint32_t cap = sis.capacity();
    EXPECT_EQ(
        cap >= ( size_tile_per_dim * size_tile_per_dim * size_tile_per_dim ),
        true );

    uint32_t new_set_cap = total_size;
    sis.reserve( new_set_cap );
    uint32_t new_cap = sis.capacity();
    EXPECT_EQ( new_cap >= new_set_cap, true );

    auto new_s = sis.size();
    EXPECT_EQ( new_s, valid_cell_num );
}

void testSparseMapReinsert()
{
    constexpr int dim_n = 3;
    constexpr int size_tile_per_dim = 8;
    constexpr int size_per_dim = size_tile_per_dim * 4;
    constexpr int total_size = size_per_dim * size_per_dim * size_per_dim;

    std::array<int, dim_n> size( { size_per_dim, size_per_dim, size_per_dim } );
    int pre_alloc_size = size_per_dim * size_per_dim;
    // Create the global mesh
    double cell_size = 0.1;
    std::array<int, 3> global_num_cell = size;
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    // Create Sparse Map
    auto sis = createSparseMap<TEST_EXECSPACE>( global_mesh, pre_alloc_size );

    constexpr int insert_cell_num = 50;
    Kokkos::View<int*, TEST_DEVICE> qid_res( "query_id", insert_cell_num );
    Kokkos::View<int*, Kokkos::HostSpace> host_cell_1did( "cell_ids_1d",
                                                          insert_cell_num );
    std::map<int, int> cell_register;
    for ( int i = 0; i < insert_cell_num; ++i )
    {
        host_cell_1did( i ) = ( std::rand() % total_size );
        int tile_k = ( host_cell_1did( i ) % size_per_dim ) >> 2;
        int tile_j =
            ( ( host_cell_1did( i ) / size_per_dim ) % size_per_dim ) >> 2;
        int tile_i = ( ( host_cell_1did( i ) / size_per_dim / size_per_dim ) %
                       size_per_dim ) >>
                     2;
        cell_register[tile_i * size_tile_per_dim * size_tile_per_dim +
                      tile_j * size_tile_per_dim + tile_k] = 1;
    }
    int valid_cell_num = cell_register.size();
    Kokkos::View<int*, TEST_DEVICE> dev_cell_1did( "cell_ids_1d_dev",
                                                   insert_cell_num );
    Kokkos::deep_copy( dev_cell_1did, host_cell_1did );

    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, insert_cell_num ),
        KOKKOS_LAMBDA( int cell_idx ) {
            int id = dev_cell_1did( cell_idx );
            int k = id % size_per_dim;
            int j = ( id / size_per_dim ) % size_per_dim;
            int i = ( id / size_per_dim / size_per_dim ) % size_per_dim;
            sis.insertCell( i, j, k );
        } );

    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, insert_cell_num ),
        KOKKOS_LAMBDA( int cell_idx ) {
            int id = dev_cell_1did( cell_idx );
            int k = id % size_per_dim;
            int j = ( id / size_per_dim ) % size_per_dim;
            int i = ( id / size_per_dim / size_per_dim ) % size_per_dim;
            auto qid = sis.queryTile( i, j, k );
            qid_res( cell_idx ) = qid;
        } );

    bool test[size_per_dim * size_per_dim * size_per_dim];

    auto qid_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), qid_res );

    for ( int i = 0; i < total_size; i++ )
    {
        test[i] = false;
    }

    for ( int i = 0; i < insert_cell_num; ++i )
    {
        test[qid_mirror( i )] = true;
    }

    int idx = 0;
    while ( idx < valid_cell_num )
    {
        EXPECT_EQ( test[idx], true );
        ++idx;
    }
    while ( idx < total_size )
    {
        EXPECT_EQ( test[idx], false );
        ++idx;
    }

    uint32_t oldcap = sis.capacity();
    sis.clear();
    uint32_t newcap = sis.capacity();
    auto newsize = sis.size();
    EXPECT_EQ( oldcap, newcap );
    EXPECT_EQ( newsize, 0 );

    std::map<int, int> cell_register_new;
    for ( int i = 0; i < insert_cell_num; ++i )
    {
        host_cell_1did( i ) = ( std::rand() % total_size );
        int tile_k = ( host_cell_1did( i ) % size_per_dim ) >> 2;
        int tile_j =
            ( ( host_cell_1did( i ) / size_per_dim ) % size_per_dim ) >> 2;
        int tile_i = ( ( host_cell_1did( i ) / size_per_dim / size_per_dim ) %
                       size_per_dim ) >>
                     2;
        cell_register_new[tile_i * size_tile_per_dim * size_tile_per_dim +
                          tile_j * size_tile_per_dim + tile_k] = 1;
    }
    valid_cell_num = cell_register_new.size();
    Kokkos::deep_copy( dev_cell_1did, host_cell_1did );

    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, insert_cell_num ),
        KOKKOS_LAMBDA( int cell_idx ) {
            int id = dev_cell_1did( cell_idx );
            int k = id % size_per_dim;
            int j = ( id / size_per_dim ) % size_per_dim;
            int i = ( id / size_per_dim / size_per_dim ) % size_per_dim;
            sis.insertCell( i, j, k );
        } );

    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, insert_cell_num ),
        KOKKOS_LAMBDA( int cell_idx ) {
            int id = dev_cell_1did( cell_idx );
            int k = id % size_per_dim;
            int j = ( id / size_per_dim ) % size_per_dim;
            int i = ( id / size_per_dim / size_per_dim ) % size_per_dim;
            auto qid = sis.queryTile( i, j, k );
            qid_res( cell_idx ) = qid;
        } );

    auto qid_mirror_2 =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), qid_res );

    for ( int i = 0; i < total_size; i++ )
    {
        test[i] = false;
    }

    for ( int i = 0; i < insert_cell_num; ++i )
    {
        test[qid_mirror_2( i )] = true;
    }

    idx = 0;
    while ( idx < valid_cell_num )
    {
        EXPECT_EQ( test[idx], true );
        ++idx;
    }
    while ( idx < total_size )
    {
        EXPECT_EQ( test[idx], false );
        ++idx;
    }
}

void tileIndexSpaceTest()
{
    int t0_max = 4;
    int t1_max = 6;
    int t2_max = 3;
    constexpr unsigned long long cell_bits_per_tile_dim = 2;
    constexpr unsigned long long cell_nums_per_tile_dim = 4;
    constexpr std::size_t N = 3;
    std::array<long, N> s = { (long)t0_max, (long)t1_max, (long)t2_max };
    TileIndexSpace<N, cell_bits_per_tile_dim> tis1( s );
    EXPECT_EQ( tis1.min( 0 ), 0 );
    EXPECT_EQ( tis1.max( 0 ), t0_max );
    EXPECT_EQ( tis1.min( 1 ), 0 );
    EXPECT_EQ( tis1.max( 1 ), t1_max );
    EXPECT_EQ( tis1.min( 2 ), 0 );
    EXPECT_EQ( tis1.max( 2 ), t2_max );

    EXPECT_EQ( tis1.sizeTile(), t0_max * t1_max * t2_max );
    EXPECT_EQ( tis1.sizeCell(),
               t0_max * t1_max * t2_max * cell_nums_per_tile_dim *
                   cell_nums_per_tile_dim * cell_nums_per_tile_dim );

    EXPECT_EQ( tis1.tileInRange( 2, 3, 1 ), true );
    EXPECT_EQ( tis1.tileInRange( 6, 3, 1 ), false );

    EXPECT_EQ( tis1.cellInRange( 6, 3, 1 ), true );
    EXPECT_EQ( tis1.cellInRange( 106, 3, 1 ), false );

    int t0_min = 2;
    int t1_min = 3;
    int t2_min = 1;
    TileIndexSpace<N, cell_bits_per_tile_dim> tis2(
        { (long)t0_min, (long)t1_min, (long)t2_min },
        { (long)t0_max, (long)t1_max, (long)t2_max } );
    EXPECT_EQ( tis2.min( 0 ), t0_min );
    EXPECT_EQ( tis2.max( 0 ), t0_max );
    EXPECT_EQ( tis2.min( 1 ), t1_min );
    EXPECT_EQ( tis2.max( 1 ), t1_max );
    EXPECT_EQ( tis2.min( 2 ), t2_min );
    EXPECT_EQ( tis2.max( 2 ), t2_max );

    auto tile_size_gt =
        ( t0_max - t0_min ) * ( t1_max - t1_min ) * ( t2_max - t2_min );
    EXPECT_EQ( tis2.sizeTile(), tile_size_gt );
    EXPECT_EQ( tis2.sizeCell(), tile_size_gt * cell_nums_per_tile_dim *
                                    cell_nums_per_tile_dim *
                                    cell_nums_per_tile_dim );

    EXPECT_EQ( tis2.tileInRange( 2, 3, 1 ), true );
    EXPECT_EQ( tis2.tileInRange( 3, 4, 2 ), true );
    EXPECT_EQ( tis2.tileInRange( 0, 2, 1 ), false );
    EXPECT_EQ( tis2.tileInRange( 6, 3, 1 ), false );

    EXPECT_EQ( tis2.cellInRange( 8, 12, 4 ), true );
    EXPECT_EQ( tis2.cellInRange( 13, 17, 9 ), true );
    EXPECT_EQ( tis2.cellInRange( 0, 2, 1 ), false );
    EXPECT_EQ( tis2.cellInRange( 106, 3, 1 ), false );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, atomic_test )
{
    testAtomicOr();
    testAtomicOrPro();
}
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, tile_space_test )
{
    testTileSpace<2, 4>(); // tile size 4x4x4
}
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, block_space_test )
{
    testBlockSpace<HashTypes::Naive>();
    testBlockSpace<HashTypes::Morton>();
}
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, sparse_map_space_test )
{
    testSparseMapFullInsert();
    testSparseMapSparseInsert();
    testSparseMapReinsert();
}

TEST( TEST_CATEGORY, tile_index_space_test ) { tileIndexSpaceTest(); }
//---------------------------------------------------------------------------//

} // end namespace Test
