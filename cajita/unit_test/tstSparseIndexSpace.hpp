#include <Cajita_SparseIndexSpace.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void testTileSpace()
{
    /* ijk <=> id*/
    {
        // size 4x4x4
        constexpr int size_bit = 2;
        constexpr int size = 4;
        TileIndexSpace<size_bit, size, size * size> tis;
        int idx_gt = 0;
        for ( int k = 0; k < size; k++ )
            for ( int j = 0; j < size; j++ )
                for ( int i = 0; i < size; i++ )
                {
                    auto idx = tis.coord_to_offset( i, j, k );
                    EXPECT_EQ( idx, idx_gt );
                    int ci, cj, ck;
                    tis.offset_to_coord( idx_gt, ci, cj, ck );
                    EXPECT_EQ( ci, i );
                    EXPECT_EQ( cj, j );
                    EXPECT_EQ( ck, k );
                    idx_gt++;
                }
    }

    {
        // size 2x2x2
        constexpr int size_bit = 1;
        constexpr int size = 2;
        TileIndexSpace<size_bit, size, size * size> tis;
        int idx_gt = 0;
        for ( int k = 0; k < size; k++ )
            for ( int j = 0; j < size; j++ )
                for ( int i = 0; i < size; i++ )
                {
                    auto idx = tis.coord_to_offset( i, j, k );
                    EXPECT_EQ( idx, idx_gt );
                    int ci, cj, ck;
                    tis.offset_to_coord( idx_gt, ci, cj, ck );
                    EXPECT_EQ( ci, i );
                    EXPECT_EQ( cj, j );
                    EXPECT_EQ( ck, k );
                    idx_gt++;
                }
    }

    {
        // size 8x8x8
        constexpr int size_bit = 3;
        constexpr int size = 8;
        TileIndexSpace<size_bit, size, size * size> tis;
        int idx_gt = 0;
        for ( int k = 0; k < size; k++ )
            for ( int j = 0; j < size; j++ )
                for ( int i = 0; i < size; i++ )
                {
                    auto idx = tis.coord_to_offset( i, j, k );
                    EXPECT_EQ( idx, idx_gt );
                    int ci, cj, ck;
                    tis.offset_to_coord( idx_gt, ci, cj, ck );
                    EXPECT_EQ( ci, i );
                    EXPECT_EQ( cj, j );
                    EXPECT_EQ( ck, k );
                    idx_gt++;
                }
    }
}

template <HashTypes hashType>
void testBlockSpace()
{
    constexpr int N = 3;
    constexpr unsigned long long CellBitsPerTileDim = 2;
    constexpr unsigned long long CellNumPerTileDim = 4;
    constexpr unsigned long long CellNumPerTile =
        CellNumPerTileDim * CellNumPerTileDim * CellNumPerTileDim;
    using KeyType = uint64_t;
    using ValueType = uint32_t;
    int size_per_dim = 64;
    std::array<int, N> size( {size_per_dim, size_per_dim, size_per_dim} );
    int capacity = size_per_dim * size_per_dim;
    float rehash_factor = 1.5;
    {
        // static constexpr HashTypes hashType = HashTypes::Naive;
        BlockIndexSpace<TEST_MEMSPACE, N, CellBitsPerTileDim, CellNumPerTileDim,
                        CellNumPerTile, hashType, KeyType, ValueType>
            bis( size, capacity, rehash_factor );
        int insert_num = 0;

        for ( int i = 0; i < size[0]; i++ )
            for ( int j = 0; j < size[1]; j++ )
                for ( int k = 0; k < size[2]; k++ )
                {
                    auto tileNo = bis.insert( i, j, k );
                    EXPECT_EQ( tileNo, insert_num );
                    auto tileKey = bis.ijk_to_key( i, j, k );
                    // EXPECT_EQ( tileKey, insert_num );
                    int ti, tj, tk;
                    bis.key_to_ijk( tileKey, ti, tj, tk );
                    EXPECT_EQ( ti, i );
                    EXPECT_EQ( tj, j );
                    EXPECT_EQ( tk, k );
                    insert_num++;
                }

        insert_num = 0;
        for ( int i = 0; i < size[0]; i++ )
            for ( int j = 0; j < size[1]; j++ )
                for ( int k = 0; k < size[2]; k++ )
                {
                    auto findNo = bis.find( i, j, k );
                    EXPECT_EQ( findNo, insert_num );
                    insert_num++;
                }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, sparse_index_space_test )
{
    testTileSpace();
    testBlockSpace<HashTypes::Naive>();
    testBlockSpace<HashTypes::Morton>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
