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

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, sparse_index_space_test ) { testTileSpace(); }

//---------------------------------------------------------------------------//

} // end namespace Test
