/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_AoSoA.hpp>
#include <Cabana_Sort.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
void testSortByKey()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1],
                                          int,
                                          double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create a Kokkos view for the keys.
    using KeyViewType = Kokkos::View<int*,typename AoSoA_t::memory_space>;
    KeyViewType keys( "keys", num_data );

    // Create the AoSoA data and keys. Create the data in reverse order so we
    // can see that it is sorted.
    auto v0 = aosoa.slice<0>();
    auto v1 = aosoa.slice<1>();
    auto v2 = aosoa.slice<2>();
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;

        keys( p ) = reverse_index;
    }

    // Sort the aosoa by keys.
    auto binning_data = Cabana::sortByKey( keys );
    Cabana::permute( binning_data, aosoa );

    // Check the result of the sort.
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), p + i );

        EXPECT_EQ( v1( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), p + i + j );

        EXPECT_EQ( binning_data.permutation(p), (unsigned) reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testBinByKey()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1],
                                          int,
                                          double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    using size_type = typename AoSoA_t::memory_space::size_type;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create a Kokkos view for the keys.
    using KeyViewType = Kokkos::View<int*,typename AoSoA_t::memory_space>;
    KeyViewType keys( "keys", num_data );

    // Create the AoSoA data and keys. Create the data in reverse order so we
    // can see that it is sorted.
    auto v0 = aosoa.slice<0>();
    auto v1 = aosoa.slice<1>();
    auto v2 = aosoa.slice<2>();
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;

        keys( p ) = reverse_index;
    }

    // Bin the aosoa by keys. Use one bin per data point to effectively make
    // this a sort.
    auto bin_data = Cabana::binByKey( keys, num_data-1 );
    Cabana::permute( bin_data, aosoa );

    // Check the result of the sort.
    EXPECT_EQ( bin_data.numBin(), num_data );
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), p + i );

        EXPECT_EQ( v1( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), p + i + j );

        EXPECT_EQ( bin_data.binSize(p), 1 );
        EXPECT_EQ( bin_data.binOffset(p), size_type(p) );
        EXPECT_EQ( bin_data.permutation(p), reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testSortBySlice()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1],
                                          int,
                                          double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = aosoa.slice<0>();
    auto v1 = aosoa.slice<1>();
    auto v2 = aosoa.slice<2>();
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;
    }

    // Sort the aosoa by the 1D member.
    auto binning_data = Cabana::sortByKey( aosoa.slice<1>() );
    Cabana::permute( binning_data, aosoa );

    // Check the result of the sort.
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), p + i );

        EXPECT_EQ( v1( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), p + i + j );

        EXPECT_EQ( binning_data.permutation(p), (unsigned) reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testSortBySliceDataOnly()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1],
                                          int,
                                          double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = aosoa.slice<0>();
    auto v1 = aosoa.slice<1>();
    auto v2 = aosoa.slice<2>();
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;
    }

    // Sort the aosoa by the 1D member.
    auto binning_data = Cabana::sortByKey( aosoa.slice<1>() );

    // Check that the data didn't get sorted and the permutation vector is
    // correct.
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), reverse_index + i );

        EXPECT_EQ( v1( p ), reverse_index );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), reverse_index + i + j );

        EXPECT_EQ( binning_data.permutation(p), (unsigned) reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testBinBySlice()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1],
                                          int,
                                          double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    using size_type = typename AoSoA_t::memory_space::size_type;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = aosoa.slice<0>();
    auto v1 = aosoa.slice<1>();
    auto v2 = aosoa.slice<2>();
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;
    }

    // Bin the aosoa by the 1D member. Use one bin per data point to
    // effectively make this a sort.
    auto bin_data = Cabana::binByKey( aosoa.slice<1>(), num_data-1 );
    Cabana::permute( bin_data, aosoa );

    // Check the result of the sort.
    EXPECT_EQ( bin_data.numBin(), num_data );
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), p + i );

        EXPECT_EQ( v1( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), p + i + j );

        EXPECT_EQ( bin_data.binSize(p), 1 );
        EXPECT_EQ( bin_data.binOffset(p), size_type(p) );
        EXPECT_EQ( bin_data.permutation(p), reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testBinBySliceDataOnly()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1],
                                          int,
                                          double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    using size_type = typename AoSoA_t::memory_space::size_type;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = aosoa.slice<0>();
    auto v1 = aosoa.slice<1>();
    auto v2 = aosoa.slice<2>();
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            v0( p, i ) = reverse_index + i;

        v1( p ) = reverse_index;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                v2( p, i, j ) = reverse_index + i + j;
    }

    // Bin the aosoa by the 1D member. Use one bin per data point to
    // effectively make this a sort. Don't actually move the particle data
    // though - just create the binning data.
    auto bin_data = Cabana::binByKey( aosoa.slice<1>(), num_data-1 );

    // Check the result of the sort. Make sure nothing moved execpt the
    // binning data.
    EXPECT_EQ( bin_data.numBin(), num_data );
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0( p, i ), reverse_index + i );

        EXPECT_EQ( v1( p ), reverse_index );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2( p, i, j ), reverse_index + i + j );

        EXPECT_EQ( bin_data.binSize(p), 1 );
        EXPECT_EQ( bin_data.binOffset(p), size_type(p) );
        EXPECT_EQ( bin_data.permutation(p), reverse_index );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, sort_by_key_test )
{
    testSortByKey();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, bin_by_key_test )
{
    testBinByKey();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, sort_by_member_test )
{
    testSortBySlice();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, sort_by_member_data_only_test )
{
    testSortBySliceDataOnly();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, bin_by_member_test )
{
    testBinBySlice();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, bin_by_member_data_only_test )
{
    testBinBySliceDataOnly();
}

//---------------------------------------------------------------------------//

} // end namespace Test
