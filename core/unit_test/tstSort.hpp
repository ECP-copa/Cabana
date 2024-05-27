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

#include <Cabana_AoSoA.hpp>
#include <Cabana_Sort.hpp>

#include <Kokkos_Core.hpp>

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
    using DataTypes =
        Cabana::MemberTypes<float[dim_1], int, double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( "aosoa", num_data );

    // Create a Kokkos view for the keys.
    using KeyViewType = Kokkos::View<int*, typename AoSoA_t::memory_space>;
    KeyViewType keys( "keys", num_data );

    // Create the AoSoA data and keys. Create the data in reverse order so we
    // can see that it is sorted.
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            int reverse_index = aosoa.size() - p - 1;

            for ( int i = 0; i < dim_1; ++i )
                v0( p, i ) = reverse_index + i;

            v1( p ) = reverse_index;

            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    v2( p, i, j ) = reverse_index + i + j;

            keys( p ) = reverse_index;
        } );

    // Sort the aosoa by keys.
    auto binning_data = Cabana::sortByKey( keys );
    Cabana::permute( binning_data, aosoa );

    // Copy the bin data so we can check it.
    Kokkos::View<std::size_t*, TEST_MEMSPACE> bin_permute( "bin_permute",
                                                           aosoa.size() );
    Kokkos::parallel_for(
        "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            bin_permute( p ) = binning_data.permutation( p );
        } );
    Kokkos::fence();
    auto bin_permute_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_permute );

    // Check the result of the sort.
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto v0_mirror = Cabana::slice<0>( mirror );
    auto v1_mirror = Cabana::slice<1>( mirror );
    auto v2_mirror = Cabana::slice<2>( mirror );
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0_mirror( p, i ), p + i );

        EXPECT_EQ( v1_mirror( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2_mirror( p, i, j ), p + i + j );

        EXPECT_EQ( bin_permute_mirror( p ), (unsigned)reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testBinByKey()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes =
        Cabana::MemberTypes<float[dim_1], int, double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( "aosoa", num_data );

    // Create a Kokkos view for the keys.
    using KeyViewType = Kokkos::View<int*, typename AoSoA_t::memory_space>;
    KeyViewType keys( "keys", num_data );

    // Create the AoSoA data and keys. Create the data in reverse order so we
    // can see that it is sorted.
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            int reverse_index = aosoa.size() - p - 1;

            for ( int i = 0; i < dim_1; ++i )
                v0( p, i ) = reverse_index + i;

            v1( p ) = reverse_index;

            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    v2( p, i, j ) = reverse_index + i + j;

            keys( p ) = reverse_index;
        } );
    Kokkos::fence();

    // Bin the aosoa by keys. Use one bin per data point to effectively make
    // this a sort.
    auto bin_data = Cabana::binByKey( keys, num_data - 1 );
    Cabana::permute( bin_data, aosoa );

    // Copy the bin data so we can check it.
    Kokkos::View<std::size_t*, TEST_MEMSPACE> bin_permute( "bin_permute",
                                                           aosoa.size() );
    Kokkos::View<std::size_t*, TEST_MEMSPACE> bin_offset( "bin_offset",
                                                          aosoa.size() );
    Kokkos::View<int*, TEST_MEMSPACE> bin_size( "bin_size", aosoa.size() );
    Kokkos::parallel_for(
        "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            bin_size( p ) = bin_data.binSize( p );
            bin_offset( p ) = bin_data.binOffset( p );
            bin_permute( p ) = bin_data.permutation( p );
        } );
    Kokkos::fence();
    auto bin_permute_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_permute );
    auto bin_offset_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_offset );
    auto bin_size_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_size );

    // Check the result of the sort.
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto v0_mirror = Cabana::slice<0>( mirror );
    auto v1_mirror = Cabana::slice<1>( mirror );
    auto v2_mirror = Cabana::slice<2>( mirror );
    EXPECT_EQ( bin_data.numBin(), num_data );
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0_mirror( p, i ), p + i );

        EXPECT_EQ( v1_mirror( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2_mirror( p, i, j ), p + i + j );

        EXPECT_EQ( bin_size_mirror( p ), 1 );
        EXPECT_EQ( bin_offset_mirror( p ), std::size_t( p ) );
        EXPECT_EQ( bin_permute_mirror( p ), reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testSortBySlice()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes =
        Cabana::MemberTypes<float[dim_1], int, double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( "aosoa", num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            int reverse_index = aosoa.size() - p - 1;

            for ( int i = 0; i < dim_1; ++i )
                v0( p, i ) = reverse_index + i;

            v1( p ) = reverse_index;

            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    v2( p, i, j ) = reverse_index + i + j;
        } );
    Kokkos::fence();

    // Sort the aosoa by the 1D member.
    auto binning_data = Cabana::sortByKey( Cabana::slice<1>( aosoa ) );
    Cabana::permute( binning_data, aosoa );

    // Copy the bin data so we can check it.
    Kokkos::View<std::size_t*, TEST_MEMSPACE> bin_permute( "bin_permute",
                                                           aosoa.size() );
    Kokkos::parallel_for(
        "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            bin_permute( p ) = binning_data.permutation( p );
        } );
    Kokkos::fence();
    auto bin_permute_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_permute );

    // Check the result of the sort.
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto v0_mirror = Cabana::slice<0>( mirror );
    auto v1_mirror = Cabana::slice<1>( mirror );
    auto v2_mirror = Cabana::slice<2>( mirror );
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0_mirror( p, i ), p + i );

        EXPECT_EQ( v1_mirror( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2_mirror( p, i, j ), p + i + j );

        EXPECT_EQ( bin_permute_mirror( p ), (unsigned)reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testSortBySliceDataOnly()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes =
        Cabana::MemberTypes<float[dim_1], int, double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( "aosoa", num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            int reverse_index = aosoa.size() - p - 1;

            for ( int i = 0; i < dim_1; ++i )
                v0( p, i ) = reverse_index + i;

            v1( p ) = reverse_index;

            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    v2( p, i, j ) = reverse_index + i + j;
        } );
    Kokkos::fence();

    // Sort the aosoa by the 1D member.
    auto binning_data = Cabana::sortByKey( Cabana::slice<1>( aosoa ) );

    // Copy the bin data so we can check it.
    Kokkos::View<std::size_t*, TEST_MEMSPACE> bin_permute( "bin_permute",
                                                           aosoa.size() );
    Kokkos::parallel_for(
        "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            bin_permute( p ) = binning_data.permutation( p );
        } );
    Kokkos::fence();
    auto bin_permute_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_permute );

    // Check that the data didn't get sorted and the permutation vector is
    // correct.
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto v0_mirror = Cabana::slice<0>( mirror );
    auto v1_mirror = Cabana::slice<1>( mirror );
    auto v2_mirror = Cabana::slice<2>( mirror );
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0_mirror( p, i ), reverse_index + i );

        EXPECT_EQ( v1_mirror( p ), reverse_index );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2_mirror( p, i, j ), reverse_index + i + j );

        EXPECT_EQ( bin_permute_mirror( p ), (unsigned)reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testBinBySlice()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes =
        Cabana::MemberTypes<float[dim_1], int, double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( "aosoa", num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            int reverse_index = aosoa.size() - p - 1;

            for ( int i = 0; i < dim_1; ++i )
                v0( p, i ) = reverse_index + i;

            v1( p ) = reverse_index;

            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    v2( p, i, j ) = reverse_index + i + j;
        } );
    Kokkos::fence();

    // Bin the aosoa by the 1D member. Use one bin per data point to
    // effectively make this a sort.
    auto bin_data = Cabana::binByKey( Cabana::slice<1>( aosoa ), num_data - 1 );
    Cabana::permute( bin_data, aosoa );

    // Copy the bin data so we can check it.
    Kokkos::View<std::size_t*, TEST_MEMSPACE> bin_permute( "bin_permute",
                                                           aosoa.size() );
    Kokkos::View<std::size_t*, TEST_MEMSPACE> bin_offset( "bin_offset",
                                                          aosoa.size() );
    Kokkos::View<int*, TEST_MEMSPACE> bin_size( "bin_size", aosoa.size() );
    Kokkos::parallel_for(
        "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            bin_size( p ) = bin_data.binSize( p );
            bin_offset( p ) = bin_data.binOffset( p );
            bin_permute( p ) = bin_data.permutation( p );
        } );
    Kokkos::fence();
    auto bin_permute_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_permute );
    auto bin_offset_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_offset );
    auto bin_size_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_size );

    // Check the result of the sort.
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto v0_mirror = Cabana::slice<0>( mirror );
    auto v1_mirror = Cabana::slice<1>( mirror );
    auto v2_mirror = Cabana::slice<2>( mirror );
    EXPECT_EQ( bin_data.numBin(), num_data );
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0_mirror( p, i ), p + i );

        EXPECT_EQ( v1_mirror( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2_mirror( p, i, j ), p + i + j );

        EXPECT_EQ( bin_size_mirror( p ), 1 );
        EXPECT_EQ( bin_offset_mirror( p ), std::size_t( p ) );
        EXPECT_EQ( bin_permute_mirror( p ), reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testBinBySliceDataOnly()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes =
        Cabana::MemberTypes<float[dim_1], int, double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( "aosoa", num_data );

    // Create the AoSoA data. Create the data in reverse order so we can see
    // that it is sorted.
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            int reverse_index = aosoa.size() - p - 1;

            for ( int i = 0; i < dim_1; ++i )
                v0( p, i ) = reverse_index + i;

            v1( p ) = reverse_index;

            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    v2( p, i, j ) = reverse_index + i + j;
        } );
    Kokkos::fence();

    // Bin the aosoa by the 1D member. Use one bin per data point to
    // effectively make this a sort. Don't actually move the particle data
    // though - just create the binning data.
    auto bin_data = Cabana::binByKey( Cabana::slice<1>( aosoa ), num_data - 1 );

    // Copy the bin data so we can check it.
    Kokkos::View<std::size_t*, TEST_MEMSPACE> bin_permute( "bin_permute",
                                                           aosoa.size() );
    Kokkos::View<std::size_t*, TEST_MEMSPACE> bin_offset( "bin_offset",
                                                          aosoa.size() );
    Kokkos::View<int*, TEST_MEMSPACE> bin_size( "bin_size", aosoa.size() );
    Kokkos::parallel_for(
        "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            bin_size( p ) = bin_data.binSize( p );
            bin_offset( p ) = bin_data.binOffset( p );
            bin_permute( p ) = bin_data.permutation( p );
        } );
    Kokkos::fence();
    auto bin_permute_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_permute );
    auto bin_offset_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_offset );
    auto bin_size_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_size );

    // Check the result of the sort. Make sure nothing moved except the
    // binning data.
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto v0_mirror = Cabana::slice<0>( mirror );
    auto v1_mirror = Cabana::slice<1>( mirror );
    auto v2_mirror = Cabana::slice<2>( mirror );
    EXPECT_EQ( bin_data.numBin(), num_data );
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0_mirror( p, i ), reverse_index + i );

        EXPECT_EQ( v1_mirror( p ), reverse_index );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2_mirror( p, i, j ), reverse_index + i + j );

        EXPECT_EQ( bin_size_mirror( p ), 1 );
        EXPECT_EQ( bin_offset_mirror( p ), std::size_t( p ) );
        EXPECT_EQ( bin_permute_mirror( p ), reverse_index );
    }
}

//---------------------------------------------------------------------------//
void testSortByKeySlice()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;

    // Declare data types.
    using DataTypes =
        Cabana::MemberTypes<float[dim_1], int, double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 3453;
    AoSoA_t aosoa( "aosoa", num_data );

    // Create a Kokkos view for the keys.
    using KeyViewType = Kokkos::View<int*, typename AoSoA_t::memory_space>;
    KeyViewType keys( "keys", num_data );

    // Create the AoSoA data and keys. Create the data in reverse order so we
    // can see that it is sorted.
    auto v0 = Cabana::slice<0>( aosoa );
    auto v1 = Cabana::slice<1>( aosoa );
    auto v2 = Cabana::slice<2>( aosoa );
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            int reverse_index = aosoa.size() - p - 1;

            for ( int i = 0; i < dim_1; ++i )
                v0( p, i ) = reverse_index + i;

            v1( p ) = reverse_index;

            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    v2( p, i, j ) = reverse_index + i + j;

            keys( p ) = reverse_index;
        } );

    // Sort slice 1 by keys.
    auto binning_data = Cabana::sortByKey( keys );
    Cabana::permute( binning_data, v0 );

    // Copy the bin data so we can check it.
    Kokkos::View<std::size_t*, TEST_MEMSPACE> bin_permute( "bin_permute",
                                                           aosoa.size() );
    Kokkos::parallel_for(
        "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            bin_permute( p ) = binning_data.permutation( p );
        } );
    Kokkos::fence();
    auto bin_permute_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_permute );

    // Check the result of the sort.
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto v0_mirror = Cabana::slice<0>( mirror );
    auto v1_mirror = Cabana::slice<1>( mirror );
    auto v2_mirror = Cabana::slice<2>( mirror );

    // Check the result of the slice 1 sort.
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        // Slice 1 particles should be reversed.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0_mirror( p, i ), p + i );

        EXPECT_EQ( bin_permute_mirror( p ), (unsigned)reverse_index );
    }
    // Other slices should be unchanged.
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        EXPECT_EQ( v1_mirror( p ), reverse_index );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2_mirror( p, i, j ), reverse_index + i + j );
    }

    // Sort slice 2 by keys.
    Cabana::permute( binning_data, v1 );

    // Copy the bin data so we can check it.
    Kokkos::parallel_for(
        "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            bin_permute( p ) = binning_data.permutation( p );
        } );
    Kokkos::fence();
    bin_permute_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_permute );

    // Check the result of the sort.
    mirror = Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    v0_mirror = Cabana::slice<0>( mirror );
    v1_mirror = Cabana::slice<1>( mirror );
    v2_mirror = Cabana::slice<2>( mirror );

    // Check the result of the slice 2 sort (slice 1 already done).
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        // Slice 1 & 2 particles should be reversed.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0_mirror( p, i ), p + i );

        EXPECT_EQ( v1_mirror( p ), p );

        EXPECT_EQ( bin_permute_mirror( p ), (unsigned)reverse_index );
    }
    // Other slice should be unchanged.
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2_mirror( p, i, j ), reverse_index + i + j );
    }

    // Sort slice 3 by keys.
    Cabana::permute( binning_data, v2 );

    // Copy the bin data so we can check it.
    Kokkos::parallel_for(
        "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int p ) {
            bin_permute( p ) = binning_data.permutation( p );
        } );
    Kokkos::fence();
    bin_permute_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_permute );

    // Check the result of the sort.
    mirror = Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    v0_mirror = Cabana::slice<0>( mirror );
    v1_mirror = Cabana::slice<1>( mirror );
    v2_mirror = Cabana::slice<2>( mirror );

    // Check the result of the slice 3 sort (slices 1-2 already done).
    for ( std::size_t p = 0; p < aosoa.size(); ++p )
    {
        int reverse_index = aosoa.size() - p - 1;

        // All slice particles should be reversed.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( v0_mirror( p, i ), p + i );

        EXPECT_EQ( v1_mirror( p ), p );

        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( v2_mirror( p, i, j ), p + i + j );

        EXPECT_EQ( bin_permute_mirror( p ), (unsigned)reverse_index );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, sort_by_key_test ) { testSortByKey(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, bin_by_key_test ) { testBinByKey(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, sort_by_member_test ) { testSortBySlice(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, sort_by_member_data_only_test )
{
    testSortBySliceDataOnly();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, bin_by_member_test ) { testBinBySlice(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, bin_by_member_data_only_test )
{
    testBinBySliceDataOnly();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, sort_by_key_slice_test ) { testSortByKeySlice(); }

//---------------------------------------------------------------------------//

} // end namespace Test
