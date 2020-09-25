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

#include <Cajita_IndexSpace.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void sizeConstructorTest()
{
    // Rank-1
    int s0 = 3;
    IndexSpace<1> is1( { s0 } );
    EXPECT_EQ( is1.min( 0 ), 0 );
    EXPECT_EQ( is1.max( 0 ), s0 );

    auto min1 = is1.min();
    EXPECT_EQ( min1[0], 0 );
    auto max1 = is1.max();
    EXPECT_EQ( max1[0], s0 );

    auto r10 = is1.range( 0 );
    EXPECT_EQ( r10.first, 0 );
    EXPECT_EQ( r10.second, s0 );

    EXPECT_EQ( is1.rank(), 1 );
    EXPECT_EQ( is1.extent( 0 ), s0 );

    // Rank-2
    int s1 = 5;
    IndexSpace<2> is2( { s0, s1 } );
    EXPECT_EQ( is2.min( 0 ), 0 );
    EXPECT_EQ( is2.max( 0 ), s0 );
    EXPECT_EQ( is2.min( 1 ), 0 );
    EXPECT_EQ( is2.max( 1 ), s1 );

    auto min2 = is2.min();
    EXPECT_EQ( min2[0], 0 );
    EXPECT_EQ( min2[1], 0 );
    auto max2 = is2.max();
    EXPECT_EQ( max2[0], s0 );
    EXPECT_EQ( max2[1], s1 );

    auto r20 = is2.range( 0 );
    EXPECT_EQ( r20.first, 0 );
    EXPECT_EQ( r20.second, s0 );
    auto r21 = is2.range( 1 );
    EXPECT_EQ( r21.first, 0 );
    EXPECT_EQ( r21.second, s1 );

    EXPECT_EQ( is2.rank(), 2 );
    EXPECT_EQ( is2.extent( 0 ), s0 );
    EXPECT_EQ( is2.extent( 1 ), s1 );

    // Rank-3
    int s2 = 9;
    IndexSpace<3> is3( { s0, s1, s2 } );
    EXPECT_EQ( is3.min( 0 ), 0 );
    EXPECT_EQ( is3.max( 0 ), s0 );
    EXPECT_EQ( is3.min( 1 ), 0 );
    EXPECT_EQ( is3.max( 1 ), s1 );
    EXPECT_EQ( is3.min( 2 ), 0 );
    EXPECT_EQ( is3.max( 2 ), s2 );

    auto min3 = is3.min();
    EXPECT_EQ( min3[0], 0 );
    EXPECT_EQ( min3[1], 0 );
    EXPECT_EQ( min3[2], 0 );
    auto max3 = is3.max();
    EXPECT_EQ( max3[0], s0 );
    EXPECT_EQ( max3[1], s1 );
    EXPECT_EQ( max3[2], s2 );

    auto r30 = is3.range( 0 );
    EXPECT_EQ( r30.first, 0 );
    EXPECT_EQ( r30.second, s0 );
    auto r31 = is3.range( 1 );
    EXPECT_EQ( r31.first, 0 );
    EXPECT_EQ( r31.second, s1 );
    auto r32 = is3.range( 2 );
    EXPECT_EQ( r32.first, 0 );
    EXPECT_EQ( r32.second, s2 );

    EXPECT_EQ( is3.rank(), 3 );
    EXPECT_EQ( is3.extent( 0 ), s0 );
    EXPECT_EQ( is3.extent( 1 ), s1 );
    EXPECT_EQ( is3.extent( 2 ), s2 );

    // Rank-4
    int s3 = 4;
    IndexSpace<4> is4( { s0, s1, s2, s3 } );
    EXPECT_EQ( is4.min( 0 ), 0 );
    EXPECT_EQ( is4.max( 0 ), s0 );
    EXPECT_EQ( is4.min( 1 ), 0 );
    EXPECT_EQ( is4.max( 1 ), s1 );
    EXPECT_EQ( is4.min( 2 ), 0 );
    EXPECT_EQ( is4.max( 2 ), s2 );
    EXPECT_EQ( is4.min( 3 ), 0 );
    EXPECT_EQ( is4.max( 3 ), s3 );

    auto min4 = is4.min();
    EXPECT_EQ( min4[0], 0 );
    EXPECT_EQ( min4[1], 0 );
    EXPECT_EQ( min4[2], 0 );
    EXPECT_EQ( min4[3], 0 );
    auto max4 = is4.max();
    EXPECT_EQ( max4[0], s0 );
    EXPECT_EQ( max4[1], s1 );
    EXPECT_EQ( max4[2], s2 );
    EXPECT_EQ( max4[3], s3 );

    auto r40 = is4.range( 0 );
    EXPECT_EQ( r40.first, 0 );
    EXPECT_EQ( r40.second, s0 );
    auto r41 = is4.range( 1 );
    EXPECT_EQ( r41.first, 0 );
    EXPECT_EQ( r41.second, s1 );
    auto r42 = is4.range( 2 );
    EXPECT_EQ( r42.first, 0 );
    EXPECT_EQ( r42.second, s2 );
    auto r43 = is4.range( 3 );
    EXPECT_EQ( r43.first, 0 );
    EXPECT_EQ( r43.second, s3 );

    EXPECT_EQ( is4.rank(), 4 );
    EXPECT_EQ( is4.extent( 0 ), s0 );
    EXPECT_EQ( is4.extent( 1 ), s1 );
    EXPECT_EQ( is4.extent( 2 ), s2 );
    EXPECT_EQ( is4.extent( 3 ), s3 );
}

//---------------------------------------------------------------------------//
void rangeConstructorTest()
{
    // Rank-1
    int s0 = 3;
    IndexSpace<1> is1( { 0 }, { s0 } );
    EXPECT_EQ( is1.min( 0 ), 0 );
    EXPECT_EQ( is1.max( 0 ), s0 );

    auto min1 = is1.min();
    EXPECT_EQ( min1[0], 0 );
    auto max1 = is1.max();
    EXPECT_EQ( max1[0], s0 );

    auto r10 = is1.range( 0 );
    EXPECT_EQ( r10.first, 0 );
    EXPECT_EQ( r10.second, s0 );

    EXPECT_EQ( is1.rank(), 1 );
    EXPECT_EQ( is1.extent( 0 ), s0 );

    // Rank-2
    int s1 = 5;
    IndexSpace<2> is2( { 0, 0 }, { s0, s1 } );
    EXPECT_EQ( is2.min( 0 ), 0 );
    EXPECT_EQ( is2.max( 0 ), s0 );
    EXPECT_EQ( is2.min( 1 ), 0 );
    EXPECT_EQ( is2.max( 1 ), s1 );

    auto min2 = is2.min();
    EXPECT_EQ( min2[0], 0 );
    EXPECT_EQ( min2[1], 0 );
    auto max2 = is2.max();
    EXPECT_EQ( max2[0], s0 );
    EXPECT_EQ( max2[1], s1 );

    auto r20 = is2.range( 0 );
    EXPECT_EQ( r20.first, 0 );
    EXPECT_EQ( r20.second, s0 );
    auto r21 = is2.range( 1 );
    EXPECT_EQ( r21.first, 0 );
    EXPECT_EQ( r21.second, s1 );

    EXPECT_EQ( is2.rank(), 2 );
    EXPECT_EQ( is2.extent( 0 ), s0 );
    EXPECT_EQ( is2.extent( 1 ), s1 );

    // Rank-3
    int s2 = 9;
    IndexSpace<3> is3( { 0, 0, 0 }, { s0, s1, s2 } );
    EXPECT_EQ( is3.min( 0 ), 0 );
    EXPECT_EQ( is3.max( 0 ), s0 );
    EXPECT_EQ( is3.min( 1 ), 0 );
    EXPECT_EQ( is3.max( 1 ), s1 );
    EXPECT_EQ( is3.min( 2 ), 0 );
    EXPECT_EQ( is3.max( 2 ), s2 );

    auto min3 = is3.min();
    EXPECT_EQ( min3[0], 0 );
    EXPECT_EQ( min3[1], 0 );
    EXPECT_EQ( min3[2], 0 );
    auto max3 = is3.max();
    EXPECT_EQ( max3[0], s0 );
    EXPECT_EQ( max3[1], s1 );
    EXPECT_EQ( max3[2], s2 );

    auto r30 = is3.range( 0 );
    EXPECT_EQ( r30.first, 0 );
    EXPECT_EQ( r30.second, s0 );
    auto r31 = is3.range( 1 );
    EXPECT_EQ( r31.first, 0 );
    EXPECT_EQ( r31.second, s1 );
    auto r32 = is3.range( 2 );
    EXPECT_EQ( r32.first, 0 );
    EXPECT_EQ( r32.second, s2 );

    EXPECT_EQ( is3.rank(), 3 );
    EXPECT_EQ( is3.extent( 0 ), s0 );
    EXPECT_EQ( is3.extent( 1 ), s1 );
    EXPECT_EQ( is3.extent( 2 ), s2 );

    // Rank-4
    int s3 = 4;
    IndexSpace<4> is4( { 0, 0, 0, 0 }, { s0, s1, s2, s3 } );
    EXPECT_EQ( is4.min( 0 ), 0 );
    EXPECT_EQ( is4.max( 0 ), s0 );
    EXPECT_EQ( is4.min( 1 ), 0 );
    EXPECT_EQ( is4.max( 1 ), s1 );
    EXPECT_EQ( is4.min( 2 ), 0 );
    EXPECT_EQ( is4.max( 2 ), s2 );
    EXPECT_EQ( is4.min( 3 ), 0 );
    EXPECT_EQ( is4.max( 3 ), s3 );

    auto min4 = is4.min();
    EXPECT_EQ( min4[0], 0 );
    EXPECT_EQ( min4[1], 0 );
    EXPECT_EQ( min4[2], 0 );
    EXPECT_EQ( min4[3], 0 );
    auto max4 = is4.max();
    EXPECT_EQ( max4[0], s0 );
    EXPECT_EQ( max4[1], s1 );
    EXPECT_EQ( max4[2], s2 );
    EXPECT_EQ( max4[3], s3 );

    auto r40 = is4.range( 0 );
    EXPECT_EQ( r40.first, 0 );
    EXPECT_EQ( r40.second, s0 );
    auto r41 = is4.range( 1 );
    EXPECT_EQ( r41.first, 0 );
    EXPECT_EQ( r41.second, s1 );
    auto r42 = is4.range( 2 );
    EXPECT_EQ( r42.first, 0 );
    EXPECT_EQ( r42.second, s2 );
    auto r43 = is4.range( 3 );
    EXPECT_EQ( r43.first, 0 );
    EXPECT_EQ( r43.second, s3 );

    EXPECT_EQ( is4.rank(), 4 );
    EXPECT_EQ( is4.extent( 0 ), s0 );
    EXPECT_EQ( is4.extent( 1 ), s1 );
    EXPECT_EQ( is4.extent( 2 ), s2 );
    EXPECT_EQ( is4.extent( 3 ), s3 );
}

//---------------------------------------------------------------------------//
void executionTest()
{
    // Rank-1.
    int min_i = 4;
    int max_i = 8;
    int size_i = 12;
    IndexSpace<1> is1( { min_i }, { max_i } );
    Kokkos::View<double *, TEST_DEVICE> v1( "v1", size_i );
    Kokkos::parallel_for(
        "fill_rank_1", createExecutionPolicy( is1, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i ) { v1( i ) = 1.0; } );
    auto v1_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), v1 );
    for ( int i = 0; i < size_i; ++i )
    {
        if ( is1.min( 0 ) <= i && i < is1.max( 0 ) )
            EXPECT_EQ( v1_mirror( i ), 1.0 );
        else
            EXPECT_EQ( v1_mirror( i ), 0.0 );
    }

    // Rank-2
    int min_j = 3;
    int max_j = 9;
    int size_j = 18;
    IndexSpace<2> is2( { min_i, min_j }, { max_i, max_j } );
    Kokkos::View<double **, TEST_DEVICE> v2( "v2", size_i, size_j );
    Kokkos::parallel_for(
        "fill_rank_2", createExecutionPolicy( is2, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j ) { v2( i, j ) = 1.0; } );
    auto v2_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), v2 );
    for ( int i = 0; i < size_i; ++i )
        for ( int j = 0; j < size_j; ++j )
        {
            if ( is2.min( 0 ) <= i && i < is2.max( 0 ) && is2.min( 1 ) <= j &&
                 j < is2.max( 1 ) )
                EXPECT_EQ( v2_mirror( i, j ), 1.0 );
            else
                EXPECT_EQ( v2_mirror( i, j ), 0.0 );
        }

    // Rank-3
    int min_k = 2;
    int max_k = 11;
    int size_k = 13;
    IndexSpace<3> is3( { min_i, min_j, min_k }, { max_i, max_j, max_k } );
    Kokkos::View<double ***, TEST_DEVICE> v3( "v3", size_i, size_j, size_k );
    Kokkos::parallel_for(
        "fill_rank_3", createExecutionPolicy( is3, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            v3( i, j, k ) = 1.0;
        } );
    auto v3_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), v3 );
    for ( int i = 0; i < size_i; ++i )
        for ( int j = 0; j < size_j; ++j )
            for ( int k = 0; k < size_k; ++k )
            {
                if ( is3.min( 0 ) <= i && i < is3.max( 0 ) &&
                     is3.min( 1 ) <= j && j < is3.max( 1 ) &&
                     is3.min( 2 ) <= k && k < is3.max( 2 ) )
                    EXPECT_EQ( v3_mirror( i, j, k ), 1.0 );
                else
                    EXPECT_EQ( v3_mirror( i, j, k ), 0.0 );
            }

    // Rank-4
    int min_l = 7;
    int max_l = 9;
    int size_l = 14;
    IndexSpace<4> is4( { min_i, min_j, min_k, min_l },
                       { max_i, max_j, max_k, max_l } );
    Kokkos::View<double ****, TEST_DEVICE> v4( "v4", size_i, size_j, size_k,
                                               size_l );
    Kokkos::parallel_for(
        "fill_rank_4", createExecutionPolicy( is4, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ) {
            v4( i, j, k, l ) = 1.0;
        } );
    auto v4_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), v4 );
    for ( int i = 0; i < size_i; ++i )
        for ( int j = 0; j < size_j; ++j )
            for ( int k = 0; k < size_k; ++k )
                for ( int l = 0; l < size_l; ++l )
                {
                    if ( is4.min( 0 ) <= i && i < is4.max( 0 ) &&
                         is4.min( 1 ) <= j && j < is4.max( 1 ) &&
                         is4.min( 2 ) <= k && k < is4.max( 2 ) &&
                         is4.min( 3 ) <= l && l < is4.max( 3 ) )
                        EXPECT_EQ( v4_mirror( i, j, k, l ), 1.0 );
                    else
                        EXPECT_EQ( v4_mirror( i, j, k, l ), 0.0 );
                }
}

//---------------------------------------------------------------------------//
void subviewTest()
{
    // Rank-1.
    int min_i = 4;
    int max_i = 8;
    int size_i = 12;
    IndexSpace<1> is1( { min_i }, { max_i } );
    Kokkos::View<double *, TEST_DEVICE> v1( "v1", size_i );
    auto sv1 = createSubview( v1, is1 );
    Kokkos::deep_copy( sv1, 1.0 );
    auto v1_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), v1 );
    for ( int i = 0; i < size_i; ++i )
    {
        if ( is1.range( 0 ).first <= i && i < is1.range( 0 ).second )
            EXPECT_EQ( v1_mirror( i ), 1.0 );
        else
            EXPECT_EQ( v1_mirror( i ), 0.0 );
    }

    // Rank-2
    int min_j = 3;
    int max_j = 9;
    int size_j = 18;
    IndexSpace<2> is2( { min_i, min_j }, { max_i, max_j } );
    Kokkos::View<double **, TEST_DEVICE> v2( "v2", size_i, size_j );
    auto sv2 = createSubview( v2, is2 );
    Kokkos::deep_copy( sv2, 1.0 );
    auto v2_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), v2 );
    for ( int i = 0; i < size_i; ++i )
        for ( int j = 0; j < size_j; ++j )
        {
            if ( is2.range( 0 ).first <= i && i < is2.range( 0 ).second &&
                 is2.range( 1 ).first <= j && j < is2.range( 1 ).second )
                EXPECT_EQ( v2_mirror( i, j ), 1.0 );
            else
                EXPECT_EQ( v2_mirror( i, j ), 0.0 );
        }

    // Rank-3
    int min_k = 2;
    int max_k = 11;
    int size_k = 13;
    IndexSpace<3> is3( { min_i, min_j, min_k }, { max_i, max_j, max_k } );
    Kokkos::View<double ***, TEST_DEVICE> v3( "v3", size_i, size_j, size_k );
    auto sv3 = createSubview( v3, is3 );
    Kokkos::deep_copy( sv3, 1.0 );
    auto v3_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), v3 );
    for ( int i = 0; i < size_i; ++i )
        for ( int j = 0; j < size_j; ++j )
            for ( int k = 0; k < size_k; ++k )
            {
                if ( is3.range( 0 ).first <= i && i < is3.range( 0 ).second &&
                     is3.range( 1 ).first <= j && j < is3.range( 1 ).second &&
                     is3.range( 2 ).first <= k && k < is3.range( 2 ).second )
                    EXPECT_EQ( v3_mirror( i, j, k ), 1.0 );
                else
                    EXPECT_EQ( v3_mirror( i, j, k ), 0.0 );
            }

    // Rank-4
    int min_l = 7;
    int max_l = 9;
    int size_l = 14;
    IndexSpace<4> is4( { min_i, min_j, min_k, min_l },
                       { max_i, max_j, max_k, max_l } );
    Kokkos::View<double ****, TEST_DEVICE> v4( "v4", size_i, size_j, size_k,
                                               size_l );
    auto sv4 = createSubview( v4, is4 );
    Kokkos::deep_copy( sv4, 1.0 );
    auto v4_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), v4 );
    for ( int i = 0; i < size_i; ++i )
        for ( int j = 0; j < size_j; ++j )
            for ( int k = 0; k < size_k; ++k )
                for ( int l = 0; l < size_l; ++l )
                {
                    if ( is4.range( 0 ).first <= i &&
                         i < is4.range( 0 ).second &&
                         is4.range( 1 ).first <= j &&
                         j < is4.range( 1 ).second &&
                         is4.range( 2 ).first <= k &&
                         k < is4.range( 2 ).second &&
                         is4.range( 3 ).first <= l &&
                         l < is4.range( 3 ).second )
                        EXPECT_EQ( v4_mirror( i, j, k, l ), 1.0 );
                    else
                        EXPECT_EQ( v4_mirror( i, j, k, l ), 0.0 );
                }
}

//---------------------------------------------------------------------------//
void sizeAppendTest()
{
    int s0 = 3;
    IndexSpace<1> is1( { s0 } );
    int s1 = 5;
    auto is2 = appendDimension( is1, s1 );

    EXPECT_EQ( is2.min( 0 ), 0 );
    EXPECT_EQ( is2.max( 0 ), s0 );
    EXPECT_EQ( is2.min( 1 ), 0 );
    EXPECT_EQ( is2.max( 1 ), s1 );

    auto min2 = is2.min();
    EXPECT_EQ( min2[0], 0 );
    EXPECT_EQ( min2[1], 0 );
    auto max2 = is2.max();
    EXPECT_EQ( max2[0], s0 );
    EXPECT_EQ( max2[1], s1 );

    auto r20 = is2.range( 0 );
    EXPECT_EQ( r20.first, 0 );
    EXPECT_EQ( r20.second, s0 );
    auto r21 = is2.range( 1 );
    EXPECT_EQ( r21.first, 0 );
    EXPECT_EQ( r21.second, s1 );

    EXPECT_EQ( is2.rank(), 2 );
    EXPECT_EQ( is2.extent( 0 ), s0 );
    EXPECT_EQ( is2.extent( 1 ), s1 );
}

//---------------------------------------------------------------------------//
void rangeAppendTest()
{
    int s0 = 3;
    IndexSpace<1> is1( { s0 } );
    int s1 = 5;
    auto is2 = appendDimension( is1, 0, s1 );

    EXPECT_EQ( is2.min( 0 ), 0 );
    EXPECT_EQ( is2.max( 0 ), s0 );
    EXPECT_EQ( is2.min( 1 ), 0 );
    EXPECT_EQ( is2.max( 1 ), s1 );

    auto min2 = is2.min();
    EXPECT_EQ( min2[0], 0 );
    EXPECT_EQ( min2[1], 0 );
    auto max2 = is2.max();
    EXPECT_EQ( max2[0], s0 );
    EXPECT_EQ( max2[1], s1 );

    auto r20 = is2.range( 0 );
    EXPECT_EQ( r20.first, 0 );
    EXPECT_EQ( r20.second, s0 );
    auto r21 = is2.range( 1 );
    EXPECT_EQ( r21.first, 0 );
    EXPECT_EQ( r21.second, s1 );

    EXPECT_EQ( is2.rank(), 2 );
    EXPECT_EQ( is2.extent( 0 ), s0 );
    EXPECT_EQ( is2.extent( 1 ), s1 );
}

//---------------------------------------------------------------------------//
void comparisonTest()
{
    IndexSpace<3> is1( { 9, 2, 1 }, { 12, 16, 4 } );
    IndexSpace<3> is2( { 9, 2, 1 }, { 12, 16, 4 } );
    IndexSpace<3> is3( { 9, 2, 1 }, { 12, 16, 5 } );
    EXPECT_TRUE( is1 == is2 );
    EXPECT_FALSE( is1 != is2 );
    EXPECT_FALSE( is1 == is3 );
    EXPECT_TRUE( is1 != is3 );
}

//---------------------------------------------------------------------------//
void defaultConstructorTest()
{
    IndexSpace<3> is1;
    EXPECT_EQ( is1.min( 0 ), -1 );
    EXPECT_EQ( is1.min( 1 ), -1 );
    EXPECT_EQ( is1.min( 2 ), -1 );
    EXPECT_EQ( is1.max( 0 ), -1 );
    EXPECT_EQ( is1.max( 1 ), -1 );
    EXPECT_EQ( is1.max( 2 ), -1 );

    IndexSpace<3> is2( { 9, 2, 1 }, { 12, 16, 4 } );
    is1 = is2;
    EXPECT_TRUE( is1 == is2 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, index_space_test )
{
    sizeConstructorTest();
    rangeConstructorTest();
    executionTest();
    subviewTest();
    sizeAppendTest();
    rangeAppendTest();
    comparisonTest();
    defaultConstructorTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
