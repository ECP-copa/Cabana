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

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// Initialize data members
template <class aosoa_type>
void initializeDataMembers( aosoa_type aosoa, const float fval,
                            const double dval, const int ival, const int dim_1,
                            const int dim_2, const int dim_3 )
{
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    auto slice_2 = Cabana::slice<2>( aosoa );
    auto slice_3 = Cabana::slice<3>( aosoa );

    Kokkos::parallel_for(
        "init_members", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int idx ) {
            // Member 0.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    for ( int k = 0; k < dim_3; ++k )
                        slice_0( idx, i, j, k ) = fval * ( i + j + k );

            // Member 1.
            slice_1( idx ) = ival;

            // Member 2.
            for ( int i = 0; i < dim_1; ++i )
                slice_2( idx, i ) = dval * i;

            // Member 3.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    slice_3( idx, i, j ) = dval * ( i + j );
        } );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
// Check the data given a set of values.
template <class aosoa_type>
void checkDataMembers( aosoa_type aosoa, const float fval, const double dval,
                       const int ival, const int dim_1, const int dim_2,
                       const int dim_3 )
{
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );

    auto slice_0 = Cabana::slice<0>( mirror );
    auto slice_1 = Cabana::slice<1>( mirror );
    auto slice_2 = Cabana::slice<2>( mirror );
    auto slice_3 = Cabana::slice<3>( mirror );

    for ( std::size_t idx = 0; idx != aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    EXPECT_FLOAT_EQ( slice_0( idx, i, j, k ),
                                     fval * ( i + j + k ) );

        // Member 1.
        EXPECT_EQ( slice_1( idx ), ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_DOUBLE_EQ( slice_2( idx, i ), dval * i );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_DOUBLE_EQ( slice_3( idx, i, j ), dval * ( i + j ) );
    }
}

//---------------------------------------------------------------------------//
// API test function
void apiTest()
{
    // Manually set the inner array size with the test layout.
    const int vector_length = 16;

    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          double[dim_1], double[dim_1][dim_2]>;

    // Create an AoSoA.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE, vector_length>;
    int num_data = 35;
    AoSoA_t aosoa( "aosoa", num_data );

    // Create some slices.
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    auto slice_2 = Cabana::slice<2>( aosoa );
    auto slice_3 = Cabana::slice<3>( aosoa );

    // Check that they are slices.
    EXPECT_TRUE( Cabana::is_slice<decltype( slice_0 )>::value );
    EXPECT_TRUE( Cabana::is_slice<decltype( slice_1 )>::value );
    EXPECT_TRUE( Cabana::is_slice<decltype( slice_2 )>::value );
    EXPECT_TRUE( Cabana::is_slice<decltype( slice_3 )>::value );

    // Check field sizes.
    EXPECT_EQ( slice_0.size(), 35 );
    EXPECT_EQ( slice_0.numSoA(), 3 );

    EXPECT_EQ( slice_1.size(), 35 );
    EXPECT_EQ( slice_1.numSoA(), 3 );

    EXPECT_EQ( slice_2.size(), 35 );
    EXPECT_EQ( slice_2.numSoA(), 3 );

    EXPECT_EQ( slice_3.size(), 35 );
    EXPECT_EQ( slice_3.numSoA(), 3 );

    // Initialize data with the () operator. The implementation of operator()
    // calls access() and therefore tests that as well.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    initializeDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Check the raw pointer interface sizes.
    EXPECT_EQ( slice_0.viewRank(), 5 );
    EXPECT_EQ( slice_0.rank, 4 );
    EXPECT_EQ( slice_0.extent( 0 ), 3 );
    EXPECT_EQ( slice_0.extent( 1 ), 16 );
    EXPECT_EQ( slice_0.extent( 2 ), dim_1 );
    EXPECT_EQ( slice_0.extent( 3 ), dim_2 );
    EXPECT_EQ( slice_0.extent( 4 ), dim_3 );

    EXPECT_EQ( slice_1.viewRank(), 2 );
    EXPECT_EQ( slice_1.rank, 1 );
    EXPECT_EQ( slice_1.extent( 0 ), 3 );
    EXPECT_EQ( slice_1.extent( 1 ), 16 );

    EXPECT_EQ( slice_2.viewRank(), 3 );
    EXPECT_EQ( slice_2.rank, 2 );
    EXPECT_EQ( slice_2.extent( 0 ), 3 );
    EXPECT_EQ( slice_2.extent( 1 ), 16 );
    EXPECT_EQ( slice_2.extent( 2 ), dim_1 );

    EXPECT_EQ( slice_3.viewRank(), 4 );
    EXPECT_EQ( slice_3.rank, 3 );
    EXPECT_EQ( slice_3.extent( 0 ), 3 );
    EXPECT_EQ( slice_3.extent( 1 ), 16 );
    EXPECT_EQ( slice_3.extent( 2 ), dim_1 );
    EXPECT_EQ( slice_3.extent( 3 ), dim_2 );

    // Now manipulate the data with the raw pointer interface.
    fval = 9.22;
    dval = 5.67;
    ival = 12;
    auto p0 = slice_0.data();
    auto p1 = slice_1.data();
    auto p2 = slice_2.data();
    auto p3 = slice_3.data();
    Kokkos::parallel_for(
        "raw_ptr_update",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, slice_0.numSoA() ),
        KOKKOS_LAMBDA( const int s ) {
            for ( std::size_t a = 0; a < slice_0.arraySize( s ); ++a )
            {
                // Member 0.
                for ( int i = 0; i < dim_1; ++i )
                    for ( int j = 0; j < dim_2; ++j )
                        for ( int k = 0; k < dim_3; ++k )
                            p0[s * slice_0.stride( 0 ) +
                               a * slice_0.stride( 1 ) +
                               i * slice_0.stride( 2 ) +
                               j * slice_0.stride( 3 ) +
                               k * slice_0.stride( 4 )] = fval * ( i + j + k );

                // Member 1.
                p1[s * slice_1.stride( 0 ) + a * slice_1.stride( 1 )] = ival;

                // Member 2.
                for ( int i = 0; i < dim_1; ++i )
                    p2[s * slice_2.stride( 0 ) + a * slice_2.stride( 1 ) +
                       i * slice_2.stride( 2 )] = dval * i;

                // Member 3.
                for ( int i = 0; i < dim_1; ++i )
                    for ( int j = 0; j < dim_2; ++j )
                        p3[s * slice_3.stride( 0 ) + a * slice_3.stride( 1 ) +
                           i * slice_3.stride( 2 ) + j * slice_3.stride( 3 )] =
                            dval * ( i + j );
            }
        } );
    Kokkos::fence();

    // Check the result of pointer manipulation
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// Random access function
void randomAccessTest()
{
    // Manually set the inner array size with the test layout.
    const int vector_length = 16;

    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          double[dim_1], double[dim_1][dim_2]>;

    // Create an AoSoA.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE, vector_length>;
    int num_data = 35;
    AoSoA_t aosoa( "aosoa", num_data );

    // Initialize data.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    initializeDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Create slices.
    auto da_slice_0 = Cabana::slice<0>( aosoa );
    auto da_slice_1 = Cabana::slice<1>( aosoa );
    auto da_slice_2 = Cabana::slice<2>( aosoa );
    auto da_slice_3 = Cabana::slice<3>( aosoa );

    // Create read-only random access slices.
    decltype( da_slice_0 )::random_access_slice ra_slice_0 = da_slice_0;
    decltype( da_slice_1 )::random_access_slice ra_slice_1 = da_slice_1;
    decltype( da_slice_2 )::random_access_slice ra_slice_2 = da_slice_2;
    decltype( da_slice_3 )::random_access_slice ra_slice_3 = da_slice_3;

    // Create a second aosoa.
    AoSoA_t aosoa_2( "aosoa_2", num_data );

    // Get normal slices of the data.
    auto slice_0 = Cabana::slice<0>( aosoa_2 );
    auto slice_1 = Cabana::slice<1>( aosoa_2 );
    auto slice_2 = Cabana::slice<2>( aosoa_2 );
    auto slice_3 = Cabana::slice<3>( aosoa_2 );

    // Assign the read-only data to the new aosoa.
    Kokkos::parallel_for(
        "assign read only",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int idx ) {
            // Member 0.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    for ( int k = 0; k < dim_3; ++k )
                        slice_0( idx, i, j, k ) = ra_slice_0( idx, i, j, k );

            // Member 1.
            slice_1( idx ) = ra_slice_1( idx );

            // Member 2.
            for ( int i = 0; i < dim_1; ++i )
                slice_2( idx, i ) = ra_slice_2( idx, i );

            // Member 3.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    slice_3( idx, i, j ) = ra_slice_3( idx, i, j );
        } );
    Kokkos::fence();

    // Check data members for proper assignment.
    checkDataMembers( aosoa_2, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// Random access function
void atomicAccessTest()
{
    // Manually set the inner array size with the test layout.
    const int vector_length = 16;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<int>;

    // Create an AoSoA.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE, vector_length>;
    int num_data = 35;
    AoSoA_t aosoa( "aosoa", num_data );

    // Get a slice of the data.
    auto slice = Cabana::slice<0>( aosoa );

    // Set to 0.
    Kokkos::parallel_for(
        "assign", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_data ),
        KOKKOS_LAMBDA( const int i ) { slice( i ) = 0; } );

    // Get an atomic slice of the data.
    decltype( slice )::atomic_access_slice atomic_slice = slice;

    // Have every thread increment all elements of the slice. This should
    // create contention in parallel without the atomic.
    auto increment_op = KOKKOS_LAMBDA( const int )
    {
        for ( int j = 0; j < num_data; ++j )
            atomic_slice( j ) += 1;
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> exec_policy( 0, num_data );
    Kokkos::parallel_for( exec_policy, increment_op );
    Kokkos::fence();

    // Check the results of the atomic increment.
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto mirror_slice = Cabana::slice<0>( mirror );

    for ( int i = 0; i < num_data; ++i )
        EXPECT_EQ( mirror_slice( i ), num_data );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, api_test ) { apiTest(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, random_access_test ) { randomAccessTest(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, atomic_access_test ) { atomicAccessTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
