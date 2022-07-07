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
#include <Cabana_Types.hpp>
#include <impl/Cabana_Index.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// Check the data given a set of values in an aosoa.
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

    for ( std::size_t idx = 0; idx < aosoa.size(); ++idx )
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
// Test an AoSoA.
void testAoSoA()
{
    // Manually set the inner array size.
    const int vector_length = 16;

    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          double[dim_1], double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE, vector_length>;

    // Make sure that it is actually an AoSoA.
    EXPECT_TRUE( Cabana::is_aosoa<AoSoA_t>::value );

    // Create an AoSoA.
    std::string label = "test_aosoa";
    AoSoA_t aosoa( label );
    EXPECT_EQ( aosoa.label(), label );

    // Get field slices.
    std::string s0_label = "slice_0";
    auto slice_0 = Cabana::slice<0>( aosoa, s0_label );
    EXPECT_EQ( slice_0.label(), s0_label );

    std::string s1_label = "slice_1";
    auto slice_1 = Cabana::slice<1>( aosoa, s1_label );
    EXPECT_EQ( slice_1.label(), s1_label );

    std::string s2_label = "slice_2";
    auto slice_2 = Cabana::slice<2>( aosoa, s2_label );
    EXPECT_EQ( slice_2.label(), s2_label );

    std::string s3_label = "slice_3";
    auto slice_3 = Cabana::slice<3>( aosoa, s3_label );
    EXPECT_EQ( slice_3.label(), s3_label );

    // Check sizes.
    EXPECT_EQ( aosoa.size(), int( 0 ) );
    EXPECT_EQ( aosoa.capacity(), int( 0 ) );
    EXPECT_EQ( aosoa.numSoA(), int( 0 ) );
    EXPECT_TRUE( aosoa.empty() );

    // Resize
    int num_data = 35;
    aosoa.resize( num_data );

    // Check sizes for the new allocation/size.
    EXPECT_EQ( aosoa.size(), int( 35 ) );
    EXPECT_EQ( aosoa.capacity(), int( 48 ) );
    EXPECT_EQ( aosoa.numSoA(), int( 3 ) );
    EXPECT_FALSE( aosoa.empty() );

    EXPECT_EQ( aosoa.arraySize( 0 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 1 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 2 ), int( 3 ) );

    // Test bounds.
    auto end = aosoa.size();
    int end_s = Cabana::Impl::Index<16>::s( end );
    int end_a = Cabana::Impl::Index<16>::a( end );
    EXPECT_EQ( end_s, 2 );
    EXPECT_EQ( end_a, 3 );

    // Create a mirror on the host and fill.
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto mirror_slice_0 = Cabana::slice<0>( mirror );
    auto mirror_slice_1 = Cabana::slice<1>( mirror );
    auto mirror_slice_2 = Cabana::slice<2>( mirror );
    auto mirror_slice_3 = Cabana::slice<3>( mirror );

    // Initialize data with the rank accessors.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    for ( std::size_t idx = 0; idx != aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    mirror_slice_0( idx, i, j, k ) = fval * ( i + j + k );

        // Member 1.
        mirror_slice_1( idx ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            mirror_slice_2( idx, i ) = dval * i;

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                mirror_slice_3( idx, i, j ) = dval * ( i + j );
    }
    Cabana::deep_copy( aosoa, mirror );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Now extend the capacity of the container. First make the capacity
    // smaller - this wont actually do anything because we never decrease the
    // allocation of the container.
    aosoa.reserve( 1 );

    // Make sure nothing changed.
    EXPECT_EQ( aosoa.size(), int( 35 ) );
    EXPECT_EQ( aosoa.capacity(), int( 48 ) );
    EXPECT_EQ( aosoa.numSoA(), int( 3 ) );
    EXPECT_EQ( aosoa.arraySize( 0 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 1 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 2 ), int( 3 ) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Now reserve a bunch of space.
    aosoa.reserve( 1024 );

    // Make sure capacity changed but sizes and data did not.
    EXPECT_EQ( aosoa.size(), int( 35 ) );
    EXPECT_EQ( aosoa.capacity(), int( 1024 ) );
    EXPECT_EQ( aosoa.numSoA(), int( 3 ) );
    EXPECT_EQ( aosoa.arraySize( 0 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 1 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 2 ), int( 3 ) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Now decrease the size of the container.
    aosoa.resize( 29 );

    // Make sure sizes and data changed but the capacity did not.
    EXPECT_EQ( aosoa.size(), int( 29 ) );
    EXPECT_EQ( aosoa.capacity(), int( 1024 ) );
    EXPECT_EQ( aosoa.numSoA(), int( 2 ) );
    EXPECT_EQ( aosoa.arraySize( 0 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 1 ), int( 13 ) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Now shrink to fit and check that the capacity changed to be as small as
    // possible while the remainder of the size values and elements are the
    // same.
    aosoa.shrinkToFit();
    EXPECT_EQ( aosoa.size(), int( 29 ) );
    EXPECT_EQ( aosoa.capacity(), int( 32 ) );
    EXPECT_EQ( aosoa.numSoA(), int( 2 ) );
    EXPECT_EQ( aosoa.arraySize( 0 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 1 ), int( 13 ) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Shrink again - nothing should change this time.
    aosoa.shrinkToFit();
    EXPECT_EQ( aosoa.size(), int( 29 ) );
    EXPECT_EQ( aosoa.capacity(), int( 32 ) );
    EXPECT_EQ( aosoa.numSoA(), int( 2 ) );
    EXPECT_EQ( aosoa.arraySize( 0 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 1 ), int( 13 ) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Now resize smaller, then larger again to confirm underlying
    // Kokkos::deep_copy will work without deprecated code.
    aosoa.resize( 15 );
    aosoa.resize( 47 );
    EXPECT_EQ( aosoa.size(), int( 47 ) );
    EXPECT_EQ( aosoa.capacity(), int( 48 ) );
    EXPECT_EQ( aosoa.numSoA(), int( 3 ) );
    EXPECT_EQ( aosoa.arraySize( 0 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 1 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 2 ), int( 15 ) );

    // Now resize to numSoA*vector_length
    // Make sure it works when all SoA structures are fully occupied
    aosoa.resize( 48 );
    EXPECT_EQ( aosoa.size(), int( 48 ) );
    EXPECT_EQ( aosoa.capacity(), int( 48 ) );
    EXPECT_EQ( aosoa.numSoA(), int( 3 ) );
    EXPECT_EQ( aosoa.arraySize( 0 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 1 ), int( 16 ) );
    EXPECT_EQ( aosoa.arraySize( 2 ), int( 16 ) );
}

//---------------------------------------------------------------------------//
// Raw data test.
void testRawData()
{
    // Manually set the inner array size.
    const int vector_length = 16;

    // Multi dimensional member sizes.
    const int dim_1 = 3;
    const int dim_2 = 5;

    // Declare data types. Note that this test only uses rank-0 data.
    using DataTypes =
        Cabana::MemberTypes<float, int, double[dim_1][dim_2], int, double>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE, vector_length>;

    // Create an AoSoA using the default constructor.
    int num_data = 350;
    std::string label = "test_aosoa";
    AoSoA_t aosoa( label, num_data );
    EXPECT_EQ( aosoa.label(), label );

    // Get slices of fields.
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    auto slice_2 = Cabana::slice<2>( aosoa );
    auto slice_3 = Cabana::slice<3>( aosoa );
    auto slice_4 = Cabana::slice<4>( aosoa );

    // Get raw pointers to the data as one would in a C interface (no
    // templates).
    float* p0 = slice_0.data();
    int* p1 = slice_1.data();
    double* p2 = slice_2.data();
    int* p3 = slice_3.data();
    double* p4 = slice_4.data();

    // Get the strides between the member arrays.
    int st0 = slice_0.stride( 0 );
    int st1 = slice_1.stride( 0 );
    int st2 = slice_2.stride( 0 );
    int st3 = slice_3.stride( 0 );
    int st4 = slice_4.stride( 0 );

    // Member 2 is multidimensional so get its extents.
    int m2e0 = slice_2.extent( 2 );
    int m2e1 = slice_2.extent( 3 );
    EXPECT_EQ( m2e0, dim_1 );
    EXPECT_EQ( m2e1, dim_2 );

    // Initialize the data with raw pointer/stride access. Start by looping
    // over the structs. Each struct has a group of contiguous arrays of size
    // array_size for each member.
    Kokkos::parallel_for(
        "raw_data_fill",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, slice_0.numSoA() ),
        KOKKOS_LAMBDA( const int s ) {
            // Loop over the array in each struct and set the values.
            int local_array_size = slice_0.arraySize( s );
            for ( int i = 0; i < local_array_size; ++i )
            {
                p0[s * st0 + i] = ( s + i ) * 1.0;
                p1[s * st1 + i] = ( s + i ) * 2;
                p3[s * st3 + i] = ( s + i ) * 4;
                p4[s * st4 + i] = ( s + i ) * 5.0;

                // Member 2 has some extra dimensions so add those to the
                // indexing. Note this is layout left.
                for ( int j = 0; j < m2e0; ++j )
                    for ( int k = 0; k < m2e1; ++k )
                        p2[s * st2 + j * 16 * m2e1 + k * 16 + i] =
                            ( s + i + j + k ) * 3.0;
            }
        } );
    Kokkos::fence();

    // Check the results.
    auto mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto mirror_slice_0 = Cabana::slice<0>( mirror );
    auto mirror_slice_1 = Cabana::slice<1>( mirror );
    auto mirror_slice_2 = Cabana::slice<2>( mirror );
    auto mirror_slice_3 = Cabana::slice<3>( mirror );
    auto mirror_slice_4 = Cabana::slice<4>( mirror );
    for ( std::size_t idx = 0; idx < aosoa.size(); ++idx )
    {
        int s = Cabana::Impl::Index<16>::s( idx );
        int a = Cabana::Impl::Index<16>::a( idx );

        EXPECT_FLOAT_EQ( mirror_slice_0( idx ), ( s + a ) * 1.0 );
        EXPECT_EQ( mirror_slice_1( idx ), int( ( s + a ) * 2 ) );
        EXPECT_EQ( mirror_slice_3( idx ), int( ( s + a ) * 4 ) );
        EXPECT_DOUBLE_EQ( mirror_slice_4( idx ), ( s + a ) * 5.0 );

        // Member 2 has some extra dimensions so check those too.
        for ( int j = 0; j < dim_1; ++j )
            for ( int k = 0; k < dim_2; ++k )
                EXPECT_DOUBLE_EQ( mirror_slice_2( idx, j, k ),
                                  ( s + a + j + k ) * 3.0 );
    }
}

//---------------------------------------------------------------------------//
// Tuple test.
void testTuple()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare member types.
    using T0 = float[dim_1][dim_2][dim_3];
    using T1 = int;
    using T2 = double[dim_1];
    using T3 = double[dim_1][dim_2];

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<T0, T1, T2, T3>;

    // Declare the tuple type.
    using Tuple_t = Cabana::Tuple<DataTypes>;

    // Create an AoSoA.
    int num_data = 453;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t aosoa( "label", num_data );

    // Create a slice of tuples with the same data types.
    Kokkos::View<Tuple_t*, typename AoSoA_t::memory_space> tuples( "tuples",
                                                                   num_data );

    // Initialize aosoa data.
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    auto slice_2 = Cabana::slice<2>( aosoa );
    auto slice_3 = Cabana::slice<3>( aosoa );
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    Kokkos::parallel_for(
        "TupleFill", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
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

    // Assign the AoSoA data to the tuples.
    Kokkos::parallel_for(
        "TupleAssign", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int idx ) {
            tuples( idx ) = aosoa.getTuple( idx );
        } );
    Kokkos::fence();

    // Change the tuple data.
    fval = 2.1;
    dval = 9.21;
    ival = 3;
    Kokkos::parallel_for(
        "TupleUpdate", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int idx ) {
            // Member 0.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    for ( int k = 0; k < dim_3; ++k )
                        Cabana::get<0>( tuples( idx ), i, j, k ) =
                            fval * ( i + j + k );

            // Member 1.
            Cabana::get<1>( tuples( idx ) ) = ival;

            // Member 2.
            for ( int i = 0; i < dim_1; ++i )
                Cabana::get<2>( tuples( idx ), i ) = dval * i;

            // Member 3.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    Cabana::get<3>( tuples( idx ), i, j ) = dval * ( i + j );
        } );
    Kokkos::fence();

    // Assign the tuple data back to the AoSoA.
    Kokkos::parallel_for(
        "TupleReAssign", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        KOKKOS_LAMBDA( const int idx ) {
            aosoa.setTuple( idx, tuples( idx ) );
        } );
    Kokkos::fence();

    // Check the results.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// Test an AoSoA using the access operator.
void testAccess()
{
    // Manually set the inner array size.
    const int vector_length = 16;

    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          double[dim_1], double[dim_1][dim_2]>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE, vector_length>;

    // Make sure that it is actually an AoSoA.
    EXPECT_TRUE( Cabana::is_aosoa<AoSoA_t>::value );

    // Create an AoSoA.
    int num_data = 453;
    AoSoA_t aosoa( "label", num_data );

    // Initialize data with the SoA accessor
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    Kokkos::parallel_for(
        "data_fill", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.numSoA() ),
        KOKKOS_LAMBDA( const int s ) {
            auto& soa = aosoa.access( s );

            for ( std::size_t a = 0; a < aosoa.arraySize( s ); ++a )
            {
                // Member 0.
                for ( int i = 0; i < dim_1; ++i )
                    for ( int j = 0; j < dim_2; ++j )
                        for ( int k = 0; k < dim_3; ++k )
                            Cabana::get<0>( soa, a, i, j, k ) =
                                fval * ( i + j + k );

                // Member 1.
                Cabana::get<1>( soa, a ) = ival;

                // Member 2.
                for ( int i = 0; i < dim_1; ++i )
                    Cabana::get<2>( soa, a, i ) = dval * i;

                // Member 3.
                for ( int i = 0; i < dim_1; ++i )
                    for ( int j = 0; j < dim_2; ++j )
                        Cabana::get<3>( soa, a, i, j ) = dval * ( i + j );
            }
        } );
    Kokkos::fence();

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// Manually defined SoA.
template <int VLEN, int D1, int D2, int D3>
struct MySoA
{
    float m0[D1][D2][D3][VLEN];
    int m1[VLEN];
    double m2[D1][VLEN];
    double m3[D1][D2][VLEN];
};

// Test an unmanaged AoSoA.
void testUnmanaged()
{
    // Manually set the inner array size.
    const int vector_length = 16;

    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types that are equivalent to the user defined struct.
    using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          double[dim_1], double[dim_1][dim_2]>;

    // Allocate an AoSoA manually.
    using soa_type = MySoA<vector_length, dim_1, dim_2, dim_3>;
    int num_soa = 3;
    int size = 35;
    Kokkos::View<soa_type*, TEST_MEMSPACE> user_data( "user_aosoa", 3 );

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE, vector_length,
                                  Kokkos::MemoryUnmanaged>;

    // Create an AoSoA.
    auto user_ptr =
        reinterpret_cast<typename AoSoA_t::soa_type*>( user_data.data() );
    AoSoA_t aosoa( user_ptr, num_soa, size );

    // Check sizes.
    EXPECT_EQ( aosoa.size(), size );
    EXPECT_EQ( aosoa.capacity(), num_soa * vector_length );
    EXPECT_EQ( aosoa.numSoA(), num_soa );

    // Initialize data in the user structure.
    auto user_data_mirror =
        Kokkos::create_mirror_view( Kokkos::HostSpace(), user_data );
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    for ( std::size_t idx = 0; idx != aosoa.size(); ++idx )
    {
        // Get aosoa indices.
        int s = idx / vector_length;
        int a = idx % vector_length;

        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    user_data_mirror( s ).m0[i][j][k][a] = fval * ( i + j + k );

        // Member 1.
        user_data_mirror( s ).m1[a] = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            user_data_mirror( s ).m2[i][a] = dval * i;

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                user_data_mirror( s ).m3[i][j][a] = dval * ( i + j );
    }
    Kokkos::deep_copy( user_data, user_data_mirror );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, aosoa_test ) { testAoSoA(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, aosoa_raw_data_test ) { testRawData(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, aosoa_tuple_test ) { testTuple(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, aosoa_access_test ) { testAccess(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, aosoa_unmanaged_test ) { testUnmanaged(); }

//---------------------------------------------------------------------------//

} // end namespace Test
