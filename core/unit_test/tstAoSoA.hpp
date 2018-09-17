/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <impl/Cabana_Index.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// Check the data given a set of values in an aosoa.
template<class aosoa_type>
void checkDataMembers(
    aosoa_type aosoa,
    const float fval, const double dval, const int ival,
    const int dim_1, const int dim_2,
    const int dim_3, const int dim_4 )
{
    auto slice_0 = aosoa.template slice<0>();
    auto slice_1 = aosoa.template slice<1>();
    auto slice_2 = aosoa.template slice<2>();
    auto slice_3 = aosoa.template slice<3>();
    auto slice_4 = aosoa.template slice<4>();

    for ( std::size_t idx = 0; idx < aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    EXPECT_EQ( slice_0( idx, i, j, k ),
                                fval * (i+j+k) );

        // Member 1.
        EXPECT_EQ( slice_1( idx ), ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        EXPECT_EQ( slice_2( idx, i, j, k, l ),
                                    fval * (i+j+k+l) );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( slice_3( idx, i ), dval * i );

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( slice_4( idx, i, j ), dval * (i+j) );
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
    const int dim_4 = 3;

    // Declare data types.
    using DataTypes =
        Cabana::MemberTypes<float[dim_1][dim_2][dim_3],
                                int,
                                float[dim_1][dim_2][dim_3][dim_4],
                                double[dim_1],
                                double[dim_1][dim_2]
                                >;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE,vector_length>;

    // Make sure that it is actually an AoSoA.
    EXPECT_TRUE( Cabana::is_aosoa<AoSoA_t>::value );

    // Create an AoSoA.
    AoSoA_t aosoa;

    // Get field slices.
    auto slice_0 = aosoa.slice<0>();
    auto slice_1 = aosoa.slice<1>();
    auto slice_2 = aosoa.slice<2>();
    auto slice_3 = aosoa.slice<3>();
    auto slice_4 = aosoa.slice<4>();

    // Check sizes.
    EXPECT_EQ( aosoa.size(), int(0) );
    EXPECT_EQ( aosoa.capacity(), int(0) );
    EXPECT_EQ( aosoa.numSoA(), int(0) );

    // Resize
    int num_data = 35;
    aosoa.resize( num_data );

    // Check sizes for the new allocation/size.
    EXPECT_EQ( aosoa.size(), int(35) );
    EXPECT_EQ( aosoa.capacity(), int(48) );
    EXPECT_EQ( aosoa.numSoA(), int(3) );

    EXPECT_EQ( aosoa.arraySize(0), int(16) );
    EXPECT_EQ( aosoa.arraySize(1), int(16) );
    EXPECT_EQ( aosoa.arraySize(2), int(3) );

    // Test bounds.
    auto end = aosoa.size();
    int end_s = Cabana::Impl::Index<16>::s(end);
    int end_a = Cabana::Impl::Index<16>::a(end);
    EXPECT_EQ( end_s, 2 );
    EXPECT_EQ( end_a, 3 );

    // Get field slices again. We invalidated the pointers by resizing the
    // slices.
    slice_0 = aosoa.slice<0>();
    slice_1 = aosoa.slice<1>();
    slice_2 = aosoa.slice<2>();
    slice_3 = aosoa.slice<3>();
    slice_4 = aosoa.slice<4>();

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
                    slice_0( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        slice_1( idx ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        slice_2( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            slice_3( idx, i ) = dval * i;

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                slice_4( idx, i, j ) = dval * (i+j);
    }

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now extend the capacity of the container. First make the capacity
    // smaller - this wont actually do anything because we never decrease the
    // allocation of the container.
    aosoa.reserve( 1 );

    // Make sure nothing changed.
    EXPECT_EQ( aosoa.size(), int(35) );
    EXPECT_EQ( aosoa.capacity(), int(48) );
    EXPECT_EQ( aosoa.numSoA(), int(3) );
    EXPECT_EQ( aosoa.arraySize(0), int(16) );
    EXPECT_EQ( aosoa.arraySize(1), int(16) );
    EXPECT_EQ( aosoa.arraySize(2), int(3) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now reserve a bunch of space.
    aosoa.reserve( 1024 );

    // Make sure capacity changed but sizes and data did not.
    EXPECT_EQ( aosoa.size(), int(35) );
    EXPECT_EQ( aosoa.capacity(), int(1024) );
    EXPECT_EQ( aosoa.numSoA(), int(3) );
    EXPECT_EQ( aosoa.arraySize(0), int(16) );
    EXPECT_EQ( aosoa.arraySize(1), int(16) );
    EXPECT_EQ( aosoa.arraySize(2), int(3) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );

    // Now decrease the size of the container.
    aosoa.resize( 29 );

    // Make sure sizes and data changed but the capacity did not.
    EXPECT_EQ( aosoa.size(), int(29) );
    EXPECT_EQ( aosoa.capacity(), int(1024) );
    EXPECT_EQ( aosoa.numSoA(), int(2) );
    EXPECT_EQ( aosoa.arraySize(0), int(16) );
    EXPECT_EQ( aosoa.arraySize(1), int(13) );
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
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
        Cabana::MemberTypes<float,
                                int,
                                double[dim_1][dim_2],
                                int,
                                double>;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE,vector_length>;

    // Create an AoSoA using the default constructor.
    int num_data = 350;
    AoSoA_t aosoa( num_data );

    // Get slices of fields.
    auto slice_0 = aosoa.slice<0>();
    auto slice_1 = aosoa.slice<1>();
    auto slice_2 = aosoa.slice<2>();
    auto slice_3 = aosoa.slice<3>();
    auto slice_4 = aosoa.slice<4>();

    // Get raw pointers to the data as one would in a C interface (no templates).
    float* p0 = slice_0.data();
    int* p1 = slice_1.data();
    double* p2 = slice_2.data();
    int* p3 = slice_3.data();
    double* p4 = slice_4.data();

    // Get the strides between the member arrays.
    int st0 = slice_0.stride(0);
    int st1 = slice_1.stride(0);
    int st2 = slice_2.stride(0);
    int st3 = slice_3.stride(0);
    int st4 = slice_4.stride(0);

    // Member 2 is multidimensional so get its extents.
    int m2e0 = slice_2.extent(2);
    int m2e1 = slice_2.extent(3);
    EXPECT_EQ( m2e0, dim_1 );
    EXPECT_EQ( m2e1, dim_2 );

    // Initialize the data with raw pointer/stride access. Start by looping
    // over the structs. Each struct has a group of contiguous arrays of size
    // array_size for each member.
    int num_soa = slice_0.numSoA();
    for ( int s = 0; s < num_soa; ++s )
    {
        // Loop over the array in each struct and set the values.
        int local_array_size = slice_0.arraySize( s );
        for ( int i = 0; i < local_array_size; ++i )
        {
            p0[ s * st0 + i ] = (s + i) * 1.0;
            p1[ s * st1 + i ] = (s + i) * 2;
            p3[ s * st3 + i ] = (s + i) * 4;
            p4[ s * st4 + i ] = (s + i) * 5.0;

            // Member 2 has some extra dimensions so add those to the
            // indexing. Note this is layout left.
            for ( int j = 0; j < m2e0; ++j )
                for ( int k = 0; k < m2e1; ++k )
                    p2[ s * st2 + k * 16 * m2e0 + j * 16 + i ] =
                        (s + i + j + k) * 3.0;
        }
    }

    // Check the results.
    for ( std::size_t idx = 0; idx < aosoa.size(); ++idx )
    {
        int s = Cabana::Impl::Index<16>::s( idx );
        int a = Cabana::Impl::Index<16>::a( idx );

        EXPECT_EQ( slice_0(idx), (s+a)*1.0 );
        EXPECT_EQ( slice_1(idx), int((s+a)*2) );
        EXPECT_EQ( slice_3(idx), int((s+a)*4) );
        EXPECT_EQ( slice_4(idx), (s+a)*5.0 );

        // Member 2 has some extra dimensions so check those too.
        for ( int j = 0; j < dim_1; ++j )
            for ( int k = 0; k < dim_2; ++k )
                EXPECT_EQ( slice_2(idx,j,k), (s+a+j+k)*3.0 );
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
    const int dim_4 = 3;

    // Declare member types.
    using T0 = float[dim_1][dim_2][dim_3];
    using T1 = int;
    using T2 = float[dim_1][dim_2][dim_3][dim_4];
    using T3 = double[dim_1];
    using T4 = double[dim_1][dim_2];

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<T0,T1,T2,T3,T4>;

    // Declare the tuple type.
    using Tuple_t = Cabana::Tuple<DataTypes>;

    // Create an AoSoA.
    int num_data = 453;
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    AoSoA_t aosoa( num_data );

    // Create a slice of tuples with the same data types.
    Kokkos::View<Tuple_t*,
                 typename AoSoA_t::memory_space::kokkos_memory_space>
        tuples( "tuples", num_data );

    // Initialize aosoa data.
    auto slice_0 = aosoa.slice<0>();
    auto slice_1 = aosoa.slice<1>();
    auto slice_2 = aosoa.slice<2>();
    auto slice_3 = aosoa.slice<3>();
    auto slice_4 = aosoa.slice<4>();
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    for ( std::size_t idx = 0; idx != aosoa.size(); ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    slice_0( idx, i, j, k ) = fval * (i+j+k);

        // Member 1.
        slice_1( idx ) = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        slice_2( idx, i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            slice_3( idx, i ) = dval * i;

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                slice_4( idx, i, j ) = dval * (i+j);
    }

    // Assign the AoSoA data to the tuples.
    for ( std::size_t idx = 0; idx < aosoa.size(); ++idx )
         tuples( idx ) = aosoa.getTuple( idx );

    // Change the tuple data.
    fval = 2.1;
    dval = 9.21;
    ival = 3;
    for ( int idx = 0; idx < num_data; ++idx )
    {
        // Member 0.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    tuples( idx ).get<0>( i, j, k ) = fval * (i+j+k);

        // Member 1.
        tuples( idx ).get<1>() = ival;

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                for ( int k = 0; k < dim_3; ++k )
                    for ( int l = 0; l < dim_4; ++l )
                        tuples( idx ).get<2>( i, j, k, l ) = fval * (i+j+k+l);

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            tuples( idx ).get<3>( i ) = dval * i;

        // Member 4.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                tuples( idx ).get<4>( i, j ) = dval * (i+j);
    }

    // Assign the tuple data back to the AoSoA.
    for ( std::size_t idx = 0; idx < aosoa.size(); ++idx )
        aosoa.setTuple( idx, tuples(idx) );

    // Check the results.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
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
    const int dim_4 = 3;

    // Declare data types.
    using DataTypes =
        Cabana::MemberTypes<float[dim_1][dim_2][dim_3],
                            int,
                            float[dim_1][dim_2][dim_3][dim_4],
                            double[dim_1],
                            double[dim_1][dim_2]
                            >;

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE,vector_length>;

    // Make sure that it is actually an AoSoA.
    EXPECT_TRUE( Cabana::is_aosoa<AoSoA_t>::value );

    // Create an AoSoA.
    int num_data = 453;
    AoSoA_t aosoa( num_data );

    // Initialize data with the SoA accessor
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    for ( std::size_t s = 0; s < aosoa.numSoA(); ++s )
    {
        auto& soa = aosoa.access( s );

        for ( int a = 0; a < aosoa.arraySize(s); ++a )
        {
            // Member 0.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    for ( int k = 0; k < dim_3; ++k )
                        soa.get<0>( a, i, j, k ) = fval * (i+j+k);

            // Member 1.
            soa.get<1>( a ) = ival;

            // Member 2.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    for ( int k = 0; k < dim_3; ++k )
                        for ( int l = 0; l < dim_4; ++l )
                            soa.get<2>( a, i, j, k, l ) = fval * (i+j+k+l);

            // Member 3.
            for ( int i = 0; i < dim_1; ++i )
                soa.get<3>( a, i ) = dval * i;

            // Member 4.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    soa.get<4>( a, i, j ) = dval * (i+j);
        }
    }

    // Check data members for proper initialization.
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3, dim_4 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_test )
{
    testAoSoA();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_raw_data_test )
{
    testRawData();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_tuple_test )
{
    testTuple();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, aosoa_access_test )
{
    testAccess();
}

//---------------------------------------------------------------------------//

} // end namespace Test
