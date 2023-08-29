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
#include <Cabana_DeepCopy.hpp>
#include <Cabana_Types.hpp>

#include <gtest/gtest.h>

namespace Test
{

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
// Perform a deep copy test.
template <class DstMemorySpace, class SrcMemorySpace, int DstVectorLength,
          int SrcVectorLength>
void testDeepCopy()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          double[dim_1], double[dim_1][dim_2]>;

    // Declare the AoSoA types.
    using DstAoSoA_t =
        Cabana::AoSoA<DataTypes, DstMemorySpace, DstVectorLength>;
    using SrcAoSoA_t =
        Cabana::AoSoA<DataTypes, SrcMemorySpace, SrcVectorLength>;

    // Create AoSoAs.
    int num_data = 357;
    DstAoSoA_t dst_aosoa( "dst", num_data );
    SrcAoSoA_t src_aosoa( "src", num_data );

    // Initialize data with the rank accessors.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    auto slice_0 = Cabana::slice<0>( src_aosoa );
    auto slice_1 = Cabana::slice<1>( src_aosoa );
    auto slice_2 = Cabana::slice<2>( src_aosoa );
    auto slice_3 = Cabana::slice<3>( src_aosoa );
    Kokkos::parallel_for(
        "initialize",
        Kokkos::RangePolicy<typename SrcMemorySpace::execution_space>(
            0, num_data ),
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

    // Deep copy
    Cabana::deep_copy( dst_aosoa, src_aosoa );

    // Check values.
    checkDataMembers( dst_aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Create a second AoSoA and deep copy by slice.
    DstAoSoA_t dst_aosoa_2( "dst", num_data );
    auto dst_slice_0 = Cabana::slice<0>( dst_aosoa_2 );
    auto dst_slice_1 = Cabana::slice<1>( dst_aosoa_2 );
    auto dst_slice_2 = Cabana::slice<2>( dst_aosoa_2 );
    auto dst_slice_3 = Cabana::slice<3>( dst_aosoa_2 );
    Cabana::deep_copy( dst_slice_0, slice_0 );
    Cabana::deep_copy( dst_slice_1, slice_1 );
    Cabana::deep_copy( dst_slice_2, slice_2 );
    Cabana::deep_copy( dst_slice_3, slice_3 );

    // Check values.
    checkDataMembers( dst_aosoa_2, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// Perform a mirror test.
void testMirror()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[dim_1][dim_2][dim_3], int,
                                          double[dim_1], double[dim_1][dim_2]>;

    // Create an AoSoA in the test memory space.
    int num_data = 423;
    Cabana::AoSoA<DataTypes, TEST_MEMSPACE> aosoa( "label", num_data );

    // Initialize data.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    auto slice_2 = Cabana::slice<2>( aosoa );
    auto slice_3 = Cabana::slice<3>( aosoa );
    Kokkos::parallel_for(
        "initialize", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_data ),
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

    // Create a mirror with the same memory space and copy separately.
    auto same_space_mirror =
        Cabana::create_mirror_view( TEST_MEMSPACE(), aosoa );
    Cabana::deep_copy( same_space_mirror, aosoa );
    auto host_space_mirror =
        Cabana::create_mirror_view( Kokkos::HostSpace(), aosoa );
    Cabana::deep_copy( host_space_mirror, aosoa );

    // Create a mirror with the same memory space and copy at the same time.
    auto same_space_copy =
        Cabana::create_mirror_view_and_copy( TEST_MEMSPACE(), aosoa );
    auto host_space_copy =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );

    // Check the mirrors/copies.
    using SameSpaceMirror = decltype( same_space_mirror );
    using HostSpaceMirror = decltype( host_space_mirror );
    auto check_mirrors = [&]( SameSpaceMirror same_space_mirror,
                              HostSpaceMirror host_space_mirror )
    {
        static_assert(
            std::is_same<TEST_MEMSPACE,
                         decltype( same_space_mirror )::memory_space>::value,
            "expected same memory spaces" );
        static_assert(
            std::is_same<DataTypes,
                         decltype( same_space_mirror )::member_types>::value,
            "expected same data types" );

        static_assert(
            std::is_same<Kokkos::HostSpace,
                         decltype( host_space_mirror )::memory_space>::value,
            "expected same memory spaces" );
        static_assert(
            std::is_same<DataTypes,
                         decltype( host_space_mirror )::member_types>::value,
            "expected same data types" );

        // Check sizes.
        EXPECT_EQ( same_space_mirror.size(), aosoa.size() );
        EXPECT_EQ( host_space_mirror.size(), aosoa.size() );

        // Check that the same memory space case didn't allocate any
        // memory. They should have the same pointer.
        EXPECT_EQ( aosoa.data(), same_space_mirror.data() );

        // Check values.
        checkDataMembers( same_space_mirror, fval, dval, ival, dim_1, dim_2,
                          dim_3 );
        checkDataMembers( host_space_mirror, fval, dval, ival, dim_1, dim_2,
                          dim_3 );
    };

    check_mirrors( same_space_mirror, host_space_mirror );
    check_mirrors( same_space_copy, host_space_copy );
}

//---------------------------------------------------------------------------//
// Perform an assignment test.
void testAssign()
{
    // Declare data types.
    using DataTypes = Cabana::MemberTypes<float[2], int>;

    // Create an AoSoA in the test memory space.
    int num_data = 423;
    Cabana::AoSoA<DataTypes, TEST_MEMSPACE> aosoa( "label", num_data );

    // Assign every tuple in the AoSoA to the same value.
    float fval = 3.2;
    int ival = 1;
    Cabana::Tuple<DataTypes> tp;
    Cabana::get<0>( tp, 0 ) = fval;
    Cabana::get<0>( tp, 1 ) = fval;
    Cabana::get<1>( tp ) = ival;
    Cabana::deep_copy( aosoa, tp );

    // Check the assignment
    auto host_aosoa =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto host_slice_0 = Cabana::slice<0>( host_aosoa );
    auto host_slice_1 = Cabana::slice<1>( host_aosoa );
    for ( int n = 0; n < num_data; ++n )
    {
        EXPECT_EQ( host_slice_0( n, 0 ), fval );
        EXPECT_EQ( host_slice_0( n, 1 ), fval );
        EXPECT_EQ( host_slice_1( n ), ival );
    }

    // Assign every element in slices to the same value.
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    fval = 5.4;
    ival = 12;
    Cabana::deep_copy( slice_0, fval );
    Cabana::deep_copy( slice_1, ival );
    host_aosoa =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    host_slice_0 = Cabana::slice<0>( host_aosoa );
    host_slice_1 = Cabana::slice<1>( host_aosoa );
    for ( int n = 0; n < num_data; ++n )
    {
        EXPECT_EQ( host_slice_0( n, 0 ), fval );
        EXPECT_EQ( host_slice_0( n, 1 ), fval );
        EXPECT_EQ( host_slice_1( n ), ival );
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, deep_copy_to_host_same_layout_test )
{
    testDeepCopy<Kokkos::HostSpace, TEST_MEMSPACE, 16, 16>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, deep_copy_from_host_same_layout_test )
{
    testDeepCopy<TEST_MEMSPACE, Kokkos::HostSpace, 16, 16>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, deep_copy_to_host_different_layout_test )
{
    testDeepCopy<Kokkos::HostSpace, TEST_MEMSPACE, 16, 32>();
    testDeepCopy<Kokkos::HostSpace, TEST_MEMSPACE, 64, 8>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, deep_copy_from_host_different_layout_test )
{
    testDeepCopy<TEST_MEMSPACE, Kokkos::HostSpace, 64, 8>();
    testDeepCopy<TEST_MEMSPACE, Kokkos::HostSpace, 16, 32>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, mirror_test ) { testMirror(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, assign_test ) { testAssign(); }

//---------------------------------------------------------------------------//

} // end namespace Test
