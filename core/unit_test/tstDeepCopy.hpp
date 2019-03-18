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

#include <Cabana_DeepCopy.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Types.hpp>

#include <gtest/gtest.h>

namespace Test
{

//---------------------------------------------------------------------------//
// Check the data given a set of values.
template<class aosoa_type>
void checkDataMembers(
    aosoa_type aosoa,
    const float fval, const double dval, const int ival,
    const int dim_1, const int dim_2, const int dim_3 )
{
    auto slice_0 = aosoa.template slice<0>();
    auto slice_1 = aosoa.template slice<1>();
    auto slice_2 = aosoa.template slice<2>();
    auto slice_3 = aosoa.template slice<3>();

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
            EXPECT_EQ( slice_2( idx, i ), dval * i );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( slice_3( idx, i, j ), dval * (i+j) );
    }
}

//---------------------------------------------------------------------------//
// Perform a deep copy test.
template<class DstMemorySpace, class SrcMemorySpace,
         int DstVectorLength, int SrcVectorLength>
void testDeepCopy()
{
    // Data dimensions.
    const int dim_1 = 3;
    const int dim_2 = 2;
    const int dim_3 = 4;

    // Declare data types.
    using DataTypes =
        Cabana::MemberTypes<float[dim_1][dim_2][dim_3],
                            int,
                            double[dim_1],
                            double[dim_1][dim_2]
                            >;

    // Declare the AoSoA types.
    using DstAoSoA_t = Cabana::AoSoA<DataTypes,DstMemorySpace,DstVectorLength>;
    using SrcAoSoA_t = Cabana::AoSoA<DataTypes,SrcMemorySpace,SrcVectorLength>;

    // Create AoSoAs.
    int num_data = 357;
    DstAoSoA_t dst_aosoa( num_data );
    SrcAoSoA_t src_aosoa( num_data );

    // Initialize data with the rank accessors.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    auto slice_0 = src_aosoa.template slice<0>();
    auto slice_1 = src_aosoa.template slice<1>();
    auto slice_2 = src_aosoa.template slice<2>();
    auto slice_3 = src_aosoa.template slice<3>();
    for ( std::size_t idx = 0; idx < src_aosoa.size(); ++idx )
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
            slice_2( idx, i ) = dval * i;

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                slice_3( idx, i, j ) = dval * (i+j);
    }

    // Deep copy
    Cabana::deep_copy( dst_aosoa, src_aosoa );

    // Check values.
    checkDataMembers( dst_aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );
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
    using DataTypes =
        Cabana::MemberTypes<float[dim_1][dim_2][dim_3],
                            int,
                            double[dim_1],
                            double[dim_1][dim_2]
                            >;

    // Create an AoSoA in the test memory space.
    int num_data = 423;
    Cabana::AoSoA<DataTypes,TEST_MEMSPACE> aosoa( num_data );

    // Initialize data.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    auto slice_0 = aosoa.template slice<0>();
    auto slice_1 = aosoa.template slice<1>();
    auto slice_2 = aosoa.template slice<2>();
    auto slice_3 = aosoa.template slice<3>();
    for ( std::size_t idx = 0; idx < aosoa.size(); ++idx )
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
            slice_2( idx, i ) = dval * i;

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                slice_3( idx, i, j ) = dval * (i+j);
    }

    // Create a mirror with the same memory space and copy.
    auto same_space_copy = Cabana::Experimental::create_mirror_view_and_copy(
        TEST_MEMSPACE(), aosoa );
    EXPECT_EQ( same_space_copy.size(), aosoa.size() );
    bool ssc_same_ms =
        std::is_same<TEST_MEMSPACE,
                     decltype(same_space_copy)::memory_space>::value;
    EXPECT_TRUE( ssc_same_ms );
    bool ssc_same_mt =
        std::is_same<DataTypes,
                     decltype(same_space_copy)::member_types>::value;
    EXPECT_TRUE( ssc_same_mt );

    // Check that the same memory space case didn't allocate any memory. They
    // should have the same pointer.
    EXPECT_EQ( aosoa.ptr(), same_space_copy.ptr() );

    // Check values.
    checkDataMembers( same_space_copy, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Create a mirror with the host space and copy.
    auto host_space_copy = Cabana::Experimental::create_mirror_view_and_copy(
        Kokkos::HostSpace(), aosoa );
    EXPECT_EQ( host_space_copy.size(), aosoa.size() );
    bool hsc_same_ms =
        std::is_same<Kokkos::HostSpace,
                     decltype(host_space_copy)::memory_space>::value;
    EXPECT_TRUE( hsc_same_ms );
    bool hsc_same_mt =
        std::is_same<DataTypes,
                     decltype(host_space_copy)::member_types>::value;
    EXPECT_TRUE( hsc_same_mt );

    // Check values.
    checkDataMembers( host_space_copy, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, deep_copy_to_host_same_layout_test )
{
    testDeepCopy<Cabana::HostSpace,TEST_MEMSPACE,16,16>();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, deep_copy_from_host_same_layout_test )
{
    testDeepCopy<TEST_MEMSPACE,Cabana::HostSpace,16,16>();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, deep_copy_to_host_different_layout_test )
{
    testDeepCopy<Cabana::HostSpace,TEST_MEMSPACE,16,32>();
    testDeepCopy<Cabana::HostSpace,TEST_MEMSPACE,64,8>();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, deep_copy_from_host_different_layout_test )
{
    testDeepCopy<TEST_MEMSPACE,Cabana::HostSpace,64,8>();
    testDeepCopy<TEST_MEMSPACE,Cabana::HostSpace,16,32>();
}

//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, mirror_test )
{
    testMirror();
}

//---------------------------------------------------------------------------//

} // end namespace Test
