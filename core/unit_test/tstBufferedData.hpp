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
#include <Cabana_BufferedAoSoA.hpp>
#include <Cabana_BufferedFor.hpp>
#include <Cabana_Types.hpp>

#include <gtest/gtest.h>

namespace Test
{

class TestTag
{
};

// TODO: dry with tstAoSoA
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
                    EXPECT_EQ( slice_0( idx, i, j, k ), fval * ( i + j + k ) );

        // Member 1.
        EXPECT_EQ( slice_1( idx ), ival );

        // Member 2.
        for ( int i = 0; i < dim_1; ++i )
            EXPECT_EQ( slice_2( idx, i ), dval * i );

        // Member 3.
        for ( int i = 0; i < dim_1; ++i )
            for ( int j = 0; j < dim_2; ++j )
                EXPECT_EQ( slice_3( idx, i, j ), dval * ( i + j ) );
    }
}

template <class buf_t>
class Tagfunctor_op
{
  public:
    // TODO: this needs a constructor and to actually use the data
    KOKKOS_INLINE_FUNCTION void operator()( const TestTag &,
                                            buf_t buffered_aosoa, const int s,
                                            const int a ) const
    {
    }
    KOKKOS_INLINE_FUNCTION void operator()(
        // const TestTag &,
        buf_t buffered_aosoa, const int s, const int a ) const
    {
    }
};

void testBufferedTag()
{
    std::cout << "Testing buffered tag" << std::endl;
    using DataTypes = Cabana::MemberTypes<float>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;

    // Cabana::simd_parallel_for( policy_1, func_1, "2d_test_1" );

    int num_data = 100;
    AoSoA_t aosoa( "test tag aosoa", num_data );

    const int buffer_count = 3;
    const int max_buffered_tuples = 40;

    using buf_t = Cabana::BufferedAoSoA<buffer_count, TEST_EXECSPACE, AoSoA_t>;
    buf_t buffered_aosoa_in( aosoa, max_buffered_tuples );

    Tagfunctor_op<buf_t> func_1;

    Cabana::buffered_parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE, TestTag>( 0, aosoa.size() ),
        buffered_aosoa_in, func_1, "test buffered for tag" );
}

void testBufferedDataCreation()
{
    // Create an AoSoA
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
    std::string label = "sample_aosoa";
    int num_data = 512;
    AoSoA_t aosoa( label, num_data );

    // Start by only buffering over one AoSoA at a time for stress test
    const int max_buffered_tuples = 8 * vector_length;

    // emulate a minimum of triple buffering?
    const int buffer_count = 3;

    // Hard code into OpenMP space for now
    // TODO: specify the exec space via a test param
    using target_exec_space = TEST_EXECSPACE;

    // Init the AoSoA data
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
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );

    using buf_t = Cabana::BufferedAoSoA<buffer_count, target_exec_space, AoSoA_t>;

    buf_t buffered_aosoa_in( aosoa, max_buffered_tuples );

    // Reset values so the outcome differs
    fval = 3.4;
    dval = 1.23;
    ival = 1;

    Cabana::buffered_parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, aosoa.size() ),
        buffered_aosoa_in,
        KOKKOS_LAMBDA( buf_t buffered_aosoa, const int s, const int a ) {
            // We have to call access and slice in the loop

            // We have to be really careful about how this access is
            // captured in the loop on GPU, and follow how ScatterView does
            // it safely. The `buffered_aosoa` may get captured by
            // reference, and then not be valid in a GPU context
            // auto buffered_access = buffered_aosoa.access();
            // auto buffered_access = buffered_aosoa.access();

            const auto slice_0 = buffered_aosoa.get_slice<0>();
            const auto slice_1 = buffered_aosoa.get_slice<1>();
            const auto slice_2 = buffered_aosoa.get_slice<2>();
            const auto slice_3 = buffered_aosoa.get_slice<3>();

            // Member 0.
            for ( int i = 0; i < dim_1; ++i )
            {
                for ( int j = 0; j < dim_2; ++j )
                {
                    for ( int k = 0; k < dim_3; ++k )
                    {
                        slice_0.access( s, a, i, j, k ) = fval * ( i + j + k );
                    }
                }
            }

            // Member 1.
            slice_1.access( s, a ) = ival;

            // Member 2.
            for ( int i = 0; i < dim_1; ++i )
            {
                slice_2.access( s, a, i ) = dval * i;
            }

            // Member 3.
            for ( int i = 0; i < dim_1; ++i )
            {
                for ( int j = 0; j < dim_2; ++j )
                {
                    slice_3.access( s, a, i, j ) = dval * ( i + j );
                }
            }
        },
        "test buffered for" );

    // TODO: test the data values
    checkDataMembers( aosoa, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, bufferedData_test ) { testBufferedDataCreation(); }
TEST( TEST_CATEGORY, bufferedData_tag_test ) { testBufferedTag(); }

} // namespace Test
