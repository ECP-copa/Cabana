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

#include <Cabana_Parallel.hpp>
#include <Cabana_ExecutionPolicy.hpp>
#include <Cabana_AoSoA.hpp>

#include <gtest/gtest.h>

namespace Test
{

//---------------------------------------------------------------------------//
// Check the data given a set of values.
template<class aosoa_type>
void checkDataMembers(
    aosoa_type aosoa, int begin, int end,
    const float fval, const double dval, const int ival,
    const int dim_1, const int dim_2, const int dim_3 )
{
    auto slice_0 = aosoa.template slice<0>();
    auto slice_1 = aosoa.template slice<1>();
    auto slice_2 = aosoa.template slice<2>();
    auto slice_3 = aosoa.template slice<3>();

    for ( std::size_t idx = begin; idx != end; ++idx )
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
// Assignment operator.
template<class AoSoA_t,
         class SliceType0,
         class SliceType1,
         class SliceType2,
         class SliceType3>
class AssignmentOp
{
  public:
    AssignmentOp( AoSoA_t aosoa,
                  float fval,
                  double dval,
                  int ival )
        : _aosoa( aosoa )
        , _slice_0( aosoa.template slice<0>() )
        , _slice_1( aosoa.template slice<1>() )
        , _slice_2( aosoa.template slice<2>() )
        , _slice_3( aosoa.template slice<3>() )
        , _fval( fval )
        , _dval( dval )
        , _ival( ival )
        , _dim_1( _slice_0.extent(2) )
        , _dim_2( _slice_0.extent(3) )
        , _dim_3( _slice_0.extent(4) )
    {}

    KOKKOS_INLINE_FUNCTION void operator()( const int idx ) const
    {
        // Member 0.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                for ( int k = 0; k < _dim_3; ++k )
                    _slice_0( idx, i, j, k ) = _fval * (i+j+k);

        // Member 1.
        _slice_1( idx ) = _ival;

        // Member 2.
        for ( int i = 0; i < _dim_1; ++i )
            _slice_2( idx, i ) = _dval * i;

        // Member 3.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                _slice_3( idx, i, j ) = _dval * (i+j);
    }

    KOKKOS_INLINE_FUNCTION void operator()( const int s, const int a ) const
    {
        // Member 0.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                for ( int k = 0; k < _dim_3; ++k )
                    _slice_0.access( s, a, i, j, k ) = _fval * (i+j+k);

        // Member 1.
        _slice_1.access( s, a ) = _ival;

        // Member 2.
        for ( int i = 0; i < _dim_1; ++i )
            _slice_2.access( s, a, i ) = _dval * i;

        // Member 3.
        for ( int i = 0; i < _dim_1; ++i )
            for ( int j = 0; j < _dim_2; ++j )
                _slice_3.access( s, a, i, j ) = _dval * (i+j);
    }

  private:

    AoSoA_t _aosoa;
    SliceType0 _slice_0;
    SliceType1 _slice_1;
    SliceType2 _slice_2;
    SliceType3 _slice_3;
    float _fval;
    double _dval;
    int _ival;
    int _dim_1;
    int _dim_2;
    int _dim_3;
};

//---------------------------------------------------------------------------//
// Parallel for test with linear indexing.
void runTest1d()
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

    // Declare the AoSoA type. Let the library pick an inner array size based
    // on the execution space.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 155;
    AoSoA_t aosoa( num_data );

    // Create a linear execution policy using the begin and end of the AoSoA.
    Cabana::RangePolicy1d<TEST_EXECSPACE> policy_1( 12, 135 );

    // Create a functor to operate on.
    using OpType = AssignmentOp<AoSoA_t,
                                decltype(aosoa.slice<0>()),
                                decltype(aosoa.slice<1>()),
                                decltype(aosoa.slice<2>()),
                                decltype(aosoa.slice<3>())>;
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    OpType func_1( aosoa, fval, dval, ival );

    // Loop in parallel.
    Cabana::parallel_for( policy_1, func_1, "1d_test_1" );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, 12, 135, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Change values and write a second functor.
    fval = 93.4;
    dval = 12.1;
    ival = 4;
    OpType func_2( aosoa, fval, dval, ival );

    // Create another range policy with the size constructor.
    Cabana::RangePolicy1d<TEST_EXECSPACE> policy_2( aosoa.size() );

    // Loop in parallel using 1D array parallelism.
    Cabana::parallel_for( policy_2, func_2, "1d_test_2" );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, 0, num_data, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Change values and write a third functor.
    fval = 7.7;
    dval = 3.2;
    ival = 9;
    OpType func_3( aosoa, fval, dval, ival );

    // Create another range policy with the container constructor.
    Cabana::RangePolicy1d<TEST_EXECSPACE> policy_3( aosoa );

    // Loop in parallel using 1D array parallelism.
    Cabana::parallel_for( policy_3, func_3, "1d_test_3" );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, 0, num_data, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// Parallel for test with vectorized indexing.
void runTest2d()
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

    // Declare the AoSoA type. Let the library pick an inner array size based
    // on the execution space.
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;

    // Create an AoSoA.
    int num_data = 155;
    AoSoA_t aosoa( num_data );

    // Create a vectorized execution policy using the begin and end of the
    // AoSoA.
    Cabana::RangePolicy2d<TEST_EXECSPACE,AoSoA_t::vector_length>
        policy_1( 12, 135 );

    // Create a functor to operate on.
    using OpType = AssignmentOp<AoSoA_t,
                                decltype(aosoa.slice<0>()),
                                decltype(aosoa.slice<1>()),
                                decltype(aosoa.slice<2>()),
                                decltype(aosoa.slice<3>())>;
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    OpType func_1( aosoa, fval, dval, ival );

    // Loop in parallel.
    Cabana::parallel_for( policy_1, func_1, "2d_test_1" );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, 12, 135, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Change values and write a second functor.
    fval = 93.4;
    dval = 12.1;
    ival = 4;
    OpType func_2( aosoa, fval, dval, ival );

    // Create another range policy with the size constructor.
    Cabana::RangePolicy2d<TEST_EXECSPACE,AoSoA_t::vector_length>
        policy_2( aosoa.size() );

    // Loop in parallel using 2D array parallelism.
    Cabana::parallel_for( policy_2, func_2, "2d_test_2" );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, 0, num_data, fval, dval, ival, dim_1, dim_2, dim_3 );

    // Change values and write a third functor.
    fval = 7.7;
    dval = 3.2;
    ival = 9;
    OpType func_3( aosoa, fval, dval, ival );

    // Create another range policy with the container constructor.
    Cabana::RangePolicy2d<TEST_EXECSPACE,AoSoA_t::vector_length>
        policy_3( aosoa );

    // Loop in parallel using 2D array parallelism.
    Cabana::parallel_for( policy_3, func_3, "2d_test_3" );

    // Check data members for proper initialization.
    checkDataMembers( aosoa, 0, num_data, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, parallel_for_test_1d )
{
    runTest1d();
}

TEST_F( TEST_CATEGORY, parallel_for_test_2d )
{
    runTest2d();
}

//---------------------------------------------------------------------------//

} // end namespace Test
