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

#include <Cabana_Parallel.hpp>
#include <Cabana_Tuple.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{

//---------------------------------------------------------------------------//
// Check the data given a set of values.
template <class view_type>
void checkDataMembers( view_type view, const float fval, const double dval,
                       const int ival, const std::size_t dim_1,
                       const std::size_t dim_2, const std::size_t dim_3 )
{
    auto mirror_view =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view );
    for ( std::size_t idx = 0; idx < mirror_view.extent( 0 ); ++idx )
    {
        // Member 0.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    EXPECT_FLOAT_EQ(
                        Cabana::get<0>( mirror_view( idx ), i, j, k ),
                        fval * ( i + j + k ) );

        // Member 1.
        EXPECT_EQ( Cabana::get<1>( mirror_view( idx ) ), ival );

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            EXPECT_DOUBLE_EQ( Cabana::get<2>( mirror_view( idx ), i ),
                              dval * i );

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                EXPECT_DOUBLE_EQ( Cabana::get<3>( mirror_view( idx ), i, j ),
                                  dval * ( i + j ) );
    }
}

//---------------------------------------------------------------------------//
// Tuple test
void runTest()
{
    // Data dimensions.
    const std::size_t dim_1 = 3;
    const std::size_t dim_2 = 2;
    const std::size_t dim_3 = 4;

    // Declare member types.
    using T0 = float[dim_1][dim_2][dim_3];
    using T1 = int;
    using T2 = double[dim_1];
    using T3 = double[dim_1][dim_2];

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<T0, T1, T2, T3>;

    // Declare the tuple type.
    using Tuple_t = Cabana::Tuple<DataTypes>;

    // Create a view of tuples.
    std::size_t num_data = 453;
    Kokkos::View<Tuple_t*, TEST_MEMSPACE> tuples( "tuples", num_data );

    // Initialize data.
    float fval = 3.4;
    double dval = 1.23;
    int ival = 1;
    auto init_func = KOKKOS_LAMBDA( const std::size_t idx )
    {
        // Member 0.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                for ( std::size_t k = 0; k < dim_3; ++k )
                    Cabana::get<0>( tuples( idx ), i, j, k ) =
                        fval * ( i + j + k );

        // Member 1.
        Cabana::get<1>( tuples( idx ) ) = ival;

        // Member 2.
        for ( std::size_t i = 0; i < dim_1; ++i )
            Cabana::get<2>( tuples( idx ), i ) = dval * i;

        // Member 3.
        for ( std::size_t i = 0; i < dim_1; ++i )
            for ( std::size_t j = 0; j < dim_2; ++j )
                Cabana::get<3>( tuples( idx ), i, j ) = dval * ( i + j );
    };
    Kokkos::fence();

    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_data );

    Kokkos::parallel_for( policy, init_func );
    Kokkos::fence();

    // Check data members of the for proper initialization.
    checkDataMembers( tuples, fval, dval, ival, dim_1, dim_2, dim_3 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, tuple_test ) { runTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
