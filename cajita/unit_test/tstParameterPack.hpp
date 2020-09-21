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

#include <Cajita_ParameterPack.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void captureTest()
{
    // Make some Kokkos views.
    Kokkos::View<double[1], TEST_MEMSPACE> dbl_view( "dbl_view" );
    Kokkos::View<int[1][1], TEST_MEMSPACE> int_view( "int_view" );

    // Make a parameter pack so we can capture them as a group.
    auto pack = makeParameterPack( dbl_view, int_view );

    // Update the pack in a kernel
    Kokkos::parallel_for(
        "fill_pack", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) {
            auto dv = get<0>( pack );
            auto iv = get<1>( pack );

            dv( 0 ) = 3.14;
            iv( 0, 0 ) = 12;
        } );

    // Check the capture.
    auto dbl_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), dbl_view );
    auto int_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), int_view );

    EXPECT_EQ( dbl_host( 0 ), 3.14 );
    EXPECT_EQ( int_host( 0, 0 ), 12 );
}

//---------------------------------------------------------------------------//
void emptyTest() { std::ignore = makeParameterPack(); }

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, parameter_pack_capture ) { captureTest(); }

TEST( TEST_CATEGORY, parameter_pack_empty ) { emptyTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
