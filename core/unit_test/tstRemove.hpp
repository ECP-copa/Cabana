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
#include <Cabana_Remove.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{

void testRemove()
{
    int num_particle = 200;
    Cabana::AoSoA<Cabana::MemberTypes<int>, TEST_MEMSPACE> aosoa(
        "remove", num_particle );
    auto keep_slice = Cabana::slice<0>( aosoa, "slice" );

    // Remove nothing
    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_particle ),
        KOKKOS_LAMBDA( const int p ) { keep_slice( p ) = 1; } );

    Cabana::remove( TEST_EXECSPACE{}, num_particle, keep_slice, aosoa );
    EXPECT_EQ( aosoa.size(), num_particle );

    // Remove only odd particles.
    aosoa.resize( num_particle );
    keep_slice = Cabana::slice<0>( aosoa, "slice" );

    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_particle ),
        KOKKOS_LAMBDA( const int p ) {
            if ( p % 2 )
            {
                keep_slice( p ) = 1;
            }
            else
            {
                keep_slice( p ) = 0;
            }
        } );

    int new_num_particle = num_particle / 2;
    Cabana::remove( TEST_EXECSPACE{}, new_num_particle, keep_slice, aosoa );
    EXPECT_EQ( aosoa.size(), new_num_particle );

    auto aosoa_host =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto keep_slice_host = Cabana::slice<0>( aosoa_host, "host_slice" );
    for ( int p = 0; p < new_num_particle; ++p )
        EXPECT_EQ( keep_slice_host( p ), 1 );

    // Remove the rest.
    aosoa.resize( new_num_particle );
    keep_slice = Cabana::slice<0>( aosoa, "slice" );
    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, new_num_particle ),
        KOKKOS_LAMBDA( const int p ) { keep_slice( p ) = 0; } );

    Cabana::remove( TEST_EXECSPACE{}, 0, keep_slice, aosoa );
    EXPECT_EQ( aosoa.size(), 0 );
}

TEST( TEST_CATEGORY, remove_test ) { testRemove(); }

} // namespace Test
