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
    // Purposely using zero-init here.
    Kokkos::View<int*, TEST_MEMSPACE> keep_particle( "keep", num_particle );

    auto keep_slice = Cabana::slice<0>( aosoa, "slice" );
    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_particle ),
        KOKKOS_LAMBDA( const int p ) {
            if ( p % 2 )
            {
                keep_slice( p ) = 1;
                keep_particle( p ) = 1;
            }
            else
            {
                keep_slice( p ) = 0;
            }
        } );

    // Remove only odd particles.
    int new_num_particle = num_particle / 2;
    Cabana::remove( TEST_EXECSPACE{}, new_num_particle, keep_particle, aosoa );
    EXPECT_EQ( aosoa.size(), new_num_particle );

    // Remove the rest.
    Kokkos::resize( keep_particle, new_num_particle );
    keep_slice = Cabana::slice<0>( aosoa, "slice" );
    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, new_num_particle ),
        KOKKOS_LAMBDA( const int p ) {
            keep_particle( p ) = 1;
            keep_slice( p ) = 0;
        } );

    Cabana::remove( TEST_EXECSPACE{}, 0, keep_particle, aosoa );
    EXPECT_EQ( aosoa.size(), 0 );
}

TEST( TEST_CATEGORY, remove_test ) { testRemove(); }

} // namespace Test
