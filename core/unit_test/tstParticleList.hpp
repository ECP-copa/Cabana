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

#include <Kokkos_Core.hpp>

#include <Cabana_Fields.hpp>
#include <Cabana_ParticleList.hpp>

#include <particle_list_unit_test.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, particle_test )
{
    Cabana::ParticleList<TEST_MEMSPACE, Cabana::Field::Position<3>, Foo,
                         CommRank, Bar>
        plist( "test_particles" );

    particleListTest( plist );
}

TEST( TEST_CATEGORY, particle_view_test )
{
    Cabana::ParticleList<TEST_MEMSPACE, Cabana::Field::Position<3>, Foo,
                         CommRank, Bar>
        plist( "test_particles" );

    particleViewTest( plist );
}

//---------------------------------------------------------------------------//

} // end namespace Test
