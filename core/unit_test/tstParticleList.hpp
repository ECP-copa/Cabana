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
    auto fields = Cabana::ParticleTraits<Cabana::Field::Position<3>, Foo,
                                         CommRank, Bar>();
    auto plist =
        Cabana::createParticleList<TEST_MEMSPACE>( "test_particles", fields );

    particleListTest( plist );
}

TEST( TEST_CATEGORY, particle_view_test )
{
    auto fields = Cabana::ParticleTraits<Cabana::Field::Position<3>, Foo,
                                         CommRank, Bar>();
    auto plist =
        Cabana::createParticleList<TEST_MEMSPACE>( "test_particles", fields );

    particleViewTest( plist );
}

//---------------------------------------------------------------------------//

} // end namespace Test
