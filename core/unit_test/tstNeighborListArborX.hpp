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

#include <Cabana_AoSoA.hpp>
#include <Cabana_Experimental_NeighborList.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

#include <neighbor_unit_test.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
void testArborXListFull()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    double test_radius = 2.32;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );
    auto position = Cabana::slice<0>( aosoa );

    // Create the neighbor list.
    using device_type = TEST_MEMSPACE; // sigh...
    auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::FullNeighborTag{}, position, 0, aosoa.size(), test_radius );

    // Check the neighbor list.
    checkFullNeighborList( nlist, position, test_radius );
}

//---------------------------------------------------------------------------//
void testArborXListHalf()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    double test_radius = 2.32;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );
    auto position = Cabana::slice<0>( aosoa );

    // Create the neighbor list.
    using device_type = TEST_MEMSPACE; // sigh...
    auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::HalfNeighborTag{}, position, 0, aosoa.size(), test_radius );

    // Check the neighbor list.
    checkHalfNeighborList( nlist, position, test_radius );
}

//---------------------------------------------------------------------------//
void testArborXListFullPartialRange()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    int num_ignore = 800;
    double test_radius = 2.32;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );
    auto position = Cabana::slice<0>( aosoa );

    // Create the neighbor list.
    using device_type = TEST_MEMSPACE; // sigh...
    auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::FullNeighborTag{}, position, 0, num_ignore, test_radius );

    // Check the neighbor list.
    checkFullNeighborListPartialRange( nlist, position, test_radius,
                                       num_ignore );
}

//---------------------------------------------------------------------------//
void testNeighborArborXParallelFor()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    double test_radius = 2.32;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );
    auto position = Cabana::slice<0>( aosoa );

    // Create the neighbor list.
    using device_type = TEST_MEMSPACE; // sigh...
    auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::FullNeighborTag{}, position, 0, aosoa.size(), test_radius );

    checkFirstNeighborParallelFor( nlist, position, test_radius );

    checkSecondNeighborParallelFor( nlist, position, test_radius );
}

//---------------------------------------------------------------------------//
void testNeighborArborXParallelReduce()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    double test_radius = 2.32;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );
    auto position = Cabana::slice<0>( aosoa );

    // Create the neighbor list.
    using device_type = TEST_MEMSPACE; // sigh...
    auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::FullNeighborTag{}, position, 0, aosoa.size(), test_radius );

    checkFirstNeighborParallelReduce( nlist, aosoa, test_radius );

    checkSecondNeighborParallelReduce( nlist, aosoa, test_radius );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_list_full_test ) { testArborXListFull(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_list_half_test ) { testArborXListHalf(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_list_full_range_test )
{
    testArborXListFullPartialRange();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, parallel_for_test ) { testNeighborArborXParallelFor(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, parallel_reduce_test )
{
    testNeighborArborXParallelReduce();
}
//---------------------------------------------------------------------------//

} // end namespace Test
