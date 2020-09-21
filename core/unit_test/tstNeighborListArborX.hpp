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
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    using device_type = TEST_MEMSPACE; // sigh...
    auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::FullNeighborTag{}, position, 0, position.size(),
        test_data.test_radius );

    // Check the neighbor list.
    checkFullNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle );
}

//---------------------------------------------------------------------------//
void testArborXListHalf()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
            Cabana::HalfNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );

        // Check the neighbor list.
        checkHalfNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist =
            Cabana::Experimental::make2DNeighborList<device_type>(
                Cabana::HalfNeighborTag{}, position, 0, position.size(),
                test_data.test_radius );

        // Check the neighbor list.
        checkHalfNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
}

//---------------------------------------------------------------------------//
void testArborXListFullPartialRange()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    int num_ignore = 800;
    auto position = Cabana::slice<0>( test_data.aosoa );

    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
            Cabana::FullNeighborTag{}, position, 0, num_ignore,
            test_data.test_radius );

        // Check the neighbor list.
        checkFullNeighborListPartialRange( nlist, test_data.N2_list_copy,
                                           test_data.num_particle, num_ignore );
    }
    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist =
            Cabana::Experimental::make2DNeighborList<device_type>(
                Cabana::FullNeighborTag{}, position, 0, num_ignore,
                test_data.test_radius );

        // Check the neighbor list.
        checkFullNeighborListPartialRange( nlist, test_data.N2_list_copy,
                                           test_data.num_particle, num_ignore );
    }
}

//---------------------------------------------------------------------------//
void testNeighborArborXParallelFor()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );

        checkFirstNeighborParallelFor( nlist, test_data.N2_list_copy,
                                       test_data.num_particle );

        checkSecondNeighborParallelFor( nlist, test_data.N2_list_copy,
                                        test_data.num_particle );
    }
    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist =
            Cabana::Experimental::make2DNeighborList<device_type>(
                Cabana::FullNeighborTag{}, position, 0, position.size(),
                test_data.test_radius );

        checkFirstNeighborParallelFor( nlist, test_data.N2_list_copy,
                                       test_data.num_particle );

        checkSecondNeighborParallelFor( nlist, test_data.N2_list_copy,
                                        test_data.num_particle );
    }
}

//---------------------------------------------------------------------------//
void testNeighborArborXParallelReduce()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );

        checkFirstNeighborParallelReduce( nlist, test_data.N2_list_copy,
                                          test_data.aosoa );

        checkSecondNeighborParallelReduce( nlist, test_data.N2_list_copy,
                                           test_data.aosoa );
    }
    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist =
            Cabana::Experimental::make2DNeighborList<device_type>(
                Cabana::FullNeighborTag{}, position, 0, position.size(),
                test_data.test_radius );

        checkFirstNeighborParallelReduce( nlist, test_data.N2_list_copy,
                                          test_data.aosoa );

        checkSecondNeighborParallelReduce( nlist, test_data.N2_list_copy,
                                           test_data.aosoa );
    }
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
