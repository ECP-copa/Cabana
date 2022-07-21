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
template <class AlgorithmTag>
void testArborXList()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    using device_type = TEST_MEMSPACE; // sigh...

    // Check CSR neighbor lists.
    {
        // Create the neighbor list.
        auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
            AlgorithmTag{}, position, 0, position.size(),
            test_data.test_radius );

        // Check the neighbor list.
        checkNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle, AlgorithmTag{} );
    }
    // Check CSR again, building with a large array allocation guess.
    {
        auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
            AlgorithmTag{}, position, 0, position.size(), test_data.test_radius,
            100 );
        checkNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle, AlgorithmTag{} );
    }
    // Check CSR again, building with a small array allocation guess.
    {
        auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
            AlgorithmTag{}, position, 0, position.size(), test_data.test_radius,
            2 );
        checkNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle, AlgorithmTag{} );
    }

    // Check 2D neighbor lists.
    {
        auto const nlist =
            Cabana::Experimental::make2DNeighborList<device_type>(
                AlgorithmTag{}, position, 0, position.size(),
                test_data.test_radius );
        checkNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle, AlgorithmTag{} );
    }
    // Check 2D again, building with a large array allocation guess.
    {
        auto const nlist =
            Cabana::Experimental::make2DNeighborList<device_type>(
                AlgorithmTag{}, position, 0, position.size(),
                test_data.test_radius, 100 );
        checkNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle, AlgorithmTag{} );
    }
    // Check 2D again, building with a small array allocation guess.
    {
        auto const nlist =
            Cabana::Experimental::make2DNeighborList<device_type>(
                AlgorithmTag{}, position, 0, position.size(),
                test_data.test_radius, 2 );
        checkNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle, AlgorithmTag{} );
    }
}

//---------------------------------------------------------------------------//
void testArborXListFullPartialRange()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
            Cabana::FullNeighborTag{}, position, 0, test_data.num_ignore,
            test_data.test_radius );

        // Check the neighbor list.
        checkFullNeighborListPartialRange( nlist, test_data.N2_list_copy,
                                           test_data.num_particle,
                                           test_data.num_ignore );
    }
    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist =
            Cabana::Experimental::make2DNeighborList<device_type>(
                Cabana::FullNeighborTag{}, position, 0, test_data.num_ignore,
                test_data.test_radius );

        // Check the neighbor list.
        checkFullNeighborListPartialRange( nlist, test_data.N2_list_copy,
                                           test_data.num_particle,
                                           test_data.num_ignore );
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

        checkNeighborParallelFor( nlist, test_data.N2_list_copy,
                                  test_data.num_particle );
    }
    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist =
            Cabana::Experimental::make2DNeighborList<device_type>(
                Cabana::FullNeighborTag{}, position, 0, position.size(),
                test_data.test_radius );

        checkNeighborParallelFor( nlist, test_data.N2_list_copy,
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

        checkNeighborParallelReduce( nlist, test_data.N2_list_copy,
                                     test_data.aosoa );
    }
    {
        // Create the neighbor list.
        using device_type = TEST_MEMSPACE; // sigh...
        auto const nlist =
            Cabana::Experimental::make2DNeighborList<device_type>(
                Cabana::FullNeighborTag{}, position, 0, position.size(),
                test_data.test_radius );

        checkNeighborParallelReduce( nlist, test_data.N2_list_copy,
                                     test_data.aosoa );
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, full_test ) { testArborXList<Cabana::FullNeighborTag>(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, half_test ) { testArborXList<Cabana::HalfNeighborTag>(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, full_range_test ) { testArborXListFullPartialRange(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, parallel_for_test ) { testNeighborArborXParallelFor(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, parallel_reduce_test )
{
    testNeighborArborXParallelReduce();
}
//---------------------------------------------------------------------------//

} // end namespace Test
