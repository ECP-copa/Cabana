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
#include <Cabana_Experimental_NeighborList.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

#include <neighbor_unit_test.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
template <std::size_t Dim>
void testArborXListFull()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Check CSR neighbor lists.
    {
        // Create the neighbor list.
        auto const nlist = Cabana::Experimental::makeNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );

        // Check the neighbor list.
        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check CSR again, building with a large array allocation guess.
    {
        auto const nlist = Cabana::Experimental::makeNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius, 100 );
        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check CSR again, building with a small array allocation guess.
    {
        auto const nlist = Cabana::Experimental::makeNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius, 2 );
        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }

    // Check 2D neighbor lists.
    {
        auto const nlist = Cabana::Experimental::make2DNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );
        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check 2D again, building with a large array allocation guess.
    {
        auto const nlist = Cabana::Experimental::make2DNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius, 100 );
        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check 2D again, building with a small array allocation guess.
    {
        auto const nlist = Cabana::Experimental::make2DNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius, 2 );
        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
}

//---------------------------------------------------------------------------//
template <std::size_t Dim>
void testArborXListHalf()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Check CSR neighbor lists.
    {
        // Create the neighbor list.
        auto const nlist = Cabana::Experimental::makeNeighborList(
            Cabana::HalfNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );

        // Check the neighbor list.
        checkHalfNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check CSR again, building with a large array allocation guess.
    {
        auto const nlist = Cabana::Experimental::makeNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius, 100 );
        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check CSR again, building with a small array allocation guess.
    {
        auto const nlist = Cabana::Experimental::makeNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius, 2 );
        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }

    // Check 2D neighbor lists.
    {
        auto const nlist = Cabana::Experimental::make2DNeighborList(
            Cabana::HalfNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );
        checkHalfNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check 2D again, building with a large array allocation guess.
    {
        auto const nlist = Cabana::Experimental::make2DNeighborList(
            Cabana::HalfNeighborTag{}, position, 0, position.size(),
            test_data.test_radius, 100 );
        checkHalfNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check 2D again, building with a small array allocation guess.
    {
        auto const nlist = Cabana::Experimental::make2DNeighborList(
            Cabana::HalfNeighborTag{}, position, 0, position.size(),
            test_data.test_radius, 2 );
        checkHalfNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
}

//---------------------------------------------------------------------------//
template <std::size_t Dim>
void testArborXListFullPartialRange()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    {
        // Create the neighbor list.
        auto const nlist = Cabana::Experimental::makeNeighborList(
            Cabana::FullNeighborTag{}, position, 0, test_data.end,
            test_data.test_radius );

        // Check the neighbor list.
        checkFullNeighborListPartialRange( nlist, test_data.N2_list_copy,
                                           test_data.num_particle, 0,
                                           test_data.end );
    }
    {
        // Create the neighbor list.
        auto const nlist = Cabana::Experimental::make2DNeighborList(
            Cabana::FullNeighborTag{}, position, 0, test_data.end,
            test_data.test_radius );

        // Check the neighbor list.
        checkFullNeighborListPartialRange( nlist, test_data.N2_list_copy,
                                           test_data.num_particle, 0,
                                           test_data.end );
    }
}

//---------------------------------------------------------------------------//
template <std::size_t Dim>
void testNeighborArborXParallelFor()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    {
        // Create the neighbor list.
        auto const nlist = Cabana::Experimental::makeNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );

        checkFirstNeighborParallelForLambda( nlist, test_data.N2_list_copy,
                                             test_data.num_particle );

        checkSecondNeighborParallelForLambda( nlist, test_data.N2_list_copy,
                                              test_data.num_particle );

        checkSplitFirstNeighborParallelFor( nlist, test_data.N2_list_copy,
                                            test_data.num_particle );

        checkFirstNeighborParallelForFunctor( nlist, test_data.N2_list_copy,
                                              test_data.num_particle, true );
        checkFirstNeighborParallelForFunctor( nlist, test_data.N2_list_copy,
                                              test_data.num_particle, false );

        checkSecondNeighborParallelForFunctor( nlist, test_data.N2_list_copy,
                                               test_data.num_particle, true );
        checkSecondNeighborParallelForFunctor( nlist, test_data.N2_list_copy,
                                               test_data.num_particle, false );
    }
    {
        // Create the neighbor list.
        auto const nlist = Cabana::Experimental::make2DNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );

        checkFirstNeighborParallelForLambda( nlist, test_data.N2_list_copy,
                                             test_data.num_particle );

        checkSecondNeighborParallelForLambda( nlist, test_data.N2_list_copy,
                                              test_data.num_particle );

        checkSplitFirstNeighborParallelFor( nlist, test_data.N2_list_copy,
                                            test_data.num_particle );

        checkFirstNeighborParallelForFunctor( nlist, test_data.N2_list_copy,
                                              test_data.num_particle, true );
        checkFirstNeighborParallelForFunctor( nlist, test_data.N2_list_copy,
                                              test_data.num_particle, false );

        checkSecondNeighborParallelForFunctor( nlist, test_data.N2_list_copy,
                                               test_data.num_particle, true );
        checkSecondNeighborParallelForFunctor( nlist, test_data.N2_list_copy,
                                               test_data.num_particle, false );
    }
}

//---------------------------------------------------------------------------//
template <std::size_t Dim>
void testNeighborArborXParallelReduce()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    {
        // Create the neighbor list.
        auto const nlist = Cabana::Experimental::makeNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );

        checkFirstNeighborParallelReduceLambda( nlist, test_data.N2_list_copy,
                                                test_data.aosoa );

        checkSecondNeighborParallelReduceLambda( nlist, test_data.N2_list_copy,
                                                 test_data.aosoa );

        checkFirstNeighborParallelReduceFunctor( nlist, test_data.N2_list_copy,
                                                 test_data.aosoa, true );
        checkFirstNeighborParallelReduceFunctor( nlist, test_data.N2_list_copy,
                                                 test_data.aosoa, false );

        checkSecondNeighborParallelReduceFunctor( nlist, test_data.N2_list_copy,
                                                  test_data.aosoa, true );
        checkSecondNeighborParallelReduceFunctor( nlist, test_data.N2_list_copy,
                                                  test_data.aosoa, false );
    }
    {
        // Create the neighbor list.
        auto const nlist = Cabana::Experimental::make2DNeighborList(
            Cabana::FullNeighborTag{}, position, 0, position.size(),
            test_data.test_radius );

        checkFirstNeighborParallelReduceLambda( nlist, test_data.N2_list_copy,
                                                test_data.aosoa );

        checkSecondNeighborParallelReduceLambda( nlist, test_data.N2_list_copy,
                                                 test_data.aosoa );

        checkFirstNeighborParallelReduceFunctor( nlist, test_data.N2_list_copy,
                                                 test_data.aosoa, true );
        checkFirstNeighborParallelReduceFunctor( nlist, test_data.N2_list_copy,
                                                 test_data.aosoa, false );

        checkSecondNeighborParallelReduceFunctor( nlist, test_data.N2_list_copy,
                                                  test_data.aosoa, true );
        checkSecondNeighborParallelReduceFunctor( nlist, test_data.N2_list_copy,
                                                  test_data.aosoa, false );
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( ArborXList, Full3d ) { testArborXListFull<3>(); }

//---------------------------------------------------------------------------//
TEST( ArborXList, Half3d ) { testArborXListHalf<3>(); }

//---------------------------------------------------------------------------//
TEST( ArborXList, FullRange3d ) { testArborXListFullPartialRange<3>(); }

//---------------------------------------------------------------------------//
TEST( ArborXList, ParallelFor3d ) { testNeighborArborXParallelFor<3>(); }

//---------------------------------------------------------------------------//
TEST( ArborXList, ParallelReduce3d ) { testNeighborArborXParallelReduce<3>(); }
//---------------------------------------------------------------------------//

} // end namespace Test
