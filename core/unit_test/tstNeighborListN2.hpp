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
#include <Cabana_N2NeighborList.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

#include <neighbor_unit_test.hpp>

#include <gtest/gtest.h>

namespace Test
{

//---------------------------------------------------------------------------//
template <class AlgorithmTag, class LayoutTag, class BuildTag>
void testN2List()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    {
        Cabana::N2NeighborList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                               LayoutTag, BuildTag>
            nlist_full( position, 0, position.size(), test_data.test_radius );
        // Test default construction.
        Cabana::N2NeighborList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                               LayoutTag, BuildTag>
            nlist;

        nlist = nlist_full;

        checkNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle, Cabana::FullNeighborTag{} );

        // Test rebuild function with explict execution space.
        nlist.build( TEST_EXECSPACE{}, position, 0, position.size(),
                     test_data.test_radius );
        checkNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle, Cabana::FullNeighborTag{} );
    }
    // Check again, building with a large array allocation size
    {
        Cabana::N2NeighborList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                               LayoutTag, BuildTag>
            nlist_max( position, 0, position.size(), test_data.test_radius,
                       100 );
        checkNeighborList( nlist_max, test_data.N2_list_copy,
                           test_data.num_particle, Cabana::FullNeighborTag{} );
    }
    // Check again, building with a small array allocation size (refill)
    {
        Cabana::N2NeighborList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                               LayoutTag, BuildTag>
            nlist_max2( position, 0, position.size(), test_data.test_radius,
                        2 );
        checkNeighborList( nlist_max2, test_data.N2_list_copy,
                           test_data.num_particle, Cabana::FullNeighborTag{} );
    }
}

//---------------------------------------------------------------------------//
template <class LayoutTag, class BuildTag>
void testN2ListFullPartialRange()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    Cabana::N2NeighborList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                           BuildTag>
        nlist( position, 0, test_data.num_ignore, test_data.test_radius );

    // Check the neighbor list.
    checkFullNeighborListPartialRange( nlist, test_data.N2_list_copy,
                                       test_data.num_particle,
                                       test_data.num_ignore );
}

template <class LayoutTag>
void testN2NeighborParallelFor()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    using ListType =
        Cabana::N2NeighborList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                               LayoutTag, Cabana::SerialOpTag>;
    ListType nlist( position, 0, position.size(), test_data.test_radius );

    checkNeighborParallelFor( nlist, test_data.N2_list_copy,
                              test_data.num_particle );
}

template <class LayoutTag>
void testN2NeighborParallelReduce()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    using ListType =
        Cabana::N2NeighborList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                               LayoutTag, Cabana::SerialOpTag>;
    ListType nlist( position, 0, position.size(), test_data.test_radius );

    checkNeighborParallelReduce( nlist, test_data.N2_list_copy,
                                 test_data.aosoa );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, n2_list_full_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testN2List<Cabana::FullNeighborTag, Cabana::NeighborLayoutCSR,
               Cabana::SerialOpTag>();
#endif
    testN2List<Cabana::FullNeighborTag, Cabana::NeighborLayout2D,
               Cabana::SerialOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testN2List<Cabana::FullNeighborTag, Cabana::NeighborLayoutCSR,
               Cabana::TeamOpTag>();
#endif
    testN2List<Cabana::FullNeighborTag, Cabana::NeighborLayout2D,
               Cabana::TeamOpTag>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, n2_list_half_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testN2List<Cabana::FullNeighborTag, Cabana::NeighborLayoutCSR,
               Cabana::SerialOpTag>();
#endif
    testN2List<Cabana::FullNeighborTag, Cabana::NeighborLayout2D,
               Cabana::SerialOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testN2List<Cabana::FullNeighborTag, Cabana::NeighborLayoutCSR,
               Cabana::TeamOpTag>();
#endif
    testN2List<Cabana::FullNeighborTag, Cabana::NeighborLayout2D,
               Cabana::TeamOpTag>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, n2_list_full_range_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testN2ListFullPartialRange<Cabana::NeighborLayoutCSR,
                               Cabana::SerialOpTag>();
#endif
    testN2ListFullPartialRange<Cabana::NeighborLayout2D, Cabana::SerialOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testN2ListFullPartialRange<Cabana::NeighborLayoutCSR, Cabana::TeamOpTag>();
#endif
    testN2ListFullPartialRange<Cabana::NeighborLayout2D, Cabana::TeamOpTag>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, n2_parallel_for_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testN2NeighborParallelFor<Cabana::NeighborLayoutCSR>();
#endif
    testN2NeighborParallelFor<Cabana::NeighborLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, n2_parallel_reduce_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testN2NeighborParallelReduce<Cabana::NeighborLayoutCSR>();
#endif
    testN2NeighborParallelReduce<Cabana::NeighborLayout2D>();
}
//---------------------------------------------------------------------------//

} // namespace Test
