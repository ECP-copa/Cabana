/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_AoSoA.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>
#include <Cabana_VerletList.hpp>

#include <Kokkos_Core.hpp>

#include <neighbor_unit_test.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// Linked cell list cell stencil test.
void testLinkedCellStencil()
{
    // Point in the middle
    {
        double min[3] = { 0.0, 0.0, 0.0 };
        double max[3] = { 10.0, 10.0, 10.0 };
        double radius = 1.0;
        double ratio = 1.0;
        Cabana::Impl::LinkedCellStencil<double> stencil( radius, ratio, min,
                                                         max );

        double xp = 4.5;
        double yp = 5.5;
        double zp = 3.5;
        int ic, jc, kc;
        stencil.grid.locatePoint( xp, yp, zp, ic, jc, kc );
        int cell = stencil.grid.cardinalCellIndex( ic, jc, kc );
        int imin, imax, jmin, jmax, kmin, kmax;
        stencil.getCells( cell, imin, imax, jmin, jmax, kmin, kmax );
        EXPECT_EQ( imin, 3 );
        EXPECT_EQ( imax, 6 );
        EXPECT_EQ( jmin, 4 );
        EXPECT_EQ( jmax, 7 );
        EXPECT_EQ( kmin, 2 );
        EXPECT_EQ( kmax, 5 );
    }

    // Point in the lower right corner
    {
        double min[3] = { 0.0, 0.0, 0.0 };
        double max[3] = { 10.0, 10.0, 10.0 };
        double radius = 1.0;
        double ratio = 1.0;
        Cabana::Impl::LinkedCellStencil<double> stencil( radius, ratio, min,
                                                         max );

        double xp = 0.5;
        double yp = 0.5;
        double zp = 0.5;
        int ic, jc, kc;
        stencil.grid.locatePoint( xp, yp, zp, ic, jc, kc );
        int cell = stencil.grid.cardinalCellIndex( ic, jc, kc );
        int imin, imax, jmin, jmax, kmin, kmax;
        stencil.getCells( cell, imin, imax, jmin, jmax, kmin, kmax );
        EXPECT_EQ( imin, 0 );
        EXPECT_EQ( imax, 2 );
        EXPECT_EQ( jmin, 0 );
        EXPECT_EQ( jmax, 2 );
        EXPECT_EQ( kmin, 0 );
        EXPECT_EQ( kmax, 2 );
    }

    // Point in the upper left corner
    {
        double min[3] = { 0.0, 0.0, 0.0 };
        double max[3] = { 10.0, 10.0, 10.0 };
        double radius = 1.0;
        double ratio = 1.0;
        Cabana::Impl::LinkedCellStencil<double> stencil( radius, ratio, min,
                                                         max );

        double xp = 9.5;
        double yp = 9.5;
        double zp = 9.5;
        int ic, jc, kc;
        stencil.grid.locatePoint( xp, yp, zp, ic, jc, kc );
        int cell = stencil.grid.cardinalCellIndex( ic, jc, kc );
        int imin, imax, jmin, jmax, kmin, kmax;
        stencil.getCells( cell, imin, imax, jmin, jmax, kmin, kmax );
        EXPECT_EQ( imin, 8 );
        EXPECT_EQ( imax, 10 );
        EXPECT_EQ( jmin, 8 );
        EXPECT_EQ( jmax, 10 );
        EXPECT_EQ( kmin, 8 );
        EXPECT_EQ( kmax, 10 );
    }
}

//---------------------------------------------------------------------------//
template <class LayoutTag, class BuildTag>
void testVerletListFull()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                           BuildTag>
            nlist_full( position, 0, position.size(), test_data.test_radius,
                        test_data.cell_size_ratio, test_data.grid_min,
                        test_data.grid_max );
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                           BuildTag>
            nlist;

        nlist = nlist_full;

        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check again, building with a large array allocation size
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                           BuildTag>
            nlist_max( position, 0, position.size(), test_data.test_radius,
                       test_data.cell_size_ratio, test_data.grid_min,
                       test_data.grid_max, 100 );
        checkFullNeighborList( nlist_max, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check again, building with a small array allocation size (refill)
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                           BuildTag>
            nlist_max2( position, 0, position.size(), test_data.test_radius,
                        test_data.cell_size_ratio, test_data.grid_min,
                        test_data.grid_max, 2 );
        checkFullNeighborList( nlist_max2, test_data.N2_list_copy,
                               test_data.num_particle );
    }
}

//---------------------------------------------------------------------------//
template <class LayoutTag, class BuildTag>
void testVerletListHalf()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::HalfNeighborTag, LayoutTag,
                           BuildTag>
            nlist( position, 0, position.size(), test_data.test_radius,
                   test_data.cell_size_ratio, test_data.grid_min,
                   test_data.grid_max );

        // Check the neighbor list.
        checkHalfNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check again, building with a large array allocation size
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::HalfNeighborTag, LayoutTag,
                           BuildTag>
            nlist_max( position, 0, position.size(), test_data.test_radius,
                       test_data.cell_size_ratio, test_data.grid_min,
                       test_data.grid_max, 100 );
        checkHalfNeighborList( nlist_max, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check again, building with a small array allocation size (refill)
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::HalfNeighborTag, LayoutTag,
                           BuildTag>
            nlist_max2( position, 0, position.size(), test_data.test_radius,
                        test_data.cell_size_ratio, test_data.grid_min,
                        test_data.grid_max, 2 );
        checkHalfNeighborList( nlist_max2, test_data.N2_list_copy,
                               test_data.num_particle );
    }
}

//---------------------------------------------------------------------------//
template <class LayoutTag, class BuildTag>
void testVerletListFullPartialRange()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    int num_ignore = 800;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                       BuildTag>
        nlist( position, 0, num_ignore, test_data.test_radius,
               test_data.cell_size_ratio, test_data.grid_min,
               test_data.grid_max );

    // Check the neighbor list.
    checkFullNeighborListPartialRange( nlist, test_data.N2_list_copy,
                                       test_data.num_particle, num_ignore );
}

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testNeighborParallelFor()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    using ListType = Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                                        LayoutTag, Cabana::TeamOpTag>;
    ListType nlist( position, 0, position.size(), test_data.test_radius,
                    test_data.cell_size_ratio, test_data.grid_min,
                    test_data.grid_max );

    checkFirstNeighborParallelFor( nlist, test_data.N2_list_copy,
                                   test_data.num_particle );

    checkSecondNeighborParallelFor( nlist, test_data.N2_list_copy,
                                    test_data.num_particle );
}

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testNeighborParallelReduce()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    using ListType = Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                                        LayoutTag, Cabana::TeamOpTag>;
    ListType nlist( position, 0, position.size(), test_data.test_radius,
                    test_data.cell_size_ratio, test_data.grid_min,
                    test_data.grid_max );

    checkFirstNeighborParallelReduce( nlist, test_data.N2_list_copy,
                                      test_data.aosoa );

    checkSecondNeighborParallelReduce( nlist, test_data.N2_list_copy,
                                       test_data.aosoa );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, linked_cell_stencil_test ) { testLinkedCellStencil(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_list_full_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFull<Cabana::VerletLayoutCSR, Cabana::TeamOpTag>();
#endif
    testVerletListFull<Cabana::VerletLayout2D, Cabana::TeamOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFull<Cabana::VerletLayoutCSR, Cabana::TeamVectorOpTag>();
#endif
    testVerletListFull<Cabana::VerletLayout2D, Cabana::TeamVectorOpTag>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_list_half_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListHalf<Cabana::VerletLayoutCSR, Cabana::TeamOpTag>();
#endif
    testVerletListHalf<Cabana::VerletLayout2D, Cabana::TeamOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListHalf<Cabana::VerletLayoutCSR, Cabana::TeamVectorOpTag>();
#endif
    testVerletListHalf<Cabana::VerletLayout2D, Cabana::TeamVectorOpTag>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_list_full_range_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFullPartialRange<Cabana::VerletLayoutCSR,
                                   Cabana::TeamOpTag>();
#endif
    testVerletListFullPartialRange<Cabana::VerletLayout2D, Cabana::TeamOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFullPartialRange<Cabana::VerletLayoutCSR,
                                   Cabana::TeamVectorOpTag>();
#endif
    testVerletListFullPartialRange<Cabana::VerletLayout2D,
                                   Cabana::TeamVectorOpTag>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, parallel_for_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testNeighborParallelFor<Cabana::VerletLayoutCSR>();
#endif
    testNeighborParallelFor<Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, parallel_reduce_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testNeighborParallelReduce<Cabana::VerletLayoutCSR>();
#endif
    testNeighborParallelReduce<Cabana::VerletLayout2D>();
}
//---------------------------------------------------------------------------//

} // end namespace Test
