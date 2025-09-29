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
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>
#include <Cabana_VerletList.hpp>

#include <Kokkos_Core.hpp>

#include <neighbor_unit_test.hpp>

#include <gtest/gtest.h>

namespace Test
{

//---------------------------------------------------------------------------//
template <std::size_t Dim, class LayoutTag, class BuildTag>
void testVerletListFull()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                           BuildTag, Dim>
            nlist_full( position, 0, position.size(), test_data.test_radius,
                        test_data.cell_size_ratio, test_data.grid_min,
                        test_data.grid_max );
        // Test default construction.
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                           BuildTag, Dim>
            nlist;

        nlist = nlist_full;

        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );

        // Test rebuild function with explicit execution space.
        nlist.build( TEST_EXECSPACE{}, position, 0, position.size(),
                     test_data.test_radius, test_data.cell_size_ratio,
                     test_data.grid_min, test_data.grid_max );
        checkFullNeighborList( nlist, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check again, building with a large array allocation size
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                           BuildTag, Dim>
            nlist_max( position, 0, position.size(), test_data.test_radius,
                       test_data.cell_size_ratio, test_data.grid_min,
                       test_data.grid_max, 100 );
        checkFullNeighborList( nlist_max, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check again, building with a small array allocation size (refill)
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                           BuildTag, Dim>
            nlist_max2( position, 0, position.size(), test_data.test_radius,
                        test_data.cell_size_ratio, test_data.grid_min,
                        test_data.grid_max, 2 );
        checkFullNeighborList( nlist_max2, test_data.N2_list_copy,
                               test_data.num_particle );
    }
}

//---------------------------------------------------------------------------//
template <std::size_t Dim, class LayoutTag, class BuildTag>
void testVerletListHalf()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::HalfNeighborTag, LayoutTag,
                           BuildTag, Dim>
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
                           BuildTag, Dim>
            nlist_max( position, 0, position.size(), test_data.test_radius,
                       test_data.cell_size_ratio, test_data.grid_min,
                       test_data.grid_max, 100 );
        checkHalfNeighborList( nlist_max, test_data.N2_list_copy,
                               test_data.num_particle );
    }
    // Check again, building with a small array allocation size (refill)
    {
        Cabana::VerletList<TEST_MEMSPACE, Cabana::HalfNeighborTag, LayoutTag,
                           BuildTag, Dim>
            nlist_max2( position, 0, position.size(), test_data.test_radius,
                        test_data.cell_size_ratio, test_data.grid_min,
                        test_data.grid_max, 2 );
        checkHalfNeighborList( nlist_max2, test_data.N2_list_copy,
                               test_data.num_particle );
    }
}

//---------------------------------------------------------------------------//
template <std::size_t Dim, class LayoutTag, class BuildTag>
void testVerletListFullPartialRange()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag,
                       BuildTag, Dim>
        nlist( position, test_data.begin, test_data.end, test_data.test_radius,
               test_data.cell_size_ratio, test_data.grid_min,
               test_data.grid_max );

    // Check the neighbor list.
    checkFullNeighborListPartialRange( nlist, test_data.N2_list_copy,
                                       test_data.num_particle, test_data.begin,
                                       test_data.end );
}

//---------------------------------------------------------------------------//
template <std::size_t Dim, class LayoutTag>
void testNeighborParallelFor()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    auto nlist = Cabana::createVerletList<Cabana::FullNeighborTag, LayoutTag,
                                          Cabana::TeamOpTag>(
        position, 0, position.size(), test_data.test_radius,
        test_data.cell_size_ratio, test_data.grid_min, test_data.grid_max );

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

//---------------------------------------------------------------------------//
template <std::size_t Dim, class LayoutTag>
void testNeighborParallelReduce()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    using ListType = Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                                        LayoutTag, Cabana::TeamOpTag, Dim>;
    ListType nlist( position, 0, position.size(), test_data.test_radius,
                    test_data.cell_size_ratio, test_data.grid_min,
                    test_data.grid_max );

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

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testNonUniformRadius()
{
    // Create the AoSoA and fill custom particle details.
    std::size_t particle_x = 2;
    // Purposely choose radius to reach all particles.
    double large_radius = 4.05;
    // Purposely choose radius to reach nearest neighbors only.
    double small_radius = 3.32;
    // Create the AoSoA and fill with particles
    NeighborListTestDataOrdered test_data( particle_x );
    auto position = Cabana::slice<0>( test_data.aosoa );
    auto radii = Cabana::slice<1>( test_data.aosoa );

    int num_p = test_data.num_particle;
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_p );
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA( int pid ) {
            if ( pid == 0 || pid == num_p - 1 )
                radii( pid ) = large_radius;
            else
                radii( pid ) = small_radius;
        } );

    // Create the neighbor list.
    using ListType = Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                                        LayoutTag, Cabana::TeamOpTag>;
    ListType nlist( position, 0, position.size(), small_radius, radii,
                    test_data.cell_size_ratio, test_data.grid_min,
                    test_data.grid_max );

    // Allocate with known maximum neighbors.
    auto list_copy = copyListToHost( nlist, test_data.num_particle, 10 );
    // Check the results.
    for ( std::size_t p = 0; p < test_data.num_particle; ++p )
    {
        // Certain particles were manually given larger radius and
        // should therefore have more neighbors.
        if ( p == 0 || p == test_data.num_particle - 1 )
            EXPECT_EQ( list_copy.counts( p ), 6 );
        else
            EXPECT_EQ( list_copy.counts( p ), 4 );
    }
}

//---------------------------------------------------------------------------//
template <std::size_t Dim, class LayoutTag>
void testModifyNeighbors()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    using ListType = Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                                        LayoutTag, Cabana::TeamOpTag, Dim>;
    ListType nlist( position, 0, position.size(), test_data.test_radius,
                    test_data.cell_size_ratio, test_data.grid_min,
                    test_data.grid_max );

    int new_id = -1;
    auto serial_set_op = KOKKOS_LAMBDA( const int i )
    {
        for ( std::size_t n = 0;
              n < Cabana::NeighborList<ListType>::numNeighbor( nlist, i ); ++n )
            nlist.setNeighbor( i, n, new_id );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, position.size() );
    Kokkos::parallel_for( "test_modify_serial", policy, serial_set_op );
    Kokkos::fence();

    auto list_copy =
        copyListToHost( nlist, test_data.N2_list_copy.neighbors.extent( 0 ),
                        test_data.N2_list_copy.neighbors.extent( 1 ) );
    // Check the results.
    for ( std::size_t p = 0; p < test_data.num_particle; ++p )
    {
        for ( int n = 0; n < test_data.N2_list_copy.counts( p ); ++n )
            // Check that all neighbors were changed.
            for ( int n = 0; n < test_data.N2_list_copy.counts( p ); ++n )
                EXPECT_EQ( list_copy.neighbors( p, n ), new_id );
    }
}

//---------------------------------------------------------------------------//
template <std::size_t Dim, class LayoutTag>
void testNeighborView()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto slice = Cabana::slice<0>( test_data.aosoa );

    // Copy manually into a View.
    Kokkos::View<double**, TEST_MEMSPACE> view( "positions", slice.size(),
                                                Dim );
    auto view_copy = KOKKOS_LAMBDA( const int i )
    {
        for ( std::size_t d = 0; d < Dim; ++d )
            view( i, d ) = slice( i, d );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, slice.size() );
    Kokkos::parallel_for( "view_copy", policy, view_copy );
    Kokkos::fence();

    // Create the neighbor list with the View data.
    using ListType = Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                                        LayoutTag, Cabana::TeamOpTag, Dim>;
    ListType nlist( view, test_data.test_radius, test_data.cell_size_ratio,
                    test_data.grid_min, test_data.grid_max );
    nlist.build( TEST_EXECSPACE{}, view, 0, slice.size(), test_data.test_radius,
                 test_data.cell_size_ratio, test_data.grid_min,
                 test_data.grid_max );

    checkFullNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle );
}

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testNeighborHistogram()
{
    int particle_x = 10;
    // Create the AoSoA and fill with particles
    NeighborListTestDataOrdered test_data( particle_x );
    auto position = Cabana::slice<0>( test_data.aosoa );

    // Create the neighbor list.
    using ListType = Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                                        LayoutTag, Cabana::TeamOpTag>;
    ListType nlist( position, 0, position.size(), test_data.test_radius,
                    test_data.cell_size_ratio, test_data.grid_min,
                    test_data.grid_max );

    // Create the neighbor histogram
    {
        int num_bin = 10;
        auto nhist = neighborHistogram( TEST_EXECSPACE{}, position.size(),
                                        nlist, num_bin );
        auto host_histogram =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), nhist );

        // 122 is the number of neighbors expected for a full spherical shell
        // with characteristic ratio (cutoff / dx) = 3
        double bin_max[10] = { 12, 24, 36, 48, 61, 73, 85, 97, 109, 122 };
        // This is the direct copied output (zeros due to fixed spacing).
        double bin_count[10] = { 32, 72, 24, 152, 120, 168, 0, 216, 0, 152 };

        for ( int i = 0; i < num_bin; ++i )
        {
            EXPECT_EQ( bin_max[i], host_histogram( i, 0 ) );
            EXPECT_EQ( bin_count[i], host_histogram( i, 1 ) );
        }
    }

    // Create another histogram with fewer bins.
    {
        int num_bin = 5;
        auto nhist = neighborHistogram( TEST_EXECSPACE{}, position.size(),
                                        nlist, num_bin );
        auto host_histogram =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), nhist );

        double bin_max[5] = { 24, 48, 73, 97, 122 };
        // This is the direct copied output.
        double bin_count[5] = { 104, 176, 288, 216, 152 };
        for ( int i = 0; i < num_bin; ++i )
        {
            EXPECT_EQ( bin_max[i], host_histogram( i, 0 ) );
            EXPECT_EQ( bin_count[i], host_histogram( i, 1 ) );
        }
    }
}

template <std::size_t Dim, class ArrayType>
void testVerletListArrays( ArrayType grid_min, ArrayType grid_max )
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData<Dim> test_data;
    auto position = Cabana::slice<0>( test_data.aosoa );
    Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                       Cabana::VerletLayout2D, Cabana::TeamOpTag, Dim>
        nlist( position, 0, position.size(), test_data.test_radius,
               test_data.cell_size_ratio, grid_min, grid_max );

    checkFullNeighborList( nlist, test_data.N2_list_copy,
                           test_data.num_particle );
}

//---------------------------------------------------------------------------//
// 3D TESTS
//---------------------------------------------------------------------------//
TEST( VerletList, Full3d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFull<3, Cabana::VerletLayoutCSR, Cabana::TeamOpTag>();
#endif
    testVerletListFull<3, Cabana::VerletLayout2D, Cabana::TeamOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFull<3, Cabana::VerletLayoutCSR, Cabana::TeamVectorOpTag>();
#endif
    testVerletListFull<3, Cabana::VerletLayout2D, Cabana::TeamVectorOpTag>();
}

TEST( VerletList, Arrays3d )
{
    NeighborListTestData<3> test_data;
    double min = test_data.box_min;
    double max = test_data.box_max;
    {
        double grid_min[3] = { min, min, min };
        double grid_max[3] = { max, max, max };
        testVerletListArrays<3>( grid_min, grid_max );
    }
    {
        // With the current variadic template for the Kokkos::Array (deprecated
        // in 4.4, but breaks compile without deprecated code enabled), nvcc
        // cannot compile with std::array
#if !defined( KOKKOS_ENABLE_CUDA ) &&                                          \
    !defined( KOKKOS_ENABLE_DEPRECATED_CODE_4 )
        std::array<double, 3> grid_min = { min, min, min };
        std::array<double, 3> grid_max = { max, max, max };
        testVerletListArrays<3>( grid_min, grid_max );
#endif
    }
}

//---------------------------------------------------------------------------//
TEST( VerletList, Half3d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListHalf<3, Cabana::VerletLayoutCSR, Cabana::TeamOpTag>();
#endif
    testVerletListHalf<3, Cabana::VerletLayout2D, Cabana::TeamOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListHalf<3, Cabana::VerletLayoutCSR, Cabana::TeamVectorOpTag>();
#endif
    testVerletListHalf<3, Cabana::VerletLayout2D, Cabana::TeamVectorOpTag>();
}

//---------------------------------------------------------------------------//
TEST( VerletList, FullRange3d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFullPartialRange<3, Cabana::VerletLayoutCSR,
                                   Cabana::TeamOpTag>();
#endif
    testVerletListFullPartialRange<3, Cabana::VerletLayout2D,
                                   Cabana::TeamOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFullPartialRange<3, Cabana::VerletLayoutCSR,
                                   Cabana::TeamVectorOpTag>();
#endif
    testVerletListFullPartialRange<3, Cabana::VerletLayout2D,
                                   Cabana::TeamVectorOpTag>();
}

//---------------------------------------------------------------------------//
TEST( VerletList, ParallelFor3d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testNeighborParallelFor<3, Cabana::VerletLayoutCSR>();
#endif
    testNeighborParallelFor<3, Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( VerletList, ParallelReduce3d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testNeighborParallelReduce<3, Cabana::VerletLayoutCSR>();
#endif
    testNeighborParallelReduce<3, Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( VerletList, ModifyList3d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testModifyNeighbors<3, Cabana::VerletLayoutCSR>();
#endif
    testModifyNeighbors<3, Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( VerletList, NeighborHistogram3d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET
    testNeighborHistogram<Cabana::VerletLayoutCSR>();
#endif
    testNeighborHistogram<Cabana::VerletLayout2D>();
}

TEST( VerletList, non_uniform_radius_test )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET
    testNonUniformRadius<Cabana::VerletLayoutCSR>();
#endif
    testNonUniformRadius<Cabana::VerletLayout2D>();
}

TEST( VerletList, View3d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testNeighborView<3, Cabana::VerletLayoutCSR>();
#endif
    testNeighborView<3, Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
// 2D TESTS
//---------------------------------------------------------------------------//

TEST( VerletList, Full2d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFull<2, Cabana::VerletLayoutCSR, Cabana::TeamOpTag>();
#endif
    testVerletListFull<2, Cabana::VerletLayout2D, Cabana::TeamOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFull<2, Cabana::VerletLayoutCSR, Cabana::TeamVectorOpTag>();
#endif
    testVerletListFull<2, Cabana::VerletLayout2D, Cabana::TeamVectorOpTag>();
}

TEST( VerletList, Arrays2d )
{
    NeighborListTestData<2> test_data;
    double min = test_data.box_min;
    double max = test_data.box_max;
    {
        double grid_min[2] = { min, min };
        double grid_max[2] = { max, max };
        testVerletListArrays<2>( grid_min, grid_max );
    }
    {
        // With the current variadic template for the Kokkos::Array (deprecated
        // in 4.4, but breaks compile without deprecated code enabled), nvcc
        // cannot compile with std::array
#if !defined( KOKKOS_ENABLE_CUDA ) &&                                          \
    !defined( KOKKOS_ENABLE_DEPRECATED_CODE_4 )
        std::array<double, 2> grid_min = { min, min };
        std::array<double, 2> grid_max = { max, max };
        testVerletListArrays<2>( grid_min, grid_max );
#endif
    }
}

//---------------------------------------------------------------------------//

TEST( VerletList, Half2d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListHalf<2, Cabana::VerletLayoutCSR, Cabana::TeamOpTag>();
#endif
    testVerletListHalf<2, Cabana::VerletLayout2D, Cabana::TeamOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListHalf<2, Cabana::VerletLayoutCSR, Cabana::TeamVectorOpTag>();
#endif
    testVerletListHalf<2, Cabana::VerletLayout2D, Cabana::TeamVectorOpTag>();
}

//---------------------------------------------------------------------------//
TEST( VerletList, FullRange2d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFullPartialRange<2, Cabana::VerletLayoutCSR,
                                   Cabana::TeamOpTag>();
#endif
    testVerletListFullPartialRange<2, Cabana::VerletLayout2D,
                                   Cabana::TeamOpTag>();

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testVerletListFullPartialRange<2, Cabana::VerletLayoutCSR,
                                   Cabana::TeamVectorOpTag>();
#endif
    testVerletListFullPartialRange<2, Cabana::VerletLayout2D,
                                   Cabana::TeamVectorOpTag>();
}

//---------------------------------------------------------------------------//
TEST( VerletList, Parallel2d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testNeighborParallelFor<2, Cabana::VerletLayoutCSR>();
#endif
    testNeighborParallelFor<2, Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( VerletList, Reduce2d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testNeighborParallelReduce<2, Cabana::VerletLayoutCSR>();
#endif
    testNeighborParallelReduce<2, Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( VerletList, Modify2d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testModifyNeighbors<2, Cabana::VerletLayoutCSR>();
#endif
    testModifyNeighbors<2, Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( VerletList, View2d )
{
#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    testNeighborView<2, Cabana::VerletLayoutCSR>();
#endif
    testNeighborView<2, Cabana::VerletLayout2D>();
}
//---------------------------------------------------------------------------//

} // end namespace Test
