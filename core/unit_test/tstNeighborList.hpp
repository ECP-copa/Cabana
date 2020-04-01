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

#include <CabanaCore_config.hpp> // Cabana_ENABLE_ARBORX
#if defined( Cabana_ENABLE_ARBORX )
#include <Cabana_Experimental_NeighborList.hpp>
#endif

#include <Cabana_AoSoA.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>
#include <Cabana_VerletList.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#if defined( Cabana_ENABLE_ARBORX )
template <typename DeviceType, typename Tag>
Cabana::VerletList<DeviceType, Tag, Cabana::VerletLayoutCSR>
convert_crs_graph_to_verlet_list(
    Cabana::VerletLayoutCSR,
    Cabana::Experimental::CrsGraph<typename DeviceType::memory_space, Tag> const
        &crs_graph )
{
    int const n_rows = crs_graph.row_ptr.size() - 1;
    int const shift = crs_graph.shift;
    int const total = crs_graph.total;
    int const n_entries = crs_graph.col_ind.size();
    Kokkos::View<int *, DeviceType> counts(
        Kokkos::view_alloc( "verlet_list_counts" ), total );

    Kokkos::View<int *, DeviceType> offsets(
        Kokkos::view_alloc( "verlet_list_offsets" ), total );
    Kokkos::View<int *, DeviceType> neighbors(
        Kokkos::view_alloc( "verlet_list_neighbors",
                            Kokkos::WithoutInitializing ),
        n_entries );
    Kokkos::parallel_for(
        Kokkos::RangePolicy<typename DeviceType::execution_space>( 0, n_rows ),
        KOKKOS_LAMBDA( int i ) {
            auto const first = crs_graph.row_ptr( i );
            auto const last = crs_graph.row_ptr( i + 1 );
            offsets( shift + i ) = first;
            counts( shift + i ) = last - first;
            for ( int j = first; j < last; ++j )
            {
                neighbors( j ) = crs_graph.col_ind( j );
            }
        } );
    Cabana::VerletList<DeviceType, Tag, Cabana::VerletLayoutCSR> verlet_list;
    verlet_list._data = {counts, offsets, neighbors};
    return verlet_list;
}

template <typename DeviceType, typename Tag>
Cabana::VerletList<DeviceType, Tag, Cabana::VerletLayout2D>
convert_crs_graph_to_verlet_list(
    Cabana::VerletLayout2D,
    Cabana::Experimental::CrsGraph<typename DeviceType::memory_space, Tag> const
        &crs_graph )
{
    int const n_rows = crs_graph.row_ptr.size() - 1;
    int const shift = crs_graph.shift;
    int const total = crs_graph.total;
    Kokkos::View<int *, DeviceType> counts(
        Kokkos::view_alloc( "verlet_list_counts" ), total );
    Kokkos::parallel_for(
        Kokkos::RangePolicy<typename DeviceType::execution_space>( 0, n_rows ),
        KOKKOS_LAMBDA( int i ) {
            auto const first = crs_graph.row_ptr( i );
            auto const last = crs_graph.row_ptr( i + 1 );
            counts( shift + i ) = last - first;
        } );
    int const max_entries_per_row = ArborX::max( counts );
    Kokkos::View<int **, DeviceType> neighbors(
        Kokkos::view_alloc( "verlet_list_neighbors" ), total,
        max_entries_per_row );
    Kokkos::parallel_for(
        Kokkos::RangePolicy<typename DeviceType::execution_space>( 0, n_rows ),
        KOKKOS_LAMBDA( int i ) {
            for ( int j = 0; j < counts( shift + i ); ++j )
            {
                neighbors( shift + i, j ) =
                    crs_graph.col_ind( crs_graph.row_ptr( i ) + j );
            }
        } );
    Cabana::VerletList<DeviceType, Tag, Cabana::VerletLayout2D> verlet_list;
    verlet_list._data = {counts, neighbors};
    return verlet_list;
}

#endif

namespace Test
{
//---------------------------------------------------------------------------//
// Linked cell list cell stencil test.
void testLinkedCellStencil()
{
    // Point in the middle
    {
        double min[3] = {0.0, 0.0, 0.0};
        double max[3] = {10.0, 10.0, 10.0};
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
        double min[3] = {0.0, 0.0, 0.0};
        double max[3] = {10.0, 10.0, 10.0};
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
        double min[3] = {0.0, 0.0, 0.0};
        double max[3] = {10.0, 10.0, 10.0};
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
// List implementation.
template <class... Params>
struct TestNeighborList
{
    Kokkos::View<int *, Params...> counts;
    Kokkos::View<int **, Params...> neighbors;
};

template <class KokkosMemorySpace>
TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
createTestListHostCopy( const TestNeighborList<KokkosMemorySpace> &test_list )
{
    using data_layout = typename decltype( test_list.counts )::array_layout;
    TestNeighborList<data_layout, Kokkos::HostSpace> list_copy;
    Kokkos::resize( list_copy.counts, test_list.counts.extent( 0 ) );
    Kokkos::deep_copy( list_copy.counts, test_list.counts );
    Kokkos::resize( list_copy.neighbors, test_list.neighbors.extent( 0 ),
                    test_list.neighbors.extent( 1 ) );
    Kokkos::deep_copy( list_copy.neighbors, test_list.neighbors );
    return list_copy;
}

// Create a host copy of a list that implements the neighbor list interface.
template <class ListType>
TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
copyListToHost( const ListType &list, const int num_particle, const int max_n )
{
    TestNeighborList<TEST_MEMSPACE> list_copy;
    list_copy.counts =
        Kokkos::View<int *, TEST_MEMSPACE>( "counts", num_particle );
    list_copy.neighbors =
        Kokkos::View<int **, TEST_MEMSPACE>( "neighbors", num_particle, max_n );
    Kokkos::parallel_for(
        "copy list", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_particle ),
        KOKKOS_LAMBDA( const int p ) {
            list_copy.counts( p ) =
                Cabana::NeighborList<ListType>::numNeighbor( list, p );
            for ( int n = 0; n < list_copy.counts( p ); ++n )
                list_copy.neighbors( p, n ) =
                    Cabana::NeighborList<ListType>::getNeighbor( list, p, n );
        } );
    Kokkos::fence();
    return createTestListHostCopy( list_copy );
}

//---------------------------------------------------------------------------//
// Create particles.
Cabana::AoSoA<Cabana::MemberTypes<double[3]>, TEST_MEMSPACE>
createParticles( const int num_particle, const double box_min,
                 const double box_max )
{
    using DataTypes = Cabana::MemberTypes<double[3]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t aosoa( "aosoa", num_particle );

    auto position = Cabana::slice<0>( aosoa );
    using PoolType = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    using RandomType = Kokkos::Random_XorShift64<TEST_EXECSPACE>;
    PoolType pool( 342343901 );
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( int d = 0; d < 3; ++d )
            position( p, d ) =
                Kokkos::rand<RandomType, double>::draw( gen, box_min, box_max );
        pool.free_state( gen );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> exec_policy( 0, num_particle );
    Kokkos::parallel_for( exec_policy, random_coord_op );
    Kokkos::fence();
    return aosoa;
}

//---------------------------------------------------------------------------//
template <class PositionSlice>
TestNeighborList<TEST_MEMSPACE>
computeFullNeighborList( const PositionSlice &position,
                         const double neighborhood_radius )
{
    // Build a neighbor list with a brute force n^2 implementation. Count
    // first.
    TestNeighborList<TEST_MEMSPACE> list;
    int num_particle = position.size();
    double rsqr = neighborhood_radius * neighborhood_radius;
    list.counts = Kokkos::View<int *, TEST_MEMSPACE>( "test_neighbor_count",
                                                      num_particle );
    Kokkos::deep_copy( list.counts, 0 );
    auto count_op = KOKKOS_LAMBDA( const int i )
    {
        for ( int j = 0; j < num_particle; ++j )
        {
            if ( i != j )
            {
                double dsqr = 0.0;
                for ( int d = 0; d < 3; ++d )
                    dsqr += ( position( i, d ) - position( j, d ) ) *
                            ( position( i, d ) - position( j, d ) );
                if ( dsqr <= rsqr )
                    list.counts( i ) += 1;
            }
        }
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> exec_policy( 0, num_particle );
    Kokkos::parallel_for( exec_policy, count_op );
    Kokkos::fence();

    // Allocate.
    auto max_op = KOKKOS_LAMBDA( const int i, int &max_val )
    {
        if ( max_val < list.counts( i ) )
            max_val = list.counts( i );
    };
    int max_n;
    Kokkos::parallel_reduce( exec_policy, max_op, Kokkos::Max<int>( max_n ) );
    Kokkos::fence();
    list.neighbors = Kokkos::View<int **, TEST_MEMSPACE>( "test_neighbors",
                                                          num_particle, max_n );

    // Fill.
    auto fill_op = KOKKOS_LAMBDA( const int i )
    {
        int n_count = 0;
        for ( int j = 0; j < num_particle; ++j )
        {
            if ( i != j )
            {
                double dsqr = 0.0;
                for ( int d = 0; d < 3; ++d )
                    dsqr += ( position( i, d ) - position( j, d ) ) *
                            ( position( i, d ) - position( j, d ) );
                if ( dsqr <= rsqr )
                {
                    list.neighbors( i, n_count ) = j;
                    ++n_count;
                }
            }
        }
    };
    Kokkos::parallel_for( exec_policy, fill_op );
    Kokkos::fence();

    return list;
}

//---------------------------------------------------------------------------//
template <class ListType, class PositionSlice>
void checkFullNeighborList( const ListType &list, const PositionSlice &position,
                            const double neighborhood_radius )
{
    // Create a test list to check against.
    auto test_list = computeFullNeighborList( position, neighborhood_radius );
    auto test_list_copy = createTestListHostCopy( test_list );

    // Create another list and copy the contents of the list we are testing
    // onto the host.
    auto list_copy = copyListToHost( list, test_list.neighbors.extent( 0 ),
                                     test_list.neighbors.extent( 1 ) );

    // Check the results.
    int num_particle = position.size();
    for ( int p = 0; p < num_particle; ++p )
    {
        // First check that the number of neighbors are the same.
        EXPECT_EQ( list_copy.counts( p ), test_list_copy.counts( p ) );

        // Now extract the neighbors.
        std::vector<int> computed_neighbors( test_list_copy.counts( p ) );
        std::vector<int> actual_neighbors( test_list_copy.counts( p ) );
        for ( int n = 0; n < test_list_copy.counts( p ); ++n )
        {
            computed_neighbors[n] = list_copy.neighbors( p, n );
            actual_neighbors[n] = test_list_copy.neighbors( p, n );
        }

        // Sort them because we have no guarantee of the order we will find
        // them in.
        std::sort( computed_neighbors.begin(), computed_neighbors.end() );
        std::sort( actual_neighbors.begin(), actual_neighbors.end() );

        // Now compare directly.
        for ( int n = 0; n < test_list_copy.counts( p ); ++n )
            EXPECT_EQ( computed_neighbors[n], actual_neighbors[n] );
    }
}

//---------------------------------------------------------------------------//
template <class ListType, class PositionSlice>
void checkHalfNeighborList( const ListType &list, const PositionSlice &position,
                            const double neighborhood_radius )
{
    // First build a full list.
    auto full_list = computeFullNeighborList( position, neighborhood_radius );
    auto full_list_copy = createTestListHostCopy( full_list );

    // Create another list and copy the contents of the list we are testing
    // onto the host.
    auto list_copy = copyListToHost( list, full_list.neighbors.extent( 0 ),
                                     full_list.neighbors.extent( 1 ) );

    // Check that the full list is twice the size of the half list.
    int num_particle = position.size();
    int half_size = 0;
    int full_size = 0;
    for ( int p = 0; p < num_particle; ++p )
    {
        half_size += list_copy.counts( p );
        full_size += full_list_copy.counts( p );
    }
    EXPECT_EQ( full_size, 2 * half_size );

    // Check the half list by ensuring that a particle does not show up in the
    // neighbor list of its neighbors.
    for ( int p = 0; p < num_particle; ++p )
    {
        // Check each neighbor of p
        for ( int n = 0; n < list_copy.counts( p ); ++n )
        {
            // Get the id of the nth neighbor of p.
            auto p_n = list_copy.neighbors( p, n );

            // Check that p is not in the neighbor list of the nth neighbor of
            // p.
            for ( int m = 0; m < list_copy.counts( p_n ); ++m )
            {
                auto n_m = list_copy.neighbors( p_n, m );
                EXPECT_NE( n_m, p );
            }
        }
    }
}

//---------------------------------------------------------------------------//
template <class ListType, class PositionSlice>
void checkFullNeighborListPartialRange( const ListType &list,
                                        const PositionSlice &position,
                                        const double neighborhood_radius,
                                        const int num_ignore )
{
    // Build a full list to test with
    auto test_list = computeFullNeighborList( position, neighborhood_radius );
    auto test_list_copy = createTestListHostCopy( test_list );

    // Create another list and copy the contents of the list we are testing
    // onto the host.
    auto list_copy = copyListToHost( list, test_list.neighbors.extent( 0 ),
                                     test_list.neighbors.extent( 1 ) );

    // Check the results.
    int num_particle = position.size();
    for ( int p = 0; p < num_particle; ++p )
    {
        if ( p < num_ignore )
        {
            // First check that the number of neighbors are the same.
            EXPECT_EQ( list_copy.counts( p ), test_list_copy.counts( p ) );

            // Now extract the neighbors.
            std::vector<int> computed_neighbors( test_list_copy.counts( p ) );
            std::vector<int> actual_neighbors( test_list_copy.counts( p ) );
            for ( int n = 0; n < test_list_copy.counts( p ); ++n )
            {
                computed_neighbors[n] = list_copy.neighbors( p, n );
                actual_neighbors[n] = test_list_copy.neighbors( p, n );
            }

            // Sort them because we have no guarantee of the order we will find
            // them in.
            std::sort( computed_neighbors.begin(), computed_neighbors.end() );
            std::sort( actual_neighbors.begin(), actual_neighbors.end() );

            // Now compare directly.
            for ( int n = 0; n < test_list_copy.counts( p ); ++n )
                EXPECT_EQ( computed_neighbors[n], actual_neighbors[n] );
        }
        else
        {
            EXPECT_EQ( list_copy.counts( p ), 0 );
        }
    }
}

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testVerletListFull()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    double test_radius = 2.32;
    double cell_size_ratio = 0.5;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );

    // Create the neighbor list.
    double grid_min[3] = {box_min, box_min, box_min};
    double grid_max[3] = {box_max, box_max, box_max};
#if !defined( Cabana_ENABLE_ARBORX )
    Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag>
        nlist_full( Cabana::slice<0>( aosoa ), 0, aosoa.size(), test_radius,
                    cell_size_ratio, grid_min, grid_max );
#else
    using device_type = TEST_MEMSPACE; // sigh...
    auto const tmp = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::FullNeighborTag{}, Cabana::slice<0>( aosoa ), 0, aosoa.size(),
        test_radius );
    Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag>
        nlist_full = convert_crs_graph_to_verlet_list<device_type,
                                                      Cabana::FullNeighborTag>(
            LayoutTag{}, tmp );
#endif

    Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag> nlist;
    nlist = nlist_full;

    // Check the neighbor list.
    auto position = Cabana::slice<0>( aosoa );
    checkFullNeighborList( nlist, position, test_radius );

    // Check again, building with a large array allocation size
    Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag>
        nlist_max( Cabana::slice<0>( aosoa ), 0, aosoa.size(), test_radius,
                   cell_size_ratio, grid_min, grid_max, 100 );
    checkFullNeighborList( nlist_max, position, test_radius );

    // Check again, building with a small array allocation size (refill)
    Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag>
        nlist_max2( Cabana::slice<0>( aosoa ), 0, aosoa.size(), test_radius,
                    cell_size_ratio, grid_min, grid_max, 2 );
    checkFullNeighborList( nlist_max2, position, test_radius );
}

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testVerletListHalf()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    double test_radius = 2.32;
    double cell_size_ratio = 0.5;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );

    // Create the neighbor list.
    double grid_min[3] = {box_min, box_min, box_min};
    double grid_max[3] = {box_max, box_max, box_max};
#if !defined( Cabana_ENABLE_ARBORX )
    Cabana::VerletList<TEST_MEMSPACE, Cabana::HalfNeighborTag, LayoutTag> nlist(
        Cabana::slice<0>( aosoa ), 0, aosoa.size(), test_radius,
        cell_size_ratio, grid_min, grid_max );
#else
    using device_type = TEST_MEMSPACE; // sigh...
    auto const tmp = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::HalfNeighborTag{}, Cabana::slice<0>( aosoa ), 0, aosoa.size(),
        test_radius );
    Cabana::VerletList<TEST_MEMSPACE, Cabana::HalfNeighborTag, LayoutTag>
        nlist = convert_crs_graph_to_verlet_list<device_type,
                                                 Cabana::HalfNeighborTag>(
            LayoutTag{}, tmp );
#endif

    // Check the neighbor list.
    auto position = Cabana::slice<0>( aosoa );
    checkHalfNeighborList( nlist, position, test_radius );

    // Check again, building with a large array allocation size
    Cabana::VerletList<TEST_MEMSPACE, Cabana::HalfNeighborTag, LayoutTag>
        nlist_max( Cabana::slice<0>( aosoa ), 0, aosoa.size(), test_radius,
                   cell_size_ratio, grid_min, grid_max, 100 );
    checkHalfNeighborList( nlist_max, position, test_radius );

    // Check again, building with a small array allocation size (refill)
    Cabana::VerletList<TEST_MEMSPACE, Cabana::HalfNeighborTag, LayoutTag>
        nlist_max2( Cabana::slice<0>( aosoa ), 0, aosoa.size(), test_radius,
                    cell_size_ratio, grid_min, grid_max, 2 );
    checkHalfNeighborList( nlist_max2, position, test_radius );
}

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testFirstNeighborParallelFor()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    double test_radius = 2.32;
    double cell_size_ratio = 0.5;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );

    // Create the neighbor list.
#if !defined( Cabana_ENABLE_ARBORX )
    using ListType =
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag>;
    double grid_min[3] = {box_min, box_min, box_min};
    double grid_max[3] = {box_max, box_max, box_max};
    ListType nlist( Cabana::slice<0>( aosoa ), 0, aosoa.size(), test_radius,
                    cell_size_ratio, grid_min, grid_max );
#else
    (void)cell_size_ratio;
    using device_type = TEST_MEMSPACE; // sigh...
    auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::FullNeighborTag{}, Cabana::slice<0>( aosoa ), 0, aosoa.size(),
        test_radius );
#endif

    // Create Kokkos views for the write operation.
    using memory_space = typename TEST_MEMSPACE::memory_space;
    Kokkos::View<int *, Kokkos::HostSpace> test_result( "test_result",
                                                        num_particle );
    Kokkos::View<int *, memory_space> serial_result( "serial_result",
                                                     num_particle );
    Kokkos::View<int *, memory_space> team_result( "team_result",
                                                   num_particle );

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto serial_count_op = KOKKOS_LAMBDA( const int i, const int n )
    {
        Kokkos::atomic_add( &serial_result( i ), n );
    };
    auto team_count_op = KOKKOS_LAMBDA( const int i, const int n )
    {
        Kokkos::atomic_add( &team_result( i ), n );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, aosoa.size() );
    Cabana::neighbor_parallel_for( policy, serial_count_op, nlist,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(), "test_1st_serial" );
    Cabana::neighbor_parallel_for( policy, team_count_op, nlist,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::TeamOpTag(), "test_1st_team" );
    Kokkos::fence();

    // Get the expected result in serial
    auto test_list =
        computeFullNeighborList( Cabana::slice<0>( aosoa ), test_radius );
    auto test_list_copy = createTestListHostCopy( test_list );
    for ( int p = 0; p < num_particle; ++p )
        for ( int n = 0; n < test_list_copy.counts( p ); ++n )
            test_result( p ) += test_list_copy.neighbors( p, n );

    // Check the result.
    auto serial_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), serial_result );
    auto team_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), team_result );
    for ( int p = 0; p < num_particle; ++p )
    {
        EXPECT_EQ( test_result( p ), serial_mirror( p ) );
        EXPECT_EQ( test_result( p ), team_mirror( p ) );
    }
}

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testVerletListFullPartialRange()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    int num_ignore = 800;
    double test_radius = 2.32;
    double cell_size_ratio = 0.5;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );

    // Create the neighbor list.
#if !defined( Cabana_ENABLE_ARBORX )
    double grid_min[3] = {box_min, box_min, box_min};
    double grid_max[3] = {box_max, box_max, box_max};
    Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag> nlist(
        Cabana::slice<0>( aosoa ), 0, num_ignore, test_radius, cell_size_ratio,
        grid_min, grid_max );
#else
    (void)cell_size_ratio;
    using device_type = TEST_MEMSPACE; // sigh...
    auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::FullNeighborTag{}, Cabana::slice<0>( aosoa ), 0, num_ignore,
        test_radius );
#endif

    // Check the neighbor list.
    auto position = Cabana::slice<0>( aosoa );
    checkFullNeighborListPartialRange( nlist, position, test_radius,
                                       num_ignore );
}

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testSecondNeighborParallelFor()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    double test_radius = 2.32;
    double cell_size_ratio = 0.5;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );

    // Create the neighbor list.
#if !defined( Cabana_ENABLE_ARBORX )
    using ListType =
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag>;
    double grid_min[3] = {box_min, box_min, box_min};
    double grid_max[3] = {box_max, box_max, box_max};
    ListType nlist( Cabana::slice<0>( aosoa ), 0, aosoa.size(), test_radius,
                    cell_size_ratio, grid_min, grid_max );
#else
    (void)cell_size_ratio;
    using device_type = TEST_MEMSPACE; // sigh...
    auto const nlist = Cabana::Experimental::makeNeighborList<device_type>(
        Cabana::FullNeighborTag{}, Cabana::slice<0>( aosoa ), 0, aosoa.size(),
        test_radius );
#endif

    // Create Kokkos views for the write operation.
    using memory_space = typename TEST_MEMSPACE::memory_space;
    Kokkos::View<int *, Kokkos::HostSpace> test_result( "test_result",
                                                        num_particle );
    Kokkos::View<int *, memory_space> serial_result( "serial_result",
                                                     num_particle );
    Kokkos::View<int *, memory_space> team_result( "team_result",
                                                   num_particle );
    Kokkos::View<int *, memory_space> vector_result( "vector_result",
                                                     num_particle );

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto serial_count_op =
        KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        Kokkos::atomic_add( &serial_result( i ), j );
        Kokkos::atomic_add( &serial_result( i ), k );
    };
    auto team_count_op = KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        Kokkos::atomic_add( &team_result( i ), j );
        Kokkos::atomic_add( &team_result( i ), k );
    };
    auto vector_count_op =
        KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        Kokkos::atomic_add( &vector_result( i ), j );
        Kokkos::atomic_add( &vector_result( i ), k );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, aosoa.size() );
    Cabana::neighbor_parallel_for( policy, serial_count_op, nlist,
                                   Cabana::SecondNeighborsTag(),
                                   Cabana::SerialOpTag(), "test_2nd_serial" );
    Cabana::neighbor_parallel_for( policy, team_count_op, nlist,
                                   Cabana::SecondNeighborsTag(),
                                   Cabana::TeamOpTag(), "test_2nd_team" );
    Cabana::neighbor_parallel_for(
        policy, vector_count_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamVectorOpTag(), "test_2nd_vector" );
    Kokkos::fence();

    // Get the expected result in serial
    auto test_list =
        computeFullNeighborList( Cabana::slice<0>( aosoa ), test_radius );
    auto test_list_copy = createTestListHostCopy( test_list );
    for ( int p = 0; p < num_particle; ++p )
        for ( int n = 0; n < test_list_copy.counts( p ) - 1; ++n )
            for ( int a = n + 1; a < test_list_copy.counts( p ); ++a )
            {
                test_result( p ) += test_list_copy.neighbors( p, n );
                test_result( p ) += test_list_copy.neighbors( p, a );
            }

    // Check the result.
    auto serial_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), serial_result );
    auto team_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), team_result );
    auto vector_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), vector_result );
    for ( int p = 0; p < num_particle; ++p )
    {
        EXPECT_EQ( test_result( p ), serial_mirror( p ) );
        EXPECT_EQ( test_result( p ), team_mirror( p ) );
        EXPECT_EQ( test_result( p ), vector_mirror( p ) );
    }
}

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testFirstNeighborParallelReduce()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    double test_radius = 2.32;
    double cell_size_ratio = 0.5;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );
    auto positions = Cabana::slice<0>( aosoa );

    // Create the neighbor list.
    using ListType =
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag>;
    double grid_min[3] = {box_min, box_min, box_min};
    double grid_max[3] = {box_max, box_max, box_max};
    ListType nlist( positions, 0, aosoa.size(), test_radius, cell_size_ratio,
                    grid_min, grid_max );

    // Reduction variables
    double test_sum = 0;
    double serial_sum = 0;
    double team_sum = 0;

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto sum_op = KOKKOS_LAMBDA( const int i, const int n, double &sum )
    {
        sum += positions( i, 0 ) + positions( n, 0 );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, aosoa.size() );
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::FirstNeighborsTag(),
        Cabana::SerialOpTag(), serial_sum, "test_reduce_serial" );
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::FirstNeighborsTag(), Cabana::TeamOpTag(),
        team_sum, "test_reduce_team" );
    Kokkos::fence();

    // Get the expected result in serial
    auto test_list = computeFullNeighborList( positions, test_radius );
    auto test_list_copy = createTestListHostCopy( test_list );
    for ( int p = 0; p < num_particle; ++p )
        for ( int n = 0; n < test_list_copy.counts( p ); ++n )
            test_sum += positions( p, 0 ) +
                        positions( test_list_copy.neighbors( p, n ), 0 );

    // Check the result.
    EXPECT_FLOAT_EQ( test_sum, serial_sum );
    EXPECT_FLOAT_EQ( test_sum, team_sum );
}

//---------------------------------------------------------------------------//
template <class LayoutTag>
void testSecondNeighborParallelReduce()
{
    // Create the AoSoA and fill with random particle positions.
    int num_particle = 1e3;
    double test_radius = 2.32;
    double cell_size_ratio = 0.5;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;
    auto aosoa = createParticles( num_particle, box_min, box_max );
    auto positions = Cabana::slice<0>( aosoa );

    // Create the neighbor list.
    using ListType =
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag, LayoutTag>;
    double grid_min[3] = {box_min, box_min, box_min};
    double grid_max[3] = {box_max, box_max, box_max};
    ListType nlist( positions, 0, aosoa.size(), test_radius, cell_size_ratio,
                    grid_min, grid_max );

    // Reduction variables
    double test_sum = 0;
    double serial_sum = 0;
    double team_sum = 0;
    double vector_sum = 0;

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto sum_op =
        KOKKOS_LAMBDA( const int i, const int n, const int a, double &sum )
    {
        sum += positions( i, 0 ) + positions( n, 0 ) + positions( a, 0 );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, aosoa.size() );
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::SerialOpTag(), serial_sum, "test_reduce_serial" );
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamOpTag(), team_sum, "test_reduce_team" );
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamVectorOpTag(), vector_sum, "test_reduce_vector" );
    Kokkos::fence();

    // Get the expected result in serial
    auto test_list = computeFullNeighborList( positions, test_radius );
    auto test_list_copy = createTestListHostCopy( test_list );
    for ( int p = 0; p < num_particle; ++p )
        for ( int n = 0; n < test_list_copy.counts( p ); ++n )
            for ( int a = n + 1; a < test_list_copy.counts( p ); ++a )
                test_sum += positions( p, 0 ) +
                            positions( test_list_copy.neighbors( p, n ), 0 ) +
                            positions( test_list_copy.neighbors( p, a ), 0 );

    // Check the result.
    EXPECT_FLOAT_EQ( test_sum, serial_sum );
    EXPECT_FLOAT_EQ( test_sum, team_sum );
    EXPECT_FLOAT_EQ( test_sum, vector_sum );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, linked_cell_stencil_test ) { testLinkedCellStencil(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_list_full_test )
{
    testVerletListFull<Cabana::VerletLayoutCSR>();
    testVerletListFull<Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_list_half_test )
{
    testVerletListHalf<Cabana::VerletLayoutCSR>();
    testVerletListHalf<Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, verlet_list_full_range_test )
{
    testVerletListFullPartialRange<Cabana::VerletLayoutCSR>();
    testVerletListFullPartialRange<Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, parallel_for_test )
{
    testFirstNeighborParallelFor<Cabana::VerletLayoutCSR>();
    testFirstNeighborParallelFor<Cabana::VerletLayout2D>();

    testSecondNeighborParallelFor<Cabana::VerletLayoutCSR>();
    testSecondNeighborParallelFor<Cabana::VerletLayout2D>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, parallel_reduce_test )
{
    testFirstNeighborParallelReduce<Cabana::VerletLayoutCSR>();
    testFirstNeighborParallelReduce<Cabana::VerletLayout2D>();

    testSecondNeighborParallelReduce<Cabana::VerletLayoutCSR>();
    testSecondNeighborParallelReduce<Cabana::VerletLayout2D>();
}
//---------------------------------------------------------------------------//

} // end namespace Test
