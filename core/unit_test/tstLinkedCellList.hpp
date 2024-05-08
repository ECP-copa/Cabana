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
#include <Cabana_DeepCopy.hpp>
#include <Cabana_LinkedCellList.hpp>

#include <neighbor_unit_test.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// Test data
//---------------------------------------------------------------------------//
struct LCLTestData
{
    enum MyFields
    {
        Position = 0,
        CellId = 1
    };
    using data_types = Cabana::MemberTypes<double[3], int[3]>;
    using aosoa_type = Cabana::AoSoA<data_types, TEST_MEMSPACE>;
    using size_type = typename aosoa_type::memory_space::size_type;
    std::size_t num_p = 1000;
    std::size_t begin = 250;
    std::size_t end = 750;
    aosoa_type aosoa;

    // Set the problem so each particle lives in the center of a cell on a
    // regular grid of cell size 1 and total size 10x10x10. We are making them
    // in the reverse order we expect the sort to happen. The sort binary
    // operator should order by i first and k last.
    int nx = 10;
    double dx = 1.0;
    double x_min = 0.0;
    double x_max = x_min + nx * dx;

    // Create a grid.
    double grid_delta[3] = { dx, dx, dx };
    double grid_min[3] = { x_min, x_min, x_min };
    double grid_max[3] = { x_max, x_max, x_max };

    using IDViewType = Kokkos::View<int* [3], TEST_MEMSPACE>;
    using PosViewType = Kokkos::View<double* [3], TEST_MEMSPACE>;
    using BinViewType = Kokkos::View<size_type***, TEST_MEMSPACE>;

    using layout = typename TEST_EXECSPACE::array_layout;
    Kokkos::View<int* [3], layout, Kokkos::HostSpace> ids_mirror;
    Kokkos::View<double* [3], layout, Kokkos::HostSpace> pos_mirror;
    Kokkos::View<size_type***, layout, Kokkos::HostSpace> bin_size_mirror;
    Kokkos::View<size_type***, layout, Kokkos::HostSpace> bin_offset_mirror;

    LCLTestData()
    {
        aosoa = aosoa_type( "aosoa", num_p );

        createParticles();
    }

    void createParticles()
    {
        // Create local variables for lambda capture
        auto nx_ = nx;
        auto x_min_ = x_min;
        auto dx_ = dx;

        // Fill the AoSoA with positions and ijk cell ids.
        auto pos = Cabana::slice<Position>( aosoa, "position" );
        auto cell_id = Cabana::slice<CellId>( aosoa, "cell_id" );
        Kokkos::parallel_for(
            "initialize", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, nx_ ),
            KOKKOS_LAMBDA( const int k ) {
                for ( int j = 0; j < nx_; ++j )
                {
                    for ( int i = 0; i < nx_; ++i )
                    {
                        std::size_t particle_id = i + j * nx_ + k * nx_ * nx_;

                        cell_id( particle_id, 0 ) = i;
                        cell_id( particle_id, 1 ) = j;
                        cell_id( particle_id, 2 ) = k;

                        pos( particle_id, 0 ) = x_min_ + ( i + 0.5 ) * dx_;
                        pos( particle_id, 1 ) = x_min_ + ( j + 0.5 ) * dx_;
                        pos( particle_id, 2 ) = x_min_ + ( k + 0.5 ) * dx_;
                    }
                }
            } );
    }
};

void copyListToHost(
    LCLTestData& test_data,
    const Cabana::LinkedCellList<TEST_MEMSPACE, double> cell_list )
{
    // Copy data to the host for testing.
    auto np = test_data.num_p;
    auto nx = test_data.nx;
    auto pos_slice = Cabana::slice<LCLTestData::Position>( test_data.aosoa );
    auto id_slice = Cabana::slice<LCLTestData::CellId>( test_data.aosoa );

    LCLTestData::IDViewType ids( "cell_ids", np );
    LCLTestData::PosViewType pos( "cell_ids", np );
    LCLTestData::BinViewType bin_size( "bin_size", nx, nx, nx );
    LCLTestData::BinViewType bin_offset( "bin_offset", nx, nx, nx );
    Kokkos::parallel_for(
        "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, nx ),
        KOKKOS_LAMBDA( const int i ) {
            for ( int j = 0; j < nx; ++j )
                for ( int k = 0; k < nx; ++k )
                {
                    std::size_t original_id = i + j * nx + k * nx * nx;
                    ids( original_id, 0 ) = id_slice( original_id, 0 );
                    ids( original_id, 1 ) = id_slice( original_id, 1 );
                    ids( original_id, 2 ) = id_slice( original_id, 2 );
                    pos( original_id, 0 ) = pos_slice( original_id, 0 );
                    pos( original_id, 1 ) = pos_slice( original_id, 1 );
                    pos( original_id, 2 ) = pos_slice( original_id, 2 );
                    bin_size( i, j, k ) = cell_list.binSize( i, j, k );
                    bin_offset( i, j, k ) = cell_list.binOffset( i, j, k );
                }
        } );
    Kokkos::fence();
    test_data.ids_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ids );
    test_data.pos_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pos );
    test_data.bin_size_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_size );
    test_data.bin_offset_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), bin_offset );
}

//---------------------------------------------------------------------------//
// Main Linked Cell List tests
//---------------------------------------------------------------------------//
void checkBins( const LCLTestData test_data,
                const Cabana::LinkedCellList<TEST_MEMSPACE, double> cell_list )
{
    auto nx = test_data.nx;

    // Checking the binning.
    EXPECT_EQ( cell_list.totalBins(), nx * nx * nx );
    EXPECT_EQ( cell_list.numBin( 0 ), nx );
    EXPECT_EQ( cell_list.numBin( 1 ), nx );
    EXPECT_EQ( cell_list.numBin( 2 ), nx );
}

// Check LinkedCell data, where either a subset (begin->end) or all data is
// sorted and where the IDs are sorted or not based on whether the entire AoSoA
// or only the position slice was permuted.
void checkLinkedCell( const LCLTestData test_data,
                      const std::size_t check_begin,
                      const std::size_t check_end, const bool sorted_ids )
{
    auto nx = test_data.nx;
    auto ids_mirror = test_data.ids_mirror;
    auto bin_size_mirror = test_data.bin_size_mirror;
    auto bin_offset_mirror = test_data.bin_offset_mirror;

    // The order should be reversed with the i index moving the slowest
    // for those that are actually in the binning range. Do this pass
    // first. We do this by looping through in the sorted order and
    // check those that had original indices in the sorting range.
    std::size_t particle_id = 0;
    for ( int i = 0; i < nx; ++i )
    {
        for ( int j = 0; j < nx; ++j )
        {
            for ( int k = 0; k < nx; ++k )
            {
                std::size_t original_id = i + j * nx + k * nx * nx;
                if ( check_begin <= original_id && original_id < check_end )
                {
                    // Get what should be the local id of the particle
                    // in the newly sorted decomposition. We are looping
                    // through this in k-fastest order (the indexing of
                    // the grid cells) and therefore should get the
                    // particles in their sorted order.
                    int sort_id = check_begin + particle_id;

                    // Back calculate what we think the ijk indices of
                    // the particle are based on k-fastest ordering.
                    int grid_id = i * nx * nx + j * nx + k;
                    int grid_i = grid_id / ( nx * nx );
                    int grid_j = ( grid_id / nx ) % nx;
                    int grid_k = grid_id % nx;

                    // Check the indices of the particle, if sorted.
                    if ( sorted_ids )
                    {
                        EXPECT_EQ( ids_mirror( sort_id, 0 ), grid_i );
                        EXPECT_EQ( ids_mirror( sort_id, 1 ), grid_j );
                        EXPECT_EQ( ids_mirror( sort_id, 2 ), grid_k );
                    }
                    // Check that we binned the particle and got the
                    // right offset.
                    EXPECT_EQ( bin_size_mirror( i, j, k ), 1 );
                    EXPECT_EQ( bin_offset_mirror( i, j, k ),
                               LCLTestData::size_type( particle_id ) );

                    // Increment the particle id.
                    ++particle_id;
                }
            }
        }
    }

    // For those that are outside the binned range IDs should be
    // unchanged and the bins should empty.
    particle_id = 0;
    for ( int k = 0; k < nx; ++k )
    {
        for ( int j = 0; j < nx; ++j )
        {
            for ( int i = 0; i < nx; ++i, ++particle_id )
            {
                if ( check_begin > particle_id || particle_id >= check_end )
                {
                    EXPECT_EQ( ids_mirror( particle_id, 0 ), i );
                    EXPECT_EQ( ids_mirror( particle_id, 1 ), j );
                    EXPECT_EQ( ids_mirror( particle_id, 2 ), k );
                    EXPECT_EQ( bin_size_mirror( i, j, k ), 0 );
                }
                else if ( not sorted_ids )
                {
                    // If sorted by position slice, IDs in range should
                    // still not be sorted.
                    EXPECT_EQ( ids_mirror( particle_id, 0 ), i );
                    EXPECT_EQ( ids_mirror( particle_id, 1 ), j );
                    EXPECT_EQ( ids_mirror( particle_id, 2 ), k );
                }
            }
        }
    }
}

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
        Cabana::LinkedCellStencil<double> stencil( radius, ratio, min, max );

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
        Cabana::LinkedCellStencil<double> stencil( radius, ratio, min, max );

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
        Cabana::LinkedCellStencil<double> stencil( radius, ratio, min, max );

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
void testLinkedList()
{
    LCLTestData test_data;
    auto grid_delta = test_data.grid_delta;
    auto grid_min = test_data.grid_min;
    auto grid_max = test_data.grid_max;
    auto pos = Cabana::slice<LCLTestData::Position>( test_data.aosoa );

    // Bin and permute the particles in the grid. First do this by only
    // operating on a subset of the particles.
    {
        auto begin = test_data.begin;
        auto end = test_data.end;
        auto cell_list = Cabana::createLinkedCellList(
            pos, begin, end, grid_delta, grid_min, grid_max,
            test_data.grid_delta[0], 1.0 );
        Cabana::permute( cell_list, test_data.aosoa );

        copyListToHost( test_data, cell_list );

        checkBins( test_data, cell_list );
        checkLinkedCell( test_data, begin, end, true );
    }

    // Now bin and permute all of the particles.
    {
        auto cell_list = Cabana::createLinkedCellList(
            pos, grid_delta, grid_min, grid_max, test_data.grid_delta[0] );
        Cabana::permute( cell_list, test_data.aosoa );

        copyListToHost( test_data, cell_list );

        checkBins( test_data, cell_list );
        checkLinkedCell( test_data, 0, test_data.num_p, true );
    }
}

//---------------------------------------------------------------------------//
void testLinkedListSlice()
{
    LCLTestData test_data;
    auto grid_delta = test_data.grid_delta;
    auto grid_min = test_data.grid_min;
    auto grid_max = test_data.grid_max;
    auto pos = Cabana::slice<LCLTestData::Position>( test_data.aosoa );

    // Bin the particles in the grid and permute only the position slice.
    // First do this by only operating on a subset of the particles.
    {
        auto begin = test_data.begin;
        auto end = test_data.end;
        auto cell_list = Cabana::createLinkedCellList(
            pos, begin, end, grid_delta, grid_min, grid_max, grid_delta[0] );
        Cabana::permute( cell_list, pos );

        copyListToHost( test_data, cell_list );

        checkBins( test_data, cell_list );
        checkLinkedCell( test_data, begin, end, false );
    }
    // Now bin and permute all of the particles.
    {
        auto cell_list = Cabana::createLinkedCellList(
            pos, grid_delta, grid_min, grid_max, grid_delta[0] );
        Cabana::permute( cell_list, pos );

        copyListToHost( test_data, cell_list );

        checkBins( test_data, cell_list );
        checkLinkedCell( test_data, 0, test_data.num_p, false );

        // Rebuild and make sure nothing changed.
        cell_list.build( pos );
        Cabana::permute( cell_list, pos );

        copyListToHost( test_data, cell_list );

        checkBins( test_data, cell_list );
        checkLinkedCell( test_data, 0, test_data.num_p, false );
    }
}

template <class ListType, class TestListType, class PositionType>
void checkLinkedCellNeighborInterface( const ListType& nlist,
                                       const TestListType& N2_list_copy,
                                       const std::size_t begin,
                                       const std::size_t end,
                                       const PositionType positions,
                                       const double cutoff )
{
    using memory_space = typename TEST_MEMSPACE::memory_space;

    // Purposely using zero-init.
    Kokkos::View<std::size_t*, memory_space> num_n2_neighbors(
        "num_n2_neighbors", positions.size() );
    Kokkos::View<std::size_t*, Kokkos::HostSpace> N2_copy_neighbors(
        "num_n2_neighbors", positions.size() );

    Cabana::NeighborDiscriminator<Cabana::FullNeighborTag> _discriminator;

    std::size_t max_n2_neighbors = 0;
    std::size_t sum_n2_neighbors = 0;

    std::size_t N2_copy_max = 0;
    std::size_t N2_copy_sum = 0;

    for ( std::size_t p = begin; p < end; ++p )
    {
        for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
        {
            if ( N2_list_copy.neighbors( p, n ) >= static_cast<int>( begin ) &&
                 N2_list_copy.neighbors( p, n ) < static_cast<int>( end ) )
            {
                N2_copy_neighbors( p ) += 1;
            }
        }
        if ( N2_copy_neighbors( p ) > N2_copy_max )
            N2_copy_max = N2_copy_neighbors( p );
        N2_copy_sum += N2_copy_neighbors( p );
    }

    double c2 = cutoff * cutoff;

    Kokkos::RangePolicy<TEST_EXECSPACE> policy( begin, end );
    Kokkos::parallel_for(
        "Neighbor_Interface", policy, KOKKOS_LAMBDA( const int pid ) {
            // Test the number of neighbors interface
            std::size_t num_lcl_neighbors =
                Cabana::NeighborList<ListType>::numNeighbor( nlist, pid );

            for ( std::size_t i = 0; i < num_lcl_neighbors; ++i )
            {
                std::size_t np = Cabana::NeighborList<ListType>::getNeighbor(
                    nlist, pid, i );

                const double dx = positions( pid, 0 ) - positions( np, 0 );
                const double dy = positions( pid, 1 ) - positions( np, 1 );
                const double dz = positions( pid, 2 ) - positions( np, 2 );
                const double r2 = dx * dx + dy * dy + dz * dz;

                if ( r2 <= c2 &&
                     _discriminator.isValid( pid, 0, 0, 0, np, 0, 0, 0 ) )
                {
                    if ( nlist.sorted() )
                        Kokkos::atomic_add(
                            &num_n2_neighbors(
                                nlist.permutation( pid - begin ) ),
                            1 );
                    else
                        Kokkos::atomic_add( &num_n2_neighbors( pid ), 1 );
                }
            }
        } );

    auto num_n2_neighbors_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), num_n2_neighbors );

    for ( std::size_t pid = 0; pid < positions.size(); ++pid )
    {
        if ( pid >= begin && pid < end )
        {
            EXPECT_EQ( num_n2_neighbors_host( pid ), N2_copy_neighbors( pid ) );
        }
        else
        {
            EXPECT_EQ( num_n2_neighbors_host( pid ), 0 );
        }

        sum_n2_neighbors += num_n2_neighbors_host( pid );
        if ( num_n2_neighbors_host( pid ) > max_n2_neighbors )
            max_n2_neighbors = num_n2_neighbors_host( pid );
    }

    EXPECT_EQ( max_n2_neighbors, N2_copy_max );
    EXPECT_EQ( sum_n2_neighbors, N2_copy_sum );
}

//---------------------------------------------------------------------------//
// linked_list_parallel
//---------------------------------------------------------------------------//
template <class ListType, class TestListType, class PositionType>
void checkLinkedCellNeighborParallel( const ListType& nlist,
                                      const TestListType& N2_list_copy,
                                      const std::size_t begin,
                                      const std::size_t end,
                                      const PositionType positions,
                                      const double cutoff )
{
    // Create Kokkos views for the write operation.
    using memory_space = typename TEST_MEMSPACE::memory_space;
    Kokkos::View<int*, memory_space> serial_result( "serial_result",
                                                    positions.size() );
    Kokkos::View<int*, memory_space> team_result( "team_result",
                                                  positions.size() );

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle (within cutoff) and compare to counts.
    auto c2 = cutoff * cutoff;

    Cabana::NeighborDiscriminator<Cabana::FullNeighborTag> _discriminator;

    auto serial_count_op = KOKKOS_LAMBDA( const int i, const int j )
    {
        const double dx = positions( i, 0 ) - positions( j, 0 );
        const double dy = positions( i, 1 ) - positions( j, 1 );
        const double dz = positions( i, 2 ) - positions( j, 2 );
        const double r2 = dx * dx + dy * dy + dz * dz;
        if ( r2 <= c2 && _discriminator.isValid( i, 0, 0, 0, j, 0, 0, 0 ) )
        {
            if ( nlist.sorted() )
            {
                Kokkos::atomic_add(
                    &serial_result( nlist.permutation( i - begin ) ),
                    nlist.permutation( j - begin ) );
            }
            else
            {
                Kokkos::atomic_add( &serial_result( i ), j );
            }
        }
    };
    auto team_count_op = KOKKOS_LAMBDA( const int i, const int j )
    {
        const double dx = positions( i, 0 ) - positions( j, 0 );
        const double dy = positions( i, 1 ) - positions( j, 1 );
        const double dz = positions( i, 2 ) - positions( j, 2 );
        const double r2 = dx * dx + dy * dy + dz * dz;
        if ( r2 <= c2 && _discriminator.isValid( i, 0, 0, 0, j, 0, 0, 0 ) )
        {
            if ( nlist.sorted() )
            {
                Kokkos::atomic_add(
                    &team_result( nlist.permutation( i - begin ) ),
                    nlist.permutation( j - begin ) );
            }
            else
            {
                Kokkos::atomic_add( &team_result( i ), j );
            }
        }
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( begin, end );
    Cabana::neighbor_parallel_for( policy, serial_count_op, nlist,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(), "test_1st_serial" );
    Cabana::neighbor_parallel_for( policy, team_count_op, nlist,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::TeamOpTag(), "test_1st_team" );
    Kokkos::fence();

    checkFirstNeighborParallelFor( N2_list_copy, serial_result, team_result, 1,
                                   begin, end );
}

//---------------------------------------------------------------------------//
// linked_list_parallel
//---------------------------------------------------------------------------//
template <class ListType, class TestListType, class AoSoAType>
void checkLinkedCellNeighborReduce( const ListType& nlist,
                                    const TestListType& N2_list_copy,
                                    const AoSoAType aosoa,
                                    const std::size_t begin,
                                    const std::size_t end, const double cutoff )
{
    auto position = Cabana::slice<0>( aosoa );

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle (within cutoff) and compare to counts.
    auto c2 = cutoff * cutoff;

    Cabana::NeighborDiscriminator<Cabana::FullNeighborTag> _discriminator;

    auto sum_op = KOKKOS_LAMBDA( const int i, const int j, double& sum )
    {
        const double dx = position( i, 0 ) - position( j, 0 );
        const double dy = position( i, 1 ) - position( j, 1 );
        const double dz = position( i, 2 ) - position( j, 2 );
        const double r2 = dx * dx + dy * dy + dz * dz;
        if ( r2 <= c2 && _discriminator.isValid( i, 0, 0, 0, j, 0, 0, 0 ) )
        {
            if ( nlist.sorted() )
            {
                sum += position( nlist.permutation( i - begin ), 0 ) +
                       position( nlist.permutation( j - begin ), 0 );
            }
            else
            {
                sum += position( i, 0 ) + position( j, 0 );
            }
        }
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( begin, end );
    double serial_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::FirstNeighborsTag(),
        Cabana::SerialOpTag(), serial_sum, "test_1st_serial" );
    double team_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::FirstNeighborsTag(), Cabana::TeamOpTag(),
        team_sum, "test_1st_team" );
    Kokkos::fence();

    checkFirstNeighborParallelReduce( N2_list_copy, aosoa, serial_sum, team_sum,
                                      1, begin, end );
}

//---------------------------------------------------------------------------//
void testLinkedCellNeighborInterface()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto positions = Cabana::slice<0>( test_data.aosoa );
    // Create the linked cell list.
    double grid_size = test_data.cell_size_ratio * test_data.test_radius;
    double grid_delta[3] = { grid_size, grid_size, grid_size };
    auto nlist = Cabana::createLinkedCellList(
        positions, test_data.begin, test_data.end, grid_delta,
        test_data.grid_min, test_data.grid_max, test_data.test_radius,
        test_data.cell_size_ratio );

    checkLinkedCellNeighborInterface( nlist, test_data.N2_list_copy,
                                      test_data.begin, test_data.end, positions,
                                      test_data.test_radius );

    Cabana::permute( nlist, positions );

    checkLinkedCellNeighborInterface( nlist, test_data.N2_list_copy,
                                      test_data.begin, test_data.end, positions,
                                      test_data.test_radius );
}

//---------------------------------------------------------------------------//
void testLinkedCellParallel()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto positions = Cabana::slice<0>( test_data.aosoa );
    // Create the linked cell list.
    double grid_size = test_data.cell_size_ratio * test_data.test_radius;
    double grid_delta[3] = { grid_size, grid_size, grid_size };
    auto nlist = Cabana::createLinkedCellList(
        positions, test_data.begin, test_data.end, grid_delta,
        test_data.grid_min, test_data.grid_max, test_data.test_radius,
        test_data.cell_size_ratio );

    checkLinkedCellNeighborParallel( nlist, test_data.N2_list_copy,
                                     test_data.begin, test_data.end, positions,
                                     test_data.test_radius );

    Cabana::permute( nlist, positions );

    checkLinkedCellNeighborParallel( nlist, test_data.N2_list_copy,
                                     test_data.begin, test_data.end, positions,
                                     test_data.test_radius );
}

//---------------------------------------------------------------------------//
void testLinkedCellReduce()
{
    // Create the AoSoA and fill with random particle positions.
    NeighborListTestData test_data;
    auto positions = Cabana::slice<0>( test_data.aosoa );
    // Create the linked cell list.
    double grid_size = test_data.cell_size_ratio * test_data.test_radius;
    double grid_delta[3] = { grid_size, grid_size, grid_size };
    auto nlist = Cabana::createLinkedCellList(
        positions, test_data.begin, test_data.end, grid_delta,
        test_data.grid_min, test_data.grid_max, test_data.test_radius,
        test_data.cell_size_ratio );

    checkLinkedCellNeighborReduce( nlist, test_data.N2_list_copy,
                                   test_data.aosoa, test_data.begin,
                                   test_data.end, test_data.test_radius );

    Cabana::permute( nlist, positions );

    checkLinkedCellNeighborReduce( nlist, test_data.N2_list_copy,
                                   test_data.aosoa, test_data.begin,
                                   test_data.end, test_data.test_radius );
}

//---------------------------------------------------------------------------//
void testLinkedListView()
{
    LCLTestData test_data;
    auto grid_delta = test_data.grid_delta;
    auto grid_min = test_data.grid_min;
    auto grid_max = test_data.grid_max;
    auto slice = Cabana::slice<LCLTestData::Position>( test_data.aosoa );

    // Copy manually into a View.
    Kokkos::View<double**, TEST_MEMSPACE> view( "positions", slice.size(), 3 );
    copySliceToView( view, slice, 0, slice.size() );

    // Bin the particles in the grid and permute only the position slice.
    // First do this by only operating on a subset of the particles.
    {
        auto begin = test_data.begin;
        auto end = test_data.end;
        Cabana::LinkedCellList<TEST_MEMSPACE> cell_list(
            view, begin, end, grid_delta, grid_min, grid_max );
        Cabana::permute( cell_list, view );

        // Copy manually back into the AoSoA to check values.
        copyViewToSlice( slice, view, 0, slice.size() );

        copyListToHost( test_data, cell_list );

        checkBins( test_data, cell_list );
        checkLinkedCell( test_data, begin, end, false );
    }
    // Now bin and permute all of the particles.
    {
        Cabana::LinkedCellList<TEST_MEMSPACE> cell_list( view, grid_delta,
                                                         grid_min, grid_max );

        // Copy manually into a View.
        Kokkos::View<double**, TEST_MEMSPACE> view( "positions", slice.size(),
                                                    3 );
        copySliceToView( view, slice, 0, slice.size() );

        Cabana::permute( cell_list, view );

        copyListToHost( test_data, cell_list );

        checkBins( test_data, cell_list );
        checkLinkedCell( test_data, 0, test_data.num_p, false );

        // Rebuild and make sure nothing changed.
        cell_list.build( view );
        Cabana::permute( cell_list, view );

        // Copy manually back into the AoSoA to check values.
        copyViewToSlice( slice, view, 0, slice.size() );

        copyListToHost( test_data, cell_list );

        checkBins( test_data, cell_list );
        checkLinkedCell( test_data, 0, test_data.num_p, false );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
TEST( LinkedCellList, Stencil ) { testLinkedCellStencil(); }

TEST( LinkedCellList, AoSoA ) { testLinkedList(); }

TEST( LinkedCellList, Slice ) { testLinkedListSlice(); }

TEST( LinkedCellList, Neighbor ) { testLinkedCellNeighborInterface(); }

TEST( LinkedCellList, ParallelFor ) { testLinkedCellParallel(); }

TEST( LinkedCellList, ParallelReduce ) { testLinkedCellReduce(); }

//---------------------------------------------------------------------------//
TEST( LinkedCellList, linked_list_view_test ) { testLinkedListView(); }

//---------------------------------------------------------------------------//

} // end namespace Test
