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
#include <Cabana_DeepCopy.hpp>
#include <Cabana_LinkedCellList.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
void testLinkedList()
{
    // Make an AoSoA with positions and ijk cell ids.
    enum MyFields
    {
        Position = 0,
        CellId = 1
    };
    using DataTypes = Cabana::MemberTypes<double[3], int[3]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    using size_type = typename AoSoA_t::memory_space::size_type;
    int num_p = 1000;
    AoSoA_t aosoa( "aosoa", num_p );

    // Set the problem so each particle lives in the center of a cell on a
    // regular grid of cell size 1 and total size 10x10x10. We are making them
    // in the reverse order we expect the sort to happen. The sort binary
    // operator should order by i first and k last.
    int nx = 10;
    double dx = 1.0;
    double x_min = 0.0;
    double x_max = x_min + nx * dx;
    auto pos = Cabana::slice<Position>( aosoa, "position" );
    auto cell_id = Cabana::slice<CellId>( aosoa, "cell_id" );
    Kokkos::parallel_for(
        "initialize", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, nx ),
        KOKKOS_LAMBDA( const int k ) {
            for ( int j = 0; j < nx; ++j )
            {
                for ( int i = 0; i < nx; ++i )
                {
                    std::size_t particle_id = i + j * nx + k * nx * nx;

                    cell_id( particle_id, 0 ) = i;
                    cell_id( particle_id, 1 ) = j;
                    cell_id( particle_id, 2 ) = k;

                    pos( particle_id, 0 ) = x_min + ( i + 0.5 ) * dx;
                    pos( particle_id, 1 ) = x_min + ( j + 0.5 ) * dx;
                    pos( particle_id, 2 ) = x_min + ( k + 0.5 ) * dx;
                }
            }
        } );

    // Create a grid.
    double grid_delta[3] = { dx, dx, dx };
    double grid_min[3] = { x_min, x_min, x_min };
    double grid_max[3] = { x_max, x_max, x_max };

    // Bin and permute the particles in the grid. First do this by only
    // operating on a subset of the particles.
    {
        std::size_t begin = 250;
        std::size_t end = 750;
        Cabana::LinkedCellList<typename AoSoA_t::memory_space> cell_list(
            pos, begin, end, grid_delta, grid_min, grid_max );
        Cabana::permute( cell_list, aosoa );

        // Checking the binning.
        EXPECT_EQ( cell_list.totalBins(), nx * nx * nx );
        EXPECT_EQ( cell_list.numBin( 0 ), nx );
        EXPECT_EQ( cell_list.numBin( 1 ), nx );
        EXPECT_EQ( cell_list.numBin( 2 ), nx );

        // Copy data to the host for testing.
        Kokkos::View<int *[3], TEST_MEMSPACE> ids( "cell_ids", num_p );
        Kokkos::View<size_type ***, TEST_MEMSPACE> bin_size( "bin_size", nx, nx,
                                                             nx );
        Kokkos::View<size_type ***, TEST_MEMSPACE> bin_offset( "bin_offset", nx,
                                                               nx, nx );
        Kokkos::parallel_for(
            "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, nx ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int j = 0; j < nx; ++j )
                    for ( int k = 0; k < nx; ++k )
                    {
                        std::size_t original_id = i + j * nx + k * nx * nx;
                        ids( original_id, 0 ) = cell_id( original_id, 0 );
                        ids( original_id, 1 ) = cell_id( original_id, 1 );
                        ids( original_id, 2 ) = cell_id( original_id, 2 );
                        bin_size( i, j, k ) = cell_list.binSize( i, j, k );
                        bin_offset( i, j, k ) = cell_list.binOffset( i, j, k );
                    }
            } );
        Kokkos::fence();
        auto ids_mirror =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ids );
        auto bin_size_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), bin_size );
        auto bin_offset_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), bin_offset );

        // The order should be reversed with the i index moving the slowest
        // for those that are actually in the binning range. Do this pass
        // first. We do this by looping through in the sorted order and check
        // those that had original indices in the sorting range.
        std::size_t particle_id = 0;
        for ( int i = 0; i < nx; ++i )
        {
            for ( int j = 0; j < nx; ++j )
            {
                for ( int k = 0; k < nx; ++k )
                {
                    std::size_t original_id = i + j * nx + k * nx * nx;
                    if ( begin <= original_id && original_id < end )
                    {
                        // Get what should be the local id of the particle in
                        // the newly sorted decomposition. We are looping
                        // through this in k-fastest order (the indexing of
                        // the grid cells) and therefore should get the
                        // particles in their sorted order.
                        int sort_id = begin + particle_id;

                        // Back calculate what we think the ijk indices of
                        // the particle are based on k-fastest ordering.
                        int grid_id = i * nx * nx + j * nx + k;
                        int grid_i = grid_id / ( nx * nx );
                        int grid_j = ( grid_id / nx ) % nx;
                        int grid_k = grid_id % nx;

                        // Check the indices of the particle
                        EXPECT_EQ( ids_mirror( sort_id, 0 ), grid_i );
                        EXPECT_EQ( ids_mirror( sort_id, 1 ), grid_j );
                        EXPECT_EQ( ids_mirror( sort_id, 2 ), grid_k );

                        // Check that we binned the particle and got the right
                        // offset.
                        EXPECT_EQ( bin_size_mirror( i, j, k ), 1 );
                        EXPECT_EQ( bin_offset_mirror( i, j, k ),
                                   size_type( particle_id ) );

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
                    if ( begin > particle_id || particle_id >= end )
                    {
                        EXPECT_EQ( ids_mirror( particle_id, 0 ), i );
                        EXPECT_EQ( ids_mirror( particle_id, 1 ), j );
                        EXPECT_EQ( ids_mirror( particle_id, 2 ), k );
                        EXPECT_EQ( bin_size_mirror( i, j, k ), 0 );
                    }
                }
            }
        }
    }

    // Now bin and permute all of the particles.
    {
        Cabana::LinkedCellList<typename AoSoA_t::memory_space> cell_list(
            Cabana::slice<Position>( aosoa ), grid_delta, grid_min, grid_max );
        Cabana::permute( cell_list, aosoa );

        // Copy data to the host for testing.
        Kokkos::View<int *[3], TEST_MEMSPACE> ids( "cell_ids", num_p );
        Kokkos::View<size_type ***, TEST_MEMSPACE> bin_size( "bin_size", nx, nx,
                                                             nx );
        Kokkos::View<size_type ***, TEST_MEMSPACE> bin_offset( "bin_offset", nx,
                                                               nx, nx );
        Kokkos::parallel_for(
            "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, nx ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int j = 0; j < nx; ++j )
                    for ( int k = 0; k < nx; ++k )
                    {
                        std::size_t original_id = i + j * nx + k * nx * nx;
                        ids( original_id, 0 ) = cell_id( original_id, 0 );
                        ids( original_id, 1 ) = cell_id( original_id, 1 );
                        ids( original_id, 2 ) = cell_id( original_id, 2 );
                        bin_size( i, j, k ) = cell_list.binSize( i, j, k );
                        bin_offset( i, j, k ) = cell_list.binOffset( i, j, k );
                    }
            } );
        Kokkos::fence();
        auto ids_mirror =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ids );
        auto bin_size_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), bin_size );
        auto bin_offset_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), bin_offset );

        // Checking the binning. The order should be reversed with the i index
        // moving the slowest.
        EXPECT_EQ( cell_list.totalBins(), nx * nx * nx );
        EXPECT_EQ( cell_list.numBin( 0 ), nx );
        EXPECT_EQ( cell_list.numBin( 1 ), nx );
        EXPECT_EQ( cell_list.numBin( 2 ), nx );
        std::size_t particle_id = 0;
        for ( int i = 0; i < nx; ++i )
        {
            for ( int j = 0; j < nx; ++j )
            {
                for ( int k = 0; k < nx; ++k, ++particle_id )
                {
                    EXPECT_EQ( ids_mirror( particle_id, 0 ), i );
                    EXPECT_EQ( ids_mirror( particle_id, 1 ), j );
                    EXPECT_EQ( ids_mirror( particle_id, 2 ), k );
                    EXPECT_EQ( cell_list.cardinalBinIndex( i, j, k ),
                               particle_id );
                    EXPECT_EQ( bin_size_mirror( i, j, k ), 1 );
                    EXPECT_EQ( bin_offset_mirror( i, j, k ),
                               size_type( particle_id ) );
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
void testLinkedListSlice()
{
    // Make an AoSoA with positions and ijk cell ids.
    enum MyFields
    {
        Position = 0,
        CellId = 1
    };
    using DataTypes = Cabana::MemberTypes<double[3], int[3]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    using size_type = typename AoSoA_t::memory_space::size_type;
    int num_p = 1000;
    AoSoA_t aosoa( "aosoa", num_p );

    // Set the problem so each particle lives in the center of a cell on a
    // regular grid of cell size 1 and total size 10x10x10. We are making them
    // in the reverse order we expect the sort to happen. The sort binary
    // operator should order by i first and k last.
    int nx = 10;
    double dx = 1.0;
    double x_min = 0.0;
    double x_max = x_min + nx * dx;
    auto pos = Cabana::slice<Position>( aosoa );
    auto cell_id = Cabana::slice<CellId>( aosoa );
    Kokkos::parallel_for(
        "initialize", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, nx ),
        KOKKOS_LAMBDA( const int k ) {
            for ( int j = 0; j < nx; ++j )
            {
                for ( int i = 0; i < nx; ++i )
                {
                    std::size_t particle_id = i + j * nx + k * nx * nx;

                    cell_id( particle_id, 0 ) = i;
                    cell_id( particle_id, 1 ) = j;
                    cell_id( particle_id, 2 ) = k;

                    pos( particle_id, 0 ) = x_min + ( i + 0.5 ) * dx;
                    pos( particle_id, 1 ) = x_min + ( j + 0.5 ) * dx;
                    pos( particle_id, 2 ) = x_min + ( k + 0.5 ) * dx;
                }
            }
        } );

    // Create a grid.
    double grid_delta[3] = { dx, dx, dx };
    double grid_min[3] = { x_min, x_min, x_min };
    double grid_max[3] = { x_max, x_max, x_max };

    // Bin the particles in the grid and permute only the position slice.
    // First do this by only operating on a subset of the particles.
    {
        std::size_t begin = 250;
        std::size_t end = 750;
        Cabana::LinkedCellList<typename AoSoA_t::memory_space> cell_list(
            pos, begin, end, grid_delta, grid_min, grid_max );
        Cabana::permute( cell_list, pos );

        // Checking the binning.
        EXPECT_EQ( cell_list.totalBins(), nx * nx * nx );
        EXPECT_EQ( cell_list.numBin( 0 ), nx );
        EXPECT_EQ( cell_list.numBin( 1 ), nx );
        EXPECT_EQ( cell_list.numBin( 2 ), nx );

        // Copy data to the host for testing.
        Kokkos::View<int *[3], TEST_MEMSPACE> ids_view( "cell_ids", num_p );
        Kokkos::View<double *[3], TEST_MEMSPACE> pos_view( "positions", num_p );
        Kokkos::View<size_type ***, TEST_MEMSPACE> bin_size( "bin_size", nx, nx,
                                                             nx );
        Kokkos::View<size_type ***, TEST_MEMSPACE> bin_offset( "bin_offset", nx,
                                                               nx, nx );
        Kokkos::parallel_for(
            "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, nx ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int j = 0; j < nx; ++j )
                    for ( int k = 0; k < nx; ++k )
                    {
                        std::size_t original_id = i + j * nx + k * nx * nx;
                        ids_view( original_id, 0 ) = cell_id( original_id, 0 );
                        ids_view( original_id, 1 ) = cell_id( original_id, 1 );
                        ids_view( original_id, 2 ) = cell_id( original_id, 2 );
                        pos_view( original_id, 0 ) = pos( original_id, 0 );
                        pos_view( original_id, 1 ) = pos( original_id, 1 );
                        pos_view( original_id, 2 ) = pos( original_id, 2 );
                        bin_size( i, j, k ) = cell_list.binSize( i, j, k );
                        bin_offset( i, j, k ) = cell_list.binOffset( i, j, k );
                    }
            } );
        Kokkos::fence();
        auto ids_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), ids_view );
        auto pos_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), pos_view );
        auto bin_size_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), bin_size );
        auto bin_offset_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), bin_offset );

        // The order should be reversed with the i index moving the slowest
        // for those that are actually in the binning range. Do this pass
        // first. We do this by looping through in the sorted order and check
        // those that had original indices in the sorting range.
        std::size_t particle_id = 0;
        for ( int i = 0; i < nx; ++i )
        {
            for ( int j = 0; j < nx; ++j )
            {
                for ( int k = 0; k < nx; ++k )
                {
                    std::size_t original_id = i + j * nx + k * nx * nx;
                    if ( begin <= original_id && original_id < end )
                    {
                        int sort_id = begin + particle_id;

                        // Back calculate what we think the ijk indices of
                        // the particle are based on k-fastest ordering.
                        int grid_id = i * nx * nx + j * nx + k;
                        int grid_i = grid_id / ( nx * nx );
                        int grid_j = ( grid_id / nx ) % nx;
                        int grid_k = grid_id % nx;
                        float pos_i = x_min + ( grid_i + 0.5 ) * dx;
                        float pos_j = x_min + ( grid_j + 0.5 ) * dx;
                        float pos_k = x_min + ( grid_k + 0.5 ) * dx;

                        // Get the position of the particle in
                        // the newly sorted decomposition. We are looping
                        // through this in k-fastest order (the indexing of
                        // the grid cells) and therefore should get the
                        // particles in their sorted order.
                        EXPECT_EQ( pos_mirror( sort_id, 0 ), pos_i );
                        EXPECT_EQ( pos_mirror( sort_id, 1 ), pos_j );
                        EXPECT_EQ( pos_mirror( sort_id, 2 ), pos_k );

                        // Check that we binned the particle and got the right
                        // offset.
                        EXPECT_EQ( bin_size_mirror( i, j, k ), 1 );
                        EXPECT_EQ( bin_offset_mirror( i, j, k ),
                                   size_type( particle_id ) );

                        // Increment the particle id.
                        ++particle_id;
                    }
                }
            }
        }
        // For positions that are outside the binned range pos should be
        // unchanged and the bins should empty.
        particle_id = 0;
        for ( int k = 0; k < nx; ++k )
        {
            for ( int j = 0; j < nx; ++j )
            {
                for ( int i = 0; i < nx; ++i, ++particle_id )
                {
                    if ( begin > particle_id || particle_id >= end )
                    {
                        float pos_i = x_min + ( i + 0.5 ) * dx;
                        float pos_j = x_min + ( j + 0.5 ) * dx;
                        float pos_k = x_min + ( k + 0.5 ) * dx;

                        // Check the positions of the particle
                        EXPECT_EQ( pos_mirror( particle_id, 0 ), pos_i );
                        EXPECT_EQ( pos_mirror( particle_id, 1 ), pos_j );
                        EXPECT_EQ( pos_mirror( particle_id, 2 ), pos_k );
                        EXPECT_EQ( bin_size_mirror( i, j, k ), 0 );
                    }

                    // All IDs should be unsorted regardless of range
                    EXPECT_EQ( ids_mirror( particle_id, 0 ), i );
                    EXPECT_EQ( ids_mirror( particle_id, 1 ), j );
                    EXPECT_EQ( ids_mirror( particle_id, 2 ), k );
                }
            }
        }
    }
    // Now bin and permute all of the particles.
    {
        Cabana::LinkedCellList<typename AoSoA_t::memory_space> cell_list(
            pos, grid_delta, grid_min, grid_max );
        Cabana::permute( cell_list, pos );

        // Checking the binning. The order should be reversed with the i index
        // moving the slowest.
        EXPECT_EQ( cell_list.totalBins(), nx * nx * nx );
        EXPECT_EQ( cell_list.numBin( 0 ), nx );
        EXPECT_EQ( cell_list.numBin( 1 ), nx );
        EXPECT_EQ( cell_list.numBin( 2 ), nx );

        // Copy data to the host for testing.
        Kokkos::View<int *[3], TEST_MEMSPACE> ids_view( "cell_ids", num_p );
        Kokkos::View<double *[3], TEST_MEMSPACE> pos_view( "positions", num_p );
        Kokkos::View<size_type ***, TEST_MEMSPACE> bin_size( "bin_size", nx, nx,
                                                             nx );
        Kokkos::View<size_type ***, TEST_MEMSPACE> bin_offset( "bin_offset", nx,
                                                               nx, nx );
        Kokkos::parallel_for(
            "copy bin data", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, nx ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int j = 0; j < nx; ++j )
                    for ( int k = 0; k < nx; ++k )
                    {
                        std::size_t original_id = i + j * nx + k * nx * nx;
                        ids_view( original_id, 0 ) = cell_id( original_id, 0 );
                        ids_view( original_id, 1 ) = cell_id( original_id, 1 );
                        ids_view( original_id, 2 ) = cell_id( original_id, 2 );
                        pos_view( original_id, 0 ) = pos( original_id, 0 );
                        pos_view( original_id, 1 ) = pos( original_id, 1 );
                        pos_view( original_id, 2 ) = pos( original_id, 2 );
                        bin_size( i, j, k ) = cell_list.binSize( i, j, k );
                        bin_offset( i, j, k ) = cell_list.binOffset( i, j, k );
                    }
            } );
        Kokkos::fence();
        auto ids_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), ids_view );
        auto pos_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), pos_view );
        auto bin_size_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), bin_size );
        auto bin_offset_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), bin_offset );

        std::size_t particle_id = 0;
        for ( int i = 0; i < nx; ++i )
        {
            for ( int j = 0; j < nx; ++j )
            {
                for ( int k = 0; k < nx; ++k, ++particle_id )
                {
                    float pos_i = x_min + ( i + 0.5 ) * dx;
                    float pos_j = x_min + ( j + 0.5 ) * dx;
                    float pos_k = x_min + ( k + 0.5 ) * dx;

                    EXPECT_EQ( pos_mirror( particle_id, 0 ), pos_i );
                    EXPECT_EQ( pos_mirror( particle_id, 1 ), pos_j );
                    EXPECT_EQ( pos_mirror( particle_id, 2 ), pos_k );
                    EXPECT_EQ( cell_list.cardinalBinIndex( i, j, k ),
                               particle_id );
                    EXPECT_EQ( bin_size_mirror( i, j, k ), 1 );
                    EXPECT_EQ( bin_offset_mirror( i, j, k ),
                               size_type( particle_id ) );
                }
            }
        }
        particle_id = 0;
        for ( int k = 0; k < nx; ++k )
        {
            for ( int j = 0; j < nx; ++j )
            {
                for ( int i = 0; i < nx; ++i, ++particle_id )
                {
                    // All IDs should be unsorted
                    EXPECT_EQ( ids_mirror( particle_id, 0 ), i );
                    EXPECT_EQ( ids_mirror( particle_id, 1 ), j );
                    EXPECT_EQ( ids_mirror( particle_id, 2 ), k );
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, linked_list_test ) { testLinkedList(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, linked_list_slice_test ) { testLinkedListSlice(); }

//---------------------------------------------------------------------------//

} // end namespace Test
