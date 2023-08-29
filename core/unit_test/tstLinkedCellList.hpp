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

#include <gtest/gtest.h>

namespace Test
{
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

void copyListToHost( LCLTestData& test_data,
                     const Cabana::LinkedCellList<TEST_MEMSPACE> cell_list )
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

void checkBins( const LCLTestData test_data,
                const Cabana::LinkedCellList<TEST_MEMSPACE> cell_list )
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
        Cabana::LinkedCellList<TEST_MEMSPACE> cell_list(
            pos, begin, end, grid_delta, grid_min, grid_max );
        Cabana::permute( cell_list, test_data.aosoa );

        copyListToHost( test_data, cell_list );

        checkBins( test_data, cell_list );
        checkLinkedCell( test_data, begin, end, true );
    }

    // Now bin and permute all of the particles.
    {
        Cabana::LinkedCellList<TEST_MEMSPACE> cell_list( pos, grid_delta,
                                                         grid_min, grid_max );
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
        Cabana::LinkedCellList<TEST_MEMSPACE> cell_list(
            pos, begin, end, grid_delta, grid_min, grid_max );
        Cabana::permute( cell_list, pos );

        copyListToHost( test_data, cell_list );

        checkBins( test_data, cell_list );
        checkLinkedCell( test_data, begin, end, false );
    }
    // Now bin and permute all of the particles.
    {
        Cabana::LinkedCellList<TEST_MEMSPACE> cell_list( pos, grid_delta,
                                                         grid_min, grid_max );
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

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, linked_list_test ) { testLinkedList(); }

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, linked_list_slice_test ) { testLinkedListSlice(); }

//---------------------------------------------------------------------------//

} // end namespace Test
