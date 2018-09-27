/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_AoSoA.hpp>
#include <Cabana_LinkedCellList.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
void testLinkedList()
{
    // Make an AoSoA with positions and ijk cell ids.
    enum MyFields { Position = 0, CellId = 1 };
    using DataTypes = Cabana::MemberTypes<double[3],int[3]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    using size_type =
        typename AoSoA_t::memory_space::kokkos_memory_space::size_type;
    int num_p = 1000;
    AoSoA_t aosoa( num_p );

    // Set the problem so each particle lives in the center of a cell on a
    // regular grid of cell size 1 and total size 10x10x10. We are making them
    // in the reverse order we expect the sort to happen. The sort binary
    // operator should order by i first and k last.
    int nx = 10;
    double dx = 1.0;
    double x_min = 0.0;
    double x_max = x_min + nx * dx;
    auto pos = aosoa.slice<Position>();
    auto cell_id = aosoa.slice<CellId>();
    std::size_t particle_id = 0;
    for ( int k = 0; k < nx; ++k )
    {
        for ( int j = 0; j < nx; ++j )
        {
            for ( int i = 0; i < nx; ++i, ++particle_id )
            {
                cell_id( particle_id, 0 ) = i;
                cell_id( particle_id, 1 ) = j;
                cell_id( particle_id, 2 ) = k;

                pos( particle_id, 0 ) = x_min + (i + 0.5) * dx;
                pos( particle_id, 1 ) = x_min + (j + 0.5) * dx;
                pos( particle_id, 2 ) = x_min + (k + 0.5) * dx;
            }
        }
    }

    // Create a grid.
    double grid_delta[3] = {dx,dx,dx};
    double grid_min[3] = {x_min,x_min,x_min};
    double grid_max[3] = {x_max,x_max,x_max};

    // Bin and permute the particles in the grid. First do this by only
    // operating on a subset of the particles.
    {
        std::size_t begin = 250;
        std::size_t end = 750;
        Cabana::LinkedCellList<typename AoSoA_t::memory_space>
            cell_list( aosoa.slice<Position>(), begin, end,
                       grid_delta, grid_min, grid_max );
        Cabana::permute( cell_list, aosoa );

        // Checking the binning.
        EXPECT_EQ( cell_list.totalBins(), nx*nx*nx );
        EXPECT_EQ( cell_list.numBin(0), nx );
        EXPECT_EQ( cell_list.numBin(1), nx );
        EXPECT_EQ( cell_list.numBin(2), nx );

        // The order should be reversed with the i index moving the slowest
        // for those that are actually in the binning range. Do this pass
        // first. We do this by looping through in the sorted order and check
        // those that had original indices in the sorting range.
        particle_id = 0;
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
                        int grid_i = grid_id / (nx*nx);
                        int grid_j = (grid_id / nx) % nx;
                        int grid_k = grid_id % nx;

                        // Check the indices of the particle
                        EXPECT_EQ( cell_id( sort_id, 0 ), grid_i );
                        EXPECT_EQ( cell_id( sort_id, 1 ), grid_j );
                        EXPECT_EQ( cell_id( sort_id, 2 ), grid_k );

                        // Check that we binned the particle and got the right
                        // offset.
                        EXPECT_EQ( cell_list.binSize(i,j,k), 1 );
                        EXPECT_EQ( cell_list.binOffset(i,j,k),
                                   size_type(particle_id) );

                        // Increment the particle id.
                        ++particle_id;
                    }
                }
            }
        }

        // For those that are outside the binned range things should be
        // unchanged and the bins should be unchanged.
        particle_id = 0;
        for ( int k = 0; k < nx; ++k )
        {
            for ( int j = 0; j < nx; ++j )
            {
                for ( int i = 0; i < nx; ++i, ++particle_id )
                {
                    if ( begin > particle_id || particle_id >= end )
                    {
                        EXPECT_EQ( cell_id( particle_id, 0 ), i );
                        EXPECT_EQ( cell_id( particle_id, 1 ), j );
                        EXPECT_EQ( cell_id( particle_id, 2 ), k );
                        EXPECT_EQ( cell_list.binSize(i,j,k), 0 );
                    }
                }
            }
        }
    }

    // Now bin and permute all of the particles.
    {
        Cabana::LinkedCellList<typename AoSoA_t::memory_space>
            cell_list( aosoa.slice<Position>(), grid_delta, grid_min, grid_max );
        Cabana::permute( cell_list, aosoa );

        // Checking the binning. The order should be reversed with the i index
        // moving the slowest.
        EXPECT_EQ( cell_list.totalBins(), nx*nx*nx );
        EXPECT_EQ( cell_list.numBin(0), nx );
        EXPECT_EQ( cell_list.numBin(1), nx );
        EXPECT_EQ( cell_list.numBin(2), nx );
        particle_id = 0;
        for ( int i = 0; i < nx; ++i )
        {
            for ( int j = 0; j < nx; ++j )
            {
                for ( int k = 0; k < nx; ++k, ++particle_id )
                {
                    EXPECT_EQ( cell_id( particle_id, 0 ), i );
                    EXPECT_EQ( cell_id( particle_id, 1 ), j );
                    EXPECT_EQ( cell_id( particle_id, 2 ), k );
                    EXPECT_EQ( cell_list.cardinalBinIndex(i,j,k), particle_id );
                    EXPECT_EQ( cell_list.binSize(i,j,k), 1 );
                    EXPECT_EQ( cell_list.binOffset(i,j,k),
                               size_type(particle_id) );
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, linked_list_test )
{
    testLinkedList();
}

//---------------------------------------------------------------------------//

} // end namespace Test
