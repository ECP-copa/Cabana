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
void testFullLinkedList()
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

    // Bin the particles in the grid.
    double grid_delta[3] = {dx,dx,dx};
    double grid_min[3] = {x_min,x_min,x_min};
    double grid_max[3] = {x_max,x_max,x_max};
    Cabana::LinkedCellList<typename AoSoA_t::memory_space>
        bin_data( aosoa.slice<Position>(), grid_delta, grid_min, grid_max );
    Cabana::permute( bin_data.permuteVector(), aosoa );

    // Checking the binning. The order should be reversed with the i index
    // moving the slowest.
    EXPECT_EQ( bin_data.totalBins(), nx*nx*nx );
    EXPECT_EQ( bin_data.numBin(0), nx );
    EXPECT_EQ( bin_data.numBin(1), nx );
    EXPECT_EQ( bin_data.numBin(2), nx );
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
                EXPECT_EQ( bin_data.cardinalBinIndex(i,j,k), particle_id );
                EXPECT_EQ( bin_data.binSize(i,j,k), 1 );
                EXPECT_EQ( bin_data.binOffset(i,j,k),
                           size_type(particle_id) );
            }
        }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST_F( TEST_CATEGORY, full_linked_list_test )
{
    testFullLinkedList();
}

//---------------------------------------------------------------------------//

} // end namespace Test
