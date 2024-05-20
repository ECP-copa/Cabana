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

#include <impl/Cabana_CartesianGrid.hpp>

#include <gtest/gtest.h>

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
namespace Test
{

TEST( cabana_cartesian_grid, grid_test )
{
    double min[3] = { -1.0, -0.5, -0.6 };
    double max[3] = { 2.5, 1.5, 1.9 };
    double delta[3] = { 0.5, 0.125, 0.25 };

    Cabana::Impl::CartesianGrid<double> grid( min[0], min[1], min[2], max[0],
                                              max[1], max[2], delta[0],
                                              delta[1], delta[2] );

    int nx, ny, nz;
    grid.numCells( nx, ny, nz );
    EXPECT_EQ( nx, 7 );
    EXPECT_EQ( ny, 16 );
    EXPECT_EQ( nz, 10 );
    auto total = grid.totalNumCells();
    EXPECT_EQ( total, nx * ny * nz );

    double xp = -0.9;
    double yp = 1.4;
    double zp = 0.1;
    int ic, jc, kc;
    grid.locatePoint( xp, yp, zp, ic, jc, kc );
    EXPECT_EQ( ic, 0 );
    EXPECT_EQ( jc, 15 );
    EXPECT_EQ( kc, 2 );

    double min_dist = grid.minDistanceToPoint( xp, yp, zp, ic, jc, kc );
    EXPECT_DOUBLE_EQ( min_dist, 0.0 );

    xp = 2.5;
    yp = 1.5;
    zp = 1.9;
    grid.locatePoint( xp, yp, zp, ic, jc, kc );
    EXPECT_EQ( ic, 6 );
    EXPECT_EQ( jc, 15 );
    EXPECT_EQ( kc, 9 );
}

} // end namespace Test
