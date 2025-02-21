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

#include <Cabana_Grid_Splines.hpp>

#include <vector>

#include <gtest/gtest.h>

using namespace Cabana::Grid;

namespace Test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( Splines, Zero )
{
    // Check partition of unity for the quadratic spline.
    double xp = -1.4;
    double low_x = -3.43;
    double dx = 0.27;
    double rdx = 1.0 / dx;
    double values[1];

    double x0 = Spline<0>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<0>::value( x0, values );
    double sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = 2.1789;
    x0 = Spline<0>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<0>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = low_x + 5 * dx;
    x0 = Spline<0>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<0>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    // Check the stencil by putting a point in the center of a dual cell (on a
    // node).
    int node_id = 4;
    xp = low_x + ( node_id + 0.25 ) * dx;
    x0 = Spline<0>::mapToLogicalGrid( xp, rdx, low_x );
    int offsets[1];
    Spline<0>::offsets( offsets );
    EXPECT_EQ( int( x0 ) + offsets[0], node_id );

    int stencil[1];
    Spline<0>::stencil( x0, stencil );
    EXPECT_EQ( stencil[0], node_id );

    // Check the interpolation of a function.
    auto grid_func = [=]( const double x ) { return 4.32 * x - 0.31; };
    double field[Spline<0>::num_knot];
    field[0] = grid_func( low_x + node_id * dx );
    Spline<0>::value( x0, values );
    double field_xp = field[0] * values[0];
    EXPECT_FLOAT_EQ( field_xp, field[0] );

    // Check the derivative of a function.
    Spline<0>::gradient( x0, rdx, values );
    double field_grad = field[0] * values[0];
    EXPECT_FLOAT_EQ( field_grad, 0.0 );
}

TEST( Splines, Linear )
{
    // Check partition of unity for the linear spline.
    double xp = -1.4;
    double low_x = -3.43;
    double dx = 0.27;
    double rdx = 1.0 / dx;
    double values[2];

    double x0 = Spline<1>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<1>::value( x0, values );
    double sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = 2.1789;
    x0 = Spline<1>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<1>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = low_x + 5 * dx;
    x0 = Spline<1>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<1>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    // Check the stencil by putting a point in the center of a primal cell.
    int cell_id = 4;
    xp = low_x + ( cell_id + 0.5 ) * dx;
    x0 = Spline<1>::mapToLogicalGrid( xp, rdx, low_x );
    int offsets[2];
    Spline<1>::offsets( offsets );
    EXPECT_EQ( int( x0 ) + offsets[0], cell_id );
    EXPECT_EQ( int( x0 ) + offsets[1], cell_id + 1 );

    int stencil[2];
    Spline<1>::stencil( x0, stencil );
    EXPECT_EQ( stencil[0], cell_id );
    EXPECT_EQ( stencil[1], cell_id + 1 );

    // Check the interpolation of a function.
    auto grid_func = [=]( const double x ) { return 4.32 * x - 0.31; };
    double field[Spline<1>::num_knot];
    field[0] = grid_func( low_x + cell_id * dx );
    field[1] = grid_func( low_x + ( cell_id + 1 ) * dx );
    Spline<1>::value( x0, values );
    double field_xp = field[0] * values[0] + field[1] * values[1];
    EXPECT_FLOAT_EQ( field_xp, grid_func( xp ) );

    // Check the derivative of a function.
    Spline<1>::gradient( x0, rdx, values );
    double field_grad = field[0] * values[0] + field[1] * values[1];
    auto grid_deriv = [=]( const double ) { return 4.32; };
    EXPECT_FLOAT_EQ( field_grad, grid_deriv( xp ) );
}

TEST( Splines, Quadratic )
{
    // Check partition of unity for the quadratic spline.
    double xp = -1.4;
    double low_x = -3.43;
    double dx = 0.27;
    double rdx = 1.0 / dx;
    double values[3];

    double x0 = Spline<2>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<2>::value( x0, values );
    double sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = 2.1789;
    x0 = Spline<2>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<2>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = low_x + 5 * dx;
    x0 = Spline<2>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<2>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    // Check the stencil by putting a point in the center of a dual cell (on a
    // node).
    int node_id = 4;
    xp = low_x + ( node_id + 0.25 ) * dx;
    x0 = Spline<2>::mapToLogicalGrid( xp, rdx, low_x );
    int offsets[3];
    Spline<2>::offsets( offsets );
    EXPECT_EQ( int( x0 ) + offsets[0], node_id - 1 );
    EXPECT_EQ( int( x0 ) + offsets[1], node_id );
    EXPECT_EQ( int( x0 ) + offsets[2], node_id + 1 );

    int stencil[3];
    Spline<2>::stencil( x0, stencil );
    EXPECT_EQ( stencil[0], node_id - 1 );
    EXPECT_EQ( stencil[1], node_id );
    EXPECT_EQ( stencil[2], node_id + 1 );

    // Check the interpolation of a function.
    auto grid_func = [=]( const double x ) { return 4.32 * x - 0.31; };
    double field[Spline<2>::num_knot];
    field[0] = grid_func( low_x + ( node_id - 1 ) * dx );
    field[1] = grid_func( low_x + node_id * dx );
    field[2] = grid_func( low_x + ( node_id + 1 ) * dx );
    Spline<2>::value( x0, values );
    double field_xp =
        field[0] * values[0] + field[1] * values[1] + field[2] * values[2];
    EXPECT_FLOAT_EQ( field_xp, grid_func( xp ) );

    // Check the derivative of a function.
    Spline<2>::gradient( x0, rdx, values );
    double field_grad =
        field[0] * values[0] + field[1] * values[1] + field[2] * values[2];
    auto grid_deriv = [=]( const double ) { return 4.32; };
    EXPECT_FLOAT_EQ( field_grad, grid_deriv( xp ) );
}

TEST( Splines, Cubic )
{
    // Check partition of unity for the cubic spline.
    double xp = -1.4;
    double low_x = -3.43;
    double dx = 0.27;
    double rdx = 1.0 / dx;
    double values[4];

    double x0 = Spline<3>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<3>::value( x0, values );
    double sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = 2.1789;
    x0 = Spline<3>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<3>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = low_x + 5 * dx;
    x0 = Spline<3>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<3>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    // Check the stencil by putting a point in the center of a primal cell.
    int cell_id = 4;
    xp = low_x + ( cell_id + 0.75 ) * dx;
    x0 = Spline<3>::mapToLogicalGrid( xp, rdx, low_x );
    int offsets[4];
    Spline<3>::offsets( offsets );
    EXPECT_EQ( int( x0 ) + offsets[0], cell_id - 1 );
    EXPECT_EQ( int( x0 ) + offsets[1], cell_id );
    EXPECT_EQ( int( x0 ) + offsets[2], cell_id + 1 );
    EXPECT_EQ( int( x0 ) + offsets[3], cell_id + 2 );

    int stencil[4];
    Spline<3>::stencil( x0, stencil );
    EXPECT_EQ( stencil[0], cell_id - 1 );
    EXPECT_EQ( stencil[1], cell_id );
    EXPECT_EQ( stencil[2], cell_id + 1 );
    EXPECT_EQ( stencil[3], cell_id + 2 );

    // Check the interpolation of a function.
    auto grid_func = [=]( const double x ) { return 4.32 * x - 0.31; };
    double field[Spline<3>::num_knot];
    field[0] = grid_func( low_x + ( cell_id - 1 ) * dx );
    field[1] = grid_func( low_x + cell_id * dx );
    field[2] = grid_func( low_x + ( cell_id + 1 ) * dx );
    field[3] = grid_func( low_x + ( cell_id + 2 ) * dx );
    Spline<3>::value( x0, values );
    double field_xp = field[0] * values[0] + field[1] * values[1] +
                      field[2] * values[2] + field[3] * values[3];
    EXPECT_FLOAT_EQ( field_xp, grid_func( xp ) );

    // Check the derivative of a function.
    Spline<3>::gradient( x0, rdx, values );
    double field_grad = field[0] * values[0] + field[1] * values[1] +
                        field[2] * values[2] + field[3] * values[3];
    auto grid_deriv = [=]( const double ) { return 4.32; };
    EXPECT_FLOAT_EQ( field_grad, grid_deriv( xp ) );
}

TEST( Splines, Quartic )
{
    // Check partition of unity for the quartic spline.
    double xp = -1.4;
    double low_x = -3.43;
    double dx = 0.27;
    double rdx = 1.0 / dx;
    double values[5];

    double x0 = Spline<4>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<4>::value( x0, values );
    double sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = 2.1789;
    x0 = Spline<4>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<4>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = low_x + 5 * dx;
    x0 = Spline<4>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<4>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    // Check the stencil by putting a point in the center of a primal cell.
    int node_id = 4;
    xp = low_x + ( node_id + 0.25 ) * dx;
    x0 = Spline<4>::mapToLogicalGrid( xp, rdx, low_x );
    int offsets[5];
    Spline<4>::offsets( offsets );
    EXPECT_EQ( int( x0 ) + offsets[0], node_id - 2 );
    EXPECT_EQ( int( x0 ) + offsets[1], node_id - 1 );
    EXPECT_EQ( int( x0 ) + offsets[2], node_id );
    EXPECT_EQ( int( x0 ) + offsets[3], node_id + 1 );
    EXPECT_EQ( int( x0 ) + offsets[4], node_id + 2 );

    int stencil[5];
    Spline<4>::stencil( x0, stencil );
    EXPECT_EQ( stencil[0], node_id - 2 );
    EXPECT_EQ( stencil[1], node_id - 1 );
    EXPECT_EQ( stencil[2], node_id );
    EXPECT_EQ( stencil[3], node_id + 1 );
    EXPECT_EQ( stencil[4], node_id + 2 );

    // Check the interpolation of a function.
    auto grid_func = [=]( const double x ) { return 4.32 * x - 0.31; };
    double field[Spline<4>::num_knot];
    field[0] = grid_func( low_x + ( node_id - 2 ) * dx );
    field[1] = grid_func( low_x + ( node_id - 1 ) * dx );
    field[2] = grid_func( low_x + node_id * dx );
    field[3] = grid_func( low_x + ( node_id + 1 ) * dx );
    field[4] = grid_func( low_x + ( node_id + 2 ) * dx );
    Spline<4>::value( x0, values );
    double field_xp = field[0] * values[0] + field[1] * values[1] +
                      field[2] * values[2] + field[3] * values[3] +
                      field[4] * values[4];
    EXPECT_FLOAT_EQ( field_xp, grid_func( xp ) );

    // Check the derivative of a function.
    Spline<4>::gradient( x0, rdx, values );
    double field_grad = field[0] * values[0] + field[1] * values[1] +
                        field[2] * values[2] + field[3] * values[3] +
                        field[4] * values[4];
    auto grid_deriv = [=]( const double ) { return 4.32; };
    EXPECT_FLOAT_EQ( field_grad, grid_deriv( xp ) );
}

TEST( Splines, Quintic )
{
    // Check partition of unity for the quintic spline.
    double xp = -1.4;
    double low_x = -3.43;
    double dx = 0.27;
    double rdx = 1.0 / dx;
    double values[6];

    double x0 = Spline<5>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<5>::value( x0, values );
    double sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = 2.1789;
    x0 = Spline<5>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<5>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = low_x + 5 * dx;
    x0 = Spline<5>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<5>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    // Check the stencil by putting a point in the center of a primal cell.
    int cell_id = 4;
    xp = low_x + ( cell_id + 0.75 ) * dx;
    x0 = Spline<5>::mapToLogicalGrid( xp, rdx, low_x );
    int offsets[6];
    Spline<5>::offsets( offsets );
    EXPECT_EQ( int( x0 ) + offsets[0], cell_id - 2 );
    EXPECT_EQ( int( x0 ) + offsets[1], cell_id - 1 );
    EXPECT_EQ( int( x0 ) + offsets[2], cell_id );
    EXPECT_EQ( int( x0 ) + offsets[3], cell_id + 1 );
    EXPECT_EQ( int( x0 ) + offsets[4], cell_id + 2 );
    EXPECT_EQ( int( x0 ) + offsets[5], cell_id + 3 );

    int stencil[6];
    Spline<5>::stencil( x0, stencil );
    EXPECT_EQ( stencil[0], cell_id - 2 );
    EXPECT_EQ( stencil[1], cell_id - 1 );
    EXPECT_EQ( stencil[2], cell_id );
    EXPECT_EQ( stencil[3], cell_id + 1 );
    EXPECT_EQ( stencil[4], cell_id + 2 );
    EXPECT_EQ( stencil[5], cell_id + 3 );

    // Check the interpolation of a function.
    auto grid_func = [=]( const double x ) { return 4.32 * x - 0.31; };
    double field[Spline<5>::num_knot];
    field[0] = grid_func( low_x + ( cell_id - 2 ) * dx );
    field[1] = grid_func( low_x + ( cell_id - 1 ) * dx );
    field[2] = grid_func( low_x + cell_id * dx );
    field[3] = grid_func( low_x + ( cell_id + 1 ) * dx );
    field[4] = grid_func( low_x + ( cell_id + 2 ) * dx );
    field[5] = grid_func( low_x + ( cell_id + 3 ) * dx );
    Spline<5>::value( x0, values );
    double field_xp = field[0] * values[0] + field[1] * values[1] +
                      field[2] * values[2] + field[3] * values[3] +
                      field[4] * values[4] + field[5] * values[5];
    EXPECT_FLOAT_EQ( field_xp, grid_func( xp ) );

    // Check the derivative of a function.
    Spline<5>::gradient( x0, rdx, values );
    double field_grad = field[0] * values[0] + field[1] * values[1] +
                        field[2] * values[2] + field[3] * values[3] +
                        field[4] * values[4] + field[5] * values[5];
    auto grid_deriv = [=]( const double ) { return 4.32; };
    EXPECT_FLOAT_EQ( field_grad, grid_deriv( xp ) );
}

TEST( Splines, Sextic )
{
    // Check partition of unity for the sextic spline.
    double xp = -1.4;
    double low_x = -3.43;
    double dx = 0.27;
    double rdx = 1.0 / dx;
    double values[7];

    double x0 = Spline<6>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<6>::value( x0, values );
    double sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = 2.1789;
    x0 = Spline<6>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<6>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    xp = low_x + 5 * dx;
    x0 = Spline<6>::mapToLogicalGrid( xp, rdx, low_x );
    Spline<6>::value( x0, values );
    sum = 0.0;
    for ( auto x : values )
        sum += x;
    EXPECT_FLOAT_EQ( sum, 1.0 );

    // Check the stencil by putting a point in the center of a primal cell.
    int node_id = 4;
    xp = low_x + ( node_id + 0.25 ) * dx;
    x0 = Spline<6>::mapToLogicalGrid( xp, rdx, low_x );
    int offsets[7];
    Spline<6>::offsets( offsets );
    EXPECT_EQ( int( x0 ) + offsets[0], node_id - 3 );
    EXPECT_EQ( int( x0 ) + offsets[1], node_id - 2 );
    EXPECT_EQ( int( x0 ) + offsets[2], node_id - 1 );
    EXPECT_EQ( int( x0 ) + offsets[3], node_id );
    EXPECT_EQ( int( x0 ) + offsets[4], node_id + 1 );
    EXPECT_EQ( int( x0 ) + offsets[5], node_id + 2 );
    EXPECT_EQ( int( x0 ) + offsets[6], node_id + 3 );

    int stencil[7];
    Spline<6>::stencil( x0, stencil );
    EXPECT_EQ( stencil[0], node_id - 3 );
    EXPECT_EQ( stencil[1], node_id - 2 );
    EXPECT_EQ( stencil[2], node_id - 1 );
    EXPECT_EQ( stencil[3], node_id );
    EXPECT_EQ( stencil[4], node_id + 1 );
    EXPECT_EQ( stencil[5], node_id + 2 );
    EXPECT_EQ( stencil[6], node_id + 3 );

    // Check the interpolation of a function.
    auto grid_func = [=]( const double x ) { return 4.32 * x - 0.31; };
    double field[Spline<6>::num_knot];
    field[0] = grid_func( low_x + ( node_id - 3 ) * dx );
    field[1] = grid_func( low_x + ( node_id - 2 ) * dx );
    field[2] = grid_func( low_x + ( node_id - 1 ) * dx );
    field[3] = grid_func( low_x + node_id * dx );
    field[4] = grid_func( low_x + ( node_id + 1 ) * dx );
    field[5] = grid_func( low_x + ( node_id + 2 ) * dx );
    field[6] = grid_func( low_x + ( node_id + 3 ) * dx );
    Spline<6>::value( x0, values );
    double field_xp = field[0] * values[0] + field[1] * values[1] +
                      field[2] * values[2] + field[3] * values[3] +
                      field[4] * values[4] + field[5] * values[5] +
                      field[6] * values[6];
    EXPECT_FLOAT_EQ( field_xp, grid_func( xp ) );

    // Check the derivative of a function.
    Spline<6>::gradient( x0, rdx, values );
    double field_grad = field[0] * values[0] + field[1] * values[1] +
                        field[2] * values[2] + field[3] * values[3] +
                        field[4] * values[4] + field[5] * values[5] +
                        field[6] * values[6];
    auto grid_deriv = [=]( const double ) { return 4.32; };
    EXPECT_FLOAT_EQ( field_grad, grid_deriv( xp ) );
}

} // end namespace Test
