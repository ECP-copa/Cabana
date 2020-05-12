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

#include <Cajita_GlobalMesh.hpp>
#include <Cajita_Types.hpp>

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
// Test uniform mesh with cubic cells.
void uniformTest1()
{
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high_corner = { -0.3, 9.5, 1.3 };
    double cell_size = 0.05;

    auto global_mesh =
        createUniformGlobalMesh( low_corner, high_corner, cell_size );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( low_corner[d], global_mesh->lowCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d], global_mesh->highCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d] - low_corner[d],
                          global_mesh->extent( d ) );

    std::array<int, 3> num_cell = { 18, 188, 4 };
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( num_cell[d], global_mesh->globalNumCell( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( global_mesh->cellSize( d ), cell_size );
}

//---------------------------------------------------------------------------//
// Test uniform mesh with number of cells constructor.
void uniformTest2()
{
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high_corner = { -0.3, 9.5, 1.3 };
    std::array<int, 3> num_cell = { 18, 188, 4 };

    auto global_mesh =
        createUniformGlobalMesh( low_corner, high_corner, num_cell );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( low_corner[d], global_mesh->lowCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d], global_mesh->highCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d] - low_corner[d],
                          global_mesh->extent( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( num_cell[d], global_mesh->globalNumCell( d ) );

    double cell_size = 0.05;
    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( global_mesh->cellSize( d ), cell_size );
}

//---------------------------------------------------------------------------//
// test uniform mesh with cells that can have a different size in each
// dimension
void uniformTest3()
{
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high_corner = { -0.3, 9.5, 1.3 };
    std::array<double, 3> cell_size = { 0.05, 0.05, 0.05 };

    auto global_mesh =
        createUniformGlobalMesh( low_corner, high_corner, cell_size );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( low_corner[d], global_mesh->lowCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d], global_mesh->highCorner( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( high_corner[d] - low_corner[d],
                          global_mesh->extent( d ) );

    std::array<int, 3> num_cell = { 18, 188, 4 };
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( num_cell[d], global_mesh->globalNumCell( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_DOUBLE_EQ( global_mesh->cellSize( d ), cell_size[d] );
}

//---------------------------------------------------------------------------//
// test a non uniform mesh
void nonUniformTest()
{
    std::vector<float> i_edge = { -0.3, 0.4, 1.1 };
    std::vector<float> j_edge = { 3.3, 8.1, 9.5, 12.2 };
    std::vector<float> k_edge = { -1.1, -0.9, 0.4, 8.8, 19.3 };

    auto global_mesh = createNonUniformGlobalMesh( i_edge, j_edge, k_edge );

    EXPECT_FLOAT_EQ( i_edge.front(), global_mesh->lowCorner( Dim::I ) );
    EXPECT_FLOAT_EQ( j_edge.front(), global_mesh->lowCorner( Dim::J ) );
    EXPECT_FLOAT_EQ( k_edge.front(), global_mesh->lowCorner( Dim::K ) );

    EXPECT_FLOAT_EQ( i_edge.back(), global_mesh->highCorner( Dim::I ) );
    EXPECT_FLOAT_EQ( j_edge.back(), global_mesh->highCorner( Dim::J ) );
    EXPECT_FLOAT_EQ( k_edge.back(), global_mesh->highCorner( Dim::K ) );

    EXPECT_FLOAT_EQ( i_edge.back() - i_edge.front(),
                     global_mesh->extent( Dim::I ) );
    EXPECT_FLOAT_EQ( j_edge.back() - j_edge.front(),
                     global_mesh->extent( Dim::J ) );
    EXPECT_FLOAT_EQ( k_edge.back() - k_edge.front(),
                     global_mesh->extent( Dim::K ) );

    EXPECT_EQ( 2, global_mesh->globalNumCell( Dim::I ) );
    EXPECT_EQ( 3, global_mesh->globalNumCell( Dim::J ) );
    EXPECT_EQ( 4, global_mesh->globalNumCell( Dim::K ) );

    const auto &mesh_i = global_mesh->nonUniformEdge( Dim::I );
    int ni = mesh_i.size();
    for ( int i = 0; i < ni; ++i )
        EXPECT_FLOAT_EQ( i_edge[i], mesh_i[i] );

    const auto &mesh_j = global_mesh->nonUniformEdge( Dim::J );
    int nj = mesh_j.size();
    for ( int j = 0; j < nj; ++j )
        EXPECT_FLOAT_EQ( j_edge[j], mesh_j[j] );

    const auto &mesh_k = global_mesh->nonUniformEdge( Dim::K );
    int nk = mesh_k.size();
    for ( int k = 0; k < nk; ++k )
        EXPECT_FLOAT_EQ( k_edge[k], mesh_k[k] );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( mesh, uniform_test )
{
    uniformTest1();
    uniformTest2();
    uniformTest3();
}

TEST( mesh, non_uniform_test ) { nonUniformTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
