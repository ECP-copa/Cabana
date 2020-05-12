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

#include <Kokkos_Core.hpp>

#include <Cajita_Array.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_Interpolation.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>
#include <Cajita_Splines.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cmath>
#include <vector>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void interpolationTest()
{
    // Create the global mesh.
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high_corner = { -0.3, 9.5, 2.3 };
    double cell_size = 0.05;
    auto global_mesh =
        createUniformGlobalMesh( low_corner, high_corner, cell_size );

    // Create the global grid.
    UniformDimPartitioner partitioner;
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a  grid local_grid.
    int halo_width = 1;
    auto local_grid = createLocalGrid( global_grid, halo_width );
    auto local_mesh = createLocalMesh<TEST_DEVICE>( *local_grid );

    // Create a point in the center of every cell.
    auto cell_space = local_grid->indexSpace( Own(), Cell(), Local() );
    int num_point = cell_space.size();
    Kokkos::View<double *[3], TEST_DEVICE> points(
        Kokkos::ViewAllocateWithoutInitializing( "points" ), num_point );
    Kokkos::parallel_for(
        "fill_points", createExecutionPolicy( cell_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            int pi = i - halo_width;
            int pj = j - halo_width;
            int pk = k - halo_width;
            int pid = pi + cell_space.extent( Dim::I ) *
                               ( pj + cell_space.extent( Dim::J ) * pk );
            int idx[3] = { i, j, k };
            double x[3];
            local_mesh.coordinates( Cell(), idx, x );
            points( pid, Dim::I ) = x[Dim::I];
            points( pid, Dim::J ) = x[Dim::J];
            points( pid, Dim::K ) = x[Dim::K];
        } );

    // Create a scalar field on the grid.
    auto scalar_layout = createArrayLayout( local_grid, 1, Node() );
    auto scalar_grid_field =
        createArray<double, TEST_DEVICE>( "scalar_grid_field", scalar_layout );
    auto scalar_halo = createHalo( *scalar_grid_field, FullHaloPattern() );
    auto scalar_grid_host =
        Kokkos::create_mirror_view( scalar_grid_field->view() );

    // Create a vector field on the grid.
    auto vector_layout = createArrayLayout( local_grid, 3, Node() );
    auto vector_grid_field =
        createArray<double, TEST_DEVICE>( "vector_grid_field", vector_layout );
    auto vector_halo = createHalo( *vector_grid_field, FullHaloPattern() );
    auto vector_grid_host =
        Kokkos::create_mirror_view( vector_grid_field->view() );

    // Create a scalar point field.
    Kokkos::View<double *, TEST_DEVICE> scalar_point_field(
        Kokkos::ViewAllocateWithoutInitializing( "scalar_point_field" ),
        num_point );
    auto scalar_point_host = Kokkos::create_mirror_view( scalar_point_field );

    // Create a vector point field.
    Kokkos::View<double *[3], TEST_DEVICE> vector_point_field(
        Kokkos::ViewAllocateWithoutInitializing( "vector_point_field" ),
        num_point );
    auto vector_point_host = Kokkos::create_mirror_view( vector_point_field );

    // Create a tensor point field.
    Kokkos::View<double *[3][3], TEST_DEVICE> tensor_point_field(
        Kokkos::ViewAllocateWithoutInitializing( "tensor_point_field" ),
        num_point );
    auto tensor_point_host = Kokkos::create_mirror_view( tensor_point_field );

    // P2G
    // ---

    Kokkos::deep_copy( scalar_point_field, 3.5 );
    Kokkos::deep_copy( vector_point_field, 3.5 );
    Kokkos::deep_copy( tensor_point_field, 3.5 );

    // Interpolate a scalar point value to the grid.
    ArrayOp::assign( *scalar_grid_field, 0.0, Ghost() );
    auto scalar_p2g = createScalarValueP2G( scalar_point_field, -0.5 );
    p2g( scalar_p2g, points, num_point, Spline<1>(), *scalar_halo,
         *scalar_grid_field );
    Kokkos::deep_copy( scalar_grid_host, scalar_grid_field->view() );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
                EXPECT_FLOAT_EQ( scalar_grid_host( i, j, k, 0 ), -1.75 );

    // Interpolate a vector point value to the grid.
    ArrayOp::assign( *vector_grid_field, 0.0, Ghost() );
    auto vector_p2g = createVectorValueP2G( vector_point_field, -0.5 );
    p2g( vector_p2g, points, num_point, Spline<1>(), *vector_halo,
         *vector_grid_field );
    Kokkos::deep_copy( vector_grid_host, vector_grid_field->view() );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
                for ( int d = 0; d < 3; ++d )
                    EXPECT_FLOAT_EQ( vector_grid_host( i, j, k, d ), -1.75 );

    // Interpolate a scalar point gradient value to the grid.
    ArrayOp::assign( *vector_grid_field, 0.0, Ghost() );
    auto scalar_grad_p2g = createScalarGradientP2G( scalar_point_field, -0.5 );
    p2g( scalar_grad_p2g, points, num_point, Spline<1>(), *vector_halo,
         *vector_grid_field );
    Kokkos::deep_copy( vector_grid_host, vector_grid_field->view() );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
                for ( int d = 0; d < 3; ++d )
                    EXPECT_FLOAT_EQ( vector_grid_host( i, j, k, d ) + 1.0,
                                     1.0 );

    // Interpolate a vector point divergence value to the grid.
    ArrayOp::assign( *scalar_grid_field, 0.0, Ghost() );
    auto vector_div_p2g = createVectorDivergenceP2G( vector_point_field, -0.5 );
    p2g( vector_div_p2g, points, num_point, Spline<1>(), *scalar_halo,
         *scalar_grid_field );
    Kokkos::deep_copy( scalar_grid_host, scalar_grid_field->view() );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
                EXPECT_FLOAT_EQ( scalar_grid_host( i, j, k, 0 ) + 1.0, 1.0 );

    // Interpolate a tensor point divergence value to the grid.
    ArrayOp::assign( *vector_grid_field, 0.0, Ghost() );
    auto tensor_div_p2g = createTensorDivergenceP2G( tensor_point_field, -0.5 );
    p2g( tensor_div_p2g, points, num_point, Spline<1>(), *vector_halo,
         *vector_grid_field );
    Kokkos::deep_copy( vector_grid_host, vector_grid_field->view() );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
                for ( int d = 0; d < 3; ++d )
                    EXPECT_FLOAT_EQ( vector_grid_host( i, j, k, d ) + 1.0,
                                     1.0 );

    // G2P
    // ---

    ArrayOp::assign( *scalar_grid_field, 3.5, Own() );
    ArrayOp::assign( *vector_grid_field, 3.5, Own() );

    // Interpolate a scalar grid value to the points.
    Kokkos::deep_copy( scalar_point_field, 0.0 );
    auto scalar_value_g2p = createScalarValueG2P( scalar_point_field, -0.5 );
    g2p( *scalar_grid_field, *scalar_halo, points, num_point, Spline<1>(),
         scalar_value_g2p );
    Kokkos::deep_copy( scalar_point_host, scalar_point_field );
    for ( int p = 0; p < num_point; ++p )
        EXPECT_FLOAT_EQ( scalar_point_host( p ), -1.75 );

    // Interpolate a vector grid value to the points.
    Kokkos::deep_copy( vector_point_field, 0.0 );
    auto vector_value_g2p = createVectorValueG2P( vector_point_field, -0.5 );
    g2p( *vector_grid_field, *vector_halo, points, num_point, Spline<1>(),
         vector_value_g2p );
    Kokkos::deep_copy( vector_point_host, vector_point_field );
    for ( int p = 0; p < num_point; ++p )
        for ( int d = 0; d < 3; ++d )
            EXPECT_FLOAT_EQ( vector_point_host( p, d ), -1.75 );

    // Interpolate a scalar grid gradient to the points.
    Kokkos::deep_copy( vector_point_field, 0.0 );
    auto scalar_gradient_g2p =
        createScalarGradientG2P( vector_point_field, -0.5 );
    g2p( *scalar_grid_field, *scalar_halo, points, num_point, Spline<1>(),
         scalar_gradient_g2p );
    Kokkos::deep_copy( vector_point_host, vector_point_field );
    for ( int p = 0; p < num_point; ++p )
        for ( int d = 0; d < 3; ++d )
            EXPECT_FLOAT_EQ( vector_point_host( p, d ) + 1.0, 1.0 );

    // Interpolate a vector grid gradient to the points.
    Kokkos::deep_copy( tensor_point_field, 0.0 );
    auto vector_gradient_g2p =
        createVectorGradientG2P( tensor_point_field, -0.5 );
    g2p( *vector_grid_field, *vector_halo, points, num_point, Spline<1>(),
         vector_gradient_g2p );
    Kokkos::deep_copy( tensor_point_host, tensor_point_field );
    for ( int p = 0; p < num_point; ++p )
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_FLOAT_EQ( tensor_point_host( p, i, j ) + 1.0, 1.0 );

    // Interpolate a vector grid divergence to the points.
    Kokkos::deep_copy( scalar_point_field, 0.0 );
    auto vector_div_g2p = createVectorDivergenceG2P( scalar_point_field, -0.5 );
    g2p( *vector_grid_field, *vector_halo, points, num_point, Spline<1>(),
         vector_div_g2p );
    Kokkos::deep_copy( scalar_point_host, scalar_point_field );
    for ( int p = 0; p < num_point; ++p )
        EXPECT_FLOAT_EQ( scalar_point_host( p ) + 1.0, 1.0 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( interpolation, interpolation_test ) { interpolationTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
