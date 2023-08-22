/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Kokkos_Core.hpp>

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_Slice.hpp>

#include <Cajita_Array.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_Interpolation.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>
#include <Cajita_Partitioner.hpp>
#include <Cajita_Splines.hpp>
#include <Cajita_Types.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cmath>
#include <vector>

using namespace Cajita;

namespace Test
{

template <class PositionType, class ScalarPoint, class VectorPoint,
          class TensorPoint, class ScalarGrid, class VectorGrid,
          class NodeIndexSpace, class ScalarHalo, class VectorHalo>
void checkP2G( const PositionType points, ScalarPoint scalar_point_field,
               VectorPoint vector_point_field, TensorPoint tensor_point_field,
               ScalarGrid scalar_grid_field, VectorGrid vector_grid_field,
               const NodeIndexSpace node_space, const ScalarHalo scalar_halo,
               const VectorHalo vector_halo, const std::size_t num_point )
{
    // Interpolate a scalar point value to the grid.
    ArrayOp::assign( *scalar_grid_field, 0.0, Ghost() );
    auto scalar_p2g = createScalarValueP2G( scalar_point_field, -0.5 );
    p2g( scalar_p2g, points, num_point, Spline<1>(), *scalar_halo,
         *scalar_grid_field );
    auto scalar_grid_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_grid_field->view() );
    for ( int i = node_space.min( Dim::I ); i < node_space.max( Dim::I ); ++i )
        for ( int j = node_space.min( Dim::J ); j < node_space.max( Dim::J );
              ++j )
            for ( int k = node_space.min( Dim::K );
                  k < node_space.max( Dim::K ); ++k )
                EXPECT_FLOAT_EQ( scalar_grid_host( i, j, k, 0 ), -1.75 );

    // Interpolate a vector point value to the grid.
    ArrayOp::assign( *vector_grid_field, 0.0, Ghost() );
    auto vector_p2g = createVectorValueP2G( vector_point_field, -0.5 );
    p2g( vector_p2g, points, num_point, Spline<1>(), *vector_halo,
         *vector_grid_field );
    auto vector_grid_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), vector_grid_field->view() );
    for ( int i = node_space.min( Dim::I ); i < node_space.max( Dim::I ); ++i )
        for ( int j = node_space.min( Dim::J ); j < node_space.max( Dim::J );
              ++j )
            for ( int k = node_space.min( Dim::K );
                  k < node_space.max( Dim::K ); ++k )
                for ( int d = 0; d < 3; ++d )
                    EXPECT_FLOAT_EQ( vector_grid_host( i, j, k, d ), -1.75 );

    // Interpolate a scalar point gradient value to the grid.
    ArrayOp::assign( *vector_grid_field, 0.0, Ghost() );
    auto scalar_grad_p2g = createScalarGradientP2G( scalar_point_field, -0.5 );
    p2g( scalar_grad_p2g, points, num_point, Spline<1>(), *vector_halo,
         *vector_grid_field );
    vector_grid_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), vector_grid_field->view() );
    for ( int i = node_space.min( Dim::I ); i < node_space.max( Dim::I ); ++i )
        for ( int j = node_space.min( Dim::J ); j < node_space.max( Dim::J );
              ++j )
            for ( int k = node_space.min( Dim::K );
                  k < node_space.max( Dim::K ); ++k )
                for ( int d = 0; d < 3; ++d )
                    EXPECT_FLOAT_EQ( vector_grid_host( i, j, k, d ) + 1.0,
                                     1.0 );

    // Interpolate a vector point divergence value to the grid.
    ArrayOp::assign( *scalar_grid_field, 0.0, Ghost() );
    auto vector_div_p2g = createVectorDivergenceP2G( vector_point_field, -0.5 );
    p2g( vector_div_p2g, points, num_point, Spline<1>(), *scalar_halo,
         *scalar_grid_field );
    scalar_grid_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_grid_field->view() );
    for ( int i = node_space.min( Dim::I ); i < node_space.max( Dim::I ); ++i )
        for ( int j = node_space.min( Dim::J ); j < node_space.max( Dim::J );
              ++j )
            for ( int k = node_space.min( Dim::K );
                  k < node_space.max( Dim::K ); ++k )
                EXPECT_FLOAT_EQ( scalar_grid_host( i, j, k, 0 ) + 1.0, 1.0 );

    // Interpolate a tensor point divergence value to the grid.
    ArrayOp::assign( *vector_grid_field, 0.0, Ghost() );
    auto tensor_div_p2g = createTensorDivergenceP2G( tensor_point_field, -0.5 );
    p2g( tensor_div_p2g, points, num_point, Spline<1>(), *vector_halo,
         *vector_grid_field );
    vector_grid_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), vector_grid_field->view() );
    for ( int i = node_space.min( Dim::I ); i < node_space.max( Dim::I ); ++i )
        for ( int j = node_space.min( Dim::J ); j < node_space.max( Dim::J );
              ++j )
            for ( int k = node_space.min( Dim::K );
                  k < node_space.max( Dim::K ); ++k )
                for ( int d = 0; d < 3; ++d )
                    EXPECT_FLOAT_EQ( vector_grid_host( i, j, k, d ) + 1.0,
                                     1.0 );
}

template <class PositionType, class ScalarPoint, class ScalarGrid,
          class ScalarHalo>
void scalarValueG2P( const PositionType points, ScalarPoint& scalar_point_field,
                     ScalarGrid& scalar_grid_field,
                     const ScalarHalo scalar_halo, const std::size_t num_point )
{
    ArrayOp::assign( *scalar_grid_field, 3.5, Own() );

    // Interpolate a scalar grid value to the points.
    auto scalar_value_g2p = createScalarValueG2P( scalar_point_field, -0.5 );
    g2p( *scalar_grid_field, *scalar_halo, points, num_point, Spline<1>(),
         scalar_value_g2p );
}

template <class ScalarPoint>
void checkScalarValueG2P( const ScalarPoint scalar_point_host,
                          const std::size_t num_point )
{
    for ( std::size_t p = 0; p < num_point; ++p )
        EXPECT_FLOAT_EQ( scalar_point_host( p ), -1.75 );
}

template <class PositionType, class VectorPoint, class VectorGrid,
          class VectorHalo>
void vectorValueG2P( const PositionType points, VectorPoint& vector_point_field,
                     VectorGrid vector_grid_field, const VectorHalo vector_halo,
                     const std::size_t num_point )
{
    ArrayOp::assign( *vector_grid_field, 3.5, Own() );

    // Interpolate a vector grid value to the points.
    auto vector_value_g2p = createVectorValueG2P( vector_point_field, -0.5 );
    g2p( *vector_grid_field, *vector_halo, points, num_point, Spline<1>(),
         vector_value_g2p );
}

template <class VectorPoint>
void checkVectorValueG2P( const VectorPoint vector_point_host,
                          const std::size_t num_point )
{
    for ( std::size_t p = 0; p < num_point; ++p )
        for ( std::size_t d = 0; d < 3; ++d )
            EXPECT_FLOAT_EQ( vector_point_host( p, d ), -1.75 );
}

template <class PositionType, class VectorPoint, class ScalarGrid,
          class ScalarHalo>
void scalarGradientG2P( const PositionType points,
                        VectorPoint& vector_point_field,
                        ScalarGrid scalar_grid_field,
                        const ScalarHalo scalar_halo,
                        const std::size_t num_point )
{
    ArrayOp::assign( *scalar_grid_field, 3.5, Own() );

    // Interpolate a scalar grid gradient to the points.
    auto scalar_gradient_g2p =
        createScalarGradientG2P( vector_point_field, -0.5 );
    g2p( *scalar_grid_field, *scalar_halo, points, num_point, Spline<1>(),
         scalar_gradient_g2p );
}

template <class VectorPoint>
void checkScalarGradientG2P( const VectorPoint vector_point_host,
                             const std::size_t num_point )
{
    for ( std::size_t p = 0; p < num_point; ++p )
        for ( std::size_t d = 0; d < 3; ++d )
            EXPECT_FLOAT_EQ( vector_point_host( p, d ) + 1.0, 1.0 );
}

template <class PositionType, class TensorPoint, class VectorGrid,
          class VectorHalo>
void vectorGradientG2P( const PositionType points,
                        TensorPoint& tensor_point_field,
                        VectorGrid vector_grid_field,
                        const VectorHalo vector_halo,
                        const std::size_t num_point )
{
    ArrayOp::assign( *vector_grid_field, 3.5, Own() );

    // Interpolate a vector grid gradient to the points.
    auto vector_gradient_g2p =
        createVectorGradientG2P( tensor_point_field, -0.5 );
    g2p( *vector_grid_field, *vector_halo, points, num_point, Spline<1>(),
         vector_gradient_g2p );
}

template <class TensorPoint>
void checkVectorGradientG2P( const TensorPoint tensor_point_host,
                             const std::size_t num_point )
{
    for ( std::size_t p = 0; p < num_point; ++p )
        for ( std::size_t i = 0; i < 3; ++i )
            for ( std::size_t j = 0; j < 3; ++j )
                EXPECT_FLOAT_EQ( tensor_point_host( p, i, j ) + 1.0, 1.0 );
}

template <class PositionType, class ScalarPoint, class VectorGrid,
          class VectorHalo>
void vectorDivergenceG2P( const PositionType points,
                          ScalarPoint& scalar_point_field,
                          VectorGrid vector_grid_field,
                          const VectorHalo vector_halo,
                          const std::size_t num_point )
{
    ArrayOp::assign( *vector_grid_field, 3.5, Own() );

    // Interpolate a vector grid divergence to the points.
    auto vector_div_g2p = createVectorDivergenceG2P( scalar_point_field, -0.5 );
    g2p( *vector_grid_field, *vector_halo, points, num_point, Spline<1>(),
         vector_div_g2p );
}

template <class ScalarPoint>
void checkVectorDivergenceG2P( const ScalarPoint scalar_point_host,
                               const std::size_t num_point )
{
    for ( std::size_t p = 0; p < num_point; ++p )
        EXPECT_FLOAT_EQ( scalar_point_host( p ) + 1.0, 1.0 );
}

auto createGrid()
{
    // Create the global mesh.
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high_corner = { -0.3, 9.5, 2.3 };
    double cell_size = 0.1;
    auto global_mesh =
        createUniformGlobalMesh( low_corner, high_corner, cell_size );

    // Create the global grid.
    DimBlockPartitioner<3> partitioner;
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    return global_grid;
}

template <class PositionType, class LocalGridType>
void setPositions( PositionType& points, const LocalGridType local_grid,
                   const int halo_width )
{
    auto local_mesh = createLocalMesh<TEST_DEVICE>( *local_grid );
    auto cell_space = local_grid->indexSpace( Own(), Cell(), Local() );

    // Create a point in the center of every cell.
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
}
//---------------------------------------------------------------------------//
void interpolationViewTest()
{
    // Create a grid local_grid.
    int halo_width = 1;
    auto global_grid = createGrid();
    auto local_grid = createLocalGrid( global_grid, halo_width );
    auto cell_space = local_grid->indexSpace( Own(), Cell(), Local() );
    auto num_point = cell_space.size();

    // Create particles.
    Kokkos::View<double* [3], TEST_DEVICE> points(
        Kokkos::ViewAllocateWithoutInitializing( "points" ), num_point );
    setPositions( points, local_grid, halo_width );

    // Node space.
    auto node_space = local_grid->indexSpace( Own(), Node(), Local() );

    // Create a scalar field on the grid.
    auto scalar_layout = createArrayLayout( local_grid, 1, Node() );
    auto scalar_grid_field =
        createArray<double, TEST_DEVICE>( "scalar_grid_field", scalar_layout );

    // Create a vector field on the grid.
    auto vector_layout = createArrayLayout( local_grid, 3, Node() );
    auto vector_grid_field =
        createArray<double, TEST_DEVICE>( "vector_grid_field", vector_layout );

    // Create a scalar, vector, and tensor point fields.
    Kokkos::View<double*, TEST_DEVICE> scalar_point_field(
        Kokkos::ViewAllocateWithoutInitializing( "scalar_point_field" ),
        num_point );
    Kokkos::View<double* [3], TEST_DEVICE> vector_point_field(
        Kokkos::ViewAllocateWithoutInitializing( "vector_point_field" ),
        num_point );
    Kokkos::View<double* [3][3], TEST_DEVICE> tensor_point_field(
        Kokkos::ViewAllocateWithoutInitializing( "tensor_point_field" ),
        num_point );

    // Create halos.
    auto scalar_halo =
        createHalo( NodeHaloPattern<3>(), halo_width, *scalar_grid_field );
    auto vector_halo =
        createHalo( NodeHaloPattern<3>(), halo_width, *vector_grid_field );

    Kokkos::deep_copy( scalar_point_field, 3.5 );
    Kokkos::deep_copy( vector_point_field, 3.5 );
    Kokkos::deep_copy( tensor_point_field, 3.5 );

    // Check all p2g together.
    checkP2G( points, scalar_point_field, vector_point_field,
              tensor_point_field, scalar_grid_field, vector_grid_field,
              node_space, scalar_halo, vector_halo, num_point );

    // Separated G2P because of reset/copies (different for AoSoA/View).
    Kokkos::deep_copy( scalar_point_field, 0.0 );
    scalarValueG2P( points, scalar_point_field, scalar_grid_field, scalar_halo,
                    num_point );
    auto scalar_point_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_point_field );
    checkScalarValueG2P( scalar_point_host, num_point );

    Kokkos::deep_copy( vector_point_field, 0.0 );
    vectorValueG2P( points, vector_point_field, vector_grid_field, vector_halo,
                    num_point );
    auto vector_point_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), vector_point_field );
    checkVectorValueG2P( vector_point_host, num_point );

    Kokkos::deep_copy( vector_point_field, 0.0 );
    scalarGradientG2P( points, vector_point_field, scalar_grid_field,
                       scalar_halo, num_point );
    vector_point_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), vector_point_field );
    checkScalarGradientG2P( vector_point_host, num_point );

    Kokkos::deep_copy( tensor_point_field, 0.0 );
    vectorGradientG2P( points, tensor_point_field, vector_grid_field,
                       vector_halo, num_point );
    auto tensor_point_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), tensor_point_field );
    checkVectorGradientG2P( tensor_point_host, num_point );

    Kokkos::deep_copy( scalar_point_field, 0.0 );
    vectorDivergenceG2P( points, scalar_point_field, vector_grid_field,
                         vector_halo, num_point );
    scalar_point_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), scalar_point_field );
    checkVectorDivergenceG2P( scalar_point_host, num_point );
}

//---------------------------------------------------------------------------//
void interpolationSliceTest()
{
    // Create a grid local_grid.
    int halo_width = 1;
    auto global_grid = createGrid();
    auto local_grid = createLocalGrid( global_grid, halo_width );
    auto cell_space = local_grid->indexSpace( Own(), Cell(), Local() );
    auto num_point = cell_space.size();

    // Create particles.
    using member_types =
        Cabana::MemberTypes<double[3][3], double[3], double[3], double>;
    Cabana::AoSoA<member_types, TEST_DEVICE> particles( "particles",
                                                        num_point );
    auto points = Cabana::slice<2>( particles, "position" );
    setPositions( points, local_grid, halo_width );

    // Node space.
    auto node_space = local_grid->indexSpace( Own(), Node(), Local() );

    // Create a scalar field on the grid.
    auto scalar_layout = createArrayLayout( local_grid, 1, Node() );
    auto scalar_grid_field =
        createArray<double, TEST_DEVICE>( "scalar_grid_field", scalar_layout );

    // Create a vector field on the grid.
    auto vector_layout = createArrayLayout( local_grid, 3, Node() );
    auto vector_grid_field =
        createArray<double, TEST_DEVICE>( "vector_grid_field", vector_layout );

    // Create a scalar, vector, and tensor point fields.
    auto tensor_point_field = Cabana::slice<0>( particles, "tensor" );
    auto vector_point_field = Cabana::slice<1>( particles, "vector" );
    auto scalar_point_field = Cabana::slice<3>( particles, "scalar" );

    // Create halos.
    auto scalar_halo =
        createHalo( NodeHaloPattern<3>(), halo_width, *scalar_grid_field );
    auto vector_halo =
        createHalo( NodeHaloPattern<3>(), halo_width, *vector_grid_field );

    Cabana::deep_copy( scalar_point_field, 3.5 );
    Cabana::deep_copy( vector_point_field, 3.5 );
    Cabana::deep_copy( tensor_point_field, 3.5 );

    // Check all p2g together.
    checkP2G( points, scalar_point_field, vector_point_field,
              tensor_point_field, scalar_grid_field, vector_grid_field,
              node_space, scalar_halo, vector_halo, num_point );

    // Separated to do the reset/copies (different for AoSoA/View).
    auto particles_host =
        Cabana::create_mirror_view( Kokkos::HostSpace(), particles );
    auto tensor_point_host = Cabana::slice<0>( particles, "tensor" );
    auto vector_point_host = Cabana::slice<1>( particles, "vector" );
    auto scalar_point_host = Cabana::slice<3>( particles, "scalar" );

    Cabana::deep_copy( scalar_point_field, 0.0 );
    scalarValueG2P( points, scalar_point_field, scalar_grid_field, scalar_halo,
                    num_point );
    Cabana::deep_copy( scalar_point_host, scalar_point_field );
    checkScalarValueG2P( scalar_point_host, num_point );

    Cabana::deep_copy( vector_point_field, 0.0 );
    vectorValueG2P( points, vector_point_field, vector_grid_field, vector_halo,
                    num_point );
    Cabana::deep_copy( vector_point_host, vector_point_field );
    checkVectorValueG2P( vector_point_host, num_point );

    Cabana::deep_copy( vector_point_field, 0.0 );
    scalarGradientG2P( points, vector_point_field, scalar_grid_field,
                       scalar_halo, num_point );
    Cabana::deep_copy( vector_point_host, vector_point_field );
    checkScalarGradientG2P( vector_point_host, num_point );

    Cabana::deep_copy( tensor_point_field, 0.0 );
    vectorGradientG2P( points, tensor_point_field, vector_grid_field,
                       vector_halo, num_point );
    Cabana::deep_copy( tensor_point_host, tensor_point_field );
    checkVectorGradientG2P( tensor_point_host, num_point );

    Cabana::deep_copy( scalar_point_field, 0.0 );
    vectorDivergenceG2P( points, scalar_point_field, vector_grid_field,
                         vector_halo, num_point );
    Cabana::deep_copy( scalar_point_host, scalar_point_field );
    checkVectorDivergenceG2P( scalar_point_host, num_point );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( interpolation, interpolation_view_test ) { interpolationViewTest(); }

TEST( interpolation, interpolation_slice_test ) { interpolationSliceTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
