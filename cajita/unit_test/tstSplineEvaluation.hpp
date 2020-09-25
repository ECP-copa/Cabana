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

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
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
template <class Scalar, class EntityType, int SplineOrder, class DeviceType,
          class DataTags>
struct PointSet
{
    // Scalar type for geometric operations.
    using scalar_type = Scalar;

    // Entity type on which the basis is defined.
    using entity_type = EntityType;

    // Basis
    static constexpr int spline_order = SplineOrder;
    using basis_type = Spline<spline_order>;

    // Number of basis values in each dimension.
    static constexpr int ns = basis_type::num_knot;

    // Spline data tags.
    using spline_data_tags = DataTags;

    // Kokkos types.
    using device_type = DeviceType;
    using execution_space = typename device_type::execution_space;
    using memory_space = typename device_type::memory_space;

    // Number of points.
    std::size_t num_point;

    // Number of allocated points.
    std::size_t num_alloc;

    // Physical cell size. (point,dim)
    Kokkos::View<Scalar *[3], device_type> cell_size;

    // Point logical position. (point,dim)
    Kokkos::View<Scalar *[3], device_type> logical_coords;

    // Point mesh stencil. (point,ns,dim)
    Kokkos::View<int *[ns][3], device_type> stencil;

    // Point basis values at entities in stencil. (point,ns,dim)
    Kokkos::View<Scalar *[ns][3], device_type> value;

    // Point basis gradient values at entities in stencil
    // (point,ni,nj,nk,dim)
    Kokkos::View<Scalar *[ns][ns][ns][3], device_type> gradient;

    // Point basis distance values at entities in stencil
    // (point,ni,nj,nk,dim)
    Kokkos::View<Scalar *[ns][3], device_type> distance;

    // Mesh uniform cell size.
    Scalar dx;

    // Inverse uniform mesh cell size.
    Scalar rdx;

    // Location of the low corner of the local mesh for the given entity
    // type.
    Kokkos::Array<Scalar, 3> low_corner;
};

//---------------------------------------------------------------------------//
template <class LocalMeshType, class EntityType, class PointCoordinates,
          class PointSetType>
void updatePointSet( const LocalMeshType &local_mesh, EntityType,
                     const PointCoordinates &points,
                     const std::size_t num_point, PointSetType &point_set )
{
    static_assert( std::is_same<typename PointCoordinates::value_type,
                                typename PointSetType::scalar_type>::value,
                   "Point coordinate/mesh scalar type mismatch" );

    // Device parameters.
    using execution_space = typename PointSetType::execution_space;

    // Scalar type
    using scalar_type = typename PointSetType::scalar_type;

    // Basis parameters.
    using Basis = typename PointSetType::basis_type;
    static constexpr int ns = PointSetType::ns;

    // Spline data tags.
    using spline_data_tags = typename PointSetType::spline_data_tags;

    // spline data type
    using sd_type =
        SplineData<scalar_type, Basis::order, EntityType, spline_data_tags>;

    // Check members.
    static_assert( sd_type::has_physical_cell_size,
                   "spline data missing physical cell size" );
    static_assert( sd_type::has_logical_position,
                   "spline data missing logical position" );
    static_assert( sd_type::has_physical_distance,
                   "spline data missing physical distance" );
    static_assert( sd_type::has_weight_values,
                   "spline data missing weight values" );
    static_assert( sd_type::has_weight_physical_gradients,
                   "spline data missing weight physical gradients" );

    // Update the size.
    if ( num_point > point_set.num_alloc )
        throw std::logic_error(
            "Attempted to update point set with more points than allocation" );
    point_set.num_point = num_point;

    // Update point set value.
    Kokkos::parallel_for(
        "updatePointSet",
        Kokkos::RangePolicy<execution_space>( 0, point_set.num_point ),
        KOKKOS_LAMBDA( const int p ) {
            // Create a spline evaluation data set. This is what we are
            // actually testing in this test.
            scalar_type px[3] = { points( p, Dim::I ), points( p, Dim::J ),
                                  points( p, Dim::K ) };
            sd_type sd;
            evaluateSpline( local_mesh, px, sd );

            // Get the cell size.
            for ( int d = 0; d < 3; ++d )
                point_set.cell_size( p, d ) = sd.dx[d];

            // Map the point coordinates to the logical space of the spline.
            for ( int d = 0; d < 3; ++d )
                point_set.logical_coords( p, d ) = sd.x[d];

            // Get the point mesh stencil.
            for ( int d = 0; d < 3; ++d )
                for ( int n = 0; n < ns; ++n )
                    point_set.stencil( p, n, d ) = sd.s[d][n];

            // Evaluate the spline values at the entities in the stencil.
            for ( int d = 0; d < 3; ++d )
                for ( int n = 0; n < ns; ++n )
                    point_set.value( p, n, d ) = sd.w[d][n];

            // Evaluate the spline gradients at the entities in the stencil.
            for ( int i = 0; i < ns; ++i )
                for ( int j = 0; j < ns; ++j )
                    for ( int k = 0; k < ns; ++k )
                    {
                        point_set.gradient( p, i, j, k, Dim::I ) =
                            sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];

                        point_set.gradient( p, i, j, k, Dim::J ) =
                            sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];

                        point_set.gradient( p, i, j, k, Dim::K ) =
                            sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
                    }

            // Evaluate the spline distances at the entities in the stencil.
            for ( int d = 0; d < 3; ++d )
                for ( int n = 0; n < ns; ++n )
                    point_set.distance( p, n, d ) = sd.d[d][n];
        } );
}

//---------------------------------------------------------------------------//
template <class DataTags, class PointCoordinates, class EntityType,
          int SplineOrder>
PointSet<typename PointCoordinates::value_type, EntityType, SplineOrder,
         typename PointCoordinates::device_type, DataTags>
createPointSet(
    const PointCoordinates &points, const std::size_t num_point,
    const std::size_t num_alloc,
    const LocalGrid<UniformMesh<typename PointCoordinates::value_type>>
        &local_grid,
    EntityType, Spline<SplineOrder> )
{
    using scalar_type = typename PointCoordinates::value_type;

    using device_type = typename PointCoordinates::device_type;

    using set_type =
        PointSet<scalar_type, EntityType, SplineOrder, device_type, DataTags>;

    set_type point_set;

    static constexpr int ns = set_type::ns;

    if ( num_point > num_alloc )
        throw std::logic_error(
            "Attempted to create point set with more points than allocation" );
    point_set.num_point = num_point;
    point_set.num_alloc = num_alloc;

    point_set.cell_size = Kokkos::View<scalar_type *[3], device_type>(
        Kokkos::ViewAllocateWithoutInitializing( "PointSet::cell_size" ),
        num_alloc );

    point_set.logical_coords = Kokkos::View<scalar_type *[3], device_type>(
        Kokkos::ViewAllocateWithoutInitializing( "PointSet::logical_coords" ),
        num_alloc );

    point_set.stencil = Kokkos::View<int *[ns][3], device_type>(
        Kokkos::ViewAllocateWithoutInitializing( "PointSet::stencil" ),
        num_alloc );

    point_set.value = Kokkos::View<scalar_type *[ns][3], device_type>(
        Kokkos::ViewAllocateWithoutInitializing( "PointSet::value" ),
        num_alloc );

    point_set.gradient =
        Kokkos::View<scalar_type *[ns][ns][ns][3], device_type>(
            Kokkos::ViewAllocateWithoutInitializing( "PointSet::gradients" ),
            num_alloc );

    point_set.distance = Kokkos::View<scalar_type *[ns][3], device_type>(
        Kokkos::ViewAllocateWithoutInitializing( "PointSet::distance" ),
        num_alloc );

    auto local_mesh = createLocalMesh<Kokkos::HostSpace>( local_grid );

    int idx_low[3] = { 0, 0, 0 };
    point_set.dx = local_mesh.measure( Edge<Dim::I>(), idx_low );

    point_set.rdx = 1.0 / point_set.dx;

    local_mesh.coordinates( EntityType(), idx_low,
                            point_set.low_corner.data() );

    updatePointSet( local_mesh, EntityType(), points, num_point, point_set );

    return point_set;
}

//---------------------------------------------------------------------------//
template <class DataTags>
void splineEvaluationTest()
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

    // Create a local grid.
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
            double x[3];
            int idx[3] = { i, j, k };
            local_mesh.coordinates( Cell(), idx, x );
            points( pid, Dim::I ) = x[Dim::I];
            points( pid, Dim::J ) = x[Dim::J];
            points( pid, Dim::K ) = x[Dim::K];
        } );

    // Create a point set with linear spline interpolation to the nodes.
    auto point_set = createPointSet<DataTags>(
        points, num_point, num_point, *local_grid, Node(), Spline<1>() );

    // Check the point set data.
    EXPECT_EQ( point_set.num_point, num_point );
    EXPECT_EQ( point_set.dx, cell_size );
    EXPECT_EQ( point_set.rdx, 1.0 / cell_size );
    double xn_low[3];
    int idx_low[3] = { 0, 0, 0 };
    local_mesh.coordinates( Node(), idx_low, xn_low );
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( point_set.low_corner[d], xn_low[d] );

    // Check cell size
    auto cell_size_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), point_set.cell_size );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
            {
                int pi = i - halo_width;
                int pj = j - halo_width;
                int pk = k - halo_width;
                int pid = pi + cell_space.extent( Dim::I ) *
                                   ( pj + cell_space.extent( Dim::J ) * pk );
                EXPECT_FLOAT_EQ( cell_size_host( pid, Dim::I ), 0.05 );
                EXPECT_FLOAT_EQ( cell_size_host( pid, Dim::J ), 0.05 );
                EXPECT_FLOAT_EQ( cell_size_host( pid, Dim::K ), 0.05 );
            }

    // Check logical coordinates
    auto logical_coords_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), point_set.logical_coords );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
            {
                int pi = i - halo_width;
                int pj = j - halo_width;
                int pk = k - halo_width;
                int pid = pi + cell_space.extent( Dim::I ) *
                                   ( pj + cell_space.extent( Dim::J ) * pk );
                EXPECT_FLOAT_EQ( logical_coords_host( pid, Dim::I ), i + 0.5 );
                EXPECT_FLOAT_EQ( logical_coords_host( pid, Dim::J ), j + 0.5 );
                EXPECT_FLOAT_EQ( logical_coords_host( pid, Dim::K ), k + 0.5 );
            }

    // Check stencil.
    auto stencil_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), point_set.stencil );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
            {
                int pi = i - halo_width;
                int pj = j - halo_width;
                int pk = k - halo_width;
                int pid = pi + cell_space.extent( Dim::I ) *
                                   ( pj + cell_space.extent( Dim::J ) * pk );
                EXPECT_EQ( stencil_host( pid, 0, Dim::I ), i );
                EXPECT_EQ( stencil_host( pid, 1, Dim::I ), i + 1 );
                EXPECT_EQ( stencil_host( pid, 0, Dim::J ), j );
                EXPECT_EQ( stencil_host( pid, 1, Dim::J ), j + 1 );
                EXPECT_EQ( stencil_host( pid, 0, Dim::K ), k );
                EXPECT_EQ( stencil_host( pid, 1, Dim::K ), k + 1 );
            }

    // Check values.
    auto values_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                            point_set.value );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
            {
                int pi = i - halo_width;
                int pj = j - halo_width;
                int pk = k - halo_width;
                int pid = pi + cell_space.extent( Dim::I ) *
                                   ( pj + cell_space.extent( Dim::J ) * pk );
                EXPECT_FLOAT_EQ( values_host( pid, 0, Dim::I ), 0.5 );
                EXPECT_FLOAT_EQ( values_host( pid, 1, Dim::I ), 0.5 );
                EXPECT_FLOAT_EQ( values_host( pid, 0, Dim::J ), 0.5 );
                EXPECT_FLOAT_EQ( values_host( pid, 1, Dim::J ), 0.5 );
                EXPECT_FLOAT_EQ( values_host( pid, 0, Dim::K ), 0.5 );
                EXPECT_FLOAT_EQ( values_host( pid, 1, Dim::K ), 0.5 );
            }

    // Check gradients.
    auto gradients_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), point_set.gradient );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
            {
                int pi = i - halo_width;
                int pj = j - halo_width;
                int pk = k - halo_width;
                int pid = pi + cell_space.extent( Dim::I ) *
                                   ( pj + cell_space.extent( Dim::J ) * pk );

                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 0, 0, Dim::I ),
                                 -0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 0, 0, Dim::J ),
                                 -0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 0, 0, Dim::K ),
                                 -0.25 / cell_size );

                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 0, 0, Dim::I ),
                                 0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 0, 0, Dim::J ),
                                 -0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 0, 0, Dim::K ),
                                 -0.25 / cell_size );

                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 1, 0, Dim::I ),
                                 0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 1, 0, Dim::J ),
                                 0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 1, 0, Dim::K ),
                                 -0.25 / cell_size );

                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 1, 0, Dim::I ),
                                 -0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 1, 0, Dim::J ),
                                 0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 1, 0, Dim::K ),
                                 -0.25 / cell_size );

                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 0, 1, Dim::I ),
                                 -0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 0, 1, Dim::J ),
                                 -0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 0, 1, Dim::K ),
                                 0.25 / cell_size );

                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 0, 1, Dim::I ),
                                 0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 0, 1, Dim::J ),
                                 -0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 0, 1, Dim::K ),
                                 0.25 / cell_size );

                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 1, 1, Dim::I ),
                                 0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 1, 1, Dim::J ),
                                 0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 1, 1, 1, Dim::K ),
                                 0.25 / cell_size );

                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 1, 1, Dim::I ),
                                 -0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 1, 1, Dim::J ),
                                 0.25 / cell_size );
                EXPECT_FLOAT_EQ( gradients_host( pid, 0, 1, 1, Dim::K ),
                                 0.25 / cell_size );
            }

    // Check distances
    auto distances_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), point_set.distance );
    for ( int i = cell_space.min( Dim::I ); i < cell_space.max( Dim::I ); ++i )
        for ( int j = cell_space.min( Dim::J ); j < cell_space.max( Dim::J );
              ++j )
            for ( int k = cell_space.min( Dim::K );
                  k < cell_space.max( Dim::K ); ++k )
            {
                int pi = i - halo_width;
                int pj = j - halo_width;
                int pk = k - halo_width;
                int pid = pi + cell_space.extent( Dim::I ) *
                                   ( pj + cell_space.extent( Dim::J ) * pk );
                for ( int d = 0; d < 3; ++d )
                {
                    double invariant = 0.0;
                    for ( int n = 0; n < 2; ++n )
                    {
                        invariant += values_host( pid, n, d ) *
                                     distances_host( pid, n, d );
                    }
                    EXPECT_FLOAT_EQ( 1.0 - invariant, 1.0 );
                }
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( splines, eval_test )
{
    // Check with default data members on spline data.
    splineEvaluationTest<void>();

    // Set each spline data member individually.
    splineEvaluationTest<SplineDataMemberTypes<
        SplinePhysicalCellSize, SplineLogicalPosition, SplinePhysicalDistance,
        SplineWeightValues, SplineWeightPhysicalGradients>>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
