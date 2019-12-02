/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_POINTSET_HPP
#define CAJITA_POINTSET_HPP

#include <Cajita_Block.hpp>
#include <Cajita_LocalMesh.hpp>
#include <Cajita_Splines.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cajita
{
//---------------------------------------------------------------------------//
/*!
  \class PointSet

  \brief Evaluation point set with a spline basis.

  \tparam Scalar The scalar type used to represent floating point values.

  \tparam EntityType The type of entity type the spline functions are
  evaluated at.

  \tparam SplineOrder The order of B-spline to use for the basis.

  \tparam DeviceType The device to use for creating point set data.

  This class defines the point set over points needed for interpolation.

  Each point is centered in a grid stencil defined by the B-spline used for
  interpolation. The point set contains the point position along with the
  entities in the stencil and their distances from the points, basis
  weights, and basis gradients.

  A point set should be regenerated every time points are redistributed and
  updated every time a new time step is started.

  Point sets are defined for uniform meshes.
*/
//---------------------------------------------------------------------------//
template <class Scalar, class EntityType, int SplineOrder, class DeviceType>
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

    // Kokkos types.
    using device_type = DeviceType;
    using execution_space = typename device_type::execution_space;
    using memory_space = typename device_type::memory_space;

    // Number of points.
    std::size_t num_point;

    // Number of allocated points.
    std::size_t num_alloc;

    // Point logical position. (point,dim)
    Kokkos::View<Scalar * [3], device_type> logical_coords;

    // Point mesh stencil. (point,ns,dim)
    Kokkos::View<int * [ns][3], device_type> stencil;

    // Point basis values at entities in stencil. (point,ns,dim)
    Kokkos::View<Scalar * [ns][3], device_type> value;

    // Point basis gradient values at entities in stencil
    // (point,ni,nj,nk,dim)
    Kokkos::View<Scalar * [ns][ns][ns][3], device_type> gradient;

    // Mesh uniform cell size.
    Scalar dx;

    // Inverse uniform mesh cell size.
    Scalar rdx;

    // Location of the low corner of the local mesh for the given entity
    // type.
    Kokkos::Array<Scalar, 3> low_corner;
};

//---------------------------------------------------------------------------//
/*!
  \brief Update a point set with new coordinates.

  \tparam PointCoordinates Container type with view traits containing the 3-d
  point coordinates. Will be indexed as (point,dim).

  \tparam PointSetType The type of point set to update.

  \param points The points over which to build the point set. Will be indexed
  as (point,dim). All points must be contained within the grid block that was
  used to generate the point set.

  \param num_point The number of points. This is the size of the first
  dimension of points. Note that the number of points must less than or equal
  to the number for which the point set was allocated. If a larger number of
  points is a needed a new point set must be created to hold the additional
  memory.

  \param point_set The point set to update.
*/
template <class PointCoordinates, class PointSetType>
void updatePointSet( const PointCoordinates &points,
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
            // Map the point coordinates to the logical space of the spline.
            for ( int d = 0; d < 3; ++d )
                point_set.logical_coords( p, d ) = Basis::mapToLogicalGrid(
                    points( p, d ), point_set.rdx, point_set.low_corner[d] );

            // Get the point mesh stencil.
            int indices[ns];
            for ( int d = 0; d < 3; ++d )
            {
                Basis::stencil( point_set.logical_coords( p, d ), indices );
                for ( int n = 0; n < ns; ++n )
                    point_set.stencil( p, n, d ) = indices[n];
            }

            // Evaluate the spline values at the entities in the stencil.
            scalar_type basis_values[ns];
            for ( int d = 0; d < 3; ++d )
            {
                Basis::value( point_set.logical_coords( p, d ), basis_values );
                for ( int n = 0; n < ns; ++n )
                    point_set.value( p, n, d ) = basis_values[n];
            }

            // Evaluate the spline gradients at the entities in the stencil.
            scalar_type basis_gradients[ns];
            for ( int d = 0; d < 3; ++d )
            {
                Basis::gradient( point_set.logical_coords( p, d ),
                                 point_set.rdx, basis_gradients );

                for ( int i = 0; i < ns; ++i )
                    for ( int j = 0; j < ns; ++j )
                        for ( int k = 0; k < ns; ++k )
                        {
                            point_set.gradient( p, i, j, k, Dim::I ) =
                                basis_gradients[i] *
                                point_set.value( p, j, Dim::J ) *
                                point_set.value( p, k, Dim::K );

                            point_set.gradient( p, i, j, k, Dim::J ) =
                                point_set.value( p, i, Dim::I ) *
                                basis_gradients[j] *
                                point_set.value( p, k, Dim::K );

                            point_set.gradient( p, i, j, k, Dim::K ) =
                                point_set.value( p, i, Dim::I ) *
                                point_set.value( p, j, Dim::J ) *
                                basis_gradients[k];
                        }
            }
        } );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a point set.

  \tparam PointCoordinates Container type with view traits containing the 3-d
  point coordinates. Will be indexed as (point,dim). The value type of the
  coordinates will define the scalar type used to generate the point set.

  \tparam EntityType The type of entity type the spline functions are
  evaluated at.

  \tparam SplineOrder The order of B-spline to use for the basis.

  \param points The points to generate the point set with. Will be indexed as
  (point,dim).

  \param num_point The number of points to generate the point set with. This
  is the size of the first dimension of points.

  \param num_alloc The number of points to allocate. Must be less than or
  equal to the input number of points.

  \param block The grid block in which the points reside. All points must be
  contained within the grid block.

  \param An instance of the entity type to which the points will interpolate.

  \param An instance of the spline function to use for interpolation.
*/
template <class PointCoordinates, class EntityType, int SplineOrder>
PointSet<typename PointCoordinates::value_type, EntityType, SplineOrder,
         typename PointCoordinates::device_type>
createPointSet(
    const PointCoordinates &points, const std::size_t num_point,
    const std::size_t num_alloc,
    const Block<UniformMesh<typename PointCoordinates::value_type>> &block,
    EntityType, Spline<SplineOrder> )
{
    using scalar_type = typename PointCoordinates::value_type;

    using device_type = typename PointCoordinates::device_type;

    using set_type =
        PointSet<scalar_type, EntityType, SplineOrder, device_type>;

    set_type point_set;

    static constexpr int ns = set_type::ns;

    if ( num_point > num_alloc )
        throw std::logic_error(
            "Attempted to create point set with more points than allocation" );
    point_set.num_point = num_point;
    point_set.num_alloc = num_alloc;

    point_set.logical_coords = Kokkos::View<scalar_type * [3], device_type>(
        Kokkos::ViewAllocateWithoutInitializing( "PointSet::logical_coords" ),
        num_alloc );

    point_set.stencil = Kokkos::View<int * [ns][3], device_type>(
        Kokkos::ViewAllocateWithoutInitializing( "PointSet::stencil" ),
        num_alloc );

    point_set.value = Kokkos::View<scalar_type * [ns][3], device_type>(
        Kokkos::ViewAllocateWithoutInitializing( "PointSet::value" ),
        num_alloc );

    point_set.gradient =
        Kokkos::View<scalar_type * [ns][ns][ns][3], device_type>(
            Kokkos::ViewAllocateWithoutInitializing( "PointSet::gradients" ),
            num_alloc );

    auto local_mesh = createLocalMesh<Kokkos::HostSpace>( block );

    point_set.dx = local_mesh.measure( Edge<Dim::I>(), 0, 0, 0 );

    point_set.rdx = 1.0 / point_set.dx;

    for ( int d = 0; d < 3; ++d )
        point_set.low_corner[d] = local_mesh.coordinate( EntityType(), 0, d );

    updatePointSet( points, num_point, point_set );

    return point_set;
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_POINTSET_HPP

//---------------------------------------------------------------------------//
