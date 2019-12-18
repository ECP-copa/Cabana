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

#ifndef CAJITA_INTERPOLATION_HPP
#define CAJITA_INTERPOLATION_HPP

#include <Cajita_Array.hpp>
#include <Cajita_Halo.hpp>
#include <Cajita_LocalMesh.hpp>
#include <Cajita_Splines.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <memory>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Point-to-Grid
//---------------------------------------------------------------------------//
/*!
  \brief Local Point-to-Grid interpolation.

  \tparam PointEvalFunctor Functor type used to evaluate the interpolated data
  for a given point at a given entity.

  \tparam PointCoordinates Container type with view traits containing the 3-d
  point coordinates. Will be indexed as (point,dim).

  \tparam ArrayScalar The scalar type used for the interpolated data.

  \tparam MeshScalar The scalar type used for the geometry/interpolation data.

  \tparam EntityType The entitytype to which the points will interpolate.

  \tparam SplineOrder The order of spline interpolation to use.

  \tparam DeviceType The device type to use for interplation

  \tparam ArrayParams Parameters for the array type.

  \param functor A functor that interpolates from a given point to a given
  entity.

  \param points The points over which to perform the interpolation. Will be
  indexed as (point,dim). The subset of indices in each point's interpolation
  stencil must be contained within the local grid that will be used for the
  interpolation

  \param num_point The number of points. This is the size of the first
  dimension of points.

  \param Spline to use for interpolation.

  \param halo The halo associated with the grid array. This hallo will be used
  to scatter the interpolated data.

  \param array The grid array to which the point data will be interpolated.
*/
template <class PointEvalFunctor, class PointCoordinates, class ArrayScalar,
          class MeshScalar, class EntityType, int SplineOrder, class DeviceType,
          class... ArrayParams>
void p2g( const PointEvalFunctor &functor, const PointCoordinates &points,
          const std::size_t num_point, Spline<SplineOrder>,
          const Halo<ArrayScalar, DeviceType> &halo,
          Array<ArrayScalar, EntityType, UniformMesh<MeshScalar>,
                ArrayParams...> &array )
{
    using array_type =
        Array<ArrayScalar, EntityType, UniformMesh<MeshScalar>, ArrayParams...>;
    static_assert(
        std::is_same<DeviceType, typename array_type::device_type>::value,
        "Mismatching points/array device types." );

    using execution_space = typename DeviceType::execution_space;

    // Create the local mesh.
    auto local_mesh =
        createLocalMesh<DeviceType>( *( array.layout()->localGrid() ) );

    // Create a scatter view of the array.
    auto array_view = array.view();
    auto array_sv = Kokkos::Experimental::create_scatter_view( array_view );

    // Loop over points and interpolate to the grid.
    Kokkos::parallel_for(
        "p2g", Kokkos::RangePolicy<execution_space>( 0, num_point ),
        KOKKOS_LAMBDA( const int p ) {
            // Create a local scatter result.
            ArrayScalar result[PointEvalFunctor::value_count];

            // Access the scatter view.
            auto array_access = array_sv.access();

            // Get the point coordinates.
            MeshScalar px[3] = {points( p, Dim::I ), points( p, Dim::J ),
                                points( p, Dim::K )};

            // Create the local spline data.
            using sd_type = SplineData<MeshScalar, SplineOrder, EntityType>;
            sd_type sd;
            evaluateSpline( local_mesh, px, sd );

            // Loop over the point stencil and evaluate the functor at each
            // entity in the stencil and apply each stencil result to the
            // array.
            for ( int i = 0; i < sd_type::num_knot; ++i )
                for ( int j = 0; j < sd_type::num_knot; ++j )
                    for ( int k = 0; k < sd_type::num_knot; ++k )
                    {
                        functor( sd, p, i, j, k, result );

                        for ( int d = 0; d < PointEvalFunctor::value_count;
                              ++d )
                            array_access( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                          sd.s[Dim::K][k], d ) += result[d];
                    }
        } );
    Kokkos::Experimental::contribute( array_view, array_sv );

    // Scatter interpolation contributions in the halo back to their owning
    // ranks.
    halo.scatter( array );
}

//---------------------------------------------------------------------------//
/*!
  \brief Point-to-grid scalar value functor.

  Interpolates a scalar function from points to entities with a given
  multiplier such that:

  f_ijk = multiplier * \sum_p weight_{pijk} * f_p

  Note that a functor implements the interpolation contribution between a
  single point, indexed with a local p index, and a single entity, indexed
  with local ijk indices. A single, potentially multi-dimensional result is
  provided as the contribution.
*/
template <class ViewType>
struct ScalarValueP2G
{
    static constexpr int value_count = 1;
    using value_type = typename ViewType::value_type;

    ViewType _x;
    value_type _multiplier;

    ScalarValueP2G( const ViewType &x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 1 == ViewType::Rank, "View must be of scalars" );
    }

    template <class SplineDataType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const SplineDataType &sd, const int p, const int i, const int j,
                const int k, value_type *result ) const
    {
        result[0] = _multiplier * _x( p ) * sd.w[Dim::I][i] * sd.w[Dim::J][j] *
                    sd.w[Dim::K][k];
    }
};

template <class ViewType>
ScalarValueP2G<ViewType>
createScalarValueP2G( const ViewType &x,
                      const typename ViewType::value_type &multiplier )
{
    return ScalarValueP2G<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Point-to-grid vector value functor.

  Interpolates a vector function from points to entities with a given
  multiplier such that:

  f_{ijkd} = multiplier * \sum_p weight_{pijk} * f_{pd}

  Note that a functor implements the interpolation contribution between a
  single point, indexed with a local p index, and a single entity, indexed
  with local ijk indices. A single, potentially multi-dimensional result is
  provided as the contribution.
*/
template <class ViewType>
struct VectorValueP2G
{
    static constexpr int value_count = 3;
    using value_type = typename ViewType::value_type;

    ViewType _x;
    value_type _multiplier;

    VectorValueP2G( const ViewType &x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 2 == ViewType::Rank, "View must be of vectors" );
    }

    template <class SplineDataType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const SplineDataType &sd, const int p, const int i, const int j,
                const int k, value_type *result ) const
    {
        value_type weight =
            _multiplier * sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
        for ( int d = 0; d < 3; ++d )
            result[d] = weight * _x( p, d );
    }
};

template <class ViewType>
VectorValueP2G<ViewType>
createVectorValueP2G( const ViewType &x,
                      const typename ViewType::value_type &multiplier )
{
    return VectorValueP2G<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Point-to-grid scalar gradient functor.

  Interpolates the gradient of a scalar function from points to entities with
  a given multiplier such that:

  f_{ijkd} = multiplier * \sum_p grad_weight_{pijkd} * f_p

  Note that a functor implements the interpolation contribution between a
  single point, indexed with a local p index, and a single entity, indexed
  with local ijk indices. A single, potentially multi-dimensional result is
  provided as the contribution.
*/
template <class ViewType>
struct ScalarGradientP2G
{
    static constexpr int value_count = 3;
    using value_type = typename ViewType::value_type;

    ViewType _x;
    value_type _multiplier;

    ScalarGradientP2G( const ViewType &x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 1 == ViewType::Rank, "View must be of scalars" );
    }

    template <class SplineDataType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const SplineDataType &sd, const int p, const int i, const int j,
                const int k, value_type *result ) const
    {
        auto mx = _multiplier * _x( p );
        result[Dim::I] =
            mx * sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
        result[Dim::J] =
            mx * sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];
        result[Dim::K] =
            mx * sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
    }
};

template <class ViewType>
ScalarGradientP2G<ViewType>
createScalarGradientP2G( const ViewType &x,
                         const typename ViewType::value_type &multiplier )
{
    return ScalarGradientP2G<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Point-to-grid vector divergence functor.

  Interpolates the divergence of a vector function from points to entities
  with a given multiplier such that:

  f_ijk = multiplier * \sum_d \sum_p grad_weight_{pijkd} * f_{pd}

  Note that a functor implements the interpolation contribution between a
  single point, indexed with a local p index, and a single entity, indexed
  with local ijk indices. A single, potentially multi-dimensional result is
  provided as the contribution.
*/
template <class ViewType>
struct VectorDivergenceP2G
{
    static constexpr int value_count = 1;
    using value_type = typename ViewType::value_type;

    ViewType _x;
    value_type _multiplier;

    VectorDivergenceP2G( const ViewType &x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 2 == ViewType::Rank, "View must be of vectors" );
    }

    template <class SplineDataType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const SplineDataType &sd, const int p, const int i, const int j,
                const int k, value_type *result ) const
    {
        result[0] = ( _x( p, Dim::I ) * sd.g[Dim::I][i] * sd.w[Dim::J][j] *
                          sd.w[Dim::K][k] +

                      _x( p, Dim::J ) * sd.w[Dim::I][i] * sd.g[Dim::J][j] *
                          sd.w[Dim::K][k] +

                      _x( p, Dim::K ) * sd.w[Dim::I][i] * sd.w[Dim::J][j] *
                          sd.g[Dim::K][k] ) *
                    _multiplier;
    }
};

template <class ViewType>
VectorDivergenceP2G<ViewType>
createVectorDivergenceP2G( const ViewType &x,
                           const typename ViewType::value_type &multiplier )
{
    return VectorDivergenceP2G<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Point-to-grid tensor divergence functor.

  Interpolates the divergence of a tensor function from points to entities
  with a given multiplier such that:

  f_ijkm = multiplier * \sum_n \sum_p grad_weight_{pijkn} * f_{pmn}

  Note that a functor implements the interpolation contribution between a
  single point, indexed with a local p index, and a single entity, indexed
  with local ijk indices. A single, potentially multi-dimensional result is
  provided as the contribution.
*/
template <class ViewType>
struct TensorDivergenceP2G
{
    static constexpr int value_count = 3;
    using value_type = typename ViewType::value_type;

    ViewType _x;
    value_type _multiplier;

    TensorDivergenceP2G( const ViewType &x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 3 == ViewType::Rank, "View must be of tensors" );
    }

    template <class SplineDataType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const SplineDataType &sd, const int p, const int i, const int j,
                const int k, value_type *result ) const
    {
        for ( int d = 0; d < 3; ++d )
            result[d] = 0.0;

        double rg[3] = {sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k],
                        sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k],
                        sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k]};

        for ( int d1 = 0; d1 < 3; ++d1 )
        {
            auto mg = _multiplier * rg[d1];
            for ( int d0 = 0; d0 < 3; ++d0 )
                result[d0] += mg * _x( p, d0, d1 );
        }
    }
};

template <class ViewType>
TensorDivergenceP2G<ViewType>
createTensorDivergenceP2G( const ViewType &x,
                           const typename ViewType::value_type &multiplier )
{
    return TensorDivergenceP2G<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
// Grid-to-Point
//---------------------------------------------------------------------------//
/*
 \brief Local Grid-to-Point interpolation.

  \tparam PointEvalFunctor Functor type used to evaluate the interpolated data
  for a given point at a given entity.

  \tparam PointCoordinates Container type with view traits containing the 3-d
  point coordinates. Will be indexed as (point,dim).

  \tparam ArrayScalar The scalar type used for the interpolated data.

  \tparam MeshScalar The scalar type used for the geometry/interpolation data.

  \tparam EntityType The entitytype to which the points will interpolate.

  \tparam SplineOrder The order of spline interpolation to use.

  \tparam DeviceType The device type to use for interplation

  \tparam ArrayParams Parameters for the array type.

  \param array The grid array from which the point data will be interpolated.

  \param halo The halo associated with the grid array. This hallo will be used
  to gather the array data before interpolation.

  \param points The points over which to perform the interpolation. Will be
  indexed as (point,dim). The subset of indices in each point's interpolation
  stencil must be contained within the local grid that will be used for the
  interpolation

  \param num_point The number of points. This is the size of the first
  dimension of points.

  \param Spline to use for interpolation.

  \param functor A functor that interpolates from a given entity to a given
  point.
*/
template <class PointEvalFunctor, class PointCoordinates, class ArrayScalar,
          class MeshScalar, class EntityType, int SplineOrder, class DeviceType,
          class... ArrayParams>
void g2p( const Array<ArrayScalar, EntityType, UniformMesh<MeshScalar>,
                      ArrayParams...> &array,
          const Halo<ArrayScalar, DeviceType> &halo,
          const PointCoordinates &points, const std::size_t num_point,
          Spline<SplineOrder>, const PointEvalFunctor &functor )
{
    using array_type =
        Array<ArrayScalar, EntityType, UniformMesh<MeshScalar>, ArrayParams...>;
    static_assert(
        std::is_same<DeviceType, typename array_type::device_type>::value,
        "Mismatching points/array device types." );

    using execution_space = typename DeviceType::execution_space;

    // Create the local mesh.
    auto local_mesh =
        createLocalMesh<DeviceType>( *( array.layout()->localGrid() ) );

    // Gather data into the halo before interpolating.
    halo.gather( array );

    // Get a view of the array data.
    auto array_view = array.view();

    // Loop over points and interpolate from the grid.
    Kokkos::parallel_for(
        "g2p", Kokkos::RangePolicy<execution_space>( 0, num_point ),
        KOKKOS_LAMBDA( const int p ) {
            // Create local gather values.
            ArrayScalar values[PointEvalFunctor::value_count];

            // Get the point coordinates.
            MeshScalar px[3] = {points( p, Dim::I ), points( p, Dim::J ),
                                points( p, Dim::K )};

            // Create the local spline data.
            using sd_type = SplineData<MeshScalar, SplineOrder, EntityType>;
            sd_type sd;
            evaluateSpline( local_mesh, px, sd );

            // Loop over the point stencil and interpolate the grid data to
            // the points.
            for ( int i = 0; i < sd_type::num_knot; ++i )
                for ( int j = 0; j < sd_type::num_knot; ++j )
                    for ( int k = 0; k < sd_type::num_knot; ++k )
                    {
                        for ( int d = 0; d < PointEvalFunctor::value_count;
                              ++d )
                            values[d] =
                                array_view( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                            sd.s[Dim::K][k], d );

                        functor( sd, p, i, j, k, values );
                    }
        } );
}

//---------------------------------------------------------------------------//
/*!
  \brief Grid-to-point scalar value functor.

  Interpolates a scalar function from entities to points with a given
  multiplier such that:

  f_p = multiplier * \sum_{ijk} weight_{pijk} * f_{ijk}

  Note that a functor implements the interpolation contribution between a
  single point, indexed with a local p index, and a single entity, indexed
  with local ijk indices. A single, potentially multi-dimensional result is
  provided as the contribution.
*/
template <class ViewType>
struct ScalarValueG2P
{
    static constexpr int value_count = 1;
    using value_type = typename ViewType::value_type;

    ViewType _x;
    value_type _multiplier;

    ScalarValueG2P( const ViewType &x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 1 == ViewType::Rank, "View must be of scalars" );
    }

    template <class SplineDataType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const SplineDataType &sd, const int p, const int i, const int j,
                const int k, const value_type *values ) const
    {
        _x( p ) += _multiplier * values[0] * sd.w[Dim::I][i] * sd.w[Dim::J][j] *
                   sd.w[Dim::K][k];
    }
};

template <class ViewType>
ScalarValueG2P<ViewType>
createScalarValueG2P( const ViewType &x,
                      const typename ViewType::value_type &multiplier )
{
    return ScalarValueG2P<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Grid-to-point vector value functor.

  Interpolates a vector function from entities to points with a given
  multiplier such that:

  f_{pd} = multiplier * \sum_{ijk} weight_{pijk} * f_{ijkd}

  Note that a functor implements the interpolation contribution between a
  single point, indexed with a local p index, and a single entity, indexed
  with local ijk indices. A single, potentially multi-dimensional result is
  provided as the contribution.
*/
template <class ViewType>
struct VectorValueG2P
{
    static constexpr int value_count = 3;
    using value_type = typename ViewType::value_type;

    ViewType _x;
    value_type _multiplier;

    VectorValueG2P( const ViewType &x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 2 == ViewType::Rank, "View must be of vectors" );
    }

    template <class SplineDataType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const SplineDataType &sd, const int p, const int i, const int j,
                const int k, const value_type *values ) const
    {
        value_type weight =
            _multiplier * sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
        for ( int d = 0; d < 3; ++d )
            _x( p, d ) += weight * values[d];
    }
};

template <class ViewType>
VectorValueG2P<ViewType>
createVectorValueG2P( const ViewType &x,
                      const typename ViewType::value_type &multiplier )
{
    return VectorValueG2P<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Grid-to-point scalar gradient functor.

  Interpolates the gradient of a scalar function from entities to points with
  a given multiplier such that:

  f_{pd} = multiplier * \sum_{ijk} grad_weight_{pijkd} * f_{ijk}

  Note that a functor implements the interpolation contribution between a
  single point, indexed with a local p index, and a single entity, indexed
  with local ijk indices. A single, potentially multi-dimensional result is
  provided as the contribution.
*/
template <class ViewType>
struct ScalarGradientG2P
{
    static constexpr int value_count = 1;
    using value_type = typename ViewType::value_type;

    ViewType _x;
    value_type _multiplier;

    ScalarGradientG2P( const ViewType &x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 2 == ViewType::Rank, "View must be of vectors" );
    }

    template <class SplineDataType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const SplineDataType &sd, const int p, const int i, const int j,
                const int k, const value_type *values ) const
    {
        auto mx = _multiplier * values[0];
        _x( p, Dim::I ) +=
            mx * sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
        _x( p, Dim::J ) +=
            mx * sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k];
        _x( p, Dim::K ) +=
            mx * sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
    }
};

template <class ViewType>
ScalarGradientG2P<ViewType>
createScalarGradientG2P( const ViewType &x,
                         const typename ViewType::value_type &multiplier )
{
    return ScalarGradientG2P<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Grid-to-point vector gradient functor.

  Interpolates the gradient of a vector function from entities to points with
  a given multiplier such that:

  f_{pmn} = multiplier * \sum_{ijk} grad_weight_{pijkm} * f_{ijkn}

  Note that a functor implements the interpolation contribution between a
  single point, indexed with a local p index, and a single entity, indexed
  with local ijk indices. A single, potentially multi-dimensional result is
  provided as the contribution.
*/
template <class ViewType>
struct VectorGradientG2P
{
    static constexpr int value_count = 3;
    using value_type = typename ViewType::value_type;

    ViewType _x;
    value_type _multiplier;

    VectorGradientG2P( const ViewType &x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 3 == ViewType::Rank, "View must be of tensors" );
    }

    template <class SplineDataType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const SplineDataType &sd, const int p, const int i, const int j,
                const int k, const value_type *values ) const
    {
        double rg[3] = {sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k],
                        sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k],
                        sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k]};

        for ( int d0 = 0; d0 < 3; ++d0 )
        {
            auto mg = _multiplier * rg[d0];
            for ( int d1 = 0; d1 < 3; ++d1 )
                _x( p, d0, d1 ) += mg * values[d1];
        }
    }
};

template <class ViewType>
VectorGradientG2P<ViewType>
createVectorGradientG2P( const ViewType &x,
                         const typename ViewType::value_type &multiplier )
{
    return VectorGradientG2P<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Grid-to-point vector value functor.

  Interpolates the divergence of a vector function from entities to points
  with a given multiplier such that:

  f_p = multiplier * \sum_d \sum_{ijk} grad_weight_{pijkd} * f_{ijkd}

  Note that a functor implements the interpolation contribution between a
  single point, indexed with a local p index, and a single entity, indexed
  with local ijk indices. A single, potentially multi-dimensional result is
  provided as the contribution.
*/
template <class ViewType>
struct VectorDivergenceG2P
{
    static constexpr int value_count = 3;
    using value_type = typename ViewType::value_type;

    ViewType _x;
    value_type _multiplier;

    VectorDivergenceG2P( const ViewType &x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 1 == ViewType::Rank, "View must be of scalars" );
    }

    template <class SplineDataType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const SplineDataType &sd, const int p, const int i, const int j,
                const int k, const value_type *values ) const
    {
        _x( p ) += ( values[Dim::I] * sd.g[Dim::I][i] * sd.w[Dim::J][j] *
                         sd.w[Dim::K][k] +

                     values[Dim::J] * sd.w[Dim::I][i] * sd.g[Dim::J][j] *
                         sd.w[Dim::K][k] +

                     values[Dim::K] * sd.w[Dim::I][i] * sd.w[Dim::J][j] *
                         sd.g[Dim::K][k] ) *
                   _multiplier;
    }
};

template <class ViewType>
VectorDivergenceG2P<ViewType>
createVectorDivergenceG2P( const ViewType &x,
                           const typename ViewType::value_type &multiplier )
{
    return VectorDivergenceG2P<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_INTERPOLATION_HPP
