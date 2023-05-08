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

/*!
  \file Cajita_Interpolation.hpp
  \brief Particle to grid and grid to particle interpolation
*/
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
#include <type_traits>

namespace Cajita
{
//---------------------------------------------------------------------------//
// LOCAL INTERPOLATION
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Local grid-to-point.
//---------------------------------------------------------------------------//
namespace G2P
{
//---------------------------------------------------------------------------//
/*!
  \brief Interpolate a scalar value to a point. 3D specialization.
  \param view A functor with view semantics of scalar grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The scalar value at the point.
*/
template <class ViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<3 == SplineDataType::num_space_dim, void>
    value( const ViewType& view, const SplineDataType& sd,
           PointDataType& result,
           typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                   void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "G2P::value requires spline weight values" );

    result = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
                result += view( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                sd.s[Dim::K][k], 0 ) *
                          sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k];
}

/*!
  \brief Interpolate a scalar value to a point. 2D specialization.
  \param view A functor with view semantics of scalar grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The scalar value at the point.
*/
template <class ViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == SplineDataType::num_space_dim, void>
    value( const ViewType& view, const SplineDataType& sd,
           PointDataType& result,
           typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                   void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "G2P::value requires spline weight values" );

    result = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            result += view( sd.s[Dim::I][i], sd.s[Dim::J][j], 0 ) *
                      sd.w[Dim::I][i] * sd.w[Dim::J][j];
}

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate a vector value to a point. 3D specialization.
  \param view A functor with view semantics of vector grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The vector value at the point.
*/
template <class ViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<3 == SplineDataType::num_space_dim, void>
    value( const ViewType& view, const SplineDataType& sd,
           PointDataType result[3],
           typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                   void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "G2P::value requires spline weight values" );

    for ( int d = 0; d < 3; ++d )
        result[d] = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
                for ( int d = 0; d < 3; ++d )
                    result[d] += view( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                       sd.s[Dim::K][k], d ) *
                                 sd.w[Dim::I][i] * sd.w[Dim::J][j] *
                                 sd.w[Dim::K][k];
}

/*!
  \brief Interpolate a vector value to a point. 2D specialization.
  \param view A functor with view semantics of vector grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The vector value at the point.
*/
template <class ViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == SplineDataType::num_space_dim, void>
    value( const ViewType& view, const SplineDataType& sd,
           PointDataType result[2],
           typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                   void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "G2P::value requires spline weight values" );

    for ( int d = 0; d < 2; ++d )
        result[d] = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int d = 0; d < 2; ++d )
                result[d] += view( sd.s[Dim::I][i], sd.s[Dim::J][j], d ) *
                             sd.w[Dim::I][i] * sd.w[Dim::J][j];
}

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate a scalar gradient to a point. 3D specialization.
  \param view A functor with view semantics of scalar grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The scalar gradient at the point.
*/
template <class ViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<3 == SplineDataType::num_space_dim, void>
    gradient( const ViewType& view, const SplineDataType& sd,
              PointDataType result[3],
              typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                      void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "G2P::gradient requires spline weight values" );
    static_assert( SplineDataType::has_weight_physical_gradients,
                   "G2P::gradient requires spline weight physical gradients" );

    for ( int d = 0; d < 3; ++d )
        result[d] = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                result[Dim::I] += view( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                        sd.s[Dim::K][k], 0 ) *
                                  sd.g[Dim::I][i] * sd.w[Dim::J][j] *
                                  sd.w[Dim::K][k];

                result[Dim::J] += view( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                        sd.s[Dim::K][k], 0 ) *
                                  sd.w[Dim::I][i] * sd.g[Dim::J][j] *
                                  sd.w[Dim::K][k];

                result[Dim::K] += view( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                        sd.s[Dim::K][k], 0 ) *
                                  sd.w[Dim::I][i] * sd.w[Dim::J][j] *
                                  sd.g[Dim::K][k];
            }
}

/*!
  \brief Interpolate a scalar gradient to a point. 2D specialization.
  \param view A functor with view semantics of scalar grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The scalar gradient at the point.
*/
template <class ViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == SplineDataType::num_space_dim, void>
    gradient( const ViewType& view, const SplineDataType& sd,
              PointDataType result[2],
              typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                      void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "G2P::gradient requires spline weight values" );
    static_assert( SplineDataType::has_weight_physical_gradients,
                   "G2P::gradient requires spline weight physical gradients" );

    for ( int d = 0; d < 2; ++d )
        result[d] = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
        {
            result[Dim::I] += view( sd.s[Dim::I][i], sd.s[Dim::J][j], 0 ) *
                              sd.g[Dim::I][i] * sd.w[Dim::J][j];

            result[Dim::J] += view( sd.s[Dim::I][i], sd.s[Dim::J][j], 0 ) *
                              sd.w[Dim::I][i] * sd.g[Dim::J][j];
        }
}

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate a vector gradient to a point. 3D specialization.
  \param view A functor with view semantics of vector grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The vector gradient at the point.
*/
template <class ViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<3 == SplineDataType::num_space_dim, void>
    gradient( const ViewType& view, const SplineDataType& sd,
              PointDataType result[3][3],
              typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                      void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "G2P::gradient requires spline weight values" );
    static_assert( SplineDataType::has_weight_physical_gradients,
                   "G2P::gradient requires spline weight physical gradients" );

    for ( int d0 = 0; d0 < 3; ++d0 )
        for ( int d1 = 0; d1 < 3; ++d1 )
            result[d0][d1] = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                typename SplineDataType::scalar_type rg[3] = {
                    sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k],
                    sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k],
                    sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k] };

                for ( int d0 = 0; d0 < 3; ++d0 )
                {
                    auto mg = rg[d0];
                    for ( int d1 = 0; d1 < 3; ++d1 )
                        result[d0][d1] +=
                            mg * view( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                       sd.s[Dim::K][k], d1 );
                }
            }
}

/*!
  \brief Interpolate a vector gradient to a point. 2D specialization.
  \param view A functor with view semantics of vector grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The vector gradient at the point.
*/
template <class ViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == SplineDataType::num_space_dim, void>
    gradient( const ViewType& view, const SplineDataType& sd,
              PointDataType result[2][2],
              typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                      void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "G2P::gradient requires spline weight values" );
    static_assert( SplineDataType::has_weight_physical_gradients,
                   "G2P::gradient requires spline weight physical gradients" );

    for ( int d0 = 0; d0 < 2; ++d0 )
        for ( int d1 = 0; d1 < 2; ++d1 )
            result[d0][d1] = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
        {
            typename SplineDataType::scalar_type rg[2] = {
                sd.g[Dim::I][i] * sd.w[Dim::J][j],
                sd.w[Dim::I][i] * sd.g[Dim::J][j] };

            for ( int d0 = 0; d0 < 2; ++d0 )
            {
                auto mg = rg[d0];
                for ( int d1 = 0; d1 < 2; ++d1 )
                    result[d0][d1] +=
                        mg * view( sd.s[Dim::I][i], sd.s[Dim::J][j], d1 );
            }
        }
}

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate a vector divergence to a point. 3D specialization.
  \param view A functor with view semantics of vector grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The vector divergence at the point.
*/
template <class ViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<3 == SplineDataType::num_space_dim, void>
    divergence( const ViewType& view, const SplineDataType& sd,
                PointDataType& result,
                typename std::enable_if<
                    ( std::rank<PointDataType>::value == 0 ), void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "G2P::divergence requires spline weight values" );
    static_assert(
        SplineDataType::has_weight_physical_gradients,
        "G2P::divergence requires spline weight physical gradients" );

    result = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
                result +=
                    view( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                          Dim::I ) *
                        sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k] +

                    view( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                          Dim::J ) *
                        sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k] +

                    view( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                          Dim::K ) *
                        sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k];
}

/*!
  \brief Interpolate a vector divergence to a point. 2D specialization.
  \param view A functor with view semantics of vector grid data from which to
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param result The vector divergence at the point.
*/
template <class ViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == SplineDataType::num_space_dim, void>
    divergence( const ViewType& view, const SplineDataType& sd,
                PointDataType& result,
                typename std::enable_if<
                    ( std::rank<PointDataType>::value == 0 ), void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "G2P::divergence requires spline weight values" );
    static_assert(
        SplineDataType::has_weight_physical_gradients,
        "G2P::divergence requires spline weight physical gradients" );

    result = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            result += view( sd.s[Dim::I][i], sd.s[Dim::J][j], Dim::I ) *
                          sd.g[Dim::I][i] * sd.w[Dim::J][j] +

                      view( sd.s[Dim::I][i], sd.s[Dim::J][j], Dim::J ) *
                          sd.w[Dim::I][i] * sd.g[Dim::J][j];
}

//---------------------------------------------------------------------------//

} // end namespace G2P

//---------------------------------------------------------------------------//
// Local point-to-grid.
//---------------------------------------------------------------------------//
namespace P2G
{
//---------------------------------------------------------------------------//
//! \cond Impl
template <class T>
struct is_scatter_view_impl : public std::false_type
{
};

template <typename DataType, typename Layout, typename ExecSpace, typename Op,
          typename duplication, typename contribution>
struct is_scatter_view_impl<Kokkos::Experimental::ScatterView<
    DataType, Layout, ExecSpace, Op, duplication, contribution>>
    : public std::true_type
{
};
//! \endcond

//! Scatter-View static type checker.
template <class T>
struct is_scatter_view
    : public is_scatter_view_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate a scalar value to the grid. 3D specialization.
  \param point_data The scalar value to at the point interpolate to the grid.
  \param sd The spline data to use for the interpolation.
  \param view The scatter view of scalar grid data to interpolate to.
*/
template <class PointDataType, class ScatterViewType, class SplineDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<3 == SplineDataType::num_space_dim, void>
    value( const PointDataType& point_data, const SplineDataType& sd,
           const ScatterViewType& view,
           typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                   void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "P2G::value requires spline weight values" );

    static_assert( is_scatter_view<ScatterViewType>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto view_access = view.access();

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
                view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                             0 ) += point_data * sd.w[Dim::I][i] *
                                    sd.w[Dim::J][j] * sd.w[Dim::K][k];
}

/*!
  \brief Interpolate a scalar value to the grid. 2D specialization.
  \param point_data The scalar value to at the point interpolate to the grid.
  \param sd The spline data to use for the interpolation.
  \param view The scatter view of scalar grid data to interpolate to.
*/
template <class PointDataType, class ScatterViewType, class SplineDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == SplineDataType::num_space_dim, void>
    value( const PointDataType& point_data, const SplineDataType& sd,
           const ScatterViewType& view,
           typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                   void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "P2G::value requires spline weight values" );

    static_assert( is_scatter_view<ScatterViewType>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto view_access = view.access();

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], 0 ) +=
                point_data * sd.w[Dim::I][i] * sd.w[Dim::J][j];
}

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate a vector value to the grid. 3D specialization.
  \param point_data The vector value at the point to interpolate to the grid.
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param view The scatter view of vector grid data to interpolate to.
*/
template <class PointDataType, class ScatterViewType, class SplineDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<3 == SplineDataType::num_space_dim, void>
    value( const PointDataType point_data[3], const SplineDataType& sd,
           const ScatterViewType& view,
           typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                   void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "P2G::value requires spline weight values" );

    static_assert( is_scatter_view<ScatterViewType>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto view_access = view.access();

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
                for ( int d = 0; d < 3; ++d )
                    view_access( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                 sd.s[Dim::K][k], d ) +=
                        point_data[d] * sd.w[Dim::I][i] * sd.w[Dim::J][j] *
                        sd.w[Dim::K][k];
}

/*!
  \brief Interpolate a vector value to the grid. 2D specialization.
  \param point_data The vector value at the point to interpolate to the grid.
  interpolate.
  \param sd The spline data to use for the interpolation.
  \param view The scatter view of vector grid data to interpolate to.
*/
template <class PointDataType, class ScatterViewType, class SplineDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == SplineDataType::num_space_dim, void>
    value( const PointDataType point_data[2], const SplineDataType& sd,
           const ScatterViewType& view,
           typename std::enable_if<( std::rank<PointDataType>::value == 0 ),
                                   void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "P2G::value requires spline weight values" );

    static_assert( is_scatter_view<ScatterViewType>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto view_access = view.access();

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int d = 0; d < 2; ++d )
                view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], d ) +=
                    point_data[d] * sd.w[Dim::I][i] * sd.w[Dim::J][j];
}

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate the gradient of a scalar to the grid. 3D specialization.
  \param point_data The scalar at the point who's gradient to interpolate to
  the grid.
  \param sd The spline data to use for the interpolation.
  \param view The scatter view of scalar gradient grid data to interpolate to.
*/
template <class PointDataType, class ScatterViewType, class SplineDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<3 == SplineDataType::num_space_dim, void>
    gradient( const PointDataType point_data, const SplineDataType& sd,
              const ScatterViewType& view )
{
    static_assert( SplineDataType::has_weight_values,
                   "P2G::gradient requires spline weight values" );
    static_assert( SplineDataType::has_weight_physical_gradients,
                   "P2G::gradient requires spline weight physical gradients" );

    static_assert( is_scatter_view<ScatterViewType>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto view_access = view.access();

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                             Dim::I ) += point_data * sd.g[Dim::I][i] *
                                         sd.w[Dim::J][j] * sd.w[Dim::K][k];

                view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                             Dim::J ) += point_data * sd.w[Dim::I][i] *
                                         sd.g[Dim::J][j] * sd.w[Dim::K][k];

                view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                             Dim::K ) += point_data * sd.w[Dim::I][i] *
                                         sd.w[Dim::J][j] * sd.g[Dim::K][k];
            }
}

/*!
  \brief Interpolate the gradient of a scalar to the grid. 2D specialization.
  \param point_data The scalar at the point who's gradient to interpolate to
  the grid.
  \param sd The spline data to use for the interpolation.
  \param view The scatter view of scalar gradient grid data to interpolate to.
*/
template <class PointDataType, class ScatterViewType, class SplineDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == SplineDataType::num_space_dim, void>
    gradient( const PointDataType point_data, const SplineDataType& sd,
              const ScatterViewType& view )
{
    static_assert( SplineDataType::has_weight_values,
                   "P2G::gradient requires spline weight values" );
    static_assert( SplineDataType::has_weight_physical_gradients,
                   "P2G::gradient requires spline weight physical gradients" );

    static_assert( is_scatter_view<ScatterViewType>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto view_access = view.access();

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
        {
            view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], Dim::I ) +=
                point_data * sd.g[Dim::I][i] * sd.w[Dim::J][j];

            view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], Dim::J ) +=
                point_data * sd.w[Dim::I][i] * sd.g[Dim::J][j];
        }
}

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate the divergence of a vector to the grid. 3D specialization.
  \param point_data The vector at the point who's divergence to interpolate to
  the grid.
  \param sd The spline data to use for the interpolation.
  \param view The scatter view of vector divergence grid data to interpolate to.
*/
template <class PointDataType, class ScatterViewType, class SplineDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<3 == SplineDataType::num_space_dim, void>
    divergence( const PointDataType point_data[3], const SplineDataType& sd,
                const ScatterViewType& view,
                typename std::enable_if<
                    ( std::rank<PointDataType>::value == 0 ), void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "P2G::divergence requires spline weight values" );
    static_assert(
        SplineDataType::has_weight_physical_gradients,
        "P2G::divergence requires spline weight physical gradients" );

    static_assert( is_scatter_view<ScatterViewType>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto view_access = view.access();

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                PointDataType result = point_data[Dim::I] * sd.g[Dim::I][i] *
                                           sd.w[Dim::J][j] * sd.w[Dim::K][k] +

                                       point_data[Dim::J] * sd.w[Dim::I][i] *
                                           sd.g[Dim::J][j] * sd.w[Dim::K][k] +

                                       point_data[Dim::K] * sd.w[Dim::I][i] *
                                           sd.w[Dim::J][j] * sd.g[Dim::K][k];

                view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], sd.s[Dim::K][k],
                             0 ) += result;
            }
}

/*!
  \brief Interpolate the divergence of a vector to the grid. 2D specialization.
  \param point_data The vector at the point who's divergence to interpolate to
  the grid.
  \param sd The spline data to use for the interpolation.
  \param view The scatter view of vector divergence grid data to interpolate to.
*/
template <class PointDataType, class ScatterViewType, class SplineDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == SplineDataType::num_space_dim, void>
    divergence( const PointDataType point_data[2], const SplineDataType& sd,
                const ScatterViewType& view,
                typename std::enable_if<
                    ( std::rank<PointDataType>::value == 0 ), void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "P2G::divergence requires spline weight values" );
    static_assert(
        SplineDataType::has_weight_physical_gradients,
        "P2G::divergence requires spline weight physical gradients" );

    static_assert( is_scatter_view<ScatterViewType>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto view_access = view.access();

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
        {
            PointDataType result =
                point_data[Dim::I] * sd.g[Dim::I][i] * sd.w[Dim::J][j] +

                point_data[Dim::J] * sd.w[Dim::I][i] * sd.g[Dim::J][j];

            view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], 0 ) += result;
        }
}

//---------------------------------------------------------------------------//
/*!
  \brief Interpolate the divergence of a tensor to the grid. 3D specialization.
  \param point_data The tensor at the point who's divergence to interpolate to
  the grid.
  \param sd The spline data to use for the interpolation.
  \param view The scatter view of tensor divergence grid data to interpolate to.
*/
template <class ScatterViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<3 == SplineDataType::num_space_dim, void>
    divergence( const PointDataType point_data[3][3], const SplineDataType& sd,
                const ScatterViewType& view,
                typename std::enable_if<
                    ( std::rank<PointDataType>::value == 0 ), void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "P2G::divergence requires spline weight values" );
    static_assert(
        SplineDataType::has_weight_physical_gradients,
        "P2G::divergence requires spline weight physical gradients" );

    static_assert( is_scatter_view<ScatterViewType>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto view_access = view.access();

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                typename SplineDataType::scalar_type rg[3] = {
                    sd.g[Dim::I][i] * sd.w[Dim::J][j] * sd.w[Dim::K][k],
                    sd.w[Dim::I][i] * sd.g[Dim::J][j] * sd.w[Dim::K][k],
                    sd.w[Dim::I][i] * sd.w[Dim::J][j] * sd.g[Dim::K][k] };

                for ( int d1 = 0; d1 < 3; ++d1 )
                {
                    for ( int d0 = 0; d0 < 3; ++d0 )
                        view_access( sd.s[Dim::I][i], sd.s[Dim::J][j],
                                     sd.s[Dim::K][k], d0 ) +=
                            rg[d1] * point_data[d0][d1];
                }
            }
}

/*!
  \brief Interpolate the divergence of a tensor to the grid. 2D specialization.
  \param point_data The tensor at the point who's divergence to interpolate to
  the grid.
  \param sd The spline data to use for the interpolation.
  \param view The scatter view of tensor divergence grid data to interpolate to.
*/
template <class ScatterViewType, class SplineDataType, class PointDataType>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<2 == SplineDataType::num_space_dim, void>
    divergence( const PointDataType point_data[2][2], const SplineDataType& sd,
                const ScatterViewType& view,
                typename std::enable_if<
                    ( std::rank<PointDataType>::value == 0 ), void*>::type = 0 )
{
    static_assert( SplineDataType::has_weight_values,
                   "P2G::divergence requires spline weight values" );
    static_assert(
        SplineDataType::has_weight_physical_gradients,
        "P2G::divergence requires spline weight physical gradients" );

    static_assert( is_scatter_view<ScatterViewType>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto view_access = view.access();

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
        {
            typename SplineDataType::scalar_type rg[2] = {
                sd.g[Dim::I][i] * sd.w[Dim::J][j],
                sd.w[Dim::I][i] * sd.g[Dim::J][j] };

            for ( int d1 = 0; d1 < 2; ++d1 )
            {
                for ( int d0 = 0; d0 < 2; ++d0 )
                    view_access( sd.s[Dim::I][i], sd.s[Dim::J][j], d0 ) +=
                        rg[d1] * point_data[d0][d1];
            }
        }
}

//---------------------------------------------------------------------------//

} // end namespace P2G

//---------------------------------------------------------------------------//
// GLOBAL INTERPOLATION
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Global grid-to-Point
//---------------------------------------------------------------------------//
/*!
 \brief Global Grid-to-Point interpolation.

  \tparam PointEvalFunctor Functor type used to evaluate the interpolated data
  for a given point at a given entity.

  \tparam PointCoordinates Container type with view traits containing the
  point coordinates. Will be indexed as (point,dim).

  \tparam ArrayScalar The scalar type used for the interpolated data.

  \tparam MeshScalar The scalar type used for the geometry/interpolation data.

  \tparam NumSpaceDim The spatial dimension of the mesh.

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

  \param functor A functor that interpolates from a given entity to a given
  point.

  \note Spline of SplineOrder passed for interpolation.
*/
template <class PointEvalFunctor, class PointCoordinates, class ArrayScalar,
          class MeshScalar, class EntityType, int SplineOrder,
          std::size_t NumSpaceDim, class DeviceType, class... ArrayParams>
void g2p(
    const Array<ArrayScalar, EntityType, UniformMesh<MeshScalar, NumSpaceDim>,
                ArrayParams...>& array,
    const Halo<DeviceType>& halo, const PointCoordinates& points,
    const std::size_t num_point, Spline<SplineOrder>,
    const PointEvalFunctor& functor )
{
    using array_type =
        Array<ArrayScalar, EntityType, UniformMesh<MeshScalar, NumSpaceDim>,
              ArrayParams...>;
    static_assert( std::is_same<typename Halo<DeviceType>::memory_space,
                                typename array_type::memory_space>::value,
                   "Mismatching points/array memory space." );

    using execution_space = typename DeviceType::execution_space;

    // Create the local mesh.
    auto local_mesh =
        createLocalMesh<DeviceType>( *( array.layout()->localGrid() ) );

    // Gather data into the halo before interpolating.
    halo.gather( execution_space(), array );

    // Get a view of the array data.
    auto array_view = array.view();

    // Loop over points and interpolate from the grid.
    Kokkos::parallel_for(
        "g2p", Kokkos::RangePolicy<execution_space>( 0, num_point ),
        KOKKOS_LAMBDA( const int p ) {
            // Get the point coordinates.
            MeshScalar px[NumSpaceDim];
            for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            {
                px[d] = points( p, d );
            }

            // Create the local spline data.
            using sd_type =
                SplineData<MeshScalar, SplineOrder, NumSpaceDim, EntityType>;
            sd_type sd;
            evaluateSpline( local_mesh, px, sd );

            // Evaluate the functor.
            functor( sd, p, array_view );
        } );
}

//---------------------------------------------------------------------------//
/*!
  \brief Grid-to-point scalar value functor.

  Interpolates a scalar function from entities to points with a given
  multiplier such that:

  f_p = multiplier * \\sum_{ijk} weight_{pijk} * f_{ijk}
*/
template <class ViewType>
struct ScalarValueG2P
{
    //! Scalar value type.
    using value_type = typename ViewType::value_type;

    //! Spline evaluation locations.
    ViewType _x;
    //! Scalar multiplier.
    value_type _multiplier;

    //! Constructor
    ScalarValueG2P( const ViewType& x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 1 == ViewType::rank, "View must be of scalars" );
    }

    //! Apply spline interplation.
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        value_type result;
        G2P::value( view, sd, result );
        _x( p ) += _multiplier * result;
    }
};

//! Creation function for grid-to-point with scalar value.
template <class ViewType>
ScalarValueG2P<ViewType>
createScalarValueG2P( const ViewType& x,
                      const typename ViewType::value_type& multiplier )
{
    return ScalarValueG2P<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Grid-to-point vector value functor.

  Interpolates a vector function from entities to points with a given
  multiplier such that:

  f_{pd} = multiplier * \\sum_{ijk} weight_{pijk} * f_{ijkd}
*/
template <class ViewType>
struct VectorValueG2P
{
    //! Scalar value type.
    using value_type = typename ViewType::value_type;

    //! Spline evaluation locations.
    ViewType _x;
    //! Scalar multiplier.
    value_type _multiplier;

    //! Constructor
    VectorValueG2P( const ViewType& x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 2 == ViewType::rank, "View must be of vectors" );
    }

    //! Apply spline interplation.
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        value_type result[SplineDataType::num_space_dim];
        G2P::value( view, sd, result );
        for ( std::size_t d = 0; d < SplineDataType::num_space_dim; ++d )
            _x( p, d ) += _multiplier * result[d];
    }
};

//! Creation function for grid-to-point with vector value.
template <class ViewType>
VectorValueG2P<ViewType>
createVectorValueG2P( const ViewType& x,
                      const typename ViewType::value_type& multiplier )
{
    return VectorValueG2P<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Grid-to-point scalar gradient functor.

  Interpolates the gradient of a scalar function from entities to points with
  a given multiplier such that:

  f_{pd} = multiplier * \\sum_{ijk} grad_weight_{pijkd} * f_{ijk}
*/
template <class ViewType>
struct ScalarGradientG2P
{
    //! Scalar value type.
    using value_type = typename ViewType::value_type;

    //! Spline evaluation locations.
    ViewType _x;
    //! Scalar multiplier.
    value_type _multiplier;

    //! Constructor
    ScalarGradientG2P( const ViewType& x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 2 == ViewType::rank, "View must be of vectors" );
    }

    //! Apply spline interplation.
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        value_type result[SplineDataType::num_space_dim];
        G2P::gradient( view, sd, result );
        for ( std::size_t d = 0; d < SplineDataType::num_space_dim; ++d )
            _x( p, d ) += _multiplier * result[d];
    }
};

//! Creation function for grid-to-point with scalar gradient.
template <class ViewType>
ScalarGradientG2P<ViewType>
createScalarGradientG2P( const ViewType& x,
                         const typename ViewType::value_type& multiplier )
{
    return ScalarGradientG2P<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Grid-to-point vector gradient functor.

  Interpolates the gradient of a vector function from entities to points with
  a given multiplier such that:

  f_{pmn} = multiplier * \\sum_{ijk} grad_weight_{pijkm} * f_{ijkn}
*/
template <class ViewType>
struct VectorGradientG2P
{
    //! Scalar value type.
    using value_type = typename ViewType::value_type;

    //! Spline evaluation locations.
    ViewType _x;
    //! Scalar multiplier.
    value_type _multiplier;

    //! Constructor
    VectorGradientG2P( const ViewType& x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 3 == ViewType::rank, "View must be of tensors" );
    }

    //! Apply spline interplation.
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        value_type result[SplineDataType::num_space_dim]
                         [SplineDataType::num_space_dim];
        G2P::gradient( view, sd, result );
        for ( std::size_t d0 = 0; d0 < SplineDataType::num_space_dim; ++d0 )
            for ( std::size_t d1 = 0; d1 < SplineDataType::num_space_dim; ++d1 )
                _x( p, d0, d1 ) += _multiplier * result[d0][d1];
    }
};

//! Creation function for grid-to-point with vector gradient.
template <class ViewType>
VectorGradientG2P<ViewType>
createVectorGradientG2P( const ViewType& x,
                         const typename ViewType::value_type& multiplier )
{
    return VectorGradientG2P<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Grid-to-point vector value functor.

  Interpolates the divergence of a vector function from entities to points
  with a given multiplier such that:

  f_p = multiplier * \\sum_d \\sum_{ijk} grad_weight_{pijkd} * f_{ijkd}
*/
template <class ViewType>
struct VectorDivergenceG2P
{
    //! Scalar value type.
    using value_type = typename ViewType::value_type;

    //! Spline evaluation locations.
    ViewType _x;
    //! Scalar multiplier.
    value_type _multiplier;

    //! Constructor
    VectorDivergenceG2P( const ViewType& x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 1 == ViewType::rank, "View must be of scalars" );
    }

    //! Apply spline interplation.
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        value_type result;
        G2P::divergence( view, sd, result );
        _x( p ) += result * _multiplier;
    }
};

//! Creation function for grid-to-point with vector divergence.
template <class ViewType>
VectorDivergenceG2P<ViewType>
createVectorDivergenceG2P( const ViewType& x,
                           const typename ViewType::value_type& multiplier )
{
    return VectorDivergenceG2P<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
// Global point-to-grid
//---------------------------------------------------------------------------//
/*!
  \brief Global Point-to-Grid interpolation.

  \tparam PointEvalFunctor Functor type used to evaluate the interpolated data
  for a given point at a given entity.

  \tparam PointCoordinates Container type with view traits containing the
  point coordinates. Will be indexed as (point,dim).

  \tparam ArrayScalar The scalar type used for the interpolated data.

  \tparam MeshScalar The scalar type used for the geometry/interpolation data.

  \tparam NumSpaceDim The spatial dimension of the mesh.

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

  \param halo The halo associated with the grid array. This hallo will be used
  to scatter the interpolated data.

  \param array The grid array to which the point data will be interpolated.

  \note Spline of SplineOrder passed for interpolation.
*/
template <class PointEvalFunctor, class PointCoordinates, class ArrayScalar,
          class MeshScalar, std::size_t NumSpaceDim, class EntityType,
          int SplineOrder, class DeviceType, class... ArrayParams>
void p2g( const PointEvalFunctor& functor, const PointCoordinates& points,
          const std::size_t num_point, Spline<SplineOrder>,
          const Halo<DeviceType>& halo,
          Array<ArrayScalar, EntityType, UniformMesh<MeshScalar, NumSpaceDim>,
                ArrayParams...>& array )
{
    using array_type =
        Array<ArrayScalar, EntityType, UniformMesh<MeshScalar, NumSpaceDim>,
              ArrayParams...>;
    static_assert( std::is_same<typename Halo<DeviceType>::memory_space,
                                typename array_type::memory_space>::value,
                   "Mismatching points/array memory space." );

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
            // Get the point coordinates.
            MeshScalar px[NumSpaceDim];
            for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            {
                px[d] = points( p, d );
            }

            // Create the local spline data.
            using sd_type =
                SplineData<MeshScalar, SplineOrder, NumSpaceDim, EntityType>;
            sd_type sd;
            evaluateSpline( local_mesh, px, sd );

            // Evaluate the functor.
            functor( sd, p, array_sv );
        } );
    Kokkos::Experimental::contribute( array_view, array_sv );

    // Scatter interpolation contributions in the halo back to their owning
    // ranks.
    halo.scatter( execution_space(), ScatterReduce::Sum(), array );
}

//---------------------------------------------------------------------------//
/*!
  \brief Point-to-grid scalar value functor.

  Interpolates a scalar function from points to entities with a given
  multiplier such that:

  f_ijk = multiplier * \\sum_p weight_{pijk} * f_p
*/
template <class ViewType>
struct ScalarValueP2G
{
    //! Scalar value type.
    using value_type = typename ViewType::value_type;

    //! Spline evaluation locations.
    ViewType _x;
    //! Scalar multiplier.
    value_type _multiplier;

    //! Constructor
    ScalarValueP2G( const ViewType& x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 1 == ViewType::rank, "View must be of scalars" );
    }

    //! Apply spline interplation.
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        value_type point_data = _x( p ) * _multiplier;
        P2G::value( point_data, sd, view );
    }
};

//! Creation function for point-to-grid with scalar value.
template <class ViewType>
ScalarValueP2G<ViewType>
createScalarValueP2G( const ViewType& x,
                      const typename ViewType::value_type& multiplier )
{
    return ScalarValueP2G<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Point-to-grid vector value functor.

  Interpolates a vector function from points to entities with a given
  multiplier such that:

  f_{ijkd} = multiplier * \\sum_p weight_{pijk} * f_{pd}
*/
template <class ViewType>
struct VectorValueP2G
{
    //! Scalar value type.
    using value_type = typename ViewType::value_type;

    //! Spline evaluation locations.
    ViewType _x;
    //! Scalar multiplier.
    value_type _multiplier;

    //! Constructor
    VectorValueP2G( const ViewType& x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 2 == ViewType::rank, "View must be of vectors" );
    }

    //! Apply spline interplation.
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        value_type point_data[SplineDataType::num_space_dim];
        for ( std::size_t d = 0; d < SplineDataType::num_space_dim; ++d )
            point_data[d] = _multiplier * _x( p, d );
        P2G::value( point_data, sd, view );
    }
};

//! Creation function for point-to-grid with vector value.
template <class ViewType>
VectorValueP2G<ViewType>
createVectorValueP2G( const ViewType& x,
                      const typename ViewType::value_type& multiplier )
{
    return VectorValueP2G<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Point-to-grid scalar gradient functor.

  Interpolates the gradient of a scalar function from points to entities with
  a given multiplier such that:

  f_{ijkd} = multiplier * \\sum_p grad_weight_{pijkd} * f_p
*/
template <class ViewType>
struct ScalarGradientP2G
{
    //! Scalar value type.
    using value_type = typename ViewType::value_type;

    //! Spline evaluation locations.
    ViewType _x;
    //! Scalar multiplier.
    value_type _multiplier;

    //! Constructor
    ScalarGradientP2G( const ViewType& x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 1 == ViewType::rank, "View must be of scalars" );
    }

    //! Apply spline interplation.
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        value_type point_data = _x( p ) * _multiplier;
        P2G::gradient( point_data, sd, view );
    }
};

//! Creation function for point-to-grid with scalar gradient.
template <class ViewType>
ScalarGradientP2G<ViewType>
createScalarGradientP2G( const ViewType& x,
                         const typename ViewType::value_type& multiplier )
{
    return ScalarGradientP2G<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Point-to-grid vector divergence functor.

  Interpolates the divergence of a vector function from points to entities
  with a given multiplier such that:

  f_ijk = multiplier * \\sum_d \\sum_p grad_weight_{pijkd} * f_{pd}
*/
template <class ViewType>
struct VectorDivergenceP2G
{
    //! Scalar value type.
    using value_type = typename ViewType::value_type;

    //! Spline evaluation locations.
    ViewType _x;
    //! Scalar multiplier.
    value_type _multiplier;

    //! Constructor
    VectorDivergenceP2G( const ViewType& x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 2 == ViewType::rank, "View must be of vectors" );
    }

    //! Apply spline interplation.
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        value_type point_data[SplineDataType::num_space_dim];
        for ( std::size_t d = 0; d < SplineDataType::num_space_dim; ++d )
            point_data[d] = _multiplier * _x( p, d );
        P2G::divergence( point_data, sd, view );
    }
};

//! Creation function for point-to-grid with vector divergence.
template <class ViewType>
VectorDivergenceP2G<ViewType>
createVectorDivergenceP2G( const ViewType& x,
                           const typename ViewType::value_type& multiplier )
{
    return VectorDivergenceP2G<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//
/*!
  \brief Point-to-grid tensor divergence functor.

  Interpolates the divergence of a tensor function from points to entities
  with a given multiplier such that:

  f_ijkm = multiplier * \\sum_n \\sum_p grad_weight_{pijkn} * f_{pmn}
*/
template <class ViewType>
struct TensorDivergenceP2G
{
    //! Scalar value type.
    using value_type = typename ViewType::value_type;

    //! Spline evaluation locations.
    ViewType _x;
    //! Scalar multiplier.
    value_type _multiplier;

    //! Constructor
    TensorDivergenceP2G( const ViewType& x, const value_type multiplier )
        : _x( x )
        , _multiplier( multiplier )
    {
        static_assert( 3 == ViewType::rank, "View must be of tensors" );
    }

    //! Apply spline interplation.
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        value_type point_data[SplineDataType::num_space_dim]
                             [SplineDataType::num_space_dim];
        for ( std::size_t d0 = 0; d0 < SplineDataType::num_space_dim; ++d0 )
            for ( std::size_t d1 = 0; d1 < SplineDataType::num_space_dim; ++d1 )
                point_data[d0][d1] = _multiplier * _x( p, d0, d1 );
        P2G::divergence( point_data, sd, view );
    }
};

//! Creation function for point-to-grid with tensor divergence.
template <class ViewType>
TensorDivergenceP2G<ViewType>
createTensorDivergenceP2G( const ViewType& x,
                           const typename ViewType::value_type& multiplier )
{
    return TensorDivergenceP2G<ViewType>( x, multiplier );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_INTERPOLATION_HPP
