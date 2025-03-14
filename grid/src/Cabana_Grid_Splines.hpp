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

/*!
  \file Cabana_Grid_Splines.hpp
  \brief Spline functions
*/
#ifndef CABANA_GRID_SPLINES_HPP
#define CABANA_GRID_SPLINES_HPP

#include <Cabana_Grid_LocalMesh.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
//! B-Spline interface for uniform grids.
template <int Order>
struct Spline;

//---------------------------------------------------------------------------//
//! Zero-order (Nearest Grid Point Interpolation). Defined on the dual grid.
template <>
struct Spline<0>
{
    //! Order.
    static constexpr int order = 0;

    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = 1;

    /*!
      \brief Map a physical location to the logical space of the dual grid in
      a single dimension.
      \param xp The coordinate to map to the logical space.
      \param rdx The inverse of the physical distance between grid locations.
      \param low_x The physical location of the low corner of the dual
      grid.
      \return The coordinate in the logical dual grid space.

      \note Casting this result to an integer yields the index at the center
      of the stencil.
      \note A zero-order spline uses the dual grid.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar
    mapToLogicalGrid( const Scalar xp, const Scalar rdx, const Scalar low_x )
    {
        return ( xp - low_x ) * rdx + 0.5;
    }

    /*!
      \brief Get the logical space stencil offsets of the spline. The stencil
      defines the offsets into a grid field about a logical coordinate.
      \param indices The stencil index offsets.
    */
    KOKKOS_INLINE_FUNCTION
    static void offsets( int indices[num_knot] ) { indices[0] = 0; }

    /*!
      \brief Compute the stencil indices for a given logical space location.
      \param x0 The coordinate at which to evaluate the spline stencil.
      \param indices The indices of the stencil.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void stencil( const Scalar x0,
                                                int indices[num_knot] )
    {
        indices[0] = static_cast<int>( x0 );
    }

    /*!
      \brief Calculate the value of the spline at all knots.
      \note x0 (unused) The coordinate at which to evaluate the spline in the
      logical grid space.
      \note values Basis values at the knots. Ordered from lowest to highest
      in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void value( const Scalar,
                                              Scalar values[num_knot] )
    {
        // Knot at i
        values[0] = 1.0;
    }

    /*!
      \brief Calculate the value of the gradient of the spline in the
      physical frame.
      \note x0 (unused) The coordinate at which to evaluate the spline in the
      logical grid space.
      \note rdx (unused) The inverse of the physical distance between grid
      locations.
      \note gradients Basis gradient values at the knots in the physical
      frame. Ordered from lowest to highest in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void gradient( const Scalar, const Scalar,
                                                 Scalar gradients[num_knot] )
    {
        // Knot at i.
        gradients[0] = 0.0;
    }
};

//---------------------------------------------------------------------------//
//! Linear. Defined on the primal grid.
template <>
struct Spline<1>
{
    //! Order.
    static constexpr int order = 1;

    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = 2;

    /*!
      \brief Map a physical location to the logical space of the primal grid in
      a single dimension.
      \param xp The coordinate to map to the logical space.
      \param rdx The inverse of the physical distance between grid locations.
      \param low_x The physical location of the low corner of the primal
      grid.
      \return The coordinate in the logical primal grid space.

      \note Casting this result to an integer yields the index at the center
      of the stencil.
      \note A linear spline uses the primal grid.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar
    mapToLogicalGrid( const Scalar xp, const Scalar rdx, const Scalar low_x )
    {
        return ( xp - low_x ) * rdx;
    }

    /*!
      \brief Get the logical space stencil offsets of the spline. The stencil
      defines the offsets into a grid field about a logical coordinate.
      \param indices The stencil index offsets.
    */
    KOKKOS_INLINE_FUNCTION
    static void offsets( int indices[num_knot] )
    {
        indices[0] = 0;
        indices[1] = 1;
    }

    /*!
      \brief Compute the stencil indices for a given logical space location.
      \param x0 The coordinate at which to evaluate the spline stencil.
      \param indices The indices of the stencil.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void stencil( const Scalar x0,
                                                int indices[num_knot] )
    {
        indices[0] = static_cast<int>( x0 );
        indices[1] = indices[0] + 1;
    }

    /*!
      \brief Calculate the value of the spline at all knots.
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param values Basis values at the knots. Ordered from lowest to highest
      in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void value( const Scalar x0,
                                              Scalar values[num_knot] )
    {
        // Knot at i
        Scalar xn = x0 - static_cast<int>( x0 );
        values[0] = 1.0 - xn;

        // Knot at i + 1
        values[1] = xn;
    }

    /*!
      \brief Calculate the value of the gradient of the spline in the
      physical frame.
      \note x0 (unused) The coordinate at which to evaluate the spline in the
      logical grid space.
      \note rdx The inverse of the physical distance between grid locations.
      \note gradients Basis gradient values at the knots in the physical
      frame. Ordered from lowest to highest in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void gradient( const Scalar, const Scalar rdx,
                                                 Scalar gradients[num_knot] )
    {
        // Knot at i.
        gradients[0] = -rdx;

        // Knot at i + 1;
        gradients[1] = rdx;
    }
};

//---------------------------------------------------------------------------//
//! Quadratic. Defined on the dual grid.
template <>
struct Spline<2>
{
    //! Order.
    static constexpr int order = 2;

    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = 3;

    /*!
      \brief Map a physical location to the logical space of the dual grid in a
      single dimension.
      \param xp The coordinate to map to the logical space.
      \param rdx The inverse of the physical distance between grid locations.
      \param low_x The physical location of the low corner of the dual grid.
      \return The coordinate in the logical dual grid space.

      \note Casting this result to an integer yields the index at the center
      of the stencil.
      \note A quadratic spline uses the dual grid.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar
    mapToLogicalGrid( const Scalar xp, const Scalar rdx, const Scalar low_x )
    {
        return ( xp - low_x ) * rdx + 0.5;
    }

    /*!
      \brief Get the logical space stencil offsets of the spline. The stencil
      defines the offsets into a grid field about a logical coordinate.
      \param indices The stencil index offsets.
    */
    KOKKOS_INLINE_FUNCTION
    static void offsets( int indices[num_knot] )
    {
        indices[0] = -1;
        indices[1] = 0;
        indices[2] = 1;
    }

    /*!
      \brief Compute the stencil indices for a given logical space location.
      \param x0 The coordinate at which to evaluate the spline stencil.
      \param indices The indices of the stencil.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void stencil( const Scalar x0,
                                                int indices[num_knot] )
    {
        indices[0] = static_cast<int>( x0 ) - 1;
        indices[1] = indices[0] + 1;
        indices[2] = indices[1] + 1;
    }

    /*!
       \brief Calculate the value of the spline at all knots.
       \param x0 The coordinate at which to evaluate the spline in the logical
       grid space.
       \param values Basis values at the knots. Ordered from lowest to highest
       in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void value( const Scalar x0,
                                              Scalar values[num_knot] )
    {
        // Constants
        Scalar nine_eights = 9.0 / 8.0;

        // Knot at i - 1
        Scalar xn = x0 - static_cast<int>( x0 ) + 0.5;
        values[0] = 0.5 * xn * xn - 1.5 * xn + nine_eights;

        // Knot at i
        xn -= 1.0;
        values[1] = -xn * xn + 0.75;

        // Knot at i + 1
        xn -= 1.0;
        values[2] = 0.5 * xn * xn + 1.5 * xn + nine_eights;
    }

    /*!
      \brief Calculate the value of the gradient of the spline in the
      physical frame.
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param rdx The inverse of the physical distance between grid locations.
      \param gradients Basis gradient values at the knots in the physical
      frame. Ordered from lowest to highest in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void
    gradient( const Scalar x0, const Scalar rdx, Scalar gradients[num_knot] )
    {
        // Knot at i - 1
        Scalar xn = x0 - static_cast<int>( x0 ) + 0.5;
        gradients[0] = ( xn - 1.5 ) * rdx;

        // Knot at i
        xn -= 1.0;
        gradients[1] = ( -2.0 * xn ) * rdx;

        // Knot at i + 1
        xn -= 1.0;
        gradients[2] = ( xn + 1.5 ) * rdx;
    }
};

//---------------------------------------------------------------------------//
//! Cubic. Defined on the primal grid.
template <>
struct Spline<3>
{
    //! Order.
    static constexpr int order = 3;

    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = 4;

    /*!
      \brief Map a physical location to the logical space of the primal grid in
      a single dimension. \param xp The coordinate to map to the logical space.
      \param rdx The inverse of the physical distance between grid locations.
      \param low_x The physical location of the low corner of the primal
      grid.
      \return The coordinate in the logical primal grid space.

      \note Casting this result to an integer yields the index at the center
      of the stencil.
      \note A cubic spline uses the primal grid.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar
    mapToLogicalGrid( const Scalar xp, const Scalar rdx, const Scalar low_x )
    {
        return ( xp - low_x ) * rdx;
    }

    /*!
      \brief Get the logical space stencil offsets of the spline. The stencil
      defines the offsets into a grid field about a logical coordinate.
      \param indices The stencil index offsets.
    */
    KOKKOS_INLINE_FUNCTION
    static void offsets( int indices[num_knot] )
    {
        indices[0] = -1;
        indices[1] = 0;
        indices[2] = 1;
        indices[3] = 2;
    }

    /*!
      \brief Compute the stencil indices for a given logical space location.
      \param x0 The coordinate at which to evaluate the spline stencil.
      \param indices The indices of the stencil.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void stencil( const Scalar x0,
                                                int indices[num_knot] )
    {
        indices[0] = static_cast<int>( x0 ) - 1;
        indices[1] = indices[0] + 1;
        indices[2] = indices[1] + 1;
        indices[3] = indices[2] + 1;
    }

    /*!
       \brief Calculate the value of the spline at all knots.
       \param x0 The coordinate at which to evaluate the spline in the logical
       grid space.
       \param values Basis values at the knots. Ordered from lowest to highest
       in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void value( const Scalar x0,
                                              Scalar values[num_knot] )
    {
        // Constants
        Scalar one_sixth = 1.0 / 6.0;
        Scalar two_thirds = one_sixth * 4.0;
        Scalar four_thirds = two_thirds * 2.0;

        // Knot at i - 1
        Scalar xn = x0 - static_cast<int>( x0 ) + 1.0;
        Scalar xn2 = xn * xn;
        values[0] = -xn * xn2 * one_sixth + xn2 - 2.0 * xn + four_thirds;

        // Knot at i
        xn -= 1.0;
        xn2 = xn * xn;
        values[1] = 0.5 * xn * xn2 - xn2 + two_thirds;

        // Knot at i + 1
        xn -= 1.0;
        xn2 = xn * xn;
        values[2] = -0.5 * xn * xn2 - xn2 + two_thirds;

        // Knot at i + 2
        xn -= 1.0;
        xn2 = xn * xn;
        values[3] = xn * xn2 * one_sixth + xn2 + 2.0 * xn + four_thirds;
    }

    /*!
      \brief Calculate the value of the gradient of the spline in the
      physical frame.
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param rdx The inverse of the physical distance between grid locations.
      \param gradients Basis gradient values at the knots in the physical
      frame. Ordered from lowest to highest in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void
    gradient( const Scalar x0, const Scalar rdx, Scalar gradients[num_knot] )
    {
        // Knot at i - 1
        Scalar xn = x0 - static_cast<int>( x0 ) + 1.0;
        gradients[0] = ( -0.5 * xn * xn + 2.0 * xn - 2.0 ) * rdx;

        // Knot at i
        xn -= 1.0;
        gradients[1] = ( 1.5 * xn * xn - 2.0 * xn ) * rdx;

        // Knot at i + 1
        xn -= 1.0;
        gradients[2] = ( -1.5 * xn * xn - 2.0 * xn ) * rdx;

        // Knot at i + 2
        xn -= 1.0;
        gradients[3] = ( 0.5 * xn * xn + 2.0 * xn + 2.0 ) * rdx;
    }
};

//! Quartic. Defined on the dual grid.
template <>
struct Spline<4>
{
    //! Order.
    static constexpr int order = 4;

    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = 5;

    /*!
      \brief Map a physical location to the logical space of the primal
      grid in a single dimension.
      \param xp The coordinate to map to the logical space.
      \param rdx The inverse of the physical distance between grid locations.
      \param low_x The physical location of the low corner of the primal grid.
      \return The coordinate in the logical primal grid space.

      \note Casting this result to an integer yields the index at the center of
      the stencil.
      \note A quartic spline uses the dual grid.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar
    mapToLogicalGrid( const Scalar xp, const Scalar rdx, const Scalar low_x )
    {
        return ( xp - low_x ) * rdx + 0.5;
    }

    /*!
     *       \brief Get the logical space stencil offsets of the spline. The
     * stencil defines the offsets into a grid field about a logical coordinate.
     *                   \param indices The stencil index offsets.
     *                       */
    KOKKOS_INLINE_FUNCTION
    static void offsets( int indices[num_knot] )
    {
        indices[0] = -2;
        indices[1] = -1;
        indices[2] = 0;
        indices[3] = 1;
        indices[4] = 2;
    }

    /*!
      \brief Compute the stencil indices for a given logical space location.
      \param x0 The coordinate at which to evaluate the spline stencil.
      \param indices The indices of the stencil.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void stencil( const Scalar x0,
                                                int indices[num_knot] )
    {
        indices[0] = static_cast<int>( x0 ) - 2;
        indices[1] = indices[0] + 1;
        indices[2] = indices[1] + 1;
        indices[3] = indices[2] + 1;
        indices[4] = indices[3] + 1;
    }

    /*!
      \brief Calculate the value of the spline at all knots.
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param values Basis values at the knots. Ordered from lowest to highest
      in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void value( const Scalar x0,
                                              Scalar values[num_knot] )
    {
        // Constants
        Scalar denom_2 = 1.0 / 384.0;
        Scalar denom_1 = denom_2 * 4.0;
        Scalar denom_0 = denom_2 * 2.0;

        // Knot at i - 2
        Scalar xn = x0 - static_cast<int>( x0 ) - 0.5;
        Scalar xn2 = xn * xn;
        Scalar xn4 = xn2 * xn2;
        values[0] =
            ( 16.0 * xn4 - 32.0 * xn * xn2 + 24.0 * xn2 - 8.0 * xn + 1.0 ) *
            denom_2;

        // Knot at i - 1
        values[1] =
            ( -16.0 * xn4 + 16.0 * xn * xn2 + 24.0 * xn2 - 44.0 * xn + 19.0 ) *
            denom_1;

        // Knot at i
        values[2] = ( 48.0 * xn4 - 120.0 * xn2 + 115.0 ) * denom_0;

        // Knot at i + 1
        values[3] =
            ( -16.0 * xn4 - 16.0 * xn * xn2 + 24.0 * xn2 + 44.0 * xn + 19.0 ) *
            denom_1;

        // Knot at i + 2
        values[4] =
            ( 16.0 * xn4 + 32.0 * xn * xn2 + 24.0 * xn2 + 8.0 * xn + 1 ) *
            denom_2;
    }

    /*!
      \brief Calculate the value of the gradient of the spline in the
      physical frame.
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param rdx The inverse of the physical distance between grid locations.
      \param gradients Basis gradient values at the knots in the physical frame.
      Ordered from lowest to highest in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void
    gradient( const Scalar x0, const Scalar rdx, Scalar gradients[num_knot] )
    {
        // Constants
        Scalar denom_2 = 1.0 / 48.0;
        Scalar denom_1 = denom_2 * 2.0;
        Scalar denom_0 = denom_2 * 12.0;

        // Knot at i - 2
        Scalar xn = x0 - static_cast<int>( x0 ) - 0.5;
        Scalar xn2 = xn * xn;
        gradients[0] =
            ( 8.0 * xn * xn2 - 12.0 * xn2 + 6.0 * xn - 1.0 ) * denom_2 * rdx;

        // Knot at i - 1
        gradients[1] = ( -16.0 * xn * xn2 + 12.0 * xn2 + 12.0 * xn - 11.0 ) *
                       denom_1 * rdx;

        // Knot at i
        gradients[2] = ( 4.0 * xn * xn2 - 5.0 * xn ) * denom_0 * rdx;

        // Knot at i + 1
        gradients[3] = ( -16.0 * xn * xn2 - 12.0 * xn2 + 12.0 * xn + 11.0 ) *
                       denom_1 * rdx;

        // Knot at i + 2
        gradients[4] =
            ( 8.0 * xn * xn2 + 12.0 * xn2 + 6.0 * xn + 1.0 ) * denom_2 * rdx;
    }
};

//! Quintic. Defined on the primal grid.
template <>
struct Spline<5>
{
    //! Order.
    static constexpr int order = 5;

    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = 6;

    /*!
      \brief Map a physical location to the logical space of the primal grid in
      a single dimension.
      \param xp The coordinate to map to the logical space.
      \param rdx The inverse of the physical distance between grid locations.
      \param low_x The physical location of the low corner of the primal grid.
      \return The coordinate in the logical primal grid space.

      \note Casting this result to an integer yields the index at the center of
      the stencil.
      \note A quintic spline uses the primal grid.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar
    mapToLogicalGrid( const Scalar xp, const Scalar rdx, const Scalar low_x )
    {
        return ( xp - low_x ) * rdx;
    }

    /*!
      \brief Get the logical space stencil offsets of the spline. The stencil
      defines the offsets into a grid field about a logical coordinate.
      \param indices The stencil index offsets.
    */
    KOKKOS_INLINE_FUNCTION
    static void offsets( int indices[num_knot] )
    {
        indices[0] = -2;
        indices[1] = -1;
        indices[2] = 0;
        indices[3] = 1;
        indices[4] = 2;
        indices[5] = 3;
    }

    /*!
      \brief Compute the stencil indices for a given logical space location.
      \param x0 The coordinate at which to evaluate the spline stencil.
      \param indices The indices of the stencil.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void stencil( const Scalar x0,
                                                int indices[num_knot] )
    {
        indices[0] = static_cast<int>( x0 ) - 2;
        indices[1] = indices[0] + 1;
        indices[2] = indices[1] + 1;
        indices[3] = indices[2] + 1;
        indices[4] = indices[3] + 1;
        indices[5] = indices[4] + 1;
    }

    /*!
      \brief Calculate the value of the spline at all knots.
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param values Basis values at the knots. Ordered from lowest to highest in
      terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void value( const Scalar x0,
                                              Scalar values[num_knot] )
    {
        // Constants
        Scalar denom_2 = 1.0 / 3840.0;
        Scalar denom_1 = denom_2 * 2.0;

        // Knot at i - 2
        Scalar xn = x0 - static_cast<int>( x0 ) - 0.5;
        Scalar xn2 = xn * xn;
        Scalar xn4 = xn2 * xn2;
        values[0] = ( -32.0 * xn * xn4 + 80.0 * xn4 - 80.0 * xn * xn2 +
                      40.0 * xn2 - 10.0 * xn + 1.0 ) *
                    denom_2;

        // Knot at i - 1
        values[1] = ( 160.0 * xn * xn4 - 240.0 * xn4 - 240.0 * xn * xn2 +
                      840.0 * xn2 - 750.0 * xn + 237.0 ) *
                    denom_2;

        // Knot at i
        values[2] = ( -160.0 * xn * xn4 + 80.0 * xn4 + 560.0 * xn * xn2 -
                      440.0 * xn2 - 770.0 * xn + 841.0 ) *
                    denom_1;

        // Knot at i + 1
        values[3] = ( 160.0 * xn * xn4 + 80.0 * xn4 - 560.0 * xn * xn2 -
                      440.0 * xn2 + 770.0 * xn + 841.0 ) *
                    denom_1;

        // Knot at i + 2
        values[4] = ( -160.0 * xn * xn4 - 240.0 * xn4 + 240.0 * xn * xn2 +
                      840.0 * xn2 + 750.0 * xn + 237.0 ) *
                    denom_2;

        // Knot at i + 3
        values[5] = ( 32.0 * xn * xn4 + 80.0 * xn4 + 80.0 * xn * xn2 +
                      40.0 * xn2 + 10.0 * xn + 1.0 ) *
                    denom_2;
    }

    /*!
      \brief Calculate the value of the gradient of the spline in the physical
      frame.
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param rdx The inverse of the physical distance between grid locations.
      \param gradients Basis gradient values at the knots in the physical frame.
      Ordered from lowest to highest in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void
    gradient( const Scalar x0, const Scalar rdx, Scalar gradients[num_knot] )
    {
        // Constants
        Scalar denom_2 = 1.0 / 384.0;
        Scalar denom_1 = denom_2 * 2.0;

        // Knot at i - 2
        Scalar xn = x0 - static_cast<int>( x0 ) - 0.5;
        Scalar xn2 = xn * xn;
        Scalar xn4 = xn2 * xn2;
        gradients[0] =
            ( -16.0 * xn4 + 32.0 * xn * xn2 - 24.0 * xn2 + 8.0 * xn - 1.0 ) *
            denom_2 * rdx;

        // Knot at i - 1
        gradients[1] =
            ( 80.0 * xn4 - 96.0 * xn * xn2 - 72.0 * xn2 + 168.0 * xn - 75.0 ) *
            denom_2 * rdx;

        // Knot at i
        gradients[2] =
            ( -80.0 * xn4 + 32.0 * xn * xn2 + 168.0 * xn2 - 88.0 * xn - 77.0 ) *
            denom_1 * rdx;

        // Knot at i + 1
        gradients[3] =
            ( 80.0 * xn4 + 32.0 * xn * xn2 - 168.0 * xn2 - 88.0 * xn + 77.0 ) *
            denom_1 * rdx;

        // Knot at i + 2
        gradients[4] =
            ( -80.0 * xn4 - 96.0 * xn * xn2 + 72.0 * xn2 + 168.0 * xn + 75.0 ) *
            denom_2 * rdx;

        // Knot at i + 3
        gradients[5] =
            ( 16.0 * xn4 + 32.0 * xn * xn2 + 24.0 * xn2 + 8.0 * xn + 1.0 ) *
            denom_2 * rdx;
    }
};

//! Sextic. Defined on the dual grid.
template <>
struct Spline<6>
{
    //! Order.
    static constexpr int order = 6;

    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = 7;

    /*!
      \brief Map a physical location to the logical space of the primal grid in
      a single dimension.
      \param xp The coordinate to map to the logical space.
      \param rdx The inverse of the physical distance between grid locations.
      \param low_x The physical location of the low corner of the primal grid.
      \return The coordinate in the logical primal grid space.

      \note Casting this result to an integer yields the index at the center of
      the stencil.
      \note A sextic spline uses the primal grid.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static Scalar
    mapToLogicalGrid( const Scalar xp, const Scalar rdx, const Scalar low_x )
    {
        return ( xp - low_x ) * rdx + 0.5;
    }

    /*!
      \brief Get the logical space stencil offsets of the spline. The stencil
      defines the offsets into a grid field about a logical coordinate.
      \param indices The stencil index offsets.
    */
    KOKKOS_INLINE_FUNCTION
    static void offsets( int indices[num_knot] )
    {
        indices[0] = -3;
        indices[1] = -2;
        indices[2] = -1;
        indices[3] = 0;
        indices[4] = 1;
        indices[5] = 2;
        indices[6] = 3;
    }

    /*!
      \brief Compute the stencil indices for a given logical space location.
      \param x0 The coordinate at which to evaluate the spline stencil.
      \param indices The indices of the stencil.
    */
    template <class Scalar>
    KOKKOS_INLINE_FUNCTION static void stencil( const Scalar x0,
                                                int indices[num_knot] )
    {
        indices[0] = static_cast<int>( x0 ) - 3;
        indices[1] = indices[0] + 1;
        indices[2] = indices[1] + 1;
        indices[3] = indices[2] + 1;
        indices[4] = indices[3] + 1;
        indices[5] = indices[4] + 1;
        indices[6] = indices[5] + 1;
    }

    /*!
      \brief Calculate the value of the spline at all knots.
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param values Basis values at the knots. Ordered from lowest to highest
      in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void value( const Scalar x0,
                                              Scalar values[num_knot] )
    {
        // Constants
        Scalar denom_31 = 1.0 / 46080.0;
        Scalar denom_2 = denom_31 * 2.0;
        Scalar denom_0 = denom_2 * 2.0;

        // Knot at i - 3
        Scalar xn = x0 - static_cast<int>( x0 ) - 0.5;
        Scalar xn2 = xn * xn;
        Scalar xn4 = xn2 * xn2;
        Scalar xn6 = xn2 * xn4;
        values[0] = ( 64.0 * xn6 - 192.0 * xn * xn4 + 240.0 * xn4 -
                      160.0 * xn * xn2 + 60.0 * xn2 - 12.0 * xn + 1.0 ) *
                    denom_31;

        // Knot at i - 2
        values[1] = ( -192.0 * xn6 + 384.0 * xn * xn4 + 240.0 * xn4 -
                      1600.0 * xn * xn2 + 2220.0 * xn2 - 1416.0 * xn + 361.0 ) *
                    denom_2;

        // Knot at i - 1
        values[2] =
            ( 960.0 * xn6 - 960.0 * xn * xn4 - 4080.0 * xn4 +
              6880.0 * xn * xn2 + 4740.0 * xn2 - 17340.0 * xn + 10543.0 ) *
            denom_31;

        // Knot at i
        values[3] =
            ( -320.0 * xn6 + 1680.0 * xn4 - 4620.0 * xn2 + 5887.0 ) * denom_0;

        // Knot at i + 1
        values[4] =
            ( 960.0 * xn6 + 960.0 * xn * xn4 - 4080.0 * xn4 -
              6880.0 * xn * xn2 + 4740.0 * xn2 + 17340.0 * xn + 10543.0 ) *
            denom_31;

        // Knot at i + 2
        values[5] = ( -192.0 * xn6 - 384.0 * xn * xn4 + 240.0 * xn4 +
                      1600.0 * xn * xn2 + 2220.0 * xn2 + 1416.0 * xn + 361.0 ) *
                    denom_2;

        // Knot at i + 3
        values[6] = ( 64.0 * xn6 + 192.0 * xn * xn4 + 240.0 * xn4 +
                      160.0 * xn * xn2 + 60.0 * xn2 + 12.0 * xn + 1.0 ) *
                    denom_31;
    }

    /*!
      \brief Calculate the value of the gradient of the spline in the physical
      frame.
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param rdx The inverse of the physical distance between grid locations.
      \param gradients Basis gradient values at the knots in the physical frame.
      Ordered from lowest to highest in terms of knot location.
    */
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION static void
    gradient( const Scalar x0, const Scalar rdx, Scalar gradients[num_knot] )
    {
        // Constants
        Scalar denom_3 = 1.0 / 3840.0;
        Scalar denom_2 = denom_3 * 4.0;
        Scalar denom_1 = denom_3 * 5.0;
        Scalar denom_0 = denom_3 * 40.0;

        // Knot at i - 2
        Scalar xn = x0 - static_cast<int>( x0 ) - 0.5;
        Scalar xn2 = xn * xn;
        Scalar xn4 = xn2 * xn2;
        gradients[0] = ( 32.0 * xn * xn4 - 80.0 * xn4 + 80.0 * xn * xn2 -
                         40.0 * xn2 + 10.0 * xn - 1.0 ) *
                       denom_3 * rdx;

        // Knot at i - 2
        gradients[1] = ( -48.0 * xn * xn4 + 80.0 * xn4 + 40.0 * xn * xn2 -
                         200.0 * xn2 + 185.0 * xn - 59.0 ) *
                       denom_2 * rdx;

        // Knot at i - 1
        gradients[2] = ( 96.0 * xn * xn4 - 80.0 * xn4 - 272.0 * xn * xn2 +
                         344.0 * xn2 + 158.0 * xn - 289.0 ) *
                       denom_1 * rdx;

        // Knot at i
        gradients[3] =
            ( -16.0 * xn * xn4 + 56.0 * xn * xn2 - 77.0 * xn ) * denom_0 * rdx;

        // Knot at i + 1
        gradients[4] = ( 96.0 * xn * xn4 + 80.0 * xn4 - 272.0 * xn * xn2 -
                         344.0 * xn2 + 158.0 * xn + 289.0 ) *
                       denom_1 * rdx;

        // Knot at i + 2
        gradients[5] = ( -48.0 * xn * xn4 - 80.0 * xn4 + 40.0 * xn * xn2 +
                         200.0 * xn2 + 185.0 * xn + 59.0 ) *
                       denom_2 * rdx;

        // Knot at i + 3
        gradients[6] = ( 32.0 * xn * xn4 + 80.0 * xn4 + 80.0 * xn * xn2 +
                         40.0 * xn2 + 10.0 * xn + 1.0 ) *
                       denom_3 * rdx;
    }
};

//---------------------------------------------------------------------------//
// Spline Data
//---------------------------------------------------------------------------//
//! Spline data tag: physical cell size.
struct SplinePhysicalCellSize
{
};
//! Spline data tag: logical position.
struct SplineLogicalPosition
{
};
//! Spline data tag: physical distance.
struct SplinePhysicalDistance
{
};
//! Spline data tag: weight value.
struct SplineWeightValues
{
};
//! Spline data tag: physical gradient.
struct SplineWeightPhysicalGradients
{
};

//! Spline data members holder
template <class... DataTags>
struct SplineDataMemberTypes
{
};

//! Determine if a given spline tag is present.
template <typename T, typename SplineDataMemberTypes_t>
struct has_spline_tag;

//! \cond Impl
template <typename T>
struct has_spline_tag<T, SplineDataMemberTypes<>> : std::false_type
{
};

template <typename T, typename U, typename... Tags>
struct has_spline_tag<T, SplineDataMemberTypes<U, Tags...>>
    : has_spline_tag<T, SplineDataMemberTypes<Tags...>>
{
};
template <typename T, typename... Tags>
struct has_spline_tag<T, SplineDataMemberTypes<T, Tags...>> : std::true_type
{
};
//! \endcond

//! Spline data member.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class Tag>
struct SplineDataMember;

//! Physical cell size spline data member.
template <typename Scalar, int Order, std::size_t NumSpaceDim>
struct SplineDataMember<Scalar, Order, NumSpaceDim, SplinePhysicalCellSize>
{
    //! Physical cell size.
    Scalar dx[NumSpaceDim];
};

//! Logical position spline data member.
template <typename Scalar, int Order, std::size_t NumSpaceDim>
struct SplineDataMember<Scalar, Order, NumSpaceDim, SplineLogicalPosition>
{
    //! Logical position.
    Scalar x[NumSpaceDim];
};

//! Physical distance spline data member.
template <typename Scalar, int Order, std::size_t NumSpaceDim>
struct SplineDataMember<Scalar, Order, NumSpaceDim, SplinePhysicalDistance>
{
    //! Spline type.
    using spline_type = Spline<Order>;
    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = spline_type::num_knot;

    //! Physical distance.
    Scalar d[NumSpaceDim][num_knot];
};

//! Weight values spline data member.
template <typename Scalar, int Order, std::size_t NumSpaceDim>
struct SplineDataMember<Scalar, Order, NumSpaceDim, SplineWeightValues>
{
    //! Spline type.
    using spline_type = Spline<Order>;
    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = spline_type::num_knot;

    //! Weight values.
    Scalar w[NumSpaceDim][num_knot];
};

//! Weight physical gradients spline data member.
template <typename Scalar, int Order, std::size_t NumSpaceDim>
struct SplineDataMember<Scalar, Order, NumSpaceDim,
                        SplineWeightPhysicalGradients>
{
    //! Spline type.
    using spline_type = Spline<Order>;
    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = spline_type::num_knot;

    //! Weight physical gradients.
    Scalar g[NumSpaceDim][num_knot];
};

//! Spline data container.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class Tags = void>
struct SplineData;

//! Default of void has all data members.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType>
struct SplineData<Scalar, Order, NumSpaceDim, EntityType, void>
    : SplineDataMember<Scalar, Order, NumSpaceDim, SplinePhysicalCellSize>,
      SplineDataMember<Scalar, Order, NumSpaceDim, SplineLogicalPosition>,
      SplineDataMember<Scalar, Order, NumSpaceDim, SplinePhysicalDistance>,
      SplineDataMember<Scalar, Order, NumSpaceDim, SplineWeightValues>,
      SplineDataMember<Scalar, Order, NumSpaceDim,
                       SplineWeightPhysicalGradients>
{
    //! Spline scalar type.
    using scalar_type = Scalar;
    //! Spline order.
    static constexpr int order = Order;
    //! Spline type.
    using spline_type = Spline<Order>;
    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = spline_type::num_knot;
    //! Entity type.
    using entity_type = EntityType;
    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    //! Physical cell size present.
    static constexpr bool has_physical_cell_size = true;
    //! Logical position present.
    static constexpr bool has_logical_position = true;
    //! Physical distance present.
    static constexpr bool has_physical_distance = true;
    //! Weight values present.
    static constexpr bool has_weight_values = true;
    //! Weight physical gradients present.
    static constexpr bool has_weight_physical_gradients = true;

    //! Local interpolation stencil.
    int s[NumSpaceDim][num_knot];
};

//! Specify each data member individually through tags.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class... Tags>
struct SplineData<Scalar, Order, NumSpaceDim, EntityType,
                  SplineDataMemberTypes<Tags...>>
    : SplineDataMember<Scalar, Order, NumSpaceDim, Tags>...
{
    //! Spline scalar type.
    using scalar_type = Scalar;
    //! Spline order.
    static constexpr int order = Order;
    //! Spline type.
    using spline_type = Spline<Order>;
    //! The number of non-zero knots in the spline.
    static constexpr int num_knot = spline_type::num_knot;
    //! Entity type.
    using entity_type = EntityType;
    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    //! Spline member types.
    using member_tags = SplineDataMemberTypes<Tags...>;
    //! Is physical cell size present.
    static constexpr bool has_physical_cell_size =
        has_spline_tag<SplinePhysicalCellSize, member_tags>::value;
    //! Is logical position present.
    static constexpr bool has_logical_position =
        has_spline_tag<SplineLogicalPosition, member_tags>::value;
    //! Is physical distance present.
    static constexpr bool has_physical_distance =
        has_spline_tag<SplinePhysicalDistance, member_tags>::value;
    //! Are weight values present.
    static constexpr bool has_weight_values =
        has_spline_tag<SplineWeightValues, member_tags>::value;
    //! Are weight physical gradients present.
    static constexpr bool has_weight_physical_gradients =
        has_spline_tag<SplineWeightPhysicalGradients, member_tags>::value;

    //! Local interpolation stencil.
    int s[NumSpaceDim][num_knot];
};

//! Assign physical cell size to the spline data.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<SplineData<Scalar, Order, NumSpaceDim, EntityType,
                                DataTags>::has_physical_cell_size>
    setSplineData(
        SplinePhysicalCellSize,
        SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>& data,
        const int d, const Scalar dx )
{
    data.dx[d] = dx;
}
//! Physical cell size spline data template helper.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!SplineData<
    Scalar, Order, NumSpaceDim, EntityType, DataTags>::has_physical_cell_size>
setSplineData( SplinePhysicalCellSize,
               SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>&,
               const int, const Scalar )
{
}

//! Assign logical position to the spline data.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<SplineData<Scalar, Order, NumSpaceDim, EntityType,
                                DataTags>::has_logical_position>
    setSplineData(
        SplineLogicalPosition,
        SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>& data,
        const int d, const Scalar x )
{
    data.x[d] = x;
}
//! Logical position spline data template helper.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!SplineData<
    Scalar, Order, NumSpaceDim, EntityType, DataTags>::has_logical_position>
setSplineData( SplineLogicalPosition,
               SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>&,
               const int, const Scalar )
{
}

//! Assign weight values to the spline data.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<SplineData<Scalar, Order, NumSpaceDim, EntityType,
                                DataTags>::has_weight_values>
    setSplineData(
        SplineWeightValues,
        SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>& data,
        const Scalar x[NumSpaceDim] )

{
    using spline_type = typename SplineData<Scalar, Order, NumSpaceDim,
                                            EntityType, DataTags>::spline_type;
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
        spline_type::value( x[d], data.w[d] );
}
//! Weight value spline data template helper.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!SplineData<
    Scalar, Order, NumSpaceDim, EntityType, DataTags>::has_weight_values>
setSplineData( SplineWeightValues,
               SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>&,
               const Scalar[NumSpaceDim] )

{
}

//! Assign weight physical gradients to the spline data.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<SplineData<Scalar, Order, NumSpaceDim, EntityType,
                                DataTags>::has_weight_physical_gradients>
    setSplineData(
        SplineWeightPhysicalGradients,
        SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>& data,
        const Scalar x[NumSpaceDim], const Scalar rdx[NumSpaceDim] )

{
    using spline_type = typename SplineData<Scalar, Order, NumSpaceDim,
                                            EntityType, DataTags>::spline_type;
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
        spline_type::gradient( x[d], rdx[d], data.g[d] );
}
//! Weight physical gradients spline data template helper.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<!SplineData<Scalar, Order, NumSpaceDim, EntityType,
                                 DataTags>::has_weight_physical_gradients>
    setSplineData(
        SplineWeightPhysicalGradients,
        SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>&,
        const Scalar[NumSpaceDim], const Scalar[NumSpaceDim] )

{
}

//! Assign physical distance to the spline data.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<SplineData<Scalar, Order, NumSpaceDim, EntityType,
                                DataTags>::has_physical_distance>
    setSplineData(
        SplinePhysicalDistance,
        SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>& data,
        const Scalar low_x[NumSpaceDim], const Scalar p[NumSpaceDim],
        const Scalar dx[NumSpaceDim] )

{
    using spline_type = typename SplineData<Scalar, Order, NumSpaceDim,
                                            EntityType, DataTags>::spline_type;
    Scalar offset;
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
    {
        offset = low_x[d] - p[d];
        for ( int n = 0; n < spline_type::num_knot; ++n )
            data.d[d][n] = offset + data.s[d][n] * dx[d];
    }
}
//! Physical distance spline data template helper.
template <typename Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!SplineData<
    Scalar, Order, NumSpaceDim, EntityType, DataTags>::has_physical_distance>
setSplineData( SplinePhysicalDistance,
               SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>&,
               const Scalar[NumSpaceDim], const Scalar[NumSpaceDim],
               const Scalar[NumSpaceDim] )
{
}

//---------------------------------------------------------------------------//
//! Evaluate spline data at a point in a uniform mesh.
template <typename Scalar, int Order, std::size_t NumSpaceDim,
          class MemorySpace, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION void evaluateSpline(
    const LocalMesh<MemorySpace, UniformMesh<Scalar, NumSpaceDim>>& local_mesh,
    const Scalar p[NumSpaceDim],
    SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>& data )
{
    // data type
    using sd_type =
        SplineData<Scalar, Order, NumSpaceDim, EntityType, DataTags>;

    // spline type.

    // Get the low corner of the mesh.
    Scalar low_x[NumSpaceDim];
    Scalar low_x_p1[NumSpaceDim];
    int low_id[NumSpaceDim];
    int low_id_p1[NumSpaceDim];
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
    {
        low_id[d] = 0;
        low_id_p1[d] = 1;
    }
    local_mesh.coordinates( EntityType(), low_id, low_x );
    local_mesh.coordinates( EntityType(), low_id_p1, low_x_p1 );

    // Compute the physical cell size.
    Scalar dx[NumSpaceDim];
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
    {
        dx[d] = low_x_p1[d] - low_x[d];
        setSplineData( SplinePhysicalCellSize(), data, d, dx[d] );
    }

    // Compute the inverse physicall cell size.
    Scalar rdx[NumSpaceDim];
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
        rdx[d] = 1.0 / dx[d];

    // Compute the reference coordinates.
    Scalar x[NumSpaceDim];
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
    {
        x[d] = sd_type::spline_type::mapToLogicalGrid( p[d], rdx[d], low_x[d] );
        setSplineData( SplineLogicalPosition(), data, d, x[d] );
    }

    // Compute the stencil.
    for ( std::size_t d = 0; d < NumSpaceDim; ++d )
    {
        sd_type::spline_type::stencil( x[d], data.s[d] );
    }

    // Compute the weight values.
    setSplineData( SplineWeightValues(), data, x );

    // Compute the weight gradients.
    setSplineData( SplineWeightPhysicalGradients(), data, x, rdx );

    // Compute the physical distance.
    setSplineData( SplinePhysicalDistance(), data, low_x, p, dx );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_SPLINES_HPP
