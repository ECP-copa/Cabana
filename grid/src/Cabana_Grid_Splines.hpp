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
#include <Cabana_Utils.hpp> // FIXME: remove after next release.

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

namespace Cajita
{
//! \cond Deprecated
template <int Order>
using Spline CAJITA_DEPRECATED = Cabana::Grid::Spline<Order>;

using SplinePhysicalCellSize CAJITA_DEPRECATED =
    Cabana::Grid::SplinePhysicalCellSize;
using SplineLogicalPosition CAJITA_DEPRECATED =
    Cabana::Grid::SplineLogicalPosition;
using SplinePhysicalDistance CAJITA_DEPRECATED =
    Cabana::Grid::SplinePhysicalDistance;
using SplineWeightValues CAJITA_DEPRECATED = Cabana::Grid::SplineWeightValues;
using SplineWeightPhysicalGradients CAJITA_DEPRECATED =
    Cabana::Grid::SplineWeightPhysicalGradients;

template <class... DataTags>
using SplineDataMemberTypes CAJITA_DEPRECATED =
    Cabana::Grid::SplineDataMemberTypes<DataTags...>;

template <class T, class SplineDataMemberTypes_t>
using has_spline_tag CAJITA_DEPRECATED =
    Cabana::Grid::has_spline_tag<T, SplineDataMemberTypes_t>;

template <class Scalar, int Order, std::size_t NumSpaceDim, class Tag>
using SplineDataMember CAJITA_DEPRECATED =
    Cabana::Grid::SplineDataMember<Scalar, Order, NumSpaceDim, Tag>;

template <class Scalar, int Order, std::size_t NumSpaceDim, class EntityType,
          class Tags = void>
using SplineData CAJITA_DEPRECATED =
    Cabana::Grid::SplineData<Scalar, Order, NumSpaceDim, EntityType, Tags>;

template <class... Args>
CAJITA_DEPRECATED void KOKKOS_INLINE_FUNCTION setSplineData( Args&&... args )
{
    return Cabana::Grid::setSplineData( std::forward<Args>( args )... );
}

template <class... Args>
CAJITA_DEPRECATED void KOKKOS_INLINE_FUNCTION evaluateSpline( Args&&... args )
{
    return Cabana::Grid::evaluateSpline( std::forward<Args>( args )... );
}
//! \endcond
} // namespace Cajita

#endif // end CABANA_GRID_SPLINES_HPP
