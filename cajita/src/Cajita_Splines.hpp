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

#ifndef CAJITA_SPLINES_HPP
#define CAJITA_SPLINES_HPP

#include <Cajita_LocalMesh.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cajita
{
//---------------------------------------------------------------------------//
// B-Spline interface for uniform grids.
//---------------------------------------------------------------------------//
template <int Order>
struct Spline;

//---------------------------------------------------------------------------//
// Zero-order (Nearest Grid Point Interpolation). Defined on the dual grid.
template <>
struct Spline<0>
{
    // Order.
    static constexpr int order = 0;

    // The number of non-zero knots in the spline.
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

    /*
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
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param values Basis values at the knots. Ordered from lowest to highest
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
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param rdx The inverse of the physical distance between grid locations.
      \param gradients Basis gradient values at the knots in the physical
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
// Linear. Defined on the primal grid.
template <>
struct Spline<1>
{
    // Order.
    static constexpr int order = 1;

    // The number of non-zero knots in the spline.
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

    /*
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
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param rdx The inverse of the physical distance between grid locations.
      \param gradients Basis gradient values at the knots in the physical
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
// Quadratic. Defined on the dual grid.
template <>
struct Spline<2>
{
    // Order.
    static constexpr int order = 2;

    // The number of non-zero knots in the spline.
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

    /*
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
// Cubic. Defined on the primal grid.
template <>
struct Spline<3>
{
    // Order.
    static constexpr int order = 3;

    // The number of non-zero knots in the spline.
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

    /*
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
// Spline data tags.
struct SplinePhysicalCellSize
{
};
struct SplineLogicalPosition
{
};
struct SplinePhysicalDistance
{
};
struct SplineWeightValues
{
};
struct SplineWeightPhysicalGradients
{
};

// Data members holder
template <class... DataTags>
struct SplineDataMemberTypes
{
};

// Determine if a given tag is present.
template <typename T, typename SplineDataMemberTypes_t>
struct has_spline_tag;

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

// Spline data members.
template <typename Scalar, int Order, class Tag>
struct SplineDataMember;

template <typename Scalar, int Order>
struct SplineDataMember<Scalar, Order, SplinePhysicalCellSize>
{
    // Physical cell size.
    Scalar dx[3];
};

template <typename Scalar, int Order>
struct SplineDataMember<Scalar, Order, SplineLogicalPosition>
{
    // Logical position.
    Scalar x[3];
};

template <typename Scalar, int Order>
struct SplineDataMember<Scalar, Order, SplinePhysicalDistance>
{
    using spline_type = Spline<Order>;
    static constexpr int num_knot = spline_type::num_knot;

    // Physical distance.
    Scalar d[3][num_knot];
};

template <typename Scalar, int Order>
struct SplineDataMember<Scalar, Order, SplineWeightValues>
{
    using spline_type = Spline<Order>;
    static constexpr int num_knot = spline_type::num_knot;

    // Weight values.
    Scalar w[3][num_knot];
};

template <typename Scalar, int Order>
struct SplineDataMember<Scalar, Order, SplineWeightPhysicalGradients>
{
    using spline_type = Spline<Order>;
    static constexpr int num_knot = spline_type::num_knot;

    // Weight physical gradients.
    Scalar g[3][num_knot];
};

// Spline data container.
template <typename Scalar, int Order, class EntityType, class Tags = void>
struct SplineData;

// Default of void has all data members.
template <typename Scalar, int Order, class EntityType>
struct SplineData<Scalar, Order, EntityType, void>
    : SplineDataMember<Scalar, Order, SplinePhysicalCellSize>,
      SplineDataMember<Scalar, Order, SplineLogicalPosition>,
      SplineDataMember<Scalar, Order, SplinePhysicalDistance>,
      SplineDataMember<Scalar, Order, SplineWeightValues>,
      SplineDataMember<Scalar, Order, SplineWeightPhysicalGradients>
{
    using scalar_type = Scalar;
    static constexpr int order = Order;
    using spline_type = Spline<Order>;
    static constexpr int num_knot = spline_type::num_knot;
    using entity_type = EntityType;

    static constexpr bool has_physical_cell_size = true;
    static constexpr bool has_logical_position = true;
    static constexpr bool has_physical_distance = true;
    static constexpr bool has_weight_values = true;
    static constexpr bool has_weight_physical_gradients = true;

    // Local interpolation stencil.
    int s[3][num_knot];
};

// Specify each data member individually through tags.
template <typename Scalar, int Order, class EntityType, class... Tags>
struct SplineData<Scalar, Order, EntityType, SplineDataMemberTypes<Tags...>>
    : SplineDataMember<Scalar, Order, Tags>...
{
    using scalar_type = Scalar;
    static constexpr int order = Order;
    using spline_type = Spline<Order>;
    static constexpr int num_knot = spline_type::num_knot;
    using entity_type = EntityType;

    using member_tags = SplineDataMemberTypes<Tags...>;

    static constexpr bool has_physical_cell_size =
        has_spline_tag<SplinePhysicalCellSize, member_tags>::value;
    static constexpr bool has_logical_position =
        has_spline_tag<SplineLogicalPosition, member_tags>::value;
    static constexpr bool has_physical_distance =
        has_spline_tag<SplinePhysicalDistance, member_tags>::value;
    static constexpr bool has_weight_values =
        has_spline_tag<SplineWeightValues, member_tags>::value;
    static constexpr bool has_weight_physical_gradients =
        has_spline_tag<SplineWeightPhysicalGradients, member_tags>::value;

    // Local interpolation stencil.
    int s[3][num_knot];
};

// Assign physical cell size to the spline data.
template <typename Scalar, int Order, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<
    SplineData<Scalar, Order, EntityType, DataTags>::has_physical_cell_size>
setSplineData( SplinePhysicalCellSize,
               SplineData<Scalar, Order, EntityType, DataTags> &data,
               const int d, const Scalar dx )
{
    data.dx[d] = dx;
}

template <typename Scalar, int Order, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<
    !SplineData<Scalar, Order, EntityType, DataTags>::has_physical_cell_size>
setSplineData( SplinePhysicalCellSize,
               SplineData<Scalar, Order, EntityType, DataTags> &, const int,
               const Scalar )
{
}

// Assign logical position to the spline data.
template <typename Scalar, int Order, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<
    SplineData<Scalar, Order, EntityType, DataTags>::has_logical_position>
setSplineData( SplineLogicalPosition,
               SplineData<Scalar, Order, EntityType, DataTags> &data,
               const int d, const Scalar x )
{
    data.x[d] = x;
}

template <typename Scalar, int Order, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<
    !SplineData<Scalar, Order, EntityType, DataTags>::has_logical_position>
setSplineData( SplineLogicalPosition,
               SplineData<Scalar, Order, EntityType, DataTags> &, const int,
               const Scalar )
{
}

// Assign weight values to the spline data.
template <typename Scalar, int Order, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<
    SplineData<Scalar, Order, EntityType, DataTags>::has_weight_values>
setSplineData( SplineWeightValues,
               SplineData<Scalar, Order, EntityType, DataTags> &data,
               const Scalar x[3] )

{
    using spline_type =
        typename SplineData<Scalar, Order, EntityType, DataTags>::spline_type;
    for ( int d = 0; d < 3; ++d )
        spline_type::value( x[d], data.w[d] );
}

template <typename Scalar, int Order, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<
    !SplineData<Scalar, Order, EntityType, DataTags>::has_weight_values>
setSplineData( SplineWeightValues,
               SplineData<Scalar, Order, EntityType, DataTags> &,
               const Scalar[3] )

{
}

// Assign weight physical gradients to the spline data.
template <typename Scalar, int Order, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<SplineData<Scalar, Order, EntityType,
                                DataTags>::has_weight_physical_gradients>
    setSplineData( SplineWeightPhysicalGradients,
                   SplineData<Scalar, Order, EntityType, DataTags> &data,
                   const Scalar x[3], const Scalar rdx[3] )

{
    using spline_type =
        typename SplineData<Scalar, Order, EntityType, DataTags>::spline_type;
    for ( int d = 0; d < 3; ++d )
        spline_type::gradient( x[d], rdx[d], data.g[d] );
}

template <typename Scalar, int Order, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION
    std::enable_if_t<!SplineData<Scalar, Order, EntityType,
                                 DataTags>::has_weight_physical_gradients>
    setSplineData( SplineWeightPhysicalGradients,
                   SplineData<Scalar, Order, EntityType, DataTags> &,
                   const Scalar[3], const Scalar[3] )

{
}

// Assign physical distance to the spline data.
template <typename Scalar, int Order, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<
    SplineData<Scalar, Order, EntityType, DataTags>::has_physical_distance>
setSplineData( SplinePhysicalDistance,
               SplineData<Scalar, Order, EntityType, DataTags> &data,
               const Scalar low_x[3], const Scalar p[3], const Scalar dx[3] )

{
    using spline_type =
        typename SplineData<Scalar, Order, EntityType, DataTags>::spline_type;
    Scalar offset;
    for ( int d = 0; d < 3; ++d )
    {
        offset = low_x[d] - p[d];
        for ( int n = 0; n < spline_type::num_knot; ++n )
            data.d[d][n] = offset + data.s[d][n] * dx[d];
    }
}

template <typename Scalar, int Order, class EntityType, class DataTags>
KOKKOS_INLINE_FUNCTION std::enable_if_t<
    !SplineData<Scalar, Order, EntityType, DataTags>::has_physical_distance>
setSplineData( SplinePhysicalDistance,
               SplineData<Scalar, Order, EntityType, DataTags> &,
               const Scalar[3], const Scalar[3], const Scalar[3] )
{
}

//---------------------------------------------------------------------------//
// Evaluate spline data at a point in a uniform mesh.
template <typename Scalar, int Order, class Device, class EntityType,
          class DataTags>
KOKKOS_INLINE_FUNCTION void
evaluateSpline( const LocalMesh<Device, UniformMesh<Scalar>> &local_mesh,
                const Scalar p[3],
                SplineData<Scalar, Order, EntityType, DataTags> &data )
{
    // data type
    using sd_type = SplineData<Scalar, Order, EntityType, DataTags>;

    // spline type.

    // Get the low corner of the mesh.
    Scalar low_x[3];
    int low_id[3] = { 0, 0, 0 };
    local_mesh.coordinates( EntityType(), low_id, low_x );

    // Compute the physical cell size.
    Scalar dx[3];
    for ( int d = 0; d < 3; ++d )
    {
        dx[d] = local_mesh.measure( Edge<Dim::I>(), low_id );
        setSplineData( SplinePhysicalCellSize(), data, d, dx[d] );
    }

    // Compute the inverse physicall cell size.
    Scalar rdx[3];
    for ( int d = 0; d < 3; ++d )
        rdx[d] = 1.0 / dx[d];

    // Compute the reference coordinates.
    Scalar x[3];
    for ( int d = 0; d < 3; ++d )
    {
        x[d] = sd_type::spline_type::mapToLogicalGrid( p[d], rdx[d], low_x[d] );
        setSplineData( SplineLogicalPosition(), data, d, x[d] );
    }

    // Compute the stencil.
    for ( int d = 0; d < 3; ++d )
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

} // end namespace Cajita

#endif // end CAJITA_SPLINES_HPP
