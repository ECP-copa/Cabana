/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_ARRAY_HPP
#define CAJITA_ARRAY_HPP

#include <Cajita_IndexSpace.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_MpiTraits.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Array layout.
//---------------------------------------------------------------------------//
template <class EntityType, class MeshType>
class ArrayLayout
{
  public:
    // Entity type.
    using entity_type = EntityType;

    // Mesh type.
    using mesh_type = MeshType;

    // Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    /*!
      \brief Constructor.
      \param local_grid The local grid over which the layout will be
      constructed. \param dofs_per_entity The number of degrees-of-freedom per
      grid entity.
    */
    ArrayLayout( const std::shared_ptr<LocalGrid<MeshType>>& local_grid,
                 const int dofs_per_entity )
        : _local_grid( local_grid )
        , _dofs_per_entity( dofs_per_entity )
    {
    }

    // Get the local grid over which this layout is defined.
    const std::shared_ptr<LocalGrid<MeshType>> localGrid() const
    {
        return _local_grid;
    }

    // Get the number of degrees-of-freedom on each grid entity.
    int dofsPerEntity() const { return _dofs_per_entity; }

    // Get the index space of the array elements in the given
    // decomposition.
    template <class DecompositionTag, class IndexType>
    IndexSpace<num_space_dim + 1>
    indexSpace( DecompositionTag decomposition_tag, IndexType index_type ) const
    {
        return appendDimension( _local_grid->indexSpace( decomposition_tag,
                                                         EntityType(),
                                                         index_type ),
                                _dofs_per_entity );
    }

    // Get the local index space of the array elements we shared with the
    // given neighbor in the given decomposition. Optionally provide a halo
    // width for the shared space. This halo width must be less than or equal
    // to the halo width of the local grid. The default behavior is to use the
    // halo width of the local grid.
    template <class DecompositionTag>
    IndexSpace<num_space_dim + 1>
    sharedIndexSpace( DecompositionTag decomposition_tag,
                      const std::array<int, num_space_dim>& off_ijk,
                      const int halo_width = -1 ) const
    {
        return appendDimension(
            _local_grid->sharedIndexSpace( decomposition_tag, EntityType(),
                                           off_ijk, halo_width ),
            _dofs_per_entity );
    }

    template <class DecompositionTag, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<4>>
    sharedIndexSpace( DecompositionTag decomposition_tag, const int off_i,
                      const int off_j, const int off_k,
                      const int halo_width = -1 ) const
    {
        std::array<int, 3> off_ijk = { off_i, off_j, off_k };
        return sharedIndexSpace( decomposition_tag, off_ijk, halo_width );
    }

    template <class DecompositionTag, std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, IndexSpace<3>>
    sharedIndexSpace( DecompositionTag decomposition_tag, const int off_i,
                      const int off_j, const int halo_width = -1 ) const
    {
        std::array<int, 2> off_ijk = { off_i, off_j };
        return sharedIndexSpace( decomposition_tag, off_ijk, halo_width );
    }

  private:
    std::shared_ptr<LocalGrid<MeshType>> _local_grid;
    int _dofs_per_entity;
};

// Static type checker.
template <class>
struct is_array_layout : public std::false_type
{
};

template <class EntityType, class MeshType>
struct is_array_layout<ArrayLayout<EntityType, MeshType>>
    : public std::true_type
{
};

template <class EntityType, class MeshType>
struct is_array_layout<const ArrayLayout<EntityType, MeshType>>
    : public std::true_type
{
};

//---------------------------------------------------------------------------//
// Array layout creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create an array layout over the entities of a local grid.
  \param local_grid The local grid over which to create the layout.
  \param dofs_per_entity The number of degrees-of-freedom per grid entity.
*/
template <class EntityType, class MeshType>
std::shared_ptr<ArrayLayout<EntityType, MeshType>>
createArrayLayout( const std::shared_ptr<LocalGrid<MeshType>>& local_grid,
                   const int dofs_per_entity, EntityType )
{
    return std::make_shared<ArrayLayout<EntityType, MeshType>>(
        local_grid, dofs_per_entity );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create an array layout over the entities of a grid given local grid
  parameters. An intermediate local grid will be created and assigned to the
  layout.
  \param local_grid The local grid over which to create the layout.
  \param dofs_per_entity The number of degrees-of-freedom per grid entity.
*/
template <class EntityType, class MeshType>
std::shared_ptr<ArrayLayout<EntityType, MeshType>>
createArrayLayout( const std::shared_ptr<GlobalGrid<MeshType>>& global_grid,
                   const int halo_cell_width, const int dofs_per_entity,
                   EntityType )
{
    return std::make_shared<ArrayLayout<EntityType, MeshType>>(
        createLocalGrid( global_grid, halo_cell_width ), dofs_per_entity );
}

//---------------------------------------------------------------------------//
// Array
//---------------------------------------------------------------------------//
template <class Scalar, class EntityType, class MeshType, class... Params>
class Array
{
  public:
    // Value type.
    using value_type = Scalar;

    // Entity type.
    using entity_type = EntityType;

    // Mesh type.
    using mesh_type = MeshType;

    // Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    // Array layout type.
    using array_layout = ArrayLayout<entity_type, mesh_type>;

    // View type.
    using view_type = std::conditional_t<
        3 == num_space_dim, Kokkos::View<value_type****, Params...>,
        std::conditional_t<2 == num_space_dim,
                           Kokkos::View<value_type***, Params...>, void>>;

    // Device type.
    using device_type = typename view_type::device_type;

    // Memory space.
    using memory_space = typename view_type::memory_space;

    // Execution space.
    using execution_space = typename view_type::execution_space;

    /*!
      \brief Create an array with the given layout. Arrays are constructed
      over the ghosted index space of the layout.
      \param label A label for the array.
      \param layout The array layout over which to construct the view.
    */
    Array( const std::string& label,
           const std::shared_ptr<array_layout>& layout )
        : _layout( layout )
        , _data( createView<value_type, Params...>(
              label, layout->indexSpace( Ghost(), Local() ) ) )
    {
    }

    /*!
      \brief Create an array with the given layout and view. This view should
      match the array index spaces in size.
      \param layout The layout of the array.
      \param view The array data.
    */
    Array( const std::shared_ptr<array_layout>& layout, const view_type& view )
        : _layout( layout )
        , _data( view )
    {
        for ( std::size_t d = 0; d < num_space_dim + 1; ++d )
            if ( (long)view.extent( d ) !=
                 layout->indexSpace( Ghost(), Local() ).extent( d ) )
                throw std::runtime_error(
                    "Layout and view dimensions do not match" );
    }

    //! Get the layout of the array.
    std::shared_ptr<array_layout> layout() const { return _layout; }

    //! Get a view of the array data.
    view_type view() const { return _data; }

    //! Get the array label.
    std::string label() const { return _data.label(); }

  private:
    std::shared_ptr<array_layout> _layout;
    view_type _data;

  public:
    // Subview types.
    using subview_type = decltype(
        createSubview( _data, _layout->indexSpace( Ghost(), Local() ) ) );
    using subview_layout = typename subview_type::array_layout;
    using subview_memory_traits = typename subview_type::memory_traits;
};

//---------------------------------------------------------------------------//
// Scatic type checker.
//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_array : public std::false_type
{
};

template <class Scalar, class EntityType, class MeshType, class... Params>
struct is_array<Array<Scalar, EntityType, MeshType, Params...>>
    : public std::true_type
{
};

template <class Scalar, class EntityType, class MeshType, class... Params>
struct is_array<const Array<Scalar, EntityType, MeshType, Params...>>
    : public std::true_type
{
};

//---------------------------------------------------------------------------//
// Array creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create an array with the given array layout. Views are constructed
  over the ghosted index space of the layout.
  \param label A label for the view.
  \param layout The array layout over which to construct the view.
*/
template <class Scalar, class... Params, class EntityType, class MeshType>
std::shared_ptr<Array<Scalar, EntityType, MeshType, Params...>>
createArray( const std::string& label,
             const std::shared_ptr<ArrayLayout<EntityType, MeshType>>& layout )
{
    return std::make_shared<Array<Scalar, EntityType, MeshType, Params...>>(
        label, layout );
}

//---------------------------------------------------------------------------//
// Subarray creation.
//---------------------------------------------------------------------------//
/*!
  \brief Create a subarray of the given array over the given range of degrees
  of freedom.
  \param array The array from which to create a subarray
  \param dof_min The minimum degree-of-freedom index of the subarray.
  \param dof_max The maximum degree-of-freedom index of the subarray.
*/
template <class Scalar, class EntityType, class MeshType, class... Params>
std::shared_ptr<Array<
    Scalar, EntityType, MeshType,
    typename Array<Scalar, EntityType, MeshType, Params...>::subview_layout,
    typename Array<Scalar, EntityType, MeshType, Params...>::device_type,
    typename Array<Scalar, EntityType, MeshType,
                   Params...>::subview_memory_traits>>
createSubarray( const Array<Scalar, EntityType, MeshType, Params...>& array,
                const int dof_min, const int dof_max )
{
    if ( dof_min < 0 || dof_max > array.layout()->dofsPerEntity() )
        throw std::logic_error( "Subarray dimensions out of bounds" );

    auto space = array.layout()->indexSpace( Ghost(), Local() );
    std::array<long, MeshType::num_space_dim + 1> min;
    std::array<long, MeshType::num_space_dim + 1> max;
    for ( std::size_t d = 0; d < MeshType::num_space_dim; ++d )
    {
        min[d] = space.min( d );
        max[d] = space.max( d );
    }
    min.back() = dof_min;
    max.back() = dof_max;
    IndexSpace<MeshType::num_space_dim + 1> sub_space( min, max );
    auto sub_view = createSubview( array.view(), sub_space );
    auto sub_layout = createArrayLayout( array.layout()->localGrid(),
                                         dof_max - dof_min, EntityType() );
    return std::make_shared<Array<
        Scalar, EntityType, MeshType,
        typename Array<Scalar, EntityType, MeshType, Params...>::subview_layout,
        typename Array<Scalar, EntityType, MeshType, Params...>::device_type,
        typename Array<Scalar, EntityType, MeshType,
                       Params...>::subview_memory_traits>>( sub_layout,
                                                            sub_view );
}

//---------------------------------------------------------------------------//
// Array operations.
//---------------------------------------------------------------------------//
namespace ArrayOp
{
//---------------------------------------------------------------------------//
/*!
  \brief Clone an array. Do not initialize the clone.
  \param array The array to clone.
*/
template <class Scalar, class... Params, class EntityType, class MeshType>
std::shared_ptr<Array<Scalar, EntityType, MeshType, Params...>>
clone( const Array<Scalar, EntityType, MeshType, Params...>& array )
{
    return createArray<Scalar, Params...>( array.label(), array.layout() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Assign a scalar value to every element of an array.
  \param array The array to assign the value to.
  \param alpha The value to assign to the array.
  \param tag The tag for the decomposition over which to perform the
  operation.
*/
template <class Array_t, class DecompositionTag>
void assign( Array_t& array, const typename Array_t::value_type alpha,
             DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    auto subview = createSubview( array.view(),
                                  array.layout()->indexSpace( tag, Local() ) );
    Kokkos::deep_copy( subview, alpha );
}

//---------------------------------------------------------------------------//
/*!
  \brief Scale every element of an array by a scalar value.
  \param array The array to scale.
  \param alpha The value to scale the array by.
  \param tag The tag for the decomposition over which to perform the
  operation.
*/
template <class Array_t, class DecompositionTag>
std::enable_if_t<3 == Array_t::num_space_dim, void>
scale( Array_t& array, const typename Array_t::value_type alpha,
       DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    auto view = array.view();
    Kokkos::parallel_for(
        "ArrayOp::scale",
        createExecutionPolicy( array.layout()->indexSpace( tag, Local() ),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ) {
            view( i, j, k, l ) *= alpha;
        } );
}

template <class Array_t, class DecompositionTag>
std::enable_if_t<2 == Array_t::num_space_dim, void>
scale( Array_t& array, const typename Array_t::value_type alpha,
       DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    auto view = array.view();
    Kokkos::parallel_for(
        "ArrayOp::scale",
        createExecutionPolicy( array.layout()->indexSpace( tag, Local() ),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int l ) {
            view( i, j, l ) *= alpha;
        } );
}

//---------------------------------------------------------------------------//
/*!
  \brief Scale every element of an array by a scalar
  \param array The array to scale.
  \param alpha The values to scale the array by. A value must be provided for
  each entity degree-of-freedom in the array.
  \param tag The tag for the decomposition over which to perform the
  operation.
*/
template <class Array_t, class DecompositionTag>
std::enable_if_t<3 == Array_t::num_space_dim, void>
scale( Array_t& array, const std::vector<typename Array_t::value_type>& alpha,
       DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( alpha.size() !=
         static_cast<unsigned>( array.layout()->dofsPerEntity() ) )
        throw std::runtime_error( "Incorrect vector size" );

    Kokkos::View<const typename Array_t::value_type*, Kokkos::HostSpace,
                 Kokkos::MemoryUnmanaged>
        alpha_view_host( alpha.data(), alpha.size() );
    auto alpha_view = Kokkos::create_mirror_view_and_copy(
        typename Array_t::device_type(), alpha_view_host );

    auto array_view = array.view();
    Kokkos::parallel_for(
        "ArrayOp::scale",
        createExecutionPolicy( array.layout()->indexSpace( tag, Local() ),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ) {
            array_view( i, j, k, l ) *= alpha_view( l );
        } );
}

template <class Array_t, class DecompositionTag>
std::enable_if_t<2 == Array_t::num_space_dim, void>
scale( Array_t& array, const std::vector<typename Array_t::value_type>& alpha,
       DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( alpha.size() !=
         static_cast<unsigned>( array.layout()->dofsPerEntity() ) )
        throw std::runtime_error( "Incorrect vector size" );

    Kokkos::View<const typename Array_t::value_type*, Kokkos::HostSpace,
                 Kokkos::MemoryUnmanaged>
        alpha_view_host( alpha.data(), alpha.size() );
    auto alpha_view = Kokkos::create_mirror_view_and_copy(
        typename Array_t::device_type(), alpha_view_host );

    auto array_view = array.view();
    Kokkos::parallel_for(
        "ArrayOp::scale",
        createExecutionPolicy( array.layout()->indexSpace( tag, Local() ),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int l ) {
            array_view( i, j, l ) *= alpha_view( l );
        } );
}

//---------------------------------------------------------------------------//
/*!
  \brief Copy one array into another over the designated decomposition. A <- B
  \param a The array to which the data will be copied.
  \param b The array from which the data will be copied.
  \param tag The tag for the decomposition over which to perform the
  operation.
*/
template <class Array_t, class DecompositionTag>
void copy( Array_t& a, const Array_t& b, DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    auto a_space = a.layout()->indexSpace( tag, Local() );
    auto b_space = b.layout()->indexSpace( tag, Local() );
    if ( a_space != b_space )
        throw std::logic_error( "Incompatible index spaces" );
    auto subview_a = createSubview( a.view(), a_space );
    auto subview_b = createSubview( b.view(), b_space );
    Kokkos::deep_copy( subview_a, subview_b );
}

//---------------------------------------------------------------------------//
/*!
  \brief Clone an array and copy its contents into the clone.
  \param array The array to clone.
  \param tag The tag for the decomposition over which to perform the copy.
*/
template <class Array_t, class DecompositionTag>
std::shared_ptr<Array_t> cloneCopy( const Array_t& array, DecompositionTag tag )
{
    auto cln = clone( array );
    copy( *cln, array, tag );
    return cln;
}

//---------------------------------------------------------------------------//
/*!
  \brief Update two vectors auch that a = alpha * a + beta * b.
  \param a The array that will be updated.
  \param alpha The value to scale a by.
  \param b The array to add to a.
  \param beta The value to scale b by.
  \param tag The tag for the decomposition over which to perform the
  operation.
*/
template <class Array_t, class DecompositionTag>
std::enable_if_t<3 == Array_t::num_space_dim, void>
update( Array_t& a, const typename Array_t::value_type alpha, const Array_t& b,
        const typename Array_t::value_type beta, DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    auto a_view = a.view();
    auto b_view = b.view();
    Kokkos::parallel_for(
        "ArrayOp::update",
        createExecutionPolicy( a.layout()->indexSpace( tag, Local() ),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ) {
            a_view( i, j, k, l ) =
                alpha * a_view( i, j, k, l ) + beta * b_view( i, j, k, l );
        } );
}

template <class Array_t, class DecompositionTag>
std::enable_if_t<2 == Array_t::num_space_dim, void>
update( Array_t& a, const typename Array_t::value_type alpha, const Array_t& b,
        const typename Array_t::value_type beta, DecompositionTag tag )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    auto a_view = a.view();
    auto b_view = b.view();
    Kokkos::parallel_for(
        "ArrayOp::update",
        createExecutionPolicy( a.layout()->indexSpace( tag, Local() ),
                               typename Array_t::execution_space() ),
        KOKKOS_LAMBDA( const long i, const long j, const long l ) {
            a_view( i, j, l ) =
                alpha * a_view( i, j, l ) + beta * b_view( i, j, l );
        } );
}

//---------------------------------------------------------------------------//
// Dot product
template <class ViewType, std::size_t NumSpaceDim>
struct DotFunctor
{
    static constexpr std::size_t num_space_dim = NumSpaceDim;
    typedef typename ViewType::value_type value_type[];
    typedef typename ViewType::size_type size_type;
    size_type value_count;
    ViewType _a;
    ViewType _b;

    DotFunctor( const ViewType& a, const ViewType& b )
        : value_count( a.extent( NumSpaceDim ) )
        , _a( a )
        , _b( b )
    {
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    operator()( const size_type i, const size_type j, const size_type k,
                const size_type l, value_type sum ) const
    {
        sum[l] += _a( i, j, k, l ) * _b( i, j, k, l );
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    operator()( const size_type i, const size_type j, const size_type l,
                value_type sum ) const
    {
        sum[l] += _a( i, j, l ) * _b( i, j, l );
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile value_type dst, const volatile value_type src ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            dst[j] += src[j];
    }

    KOKKOS_INLINE_FUNCTION void init( value_type sum ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            sum[j] = 0.0;
    }
};

/*!
  \brief Compute the dot product of owned space of two arrays.
  \param a The first array in the dot product.
  \param b The second array in the dot product.
  \param products The dot product of each entity degree-of-freedom in the
  array. This vector should be pre-sized to the number of degrees-of-freedom
  per entity.
*/
template <class Array_t>
void dot( const Array_t& a, const Array_t& b,
          std::vector<typename Array_t::value_type>& products )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( products.size() !=
         static_cast<unsigned>( a.layout()->dofsPerEntity() ) )
        throw std::runtime_error( "Incorrect vector size" );

    for ( auto& p : products )
        p = 0.0;

    DotFunctor<typename Array_t::view_type, Array_t::num_space_dim> functor(
        a.view(), b.view() );
    Kokkos::parallel_reduce(
        "ArrayOp::dot",
        createExecutionPolicy( a.layout()->indexSpace( Own(), Local() ),
                               typename Array_t::execution_space() ),
        functor, products.data() );

    MPI_Allreduce( MPI_IN_PLACE, products.data(), products.size(),
                   MpiTraits<typename Array_t::value_type>::type(), MPI_SUM,
                   a.layout()->localGrid()->globalGrid().comm() );
}

//---------------------------------------------------------------------------//
// Infinity norm
template <class ViewType, std::size_t NumSpaceDim>
struct NormInfFunctor
{
    static constexpr std::size_t num_space_dim = NumSpaceDim;
    typedef typename ViewType::value_type value_type[];
    typedef typename ViewType::size_type size_type;
    size_type value_count;
    ViewType _view;

    NormInfFunctor( const ViewType& view )
        : value_count( view.extent( NumSpaceDim ) )
        , _view( view )
    {
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    operator()( const size_type i, const size_type j, const size_type k,
                const size_type l, value_type norm ) const
    {
        auto v_abs = fabs( _view( i, j, k, l ) );
        if ( v_abs > norm[l] )
            norm[l] = v_abs;
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    operator()( const size_type i, const size_type j, const size_type l,
                value_type norm ) const
    {
        auto v_abs = fabs( _view( i, j, l ) );
        if ( v_abs > norm[l] )
            norm[l] = v_abs;
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile value_type dst, const volatile value_type src ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            if ( src[j] > dst[j] )
                dst[j] = src[j];
    }

    KOKKOS_INLINE_FUNCTION void init( value_type norm ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            norm[j] = 0.0;
    }
};

/*!
  \brief Calculate the infinity-norm of the owned elements of the array.
  \param array The array to compute the norm for.
  \param norms The norms for each degree-of-freedom in the array. This vector
  should be pre-sized to the number of degrees-of-freedom per entity.
*/
template <class Array_t>
void normInf( const Array_t& array,
              std::vector<typename Array_t::value_type>& norms )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( norms.size() !=
         static_cast<unsigned>( array.layout()->dofsPerEntity() ) )
        throw std::runtime_error( "Incorrect vector size" );

    for ( auto& n : norms )
        n = 0.0;

    NormInfFunctor<typename Array_t::view_type, Array_t::num_space_dim> functor(
        array.view() );
    Kokkos::parallel_reduce(
        "ArrayOp::normInf",
        createExecutionPolicy( array.layout()->indexSpace( Own(), Local() ),
                               typename Array_t::execution_space() ),
        functor, norms.data() );

    MPI_Allreduce( MPI_IN_PLACE, norms.data(), norms.size(),
                   MpiTraits<typename Array_t::value_type>::type(), MPI_MAX,
                   array.layout()->localGrid()->globalGrid().comm() );
}

//---------------------------------------------------------------------------//
// One norm
template <class ViewType, std::size_t NumSpaceDim>
struct Norm1Functor
{
    static constexpr std::size_t num_space_dim = NumSpaceDim;
    typedef typename ViewType::value_type value_type[];
    typedef typename ViewType::size_type size_type;
    size_type value_count;
    ViewType _view;

    Norm1Functor( const ViewType& view )
        : value_count( view.extent( NumSpaceDim ) )
        , _view( view )
    {
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    operator()( const size_type i, const size_type j, const size_type k,
                const size_type l, value_type norm ) const
    {
        norm[l] += fabs( _view( i, j, k, l ) );
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    operator()( const size_type i, const size_type j, const size_type l,
                value_type norm ) const
    {
        norm[l] += fabs( _view( i, j, l ) );
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile value_type dst, const volatile value_type src ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            dst[j] += src[j];
    }

    KOKKOS_INLINE_FUNCTION void init( value_type norm ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            norm[j] = 0.0;
    }
};

/*!
  \brief Calculate the one-norm of the owned elements of the array.
  \param array The array to compute the norm for.
  \param norms The norms for each degree-of-freedom in the array. This vector
  should be pre-sized to the number of degrees-of-freedom per entity.
*/
template <class Array_t>
void norm1( const Array_t& array,
            std::vector<typename Array_t::value_type>& norms )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( norms.size() !=
         static_cast<unsigned>( array.layout()->dofsPerEntity() ) )
        throw std::runtime_error( "Incorrect vector size" );

    for ( auto& n : norms )
        n = 0.0;

    Norm1Functor<typename Array_t::view_type, Array_t::num_space_dim> functor(
        array.view() );
    Kokkos::parallel_reduce(
        "ArrayOp::norm1",
        createExecutionPolicy( array.layout()->indexSpace( Own(), Local() ),
                               typename Array_t::execution_space() ),
        functor, norms.data() );

    MPI_Allreduce( MPI_IN_PLACE, norms.data(), norms.size(),
                   MpiTraits<typename Array_t::value_type>::type(), MPI_SUM,
                   array.layout()->localGrid()->globalGrid().comm() );
}

//---------------------------------------------------------------------------//
// Two norm
template <class ViewType, std::size_t NumSpaceDim>
struct Norm2Functor
{
    static constexpr std::size_t num_space_dim = NumSpaceDim;
    typedef typename ViewType::value_type value_type[];
    typedef typename ViewType::size_type size_type;
    size_type value_count;
    ViewType _view;

    Norm2Functor( const ViewType& view )
        : value_count( view.extent( NumSpaceDim ) )
        , _view( view )
    {
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    operator()( const size_type i, const size_type j, const size_type k,
                const size_type l, value_type norm ) const
    {
        norm[l] += _view( i, j, k, l ) * _view( i, j, k, l );
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    operator()( const size_type i, const size_type j, const size_type l,
                value_type norm ) const
    {
        norm[l] += _view( i, j, l ) * _view( i, j, l );
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile value_type dst, const volatile value_type src ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            dst[j] += src[j];
    }

    KOKKOS_INLINE_FUNCTION void init( value_type norm ) const
    {
        for ( size_type j = 0; j < value_count; ++j )
            norm[j] = 0.0;
    }
};

/*!
  \brief Calculate the two-norm of the owned elements of the array.
  \param array The array to compute the norm for.
  \param norms The norms for each entity degree-of-freedom in the array. This
  vector should be pre-sized to the number of degrees-of-freedom per entity.
*/
template <class Array_t>
void norm2( const Array_t& array,
            std::vector<typename Array_t::value_type>& norms )
{
    static_assert( is_array<Array_t>::value, "Cajita::Array required" );
    if ( norms.size() !=
         static_cast<unsigned>( array.layout()->dofsPerEntity() ) )
        throw std::runtime_error( "Incorrect vector size" );

    for ( auto& n : norms )
        n = 0.0;

    Norm2Functor<typename Array_t::view_type, Array_t::num_space_dim> functor(
        array.view() );
    Kokkos::parallel_reduce(
        "ArrayOp::norm2",
        createExecutionPolicy( array.layout()->indexSpace( Own(), Local() ),
                               typename Array_t::execution_space() ),
        functor, norms.data() );

    MPI_Allreduce( MPI_IN_PLACE, norms.data(), norms.size(),
                   MpiTraits<typename Array_t::value_type>::type(), MPI_SUM,
                   array.layout()->localGrid()->globalGrid().comm() );

    for ( auto& n : norms )
        n = std::sqrt( n );
}
//---------------------------------------------------------------------------//

} // end namespace ArrayOp

//---------------------------------------------------------------------------//

} // end namespace Cajita.

#endif // end CAJITA_ARRAY_HPP
