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
  \file Cabana_Experimental_NeighborList.hpp
  \brief ArborX tree-based neighbor lists
*/
#ifndef CABANA_EXPERIMENTAL_NEIGHBOR_LIST_HPP
#define CABANA_EXPERIMENTAL_NEIGHBOR_LIST_HPP

#include <Cabana_NeighborList.hpp>
#include <Cabana_Slice.hpp>
#include <Cabana_Types.hpp> // is_accessible_from

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace Cabana
{
namespace Experimental
{
//! \cond Impl
namespace stdcxx20
{
template <class T>
struct remove_cvref
{
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;
} // namespace stdcxx20

namespace Impl
{

template <typename Positions,
          typename = std::enable_if_t<Cabana::is_slice<Positions>::value ||
                                      Kokkos::is_view_v<Positions>>>
struct SubPositionsAndRadius
{
    using positions_type = Positions;
    using memory_space = typename Positions::memory_space;
    Positions data;
    using size_type = typename Positions::size_type;
    size_type first;
    size_type last;
    using value_type = typename Positions::value_type;
    value_type radius;
};

template <typename Positions,
          typename = std::enable_if_t<
              Cabana::is_slice<std::remove_reference_t<Positions>>::value ||
              Kokkos::is_view_v<Positions>>>
auto makePredicates(
    Positions&& positions,
    typename stdcxx20::remove_cvref_t<Positions>::size_type first,
    typename stdcxx20::remove_cvref_t<Positions>::size_type last,
    typename stdcxx20::remove_cvref_t<Positions>::value_type radius )
{
    return Impl::SubPositionsAndRadius<stdcxx20::remove_cvref_t<Positions>>{
        std::forward<Positions>( positions ), first, last, radius };
}

template <typename ExecutionSpace, typename D, typename... P>
typename Kokkos::View<D, P...>::non_const_value_type
max_reduce( ExecutionSpace const& space, Kokkos::View<D, P...> const& v )
{
    using V = Kokkos::View<D, P...>;
    static_assert( V::rank == 1 );
    static_assert( Kokkos::is_execution_space<ExecutionSpace>::value );
    static_assert(
        is_accessible_from<typename V::memory_space, ExecutionSpace>::value );
    using Ret = typename Kokkos::View<D, P...>::non_const_value_type;
    Ret max_val;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>( space, 0, v.extent( 0 ) ),
        KOKKOS_LAMBDA( int i, Ret& partial_max ) {
            if ( v( i ) > partial_max )
            {
                partial_max = v( i );
            }
        },
        Kokkos::Max<Ret>( max_val ) );
    return max_val;
}
//! \endcond
} // namespace Impl
} // namespace Experimental
} // namespace Cabana

namespace ArborX
{
//! Neighbor access trait for Cabana slice and/or Kokkos View.
template <typename Positions>
struct AccessTraits<Positions, PrimitivesTag,
                    std::enable_if_t<Cabana::is_slice<Positions>{} ||
                                     Kokkos::is_view<Positions>{}>>
{
    //! Kokkos memory space.
    using memory_space = typename Positions::memory_space;
    //! Size type.
    using size_type = typename Positions::size_type;
    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = 3;
    //! Get number of particles.
    static KOKKOS_FUNCTION size_type size( Positions const& x )
    {
        return Cabana::size( x );
    }
    //! Get the particle at the index.
    template <std::size_t NSD = num_space_dim>
    static KOKKOS_FUNCTION std::enable_if_t<3 == NSD, Point>
    get( Positions const& x, size_type i )
    {
        return { static_cast<float>( x( i, 0 ) ),
                 static_cast<float>( x( i, 1 ) ),
                 static_cast<float>( x( i, 2 ) ) };
    }
    //! Get the particle at the index.
    template <std::size_t NSD = num_space_dim>
    static KOKKOS_FUNCTION std::enable_if_t<2 == NSD, Point>
    get( Positions const& x, size_type i )
    {
        return { static_cast<float>( x( i, 0 ) ),
                 static_cast<float>( x( i, 1 ) ) };
    }
};
//! Neighbor access trait.
template <typename Positions>
struct AccessTraits<
    Cabana::Experimental::Impl::SubPositionsAndRadius<Positions>, PredicatesTag>
{
    //! Position wrapper with partial range and radius information.
    using PositionLike =
        Cabana::Experimental::Impl::SubPositionsAndRadius<Positions>;
    //! Kokkos memory space.
    using memory_space = typename PositionLike::memory_space;
    //! Size type.
    using size_type = typename PositionLike::size_type;
    //! Get number of particles.
    static KOKKOS_FUNCTION size_type size( PositionLike const& x )
    {
        return x.last - x.first;
    }
    //! Get the particle at the index.
    static KOKKOS_FUNCTION auto get( PositionLike const& x, size_type i )
    {
        assert( i < size( x ) );
        auto const point =
            AccessTraits<typename PositionLike::positions_type,
                         PrimitivesTag>::get( x.data, x.first + i );
        return attach( intersects( Sphere{ point, x.radius } ), (int)i );
    }
};
} // namespace ArborX

namespace Cabana
{
namespace Experimental
{
namespace Impl
{
//! \cond Impl

template <typename Tag>
struct CollisionFilter;

template <>
struct CollisionFilter<FullNeighborTag>
{
    KOKKOS_FUNCTION bool static keep( int i, int j ) noexcept
    {
        return i != j; // discard self-collision
    }
};

template <>
struct CollisionFilter<HalfNeighborTag>
{
    KOKKOS_FUNCTION static bool keep( int i, int j ) noexcept { return i > j; }
};

// Custom callback for ArborX::BVH::query()
template <typename Tag>
struct NeighborDiscriminatorCallback
{
    template <typename Predicate, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()( Predicate const& predicate,
                                     int primitive_index,
                                     OutputFunctor const& out ) const
    {
        int const predicate_index = getData( predicate );
        if ( CollisionFilter<Tag>::keep( predicate_index, primitive_index ) )
        {
            out( primitive_index );
        }
    }
};

// Count in the first pass
template <typename Counts, typename Tag>
struct NeighborDiscriminatorCallback2D_FirstPass
{
    Counts counts;
    template <typename Predicate>
    KOKKOS_FUNCTION void operator()( Predicate const& predicate,
                                     int primitive_index ) const
    {
        int const predicate_index = getData( predicate );
        if ( CollisionFilter<Tag>::keep( predicate_index, primitive_index ) )
        {
            ++counts( predicate_index ); // WARNING see below**
        }
    }
};

// Preallocate and attempt fill in the first pass
template <typename Counts, typename Neighbors, typename Tag>
struct NeighborDiscriminatorCallback2D_FirstPass_BufferOptimization
{
    Counts counts;
    Neighbors neighbors;
    template <typename Predicate>
    KOKKOS_FUNCTION void operator()( Predicate const& predicate,
                                     int primitive_index ) const
    {
        int const predicate_index = getData( predicate );
        auto& count = counts( predicate_index );
        if ( CollisionFilter<Tag>::keep( predicate_index, primitive_index ) )
        {
            if ( count < (int)neighbors.extent( 1 ) )
            {
                neighbors( predicate_index, count++ ) =
                    primitive_index; // WARNING see below**
            }
            else
            {
                count++;
            }
        }
    }
};

// Fill in the second pass
template <typename Counts, typename Neighbors, typename Tag>
struct NeighborDiscriminatorCallback2D_SecondPass
{
    Counts counts;
    Neighbors neighbors;
    template <typename Predicate>
    KOKKOS_FUNCTION void operator()( Predicate const& predicate,
                                     int primitive_index ) const
    {
        int const predicate_index = getData( predicate );
        auto& count = counts( predicate_index );
        if ( CollisionFilter<Tag>::keep( predicate_index, primitive_index ) )
        {
            assert( count < (int)neighbors.extent( 1 ) );
            neighbors( predicate_index, count++ ) =
                primitive_index; // WARNING see below**
        }
    }
};

// NOTE** Taking advantage of the knowledge that one predicate is processed by a
// single thread.  Count increment should be atomic otherwise.

//! \endcond
} // namespace Impl

//---------------------------------------------------------------------------//
//! 1d ArborX neighbor list storage layout.
template <typename MemorySpace, typename Tag>
struct CrsGraph
{
    //! Neighbor indices
    Kokkos::View<int*, MemorySpace> col_ind;
    //! Neighbor offsets.
    Kokkos::View<int*, MemorySpace> row_ptr;
    //! Neighbor offset shift.
    typename MemorySpace::size_type shift;
    //! Total particles.
    typename MemorySpace::size_type total;
};

//---------------------------------------------------------------------------//
/*!
  \brief Neighbor list implementation using ArborX for particles within the
  interaction distance with a 1D compressed layout for particles and neighbors.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Positions The position type.
  \tparam AlgorithmTag Tag indicating whether to build a full or half neighbor
  list.

  \param space Kokkos execution space.
  \param positions The particle positions.
  \param first The beginning particle index to compute neighbors for.
  \param last The end particle index to compute neighbors for.
  \param radius The radius of the neighborhood. Particles within this radius are
  considered neighbors.
  \param buffer_size Optional guess for maximum number of neighbors.

  Neighbor list implementation most appropriate for highly varying particle
  densities.
*/
template <typename ExecutionSpace, typename Positions, typename Tag>
auto makeNeighborList( ExecutionSpace space, Tag, Positions const& positions,
                       typename Positions::size_type first,
                       typename Positions::size_type last,
                       typename Positions::value_type radius,
                       int buffer_size = 0 )
{
    assert( buffer_size >= 0 );
    assert( last >= first );
    assert( last <= positions.size() );

    using memory_space = typename Positions::memory_space;

    ArborX::BVH<memory_space> bvh( space, positions );

    Kokkos::View<int*, memory_space> indices(
        Kokkos::view_alloc( "indices", Kokkos::WithoutInitializing ), 0 );
    Kokkos::View<int*, memory_space> offset(
        Kokkos::view_alloc( "offset", Kokkos::WithoutInitializing ), 0 );
    bvh.query(
        space, Impl::makePredicates( positions, first, last, radius ),
        Impl::NeighborDiscriminatorCallback<Tag>{}, indices, offset,
        ArborX::Experimental::TraversalPolicy().setBufferSize( buffer_size ) );

    return CrsGraph<memory_space, Tag>{
        std::move( indices ), std::move( offset ), first, bvh.size() };
}

/*!
  \brief Neighbor list implementation using ArborX for particles within the
  interaction distance with a 1D compressed layout for particles and neighbors.

  \tparam Positions The position type.
  \tparam Tag Tag indicating whether to build a full or half neighbor list.

  \param tag Tag indicating whether to build a full or half neighbor list.
  \param positions The containing the particle positions.
  \param first The beginning particle index to compute neighbors for.
  \param last The end particle index to compute neighbors for.
  \param radius The radius of the neighborhood. Particles within this radius are
  considered neighbors.
  \param buffer_size Optional guess for maximum number of neighbors.

  Neighbor list implementation most appropriate for highly varying particle
  densities.
*/
template <typename Positions, typename Tag>
auto makeNeighborList( Tag tag, Positions const& positions,
                       typename Positions::size_type first,
                       typename Positions::size_type last,
                       typename Positions::value_type radius,
                       int buffer_size = 0 )
{
    typename Positions::execution_space space{};
    return makeNeighborList( space, tag, positions, first, last, radius,
                             buffer_size );
}

/*!
  \brief Neighbor list implementation using ArborX for particles within the
  interaction distance with a 1D compressed layout for particles and neighbors.

  \tparam DeviceType Kokkos device type.
  \tparam Positions The position type.
  \tparam Tag Tag indicating whether to build a full or half neighbor list.

  \param tag Tag indicating whether to build a full or half neighbor list.
  \param positions The particle positions.
  \param first The beginning particle index to compute neighbors for.
  \param last The end particle index to compute neighbors for.
  \param radius The radius of the neighborhood. Particles within this radius are
  considered neighbors.
  \param buffer_size Optional guess for maximum number of neighbors.

  Neighbor list implementation most appropriate for highly varying particle
  densities.
*/
template <typename DeviceType, typename Positions, typename Tag>
[[deprecated]] auto makeNeighborList( Tag tag, Positions const& positions,
                                      typename Positions::size_type first,
                                      typename Positions::size_type last,
                                      typename Positions::value_type radius,
                                      int buffer_size = 0 )
{
    using exec_space = typename DeviceType::execution_space;
    return makeNeighborList( exec_space{}, tag, positions, first, last, radius,
                             buffer_size );
}

//---------------------------------------------------------------------------//
//! 2d ArborX neighbor list storage layout.
template <typename MemorySpace, typename Tag>
struct Dense
{
    //! Neighbor counts.
    Kokkos::View<int*, MemorySpace> cnt;
    //! Neighbor indices.
    Kokkos::View<int**, MemorySpace> val;
    //! Neighbor offset shift.
    typename MemorySpace::size_type shift;
    //! Total particles.
    typename MemorySpace::size_type total;
};

//---------------------------------------------------------------------------//
/*!
  \brief Neighbor list implementation using ArborX for particles within the
  interaction distance with a 2D layout for particles and neighbors.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Positions The position type.
  \tparam Tag Tag indicating whether to build a full or half neighbor list.

  \param space Kokkos execution space.
  \param positions The particle positions.
  \param first The beginning particle index to compute neighbors for.
  \param last The end particle index to compute neighbors for.
  \param radius The radius of the neighborhood. Particles within this radius are
  considered neighbors.
  \param buffer_size Optional guess for maximum number of neighbors per
  particle.

  Neighbor list implementation most appropriate for highly varying particle
  densities.
*/
template <typename ExecutionSpace, typename Positions, typename Tag>
auto make2DNeighborList( ExecutionSpace space, Tag, Positions const& positions,
                         typename Positions::size_type first,
                         typename Positions::size_type last,
                         typename Positions::value_type radius,
                         int buffer_size = 0 )
{
    assert( buffer_size >= 0 );
    assert( last >= first );
    assert( last <= positions.size() );

    using memory_space = typename Positions::memory_space;

    ArborX::BVH<memory_space> bvh( space, positions );

    auto const predicates =
        Impl::makePredicates( positions, first, last, radius );

    auto const n_queries =
        ArborX::AccessTraits<std::remove_const_t<decltype( predicates )>,
                             ArborX::PredicatesTag>::size( predicates );

    Kokkos::View<int**, memory_space> neighbors;
    Kokkos::View<int*, memory_space> counts( "counts", n_queries );
    if ( buffer_size > 0 )
    {
        neighbors = Kokkos::View<int**, memory_space>(
            Kokkos::view_alloc( "neighbors", Kokkos::WithoutInitializing ),
            n_queries, buffer_size );
        bvh.query(
            space, predicates,
            Impl::NeighborDiscriminatorCallback2D_FirstPass_BufferOptimization<
                decltype( counts ), decltype( neighbors ), Tag>{ counts,
                                                                 neighbors } );
    }
    else
    {
        bvh.query(
            space, predicates,
            Impl::NeighborDiscriminatorCallback2D_FirstPass<decltype( counts ),
                                                            Tag>{ counts } );
    }

    auto const max_neighbors = Impl::max_reduce( space, counts );
    if ( max_neighbors <= buffer_size )
    {
        // NOTE We do not bother shrinking to eliminate the excess allocation.
        // NOTE If buffer_size is 0, neighbors is default constructed.  This is
        // fine with the current design/implementation of NeighborList access
        // traits.
        return Dense<memory_space, Tag>{ counts, neighbors, first, bvh.size() };
    }

    neighbors = Kokkos::View<int**, memory_space>(
        Kokkos::view_alloc( "neighbors", Kokkos::WithoutInitializing ),
        n_queries, max_neighbors ); // realloc storage for neighbors
    Kokkos::deep_copy( counts, 0 ); // reset counts to zero
    bvh.query( space, predicates,
               Impl::NeighborDiscriminatorCallback2D_SecondPass<
                   decltype( counts ), decltype( neighbors ), Tag>{
                   counts, neighbors } );

    return Dense<memory_space, Tag>{ counts, neighbors, first, bvh.size() };
}

/*!
  \brief Neighbor list implementation using ArborX for particles within the
  interaction distance with a 2D layout for particles and neighbors.

  \tparam Positions The position type.
  \tparam Tag Tag indicating whether to build a full or half neighbor list.

  \param tag Tag indicating whether to build a full or half neighbor list.
  \param positions The particle positions.
  \param first The beginning particle index to compute neighbors for.
  \param last The end particle index to compute neighbors for.
  \param radius The radius of the neighborhood. Particles within this radius are
  considered neighbors.
  \param buffer_size Optional guess for maximum number of neighbors per
  particle.

  Neighbor list implementation most appropriate for highly varying particle
  densities.
*/
template <typename Positions, typename Tag>
auto make2DNeighborList( Tag tag, Positions const& positions,
                         typename Positions::size_type first,
                         typename Positions::size_type last,
                         typename Positions::value_type radius,
                         int buffer_size = 0 )
{
    using exec_space = typename Positions::execution_space;
    return make2DNeighborList( exec_space{}, tag, positions, first, last,
                               radius, buffer_size );
}

/*!
  \brief Neighbor list implementation using ArborX for particles within the
  interaction distance with a 2D layout for particles and neighbors.

  \tparam DeviceType Kokkos device type.
  \tparam Positions The position type.
  \tparam Tag Tag indicating whether to build a full or half neighbor list.

  \param tag Tag indicating whether to build a full or half neighbor list.
  \param positions The particle positions.
  \param first The beginning particle index to compute neighbors for.
  \param last The end particle index to compute neighbors for.
  \param radius The radius of the neighborhood. Particles within this radius are
  considered neighbors.
  \param buffer_size Optional guess for maximum number of neighbors per
  particle.

  Neighbor list implementation most appropriate for highly varying particle
  densities.
*/
template <typename DeviceType, typename Positions, typename Tag>
[[deprecated]] auto make2DNeighborList( Tag tag, Positions const& positions,
                                        typename Positions::size_type first,
                                        typename Positions::size_type last,
                                        typename Positions::value_type radius,
                                        int buffer_size = 0 )
{
    using exec_space = typename DeviceType::execution_space;
    return make2DNeighborList( exec_space{}, tag, positions, first, last,
                               radius, buffer_size );
}

} // namespace Experimental

//! 1d ArborX NeighborList interface.
template <typename MemorySpace, typename Tag>
class NeighborList<Experimental::CrsGraph<MemorySpace, Tag>>
{
    //! Size type.
    using size_type = std::size_t;
    //! Neighbor storage type.
    using crs_graph_type = Experimental::CrsGraph<MemorySpace, Tag>;

  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;

    //! Get the total number of neighbors across all particles.
    KOKKOS_INLINE_FUNCTION
    static size_type totalNeighbor( crs_graph_type const& crs_graph )
    {
        return Impl::totalNeighbor( crs_graph, crs_graph.total );
    }

    //! Get the maximum number of neighbors across all particles.
    KOKKOS_INLINE_FUNCTION
    static size_type maxNeighbor( crs_graph_type const& crs_graph )
    {
        return Impl::maxNeighbor( crs_graph, crs_graph.total );
    }

    //! Get the number of neighbors for a given particle index.
    static KOKKOS_FUNCTION size_type
    numNeighbor( crs_graph_type const& crs_graph, size_type p )
    {
        assert( (int)p >= 0 && p < crs_graph.total );
        p -= crs_graph.shift;
        if ( (int)p < 0 || p >= crs_graph.row_ptr.size() - 1 )
            return 0;
        return crs_graph.row_ptr( p + 1 ) - crs_graph.row_ptr( p );
    }
    //! Get the id for a neighbor for a given particle index and neighbor index.
    static KOKKOS_FUNCTION size_type
    getNeighbor( crs_graph_type const& crs_graph, size_type p, size_type n )
    {
        assert( n < numNeighbor( crs_graph, p ) );
        p -= crs_graph.shift;
        return crs_graph.col_ind( crs_graph.row_ptr( p ) + n );
    }
};

//! 2d ArborX NeighborList interface.
template <typename MemorySpace, typename Tag>
class NeighborList<Experimental::Dense<MemorySpace, Tag>>
{
    //! Size type.
    using size_type = std::size_t;
    //! Neighbor storage type.
    using specialization_type = Experimental::Dense<MemorySpace, Tag>;

  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;

    //! Get the total number of neighbors across all particles.
    KOKKOS_INLINE_FUNCTION
    static size_type totalNeighbor( specialization_type const& d )
    {
        return Impl::totalNeighbor( d, d.total );
    }

    //! Get the maximum number of neighbors across all particles.
    KOKKOS_INLINE_FUNCTION
    static size_type maxNeighbor( specialization_type const& d )
    {
        return Impl::maxNeighbor( d, d.total );
    }

    //! Get the number of neighbors for a given particle index.
    static KOKKOS_FUNCTION size_type numNeighbor( specialization_type const& d,
                                                  size_type p )
    {
        assert( (int)p >= 0 && p < d.total );
        p -= d.shift;
        if ( (int)p < 0 || p >= d.cnt.size() )
            return 0;
        return d.cnt( p );
    }
    //! Get the id for a neighbor for a given particle index and neighbor index.
    static KOKKOS_FUNCTION size_type getNeighbor( specialization_type const& d,
                                                  size_type p, size_type n )
    {
        assert( n < numNeighbor( d, p ) );
        p -= d.shift;
        return d.val( p, n );
    }
};

} // namespace Cabana

#endif
