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
  \file Cabana_Parallel.hpp
  \brief SIMD and neighbor extension of Kokkos parallel iteration
*/
#ifndef CABANA_PARALLEL_HPP
#define CABANA_PARALLEL_HPP

#include <Cabana_ExecutionPolicy.hpp>
#include <Cabana_LinkedCellList.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Types.hpp> // is_accessible_from
#include <impl/Cabana_CartesianGrid.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
namespace Impl
{
//! \cond Impl

// No work tag was provided so call without a tag argument.
template <class WorkTag, class FunctorType, class... IndexTypes>
KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<std::is_same<WorkTag, void>::value>::type
    functorTagDispatch( const FunctorType& functor, IndexTypes&&... indices )
{
    functor( std::forward<IndexTypes>( indices )... );
}

// The user gave us a tag so call the version using that.
template <class WorkTag, class FunctorType, class... IndexTypes>
KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<!std::is_same<WorkTag, void>::value>::type
    functorTagDispatch( const FunctorType& functor, IndexTypes&&... indices )
{
    const WorkTag t{};
    functor( t, std::forward<IndexTypes>( indices )... );
}

// No work tag was provided so call reduce without a tag argument.
template <class WorkTag, class FunctorType, class... IndexTypes,
          class ReduceType>
KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<std::is_same<WorkTag, void>::value>::type
    functorTagDispatch( const FunctorType& functor, IndexTypes&&... indices,
                        ReduceType& reduce_val )
{
    functor( std::forward<IndexTypes>( indices )..., reduce_val );
}

// The user gave us a tag so call the reduce version using that.
template <class WorkTag, class FunctorType, class... IndexTypes,
          class ReduceType>
KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<!std::is_same<WorkTag, void>::value>::type
    functorTagDispatch( const FunctorType& functor, IndexTypes&&... indices,
                        ReduceType& reduce_val )
{
    const WorkTag t{};
    functor( t, std::forward<IndexTypes>( indices )..., reduce_val );
}

template <class ExecutionPolicy, class Functor>
struct ParallelFor;

template <class Functor, int VectorLength, class... Properties>
struct ParallelFor<SimdPolicy<VectorLength, Properties...>, Functor>
{
    using simd_policy = SimdPolicy<VectorLength, Properties...>;
    using team_policy = typename simd_policy::base_type;
    using work_tag = typename team_policy::work_tag;
    using index_type = typename team_policy::index_type;
    using member_type = typename team_policy::member_type;

    simd_policy exec_policy_;
    Functor functor_;

    ParallelFor( std::string label, simd_policy exec_policy, Functor functor )
        : exec_policy_( std::move( exec_policy ) )
        , functor_( std::move( functor ) )
    {
        if ( label.empty() )
            Kokkos::parallel_for(
                dynamic_cast<const team_policy&>( exec_policy_ ), *this );
        else
            Kokkos::parallel_for(
                label, dynamic_cast<const team_policy&>( exec_policy_ ),
                *this );
    }

    template <class WorkTag>
    KOKKOS_FUNCTION std::enable_if_t<!std::is_void<WorkTag>::value &&
                                     std::is_same<WorkTag, work_tag>::value>
    operator()( WorkTag, member_type const& team ) const
    {
        this->operator()( team );
    }

    KOKKOS_FUNCTION void operator()( member_type const& team ) const
    {
        index_type s = team.league_rank() + exec_policy_.structBegin();
        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange( team, exec_policy_.arrayBegin( s ),
                                       exec_policy_.arrayEnd( s ) ),
            [&]( index_type a )
            { Impl::functorTagDispatch<work_tag>( functor_, s, a ); } );
    }
};

//! \endcond
} // end namespace Impl

//---------------------------------------------------------------------------//
// SIMD Parallel For
//---------------------------------------------------------------------------//
/*!
  \brief Execute a vectorized functor in parallel with a 2d execution policy.

  \tparam FunctorType The functor type to execute.

  \tparam VectorLength The length of the vector over which to execute the
  vectorized code.

  \tparam ExecParameters Execution policy parameters.

  \param exec_policy The 2D range policy over which to execute the functor.

  \param functor The vectorized functor to execute in parallel. Must accept
  both a struct and array index.

  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_for called by this code and can be used for
  identification and profiling purposes.

  A "functor" is a callable object containing the function to execute in
  parallel, data needed for that execution, and an optional \c execution_space
  typedef.  Here is an example functor for parallel_for:

  \code
  class FunctorType {
  public:
  typedef  ...  execution_space ;
  void operator() ( const int struct, const int array ) const ;
  };
  \endcode

  In the above example, \c struct defines an index to a given AoSoA/Slice
  struct and array defines and index to the given array element in that struct.
  Its <tt>operator()</tt> method defines the operation to parallelize, over
  the range of indices <tt>idx=[begin,end]</tt>. The kernel represented by the
  functor is intended to vectorize of the array index.

  \note The work tag gets applied at the user functor level, not at the level
  of the functor in this implementation that wraps the user functor.
*/
template <class FunctorType, int VectorLength, class... ExecParameters>
inline void simd_parallel_for(
    const SimdPolicy<VectorLength, ExecParameters...>& exec_policy,
    const FunctorType& functor, const std::string& str = "" )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::simd_parallel_for" );

    Impl::ParallelFor<SimdPolicy<VectorLength, ExecParameters...>, FunctorType>(
        str, exec_policy, functor );
}

//---------------------------------------------------------------------------//
// Neighbor Parallel For
//---------------------------------------------------------------------------//
// Algorithm tags.

//! Loop over particle neighbors.
class FirstNeighborsTag
{
};

//! Loop over particle neighbors (first) and neighbor's neighbors (second)
class SecondNeighborsTag
{
};

//! Neighbor operations are executed in serial on each particle thread.
class SerialOpTag
{
};

//! Neighbor operations are executed with team parallelism.
class TeamOpTag
{
};

//! Neighbor operations are executed with team vector parallelism.
class TeamVectorOpTag
{
};

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor in parallel according to the execution policy over
  particles with a thread-local serial loop over particle first neighbors.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The neighbor list over which to execute the neighbor operations.
  \note FirstNeighborsTag Tag indicating operations over particle first
  neighbors.
  \note SerialOpTag Tag indicating a serial loop strategy over neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_for called by this code and can be used for
  identification and profiling purposes.

  A "functor" is a class containing the function to execute in parallel, data
  needed for that execution, and an optional \c execution_space typedef.  Here
  is an example functor for neighbor parallel_for:

  \code
  class FunctorType {
  public:
  typedef  ...  execution_space ;
  void operator() ( const int particle_index, const int neighbor_index ) const ;
  };
  \endcode

  In the above example, \c Index is a Cabana index to a given AoSoA element
  for a particle and its neighbor. Its <tt>operator()</tt> method defines the
  operation to parallelize, over the range of indices
  <tt>idx=[begin,end]</tt>.  This compares to a single iteration \c idx of a
  \c for loop.
*/
template <class FunctorType, class NeighborListType, class... ExecParameters>
inline void neighbor_parallel_for(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const NeighborListType& list,
    const FirstNeighborsTag, const SerialOpTag, const std::string& str = "",
    typename std::enable_if<( !is_linked_cell_list<NeighborListType>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::neighbor_parallel_for" );

    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using index_type =
        typename Kokkos::RangePolicy<ExecParameters...>::index_type;

    using neighbor_list_traits = NeighborList<NeighborListType>;

    using memory_space = typename neighbor_list_traits::memory_space;

    auto begin = exec_policy.begin();
    auto end = exec_policy.end();
    using linear_policy_type = Kokkos::RangePolicy<execution_space, void, void>;
    linear_policy_type linear_exec_policy( begin, end );

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    auto neigh_func = KOKKOS_LAMBDA( const index_type i )
    {
        for ( index_type n = 0;
              n < neighbor_list_traits::numNeighbor( list, i ); ++n )
            Impl::functorTagDispatch<work_tag>(
                functor, i,
                static_cast<index_type>(
                    neighbor_list_traits::getNeighbor( list, i, n ) ) );
    };
    if ( str.empty() )
        Kokkos::parallel_for( linear_exec_policy, neigh_func );
    else
        Kokkos::parallel_for( str, linear_exec_policy, neigh_func );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor in parallel according to the execution policy over
  particles with thread-local serial loops over particle first and second
  neighbors.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The neighbor list over which to execute the neighbor operations.
  \note SecondNeighborsTag Tag indicating operations over particle first and
  second neighbors.
  \note SerialOpTag Tag indicating a serial loop strategy over neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_for called by this code and can be used for
  identification and profiling purposes.
*/
template <class FunctorType, class NeighborListType, class... ExecParameters>
inline void neighbor_parallel_for(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const NeighborListType& list,
    const SecondNeighborsTag, const SerialOpTag, const std::string& str = "",
    typename std::enable_if<( !is_linked_cell_list<NeighborListType>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::neighbor_parallel_for" );

    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using index_type =
        typename Kokkos::RangePolicy<ExecParameters...>::index_type;

    using neighbor_list_traits = NeighborList<NeighborListType>;

    using memory_space = typename neighbor_list_traits::memory_space;

    auto begin = exec_policy.begin();
    auto end = exec_policy.end();
    using linear_policy_type = Kokkos::RangePolicy<execution_space, void, void>;
    linear_policy_type linear_exec_policy( begin, end );

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    auto neigh_func = KOKKOS_LAMBDA( const index_type i )
    {
        const index_type nn = neighbor_list_traits::numNeighbor( list, i );

        for ( index_type n = 0; n < nn; ++n )
        {
            const index_type j =
                neighbor_list_traits::getNeighbor( list, i, n );

            for ( index_type a = n + 1; a < nn; ++a )
            {
                const index_type k =
                    neighbor_list_traits::getNeighbor( list, i, a );
                Impl::functorTagDispatch<work_tag>( functor, i, j, k );
            }
        }
    };
    if ( str.empty() )
        Kokkos::parallel_for( linear_exec_policy, neigh_func );
    else
        Kokkos::parallel_for( str, linear_exec_policy, neigh_func );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor in parallel according to the execution policy over
  particles with team parallelism over particle first neighbors.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The neighbor list over which to execute the neighbor operations.
  \note FirstNeighborsTag Tag indicating operations over particle first
  neighbors.
  \note TeamOpTag Tag indicating a team parallel strategy over neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_for called by this code and can be used for
  identification and profiling purposes.
*/
template <class FunctorType, class NeighborListType, class... ExecParameters>
inline void neighbor_parallel_for(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const NeighborListType& list,
    const FirstNeighborsTag, const TeamOpTag, const std::string& str = "",
    typename std::enable_if<( !is_linked_cell_list<NeighborListType>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::neighbor_parallel_for" );

    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using kokkos_policy =
        Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>;
    kokkos_policy team_policy( exec_policy.end() - exec_policy.begin(),
                               Kokkos::AUTO );

    using index_type = typename kokkos_policy::index_type;

    using neighbor_list_traits = NeighborList<NeighborListType>;

    using memory_space = typename neighbor_list_traits::memory_space;

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    const auto range_begin = exec_policy.begin();

    auto neigh_func =
        KOKKOS_LAMBDA( const typename kokkos_policy::member_type& team )
    {
        index_type i = team.league_rank() + range_begin;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(
                team, neighbor_list_traits::numNeighbor( list, i ) ),
            [&]( const index_type n )
            {
                Impl::functorTagDispatch<work_tag>(
                    functor, i,
                    static_cast<index_type>(
                        neighbor_list_traits::getNeighbor( list, i, n ) ) );
            } );
    };
    if ( str.empty() )
        Kokkos::parallel_for( team_policy, neigh_func );
    else
        Kokkos::parallel_for( str, team_policy, neigh_func );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor in parallel according to the execution policy over
  particles with team parallelism over particle first neighbors and serial loop
  over second neighbors.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The neighbor list over which to execute the neighbor operations.
  \note SecondNeighborsTag Tag indicating operations over particle first and
  second neighbors.
  \note TeamOpTag Tag indicating a team parallel strategy over particle first
  neighbors and serial execution over second neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_for called by this code and can be used for
  identification and profiling purposes.
*/
template <class FunctorType, class NeighborListType, class... ExecParameters>
inline void neighbor_parallel_for(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const NeighborListType& list,
    const SecondNeighborsTag, const TeamOpTag, const std::string& str = "",
    typename std::enable_if<( !is_linked_cell_list<NeighborListType>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::neighbor_parallel_for" );

    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using kokkos_policy =
        Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>;
    kokkos_policy team_policy( exec_policy.end() - exec_policy.begin(),
                               Kokkos::AUTO );

    using index_type = typename kokkos_policy::index_type;

    using neighbor_list_traits = NeighborList<NeighborListType>;

    using memory_space = typename neighbor_list_traits::memory_space;

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    const auto range_begin = exec_policy.begin();

    auto neigh_func =
        KOKKOS_LAMBDA( const typename kokkos_policy::member_type& team )
    {
        index_type i = team.league_rank() + range_begin;

        const index_type nn = neighbor_list_traits::numNeighbor( list, i );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team, nn ),
            [&]( const index_type n )
            {
                const index_type j =
                    neighbor_list_traits::getNeighbor( list, i, n );

                for ( index_type a = n + 1; a < nn; ++a )
                {
                    const index_type k =
                        neighbor_list_traits::getNeighbor( list, i, a );
                    Impl::functorTagDispatch<work_tag>( functor, i, j, k );
                }
            } );
    };
    if ( str.empty() )
        Kokkos::parallel_for( team_policy, neigh_func );
    else
        Kokkos::parallel_for( str, team_policy, neigh_func );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor in parallel according to the execution policy over
  particles with team parallelism over particle first neighbors and vector loop
  parallelism over second neighbors.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The neighbor list over which to execute the neighbor operations.
  \note SecondNeighborsTag Tag indicating operations over particle first and
  second neighbors.
  \note TeamVectorOpTag Tag indicating a team parallel strategy over particle
  first neighbors and vector parallel loop strategy over second neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_for called by this code and can be used for
  identification and profiling purposes.
*/
template <class FunctorType, class NeighborListType, class... ExecParameters>
inline void neighbor_parallel_for(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const NeighborListType& list,
    const SecondNeighborsTag, const TeamVectorOpTag,
    const std::string& str = "",
    typename std::enable_if<( !is_linked_cell_list<NeighborListType>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::neighbor_parallel_for" );

    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using kokkos_policy =
        Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>;
    kokkos_policy team_policy( exec_policy.end() - exec_policy.begin(),
                               Kokkos::AUTO );

    using index_type = typename kokkos_policy::index_type;

    using neighbor_list_traits = NeighborList<NeighborListType>;

    using memory_space = typename neighbor_list_traits::memory_space;

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    const auto range_begin = exec_policy.begin();

    auto neigh_func =
        KOKKOS_LAMBDA( const typename kokkos_policy::member_type& team )
    {
        index_type i = team.league_rank() + range_begin;

        const index_type nn = neighbor_list_traits::numNeighbor( list, i );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team, nn ),
            [&]( const index_type n )
            {
                const index_type j =
                    neighbor_list_traits::getNeighbor( list, i, n );

                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange( team, n + 1, nn ),
                    [&]( const index_type a )
                    {
                        const index_type k =
                            neighbor_list_traits::getNeighbor( list, i, a );
                        Impl::functorTagDispatch<work_tag>( functor, i, j, k );
                    } );
            } );
    };
    if ( str.empty() )
        Kokkos::parallel_for( team_policy, neigh_func );
    else
        Kokkos::parallel_for( str, team_policy, neigh_func );
}

//---------------------------------------------------------------------------//
// Neighbor Parallel Reduce
//---------------------------------------------------------------------------//
/*!
  \brief Execute functor reduction in parallel according to the execution policy
  over particles with a thread-local serial loop over particle first neighbors.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.
  \tparam ReduceType The reduction type.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The neighbor list over which to execute the neighbor operations.
  \note FirstNeighborsTag Tag indicating operations over particle first
  neighbors.
  \note SerialOpTag Tag indicating a serial loop strategy over
  neighbors.
  \param reduce_val Scalar to be reduced across particles and neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_reduce called by this code and can be used for
  identification and profiling purposes.

  A "functor" is a class containing the function to execute in parallel, data
  needed for that execution, and an optional \c execution_space typedef.  Here
  is an example functor for neighbor parallel_for:

  \code
  class FunctorType {
  public:
  typedef  ...  execution_space ;
  void operator() ( const int particle_index, const int neighbor_index ) const ;
  };
  \endcode

  In the above example, \c Index is a Cabana index to a given AoSoA element
  for a particle and its neighbor. Its <tt>operator()</tt> method defines the
  operation to parallelize, over the range of indices
  <tt>idx=[begin,end]</tt>.  This compares to a single iteration \c idx of a
  \c for loop.
*/
template <class FunctorType, class NeighborListType, class ReduceType,
          class... ExecParameters>
inline void neighbor_parallel_reduce(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const NeighborListType& list,
    const FirstNeighborsTag, const SerialOpTag, ReduceType& reduce_val,
    const std::string& str = "",
    typename std::enable_if<( !is_linked_cell_list<NeighborListType>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region(
        "Cabana::neighbor_parallel_reduce" );

    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using index_type =
        typename Kokkos::RangePolicy<ExecParameters...>::index_type;

    using neighbor_list_traits = NeighborList<NeighborListType>;

    using memory_space = typename neighbor_list_traits::memory_space;

    auto begin = exec_policy.begin();
    auto end = exec_policy.end();
    using linear_policy_type = Kokkos::RangePolicy<execution_space, void, void>;
    linear_policy_type linear_exec_policy( begin, end );

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    auto neigh_reduce = KOKKOS_LAMBDA( const index_type i, ReduceType& ival )
    {
        for ( index_type n = 0;
              n < neighbor_list_traits::numNeighbor( list, i ); ++n )
            Impl::functorTagDispatch<work_tag>(
                functor, i,
                static_cast<index_type>(
                    neighbor_list_traits::getNeighbor( list, i, n ) ),
                ival );
    };
    if ( str.empty() )
        Kokkos::parallel_reduce( linear_exec_policy, neigh_reduce, reduce_val );
    else
        Kokkos::parallel_reduce( str, linear_exec_policy, neigh_reduce,
                                 reduce_val );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor reduction in parallel according to the execution policy
  over particles with thread-local serial loops over particle first and second
  neighbors.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.
  \tparam ReduceType The reduction type.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The neighbor list over which to execute the neighbor operations.
  \note SecondNeighborsTag Tag indicating operations over particle first and
  second neighbors.
  \note SerialOpTag Tag indicating a serial loop strategy over neighbors.
  \param reduce_val Scalar to be reduced across particles and neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_reduce called by this code and can be used for
  identification and profiling purposes.
*/
template <class FunctorType, class NeighborListType, class ReduceType,
          class... ExecParameters>
inline void neighbor_parallel_reduce(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const NeighborListType& list,
    const SecondNeighborsTag, const SerialOpTag, ReduceType& reduce_val,
    const std::string& str = "",
    typename std::enable_if<( !is_linked_cell_list<NeighborListType>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region(
        "Cabana::neighbor_parallel_reduce" );

    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using index_type =
        typename Kokkos::RangePolicy<ExecParameters...>::index_type;

    using neighbor_list_traits = NeighborList<NeighborListType>;

    using memory_space = typename neighbor_list_traits::memory_space;

    auto begin = exec_policy.begin();
    auto end = exec_policy.end();
    using linear_policy_type = Kokkos::RangePolicy<execution_space, void, void>;
    linear_policy_type linear_exec_policy( begin, end );

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    auto neigh_reduce = KOKKOS_LAMBDA( const index_type i, ReduceType& ival )
    {
        const index_type nn = neighbor_list_traits::numNeighbor( list, i );

        for ( index_type n = 0; n < nn; ++n )
        {
            const index_type j =
                neighbor_list_traits::getNeighbor( list, i, n );

            for ( index_type a = n + 1; a < nn; ++a )
            {
                const index_type k =
                    neighbor_list_traits::getNeighbor( list, i, a );
                Impl::functorTagDispatch<work_tag>( functor, i, j, k, ival );
            }
        }
    };
    if ( str.empty() )
        Kokkos::parallel_reduce( linear_exec_policy, neigh_reduce, reduce_val );
    else
        Kokkos::parallel_reduce( str, linear_exec_policy, neigh_reduce,
                                 reduce_val );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor reduction in parallel according to the execution policy
  over particles with team parallelism over particle first neighbors.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.
  \tparam ReduceType The reduction type.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The neighbor list over which to execute the neighbor operations.
  \note FirstNeighborsTag Tag indicating operations over particle first
  neighbors.
  \note TeamOpTag Tag indicating a team parallel strategy over particle
  neighbors.
  \param reduce_val Scalar to be reduced across particles and neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_reduce called by this code and can be used for
  identification and profiling purposes.
*/
template <class FunctorType, class NeighborListType, class ReduceType,
          class... ExecParameters>
inline void neighbor_parallel_reduce(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const NeighborListType& list,
    const FirstNeighborsTag, const TeamOpTag, ReduceType& reduce_val,
    const std::string& str = "",
    typename std::enable_if<( !is_linked_cell_list<NeighborListType>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region(
        "Cabana::neighbor_parallel_reduce" );

    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using kokkos_policy =
        Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>;
    kokkos_policy team_policy( exec_policy.end() - exec_policy.begin(),
                               Kokkos::AUTO );

    using index_type = typename kokkos_policy::index_type;

    using neighbor_list_traits = NeighborList<NeighborListType>;

    using memory_space = typename neighbor_list_traits::memory_space;

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    const auto range_begin = exec_policy.begin();

    auto neigh_reduce = KOKKOS_LAMBDA(
        const typename kokkos_policy::member_type& team, ReduceType& ival )
    {
        index_type i = team.league_rank() + range_begin;
        ReduceType reduce_n = 0;

        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(
                team, neighbor_list_traits::numNeighbor( list, i ) ),
            [&]( const index_type n, ReduceType& nval )
            {
                Impl::functorTagDispatch<work_tag>(
                    functor, i,
                    static_cast<index_type>(
                        neighbor_list_traits::getNeighbor( list, i, n ) ),
                    nval );
            },
            reduce_n );
        Kokkos::single( Kokkos::PerTeam( team ), [&]() { ival += reduce_n; } );
    };
    if ( str.empty() )
        Kokkos::parallel_reduce( team_policy, neigh_reduce, reduce_val );
    else
        Kokkos::parallel_reduce( str, team_policy, neigh_reduce, reduce_val );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor reduction in parallel according to the execution policy
  over particles with team parallelism over particle first neighbors and serial
  loop over second neighbors.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.
  \tparam ReduceType The reduction type.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The neighbor list over which to execute the neighbor operations.
  \note SecondNeighborsTag Tag indicating operations over particle first and
  second neighbors.
  \note TeamOpTag Tag indicating a team parallel strategy over particle first
  neighbors and serial loops over second neighbors.
  \param reduce_val Scalar to be reduced across particles and neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_reduce called by this code and can be used for
  identification and profiling purposes.
*/
template <class FunctorType, class NeighborListType, class ReduceType,
          class... ExecParameters>
inline void neighbor_parallel_reduce(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const NeighborListType& list,
    const SecondNeighborsTag, const TeamOpTag, ReduceType& reduce_val,
    const std::string& str = "",
    typename std::enable_if<( !is_linked_cell_list<NeighborListType>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region(
        "Cabana::neighbor_parallel_reduce" );

    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using kokkos_policy =
        Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>;
    kokkos_policy team_policy( exec_policy.end() - exec_policy.begin(),
                               Kokkos::AUTO );

    using index_type = typename kokkos_policy::index_type;

    using neighbor_list_traits = NeighborList<NeighborListType>;

    using memory_space = typename neighbor_list_traits::memory_space;

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    const auto range_begin = exec_policy.begin();

    auto neigh_reduce = KOKKOS_LAMBDA(
        const typename kokkos_policy::member_type& team, ReduceType& ival )
    {
        index_type i = team.league_rank() + range_begin;
        ReduceType reduce_n = 0;

        const index_type nn = neighbor_list_traits::numNeighbor( list, i );
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange( team, nn ),
            [&]( const index_type n, ReduceType& nval )
            {
                const index_type j =
                    neighbor_list_traits::getNeighbor( list, i, n );

                for ( index_type a = n + 1; a < nn; ++a )
                {
                    const index_type k =
                        neighbor_list_traits::getNeighbor( list, i, a );
                    Impl::functorTagDispatch<work_tag>( functor, i, j, k,
                                                        nval );
                }
            },
            reduce_n );
        Kokkos::single( Kokkos::PerTeam( team ), [&]() { ival += reduce_n; } );
    };
    if ( str.empty() )
        Kokkos::parallel_reduce( team_policy, neigh_reduce, reduce_val );
    else
        Kokkos::parallel_reduce( str, team_policy, neigh_reduce, reduce_val );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor reduction in parallel according to the execution policy
  over particles with team parallelism over particle first neighbors and vector
  loop parallelism over second neighbors.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.
  \tparam ReduceType The reduction type.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The neighbor list over which to execute the neighbor operations.
  \note SecondNeighborsTag Tag indicating operations over particle first and
  second neighbors.
  \note TeamVectorOpTag Tag indicating a team parallel strategy over particle
  first neighbors and vector loops over second neighbors.
  \param reduce_val Scalar to be reduced across particles and neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_reduce called by this code and can be used for
  identification and profiling purposes.
*/
template <class FunctorType, class NeighborListType, class ReduceType,
          class... ExecParameters>
inline void neighbor_parallel_reduce(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const NeighborListType& list,
    const SecondNeighborsTag, const TeamVectorOpTag, ReduceType& reduce_val,
    const std::string& str = "",
    typename std::enable_if<( !is_linked_cell_list<NeighborListType>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region(
        "Cabana::neighbor_parallel_reduce" );

    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using kokkos_policy =
        Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>;
    kokkos_policy team_policy( exec_policy.end() - exec_policy.begin(),
                               Kokkos::AUTO );

    using index_type = typename kokkos_policy::index_type;

    using neighbor_list_traits = NeighborList<NeighborListType>;

    using memory_space = typename neighbor_list_traits::memory_space;

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    const auto range_begin = exec_policy.begin();

    auto neigh_reduce = KOKKOS_LAMBDA(
        const typename kokkos_policy::member_type& team, ReduceType& ival )
    {
        index_type i = team.league_rank() + range_begin;
        ReduceType reduce_n = 0;

        const index_type nn = neighbor_list_traits::numNeighbor( list, i );
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange( team, nn ),
            [&]( const index_type n, ReduceType& nval )
            {
                const index_type j =
                    neighbor_list_traits::getNeighbor( list, i, n );
                ReduceType reduce_a = 0;

                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange( team, n + 1, nn ),
                    [&]( const index_type a, ReduceType& aval )
                    {
                        const index_type k =
                            neighbor_list_traits::getNeighbor( list, i, a );
                        Impl::functorTagDispatch<work_tag>( functor, i, j, k,
                                                            aval );
                    },
                    reduce_a );
                nval += reduce_a;
            },
            reduce_n );
        Kokkos::single( Kokkos::PerTeam( team ), [&]() { ival += reduce_n; } );
    };
    if ( str.empty() )
        Kokkos::parallel_reduce( team_policy, neigh_reduce, reduce_val );
    else
        Kokkos::parallel_reduce( str, team_policy, neigh_reduce, reduce_val );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor in serial within existing parallel kernel over particle
  first neighbors.

  \tparam IndexType The particle index type.
  \tparam FunctorType The neighbor functor type to execute.
  \tparam NeighborListType The neighbor list type.

  \param i Particle index.
  \param neighbor_functor The neighbor functor to execute in parallel.
  \param list The neighbor list over which to execute the neighbor operations.
  \note FirstNeighborsTag Tag indicating operations over particle first
  neighbors.

  A "functor" is a class containing the function to execute in parallel, data
  needed for that execution, and an optional \c execution_space typedef.  Here
  is an example functor for neighbor parallel_for:

  \code
  class FunctorType {
  public:
  typedef  ...  execution_space ;
  void operator() ( const int particle_index, const int neighbor_index ) const ;
  };
  \endcode

  In the above example, \c Index is a Cabana index to a given AoSoA element
  for a particle and its neighbor. Its <tt>operator()</tt> method defines the
  operation to parallelize, over the range of indices
  <tt>idx=[begin,end]</tt>.  This compares to a single iteration \c idx of a
  \c for loop.
*/
template <class IndexType, class FunctorType, class NeighborListType>
KOKKOS_INLINE_FUNCTION void
for_each_neighbor( const IndexType i, const FunctorType& neighbor_functor,
                   const NeighborListType& list, const FirstNeighborsTag )
{
    using neighbor_list_traits = NeighborList<NeighborListType>;

    for ( IndexType n = 0;
          n < static_cast<IndexType>(
                  neighbor_list_traits::numNeighbor( list, i ) );
          ++n )
        neighbor_functor(
            i, static_cast<IndexType>(
                   neighbor_list_traits::getNeighbor( list, i, n ) ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute team parallel functor within existing parallel kernel over
  particle first neighbors.

  \tparam IndexType The particle index type.
  \tparam FunctorType The neighbor functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam TeamMemberType Kokkos team policy.

  \param i Particle index.
  \param team Kokkos team.
  \param neighbor_functor The neighbor functor to execute in parallel.
  \param list The neighbor list over which to execute the neighbor operations.
  \note FirstNeighborsTag Tag indicating operations over particle first
  neighbors.
*/
template <class IndexType, class FunctorType, class NeighborListType,
          class TeamMemberType>
KOKKOS_INLINE_FUNCTION void
for_each_neighbor( const IndexType i, const TeamMemberType team,
                   const FunctorType& neighbor_functor,
                   const NeighborListType& list, const FirstNeighborsTag )
{
    using neighbor_list_traits = NeighborList<NeighborListType>;

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange( team,
                                 neighbor_list_traits::numNeighbor( list, i ) ),
        [&]( const IndexType n )
        {
            Impl::functorTagDispatch<void>(
                neighbor_functor, i,
                static_cast<IndexType>(
                    neighbor_list_traits::getNeighbor( list, i, n ) ) );
        } );
}

//---------------------------------------------------------------------------//
// Linked Cell Parallel For
//---------------------------------------------------------------------------//
namespace Impl
{
//! \cond Impl
/*!
  \brief Struct for performing a loop over linked cell bins
  and then the particles in those bins on device
*/
template <class WorkTag, class Functor, class Policy, class LinkedCellType,
          class ViewType>
struct LinkedCellParallelFor
{
    //! index type to be used for _begin
    using index_type = typename Policy::index_type;

    //! Execution policy
    Policy _exec_policy;
    //! Functor to execute
    Functor _functor;
    //! Linked cell list
    LinkedCellType _list;
    //! Spatial dimension
    static constexpr std::size_t num_space_dim = LinkedCellType::num_space_dim;

    //! beginning index of the loop
    index_type _begin;

    //! discriminator for whether a particle is a neighbor or not
    NeighborDiscriminator<SelfNeighborTag> _discriminator;

    //! Constructor
    LinkedCellParallelFor( std::string label, Policy exec_policy,
                           Functor functor, const LinkedCellType& list,
                           const index_type begin = 0 )
        : _exec_policy( exec_policy )
        , _functor( functor )
        , _list( list )
        , _begin( begin )
    {
        if ( label.empty() )
            Kokkos::parallel_for( dynamic_cast<const Policy&>( exec_policy ),
                                  *this );
        else
            Kokkos::parallel_for(
                label, dynamic_cast<const Policy&>( exec_policy ), *this );
    }

    //! Performs a loop over the particles in neighboring bins in Serial
    KOKKOS_FUNCTION void operator()( SerialOpTag, const index_type i ) const
    {
        Kokkos::Array<int, num_space_dim> min;
        Kokkos::Array<int, num_space_dim> max;
        _list.getStencilCells( _list.getParticleBin( i ), min, max );

        iterate_serial_bins( min, max, i );
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    iterate_serial_bins( const Kokkos::Array<int, 3> min,
                         const Kokkos::Array<int, 3> max,
                         const std::size_t i ) const
    {
        Kokkos::Array<int, 3> ijk;
        for ( int gi = min[0]; gi < max[0]; ++gi )
            for ( int gj = min[1]; gj < max[1]; ++gj )
                for ( int gk = min[2]; gk < max[2]; ++gk )
                {
                    ijk = { gi, gj, gk };
                    iterate_serial_particles( ijk, i );
                }
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    iterate_serial_bins( const Kokkos::Array<int, 2> min,
                         const Kokkos::Array<int, 2> max,
                         const std::size_t i ) const
    {
        Kokkos::Array<int, 2> ij;
        for ( int gi = min[0]; gi < max[0]; ++gi )
            for ( int gj = min[1]; gj < max[1]; ++gj )
            {
                ij = { gi, gj };
                iterate_serial_particles( ij, i );
            }
    }

    KOKKOS_INLINE_FUNCTION void
    iterate_serial_particles( const Kokkos::Array<int, num_space_dim> ijk,
                              const std::size_t i ) const
    {
        // Check the particles in this bin to see if they are
        // neighbors.
        auto offset = _list.binOffset( ijk );
        auto size = _list.binSize( ijk );
        for ( std::size_t n = offset; n < offset + size; ++n )
        {
            // Get the true id of the candidate neighbor.
            auto j = _list.getParticle( n );

            // Avoid self interactions (dummy position args).
            if ( _discriminator.isValid( i, j ) )
            {
                Impl::functorTagDispatch<WorkTag>( _functor, i, j );
            }
        }
    }

    //! Performs a loop over the particles in neighboring bins in TeamOp
    KOKKOS_FUNCTION void
    operator()( TeamOpTag, const typename Policy::member_type& team ) const
    {
        index_type i = team.league_rank() + _begin;
        Kokkos::Array<int, num_space_dim> min;
        Kokkos::Array<int, num_space_dim> max;
        _list.getStencilCells( _list.getParticleBin( i ), min, max );

        iterate_team_bins( team, min, max, i );
    }

    template <class TeamType, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    iterate_team_bins( TeamType team, const Kokkos::Array<int, 3> min,
                       const Kokkos::Array<int, 3> max,
                       const std::size_t i ) const
    {
        Kokkos::Array<int, 3> ijk;
        for ( int gi = min[0]; gi < max[0]; ++gi )
            for ( int gj = min[1]; gj < max[1]; ++gj )
                for ( int gk = min[2]; gk < max[2]; ++gk )
                {
                    ijk = { gi, gj, gk };
                    iterate_team_particles( team, ijk, i );
                }
    }

    template <class TeamType, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    iterate_team_bins( TeamType team, const Kokkos::Array<int, 2> min,
                       const Kokkos::Array<int, 2> max,
                       const std::size_t i ) const
    {
        Kokkos::Array<int, 2> ij;
        for ( int gi = min[0]; gi < max[0]; ++gi )
            for ( int gj = min[1]; gj < max[1]; ++gj )
            {
                ij = { gi, gj };
                iterate_team_particles( team, ij, i );
            }
    }

    template <class TeamType>
    KOKKOS_INLINE_FUNCTION void
    iterate_team_particles( TeamType team,
                            const Kokkos::Array<int, num_space_dim> ijk,
                            const std::size_t i ) const
    {
        // Check the particles in this bin to see if they
        // are neighbors.
        auto offset = _list.binOffset( ijk );
        auto size = _list.binSize( ijk );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team, offset, offset + size ),
            [&]( const index_type n )
            {
                // Get the true id of the candidate neighbor.
                auto j = _list.getParticle( n );

                // Avoid self interactions (dummy position args).
                if ( _discriminator.isValid( i, j ) )
                {
                    Impl::functorTagDispatch<WorkTag>( _functor, i, j );
                }
            } );
    }
};

/*!
  \brief Struct for performing a loop over linked cell bins
  and then the particles in those bins on device
*/
template <class WorkTag, class Functor, class Policy, class LinkedCellType,
          class ViewType, class ReduceType>
struct LinkedCellParallelReduce
{
    //! index type to be used for _begin
    using index_type = typename Policy::index_type;

    //! Execution policy
    Policy _exec_policy;
    //! Functor to execute
    Functor _functor;
    //! Linked cell list
    LinkedCellType _list;
    //! Spatial dimension
    static constexpr std::size_t num_space_dim = LinkedCellType::num_space_dim;

    //! beginning index of the loop
    index_type _begin;

    //! discriminator for whether a particle is a neighbor or not
    NeighborDiscriminator<SelfNeighborTag> _discriminator;

    //! Constructor
    LinkedCellParallelReduce( std::string label, Policy exec_policy,
                              Functor functor, const LinkedCellType& list,
                              ReduceType& reduce_val,
                              const index_type begin = 0 )
        : _exec_policy( exec_policy )
        , _functor( functor )
        , _list( list )
        , _begin( begin )
    {
        if ( label.empty() )
            Kokkos::parallel_reduce( dynamic_cast<const Policy&>( exec_policy ),
                                     *this, reduce_val );
        else
            Kokkos::parallel_reduce( label,
                                     dynamic_cast<const Policy&>( exec_policy ),
                                     *this, reduce_val );
    }

    //! Performs a loop over the particles in neighboring bins in Serial
    KOKKOS_FUNCTION void operator()( SerialOpTag, const index_type i,
                                     ReduceType& ival ) const
    {
        Kokkos::Array<int, num_space_dim> min;
        Kokkos::Array<int, num_space_dim> max;
        _list.getStencilCells( _list.getParticleBin( i ), min, max );

        iterate_serial_bins( min, max, i, ival );
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    iterate_serial_bins( const Kokkos::Array<int, 3> min,
                         const Kokkos::Array<int, 3> max, const std::size_t i,
                         ReduceType& ival ) const
    {
        Kokkos::Array<int, 3> ijk;
        for ( int gi = min[0]; gi < max[0]; ++gi )
            for ( int gj = min[1]; gj < max[1]; ++gj )
                for ( int gk = min[2]; gk < max[2]; ++gk )
                {
                    ijk = { gi, gj, gk };
                    iterate_serial_particles( ijk, i, ival );
                }
    }

    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    iterate_serial_bins( const Kokkos::Array<int, 2> min,
                         const Kokkos::Array<int, 2> max, const std::size_t i,
                         ReduceType& ival ) const
    {
        Kokkos::Array<int, 2> ij;
        for ( int gi = min[0]; gi < max[0]; ++gi )
            for ( int gj = min[1]; gj < max[1]; ++gj )
            {
                ij = { gi, gj };
                iterate_serial_particles( ij, i, ival );
            }
    }

    KOKKOS_INLINE_FUNCTION void
    iterate_serial_particles( const Kokkos::Array<int, num_space_dim> ijk,
                              const std::size_t i, ReduceType& ival ) const
    {
        // Check the particles in this bin to see if they are
        // neighbors.
        auto offset = _list.binOffset( ijk );
        auto size = _list.binSize( ijk );
        for ( std::size_t n = offset; n < offset + size; ++n )
        {
            // Get the true id of the candidate neighbor.
            auto j = _list.getParticle( n );

            // Avoid self interactions (dummy position args).
            if ( _discriminator.isValid( i, j ) )
            {
                Impl::functorTagDispatch<WorkTag>( _functor, i, j, ival );
            }
        }
    }

    //! Performs a loop over the particles in neighboring bins in TeamOp
    KOKKOS_FUNCTION void operator()( TeamOpTag,
                                     const typename Policy::member_type& team,
                                     ReduceType& ival ) const
    {
        index_type i = team.league_rank() + _begin;
        Kokkos::Array<int, num_space_dim> min;
        Kokkos::Array<int, num_space_dim> max;
        _list.getStencilCells( _list.getParticleBin( i ), min, max );

        iterate_team_bins( team, min, max, i, ival );
    }

    template <class TeamType, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    iterate_team_bins( TeamType team, const Kokkos::Array<int, 3> min,
                       const Kokkos::Array<int, 3> max, const std::size_t i,
                       ReduceType& ival ) const
    {
        Kokkos::Array<int, 3> ijk;
        for ( int gi = min[0]; gi < max[0]; ++gi )
            for ( int gj = min[1]; gj < max[1]; ++gj )
                for ( int gk = min[2]; gk < max[2]; ++gk )
                {
                    ijk = { gi, gj, gk };
                    iterate_team_particles( team, ijk, i, ival );
                }
    }

    template <class TeamType, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    iterate_team_bins( TeamType team, const Kokkos::Array<int, 2> min,
                       const Kokkos::Array<int, 2> max, const std::size_t i,
                       ReduceType& ival ) const
    {
        Kokkos::Array<int, 2> ij;
        for ( int gi = min[0]; gi < max[0]; ++gi )
            for ( int gj = min[1]; gj < max[1]; ++gj )
            {
                ij = { gi, gj };
                iterate_team_particles( team, ij, i, ival );
            }
    }

    template <class TeamType>
    KOKKOS_INLINE_FUNCTION void
    iterate_team_particles( TeamType team,
                            const Kokkos::Array<int, num_space_dim> ijk,
                            const std::size_t i, ReduceType& ival ) const
    {
        // Check the particles in this bin to see if they
        // are neighbors.
        auto offset = _list.binOffset( ijk );
        auto size = _list.binSize( ijk );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team, offset, offset + size ),
            [&]( const index_type n )
            {
                // Get the true id of the candidate neighbor.
                auto j = _list.getParticle( n );

                // Avoid self interactions (dummy position args).
                if ( _discriminator.isValid( i, j ) )
                {
                    Impl::functorTagDispatch<WorkTag>( _functor, i, j, ival );
                }
            } );
    }
};
//! \endcond
} // namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor in parallel according to the execution policy over
  particles with thread-local serial loops over linked cell list bins and
  particle first neighbors in those bins.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The linked cell list over which to execute neighbor operations.
  \note FirstNeighborsTag Tag indicating operations over particle first
  neighbors.
  \note SerialOpTag Tag indicating a serial loop strategy over neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_for called by this code and can be used for
  identification and profiling purposes.

  A "functor" is a class containing the function to execute in parallel, data
  needed for that execution, and an optional \c execution_space typedef.  Here
  is an example functor for neighbor parallel_for:

  \code
  class FunctorType {
  public:
  typedef  ...  execution_space ;
  void operator() ( const int particle_index, const int neighbor_index ) const ;
  };
  \endcode

  In the above example, \c Index is a Cabana index to a given AoSoA element
  for a particle and its neighbor. Its <tt>operator()</tt> method defines the
  operation to parallelize, over the range of indices
  <tt>idx=[begin,end]</tt>.  This compares to a single iteration \c idx of a
  \c for loop.
*/

template <class FunctorType, class LinkedCellType, class... ExecParameters>
inline void neighbor_parallel_for(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const LinkedCellType& list,
    const FirstNeighborsTag, const SerialOpTag, const std::string& str = "",
    typename std::enable_if<( is_linked_cell_list<LinkedCellType>::value ),
                            int>::type* = 0 )
{
    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;
    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using memory_space = typename LinkedCellType::memory_space;

    auto begin = exec_policy.begin();
    auto end = exec_policy.end();
    // Cannot iterate over range that was not binned.
    assert( begin == list.getParticleBegin() );
    assert( end == list.getParticleEnd() );

    using linear_policy_type =
        Kokkos::RangePolicy<SerialOpTag, execution_space>;
    linear_policy_type linear_exec_policy( begin, end );

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    Impl::LinkedCellParallelFor<work_tag, FunctorType, linear_policy_type,
                                LinkedCellType,
                                typename LinkedCellType::CountView>
        lcl_par( str, linear_exec_policy, functor, list, exec_policy.begin() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor in parallel according to the execution policy over
  particles with thread-local serial loops over linked cell list bins and
  team threading over particle first neighbors in those bins.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The linked cell list over which to execute neighbor operations.
  \note FirstNeighborsTag Tag indicating operations over particle first
  neighbors.
  \note TeamOpTag Tag indicating a team parallel strategy over particle
  neighbors.
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_for called by this code and can be used for
  identification and profiling purposes.
*/

template <class FunctorType, class LinkedCellType, class... ExecParameters>
inline void neighbor_parallel_for(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const LinkedCellType& list,
    const FirstNeighborsTag, const TeamOpTag, const std::string& str = "",
    typename std::enable_if<( is_linked_cell_list<LinkedCellType>::value ),
                            int>::type* = 0 )
{
    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;
    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using team_policy_type =
        Kokkos::TeamPolicy<TeamOpTag, execution_space,
                           Kokkos::Schedule<Kokkos::Dynamic>>;
    team_policy_type team_policy( exec_policy.end() - exec_policy.begin(),
                                  Kokkos::AUTO );

    using memory_space = typename LinkedCellType::memory_space;

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    // Cannot iterate over range that was not binned.
    assert( exec_policy.begin() == list.getParticleBegin() );
    assert( exec_policy.end() == list.getParticleEnd() );

    Impl::LinkedCellParallelFor<work_tag, FunctorType, team_policy_type,
                                LinkedCellType,
                                typename LinkedCellType::CountView>
        lcl_par( str, team_policy, functor, list, exec_policy.begin() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor in parallel according to the execution policy over
  particles with thread-local serial loops over linked cell list bins and
  particle first neighbors in those bins.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The linked cell list over which to execute neighbor operations.
  \note FirstNeighborsTag Tag indicating operations over particle first
  neighbors.
  \note SerialOpTag Tag indicating a serial loop strategy over neighbors.
  \param reduce_val the value begin reduced
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_for called by this code and can be used for
  identification and profiling purposes.

  A "functor" is a class containing the function to execute in parallel, data
  needed for that execution, and an optional \c execution_space typedef.  Here
  is an example functor for neighbor parallel_for:

  \code
  class FunctorType {
  public:
  typedef  ...  execution_space ;
  void operator() ( const int particle_index, const int neighbor_index ) const ;
  };
  \endcode

  In the above example, \c Index is a Cabana index to a given AoSoA element
  for a particle and its neighbor. Its <tt>operator()</tt> method defines the
  operation to parallelize, over the range of indices
  <tt>idx=[begin,end]</tt>.  This compares to a single iteration \c idx of a
  \c for loop.
*/
template <class FunctorType, class LinkedCellType, class ReduceType,
          class... ExecParameters>
inline void neighbor_parallel_reduce(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const LinkedCellType& list,
    const FirstNeighborsTag, const SerialOpTag, ReduceType& reduce_val,
    const std::string& str = "",
    typename std::enable_if<( is_linked_cell_list<LinkedCellType>::value ),
                            int>::type* = 0 )
{
    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;
    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using memory_space = typename LinkedCellType::memory_space;

    auto begin = exec_policy.begin();
    auto end = exec_policy.end();
    // Cannot iterate over range that was not binned.
    assert( begin == list.getParticleBegin() );
    assert( end == list.getParticleEnd() );

    using linear_policy_type =
        Kokkos::RangePolicy<SerialOpTag, execution_space>;
    linear_policy_type linear_exec_policy( begin, end );

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    Impl::LinkedCellParallelReduce<
        work_tag, FunctorType, linear_policy_type, LinkedCellType,
        typename LinkedCellType::CountView, ReduceType>
        lcl_par( str, linear_exec_policy, functor, list, reduce_val,
                 exec_policy.begin() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute functor in parallel according to the execution policy over
  particles with thread-local serial loops over linked cell list bins and
  team threading over particle first neighbors in those bins.

  \tparam FunctorType The functor type to execute.
  \tparam NeighborListType The neighbor list type.
  \tparam ExecParams The Kokkos range policy parameters.

  \param exec_policy The policy over which to execute the functor.
  \param functor The functor to execute in parallel
  \param list The linked cell list over which to execute neighbor operations.
  \note FirstNeighborsTag Tag indicating operations over particle first
  neighbors.
  \note TeamOpTag Tag indicating a team parallel strategy over particle
  neighbors.
  \param reduce_val the value begin reduced
  \param str Optional name for the functor. Will be forwarded if non-empty to
  the Kokkos::parallel_for called by this code and can be used for
  identification and profiling purposes.
*/
template <class FunctorType, class LinkedCellType, class ReduceType,
          class... ExecParameters>
inline void neighbor_parallel_reduce(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor, const LinkedCellType& list,
    const FirstNeighborsTag, const TeamOpTag, ReduceType& reduce_val,
    const std::string& str = "",
    typename std::enable_if<( is_linked_cell_list<LinkedCellType>::value ),
                            int>::type* = 0 )
{
    using work_tag = typename Kokkos::RangePolicy<ExecParameters...>::work_tag;
    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using team_policy_type =
        Kokkos::TeamPolicy<TeamOpTag, execution_space,
                           Kokkos::Schedule<Kokkos::Dynamic>>;
    team_policy_type team_policy( exec_policy.end() - exec_policy.begin(),
                                  Kokkos::AUTO );

    using memory_space = typename LinkedCellType::memory_space;

    static_assert( is_accessible_from<memory_space, execution_space>{}, "" );

    // Cannot iterate over range that was not binned.
    assert( exec_policy.begin() == list.getParticleBegin() );
    assert( exec_policy.end() == list.getParticleEnd() );

    Impl::LinkedCellParallelReduce<
        work_tag, FunctorType, team_policy_type, LinkedCellType,
        typename LinkedCellType::CountView, ReduceType>
        lcl_par( str, team_policy, functor, list, reduce_val,
                 exec_policy.begin() );
}

} // end namespace Cabana

#endif // end CABANA_PARALLEL_HPP
