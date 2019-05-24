/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_PARALLEL_HPP
#define CABANA_PARALLEL_HPP

#include <Cabana_ExecutionPolicy.hpp>
#include <Cabana_NeighborList.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
namespace Impl
{

// No work tag was provided so call without a tag argument.
template<class WorkTag, class FunctorType, class IndexType>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<std::is_same<WorkTag,void>::value>::type
functorTagDispatch( const FunctorType& functor,
                    const IndexType s,
                    const IndexType a )
{
    functor(s,a);
}

// The user gave us a tag so call the version using that.
template<class WorkTag, class FunctorType, class IndexType>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<!std::is_same<WorkTag,void>::value>::type
functorTagDispatch( const FunctorType& functor,
                    const IndexType s,
                    const IndexType a )
{
    const WorkTag t{};
    functor(t,s,a);
}

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

  \param str An optional name for the functor. Will be forwarded to the
  Kokkos::parallel_for called by this code and can be used for identification
  and profiling purposes.

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
template<class FunctorType, int VectorLength, class ... ExecParameters>
inline void simd_parallel_for(
    const SimdPolicy<VectorLength,ExecParameters...>& exec_policy,
    const FunctorType& functor,
    const std::string& str = "" )
{
    using simd_policy = SimdPolicy<VectorLength,ExecParameters...>;

    using work_tag = typename simd_policy::work_tag;

    using team_policy = typename simd_policy::base_type;

    using index_type = typename team_policy::index_type;

    Kokkos::parallel_for(
        str,
        dynamic_cast<const team_policy&>(exec_policy),
        KOKKOS_LAMBDA( const typename team_policy::member_type& team )
        {
            index_type s = team.league_rank() + exec_policy.structBegin();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange( team,
                                           exec_policy.arrayBegin(s),
                                           exec_policy.arrayEnd(s)),
                [&]( const index_type a )
                { Impl::functorTagDispatch<work_tag>(functor,s,a);});
        });
}


//---------------------------------------------------------------------------//
// Neighbor Parallel For
//---------------------------------------------------------------------------//
// Algorithm tags.

//! Neighbor operations are executed in serial on each particle thread.
class SerialNeighborOpTag {};

//! Neighbor operations are executed in parallel in a team on each particle
//! thread.
class TeamNeighborOpTag {};

//---------------------------------------------------------------------------//
/*!
  \brief Execute \c functor in parallel according to the execution \c policy
  with a thread-local serial loop over particle neighbors.

  \tparam FunctorType The functor type to execute.

  \tparam NeighborListType The neighbor list type.

  \tparam ExecParams The Kokkos range policy parameters.

  \param exec_policy The policy over which to execute the functor.

  \param functor The functor to execute in parallel

  \param list The neighbor list over which to execute the neighbor operations.

  \param tag Algorithm tag indicating a serial loop strategy over particle
  neighbors.

  \param str An optional name for the functor. Will be forwarded to the
  Kokkos::parallel_for called by this code and can be used for identification
  and profiling purposes.

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
template<class FunctorType, class NeighborListType, class ... ExecParameters>
inline void neighbor_parallel_for(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor,
    const NeighborListType& list,
    const SerialNeighborOpTag& tag,
    const std::string& str = "" )
{
    std::ignore = tag;

    using work_tag =
        typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using index_type =
        typename Kokkos::RangePolicy<ExecParameters...>::index_type;

    Kokkos::parallel_for(
        str,
        exec_policy,
        KOKKOS_LAMBDA( const index_type i )
        {
            for ( index_type n = 0;
                  n < NeighborList<NeighborListType>::numNeighbor(list,i);
                  ++n )
                Impl::functorTagDispatch<work_tag>(
                    functor,
                    i,
                    static_cast<index_type>(
                        NeighborList<NeighborListType>::getNeighbor(list,i,n))
                    );
        } );
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute \c functor in parallel according to the execution \c policy
  with team parallelism over particle neighbors.

  \tparam ExecutionSpace The execution space in which to execute the functor.

  \tparam FunctorType The functor type to execute.

  \tparam NeighborListType The neighbor list type.

  \param exec_policy The policy over which to execute the functor.

  \param functor The functor to execute in parallel

  \param list The neighbor list over which to execute the neighbor operations.

  \param tag Algorithm tag indicating a team parallel strategy over particle
  neighbors.

  \param str An optional name for the functor. Will be forwarded to the
  Kokkos::parallel_for called by this code and can be used for identification
  and profiling purposes.

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
template<class FunctorType, class NeighborListType, class ... ExecParameters>
inline void neighbor_parallel_for(
    const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
    const FunctorType& functor,
    const NeighborListType& list,
    const TeamNeighborOpTag& tag,
    const std::string& str = "" )
{
    std::ignore = tag;

    using work_tag =
        typename Kokkos::RangePolicy<ExecParameters...>::work_tag;

    using execution_space =
        typename Kokkos::RangePolicy<ExecParameters...>::execution_space;

    using kokkos_policy =
        Kokkos::TeamPolicy<execution_space,
                           Kokkos::Schedule<Kokkos::Dynamic> >;
    kokkos_policy team_policy( exec_policy.end() - exec_policy.begin(),
                               Kokkos::AUTO );

    using index_type = typename kokkos_policy::index_type;

    const auto range_begin = exec_policy.begin();

    Kokkos::parallel_for(
        str,
        team_policy,
        KOKKOS_LAMBDA( const typename kokkos_policy::member_type& team )
        {
            index_type i = team.league_rank() + range_begin;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(
                    team,NeighborList<NeighborListType>::numNeighbor(list,i)),
                [&]( const index_type n ) {
                Impl::functorTagDispatch<work_tag>(
                    functor,
                    i,
                    static_cast<index_type>(
                        NeighborList<NeighborListType>::getNeighbor(list,i,n) )
                    );
                });
        } );
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_PARALLEL_HPP
