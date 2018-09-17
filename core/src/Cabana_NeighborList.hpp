/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_NEIGHBORLIST_HPP
#define CABANA_NEIGHBORLIST_HPP

#include <Cabana_Macros.hpp>

#include <Kokkos_Core.hpp>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Neighbor List Interface
//---------------------------------------------------------------------------//
/*!
  \class FullNeighborTag
  \brief Tag for full neighbor lists.

  In this case every particle has its neighbors stored in the list. So, if
  particle "i" neighbors particle "j" then "j" will be in the neighbor list
  for "i" and "i" will bin the the neighbor list for "j".
*/
class FullNeighborTag {};

//---------------------------------------------------------------------------//
/*!
  \class HalfNeighborTag
  \brief Tag for half neighbor lists.

  In this case only half of the neighbors are stored and the inverse
  relationship is implied. So, if particle "i" neighbors particle "j" then "j"
  will be in the neighbor list for "i" while the fact that "i" is a neighbor
  of "j" is implied.
*/
class HalfNeighborTag {};

//---------------------------------------------------------------------------//
/*!
  \class NeighborList

  \brief Neighbor list interface. Provides an interface callable at the
  functor level that gives access to neighbor data for particles.
*/
template<class NeighborListType>
class NeighborList
{
  public:

    // Get the list type tag. Either full or half.
    using TypeTag = typename NeighborListType::TypeTag;

    // Get the number of neighbors for a given particle index.
    CABANA_INLINE_FUNCTION
    static int numNeighbor( const NeighborListType& list,
                            const std::size_t particle_index );

    // Get the id for a neighbor for a given particle index and the index of
    // the neighbor relative to the particle.
    CABANA_INLINE_FUNCTION
    static int getNeighbor( const NeighborListType& list,
                            const std::size_t particle_index,
                            const int neighbor_index );
};

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

  \tparam ExecutionPolicy The execution policy type over which to execute the
  functor.

  \tparam FunctorType The functor type to execute.

  \tparam NeighborListType The neighbor list type.

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
template<class ExecutionPolicy, class FunctorType, class NeighborListType>
inline void neighbor_parallel_for( const ExecutionPolicy& exec_policy,
                                   const FunctorType& functor,
                                   const NeighborListType& list,
                                   const SerialNeighborOpTag& tag,
                                   const std::string& str = "" )
{
    std::ignore = tag;

    // Serial neighbor operation.
    auto functor_wrapper =
        KOKKOS_LAMBDA( const int i )
        {
            for ( int n = 0;
                  n < NeighborList<NeighborListType>::numNeighbor(list,i);
                  ++n )
            {
                functor(
                    i, NeighborList<NeighborListType>::getNeighbor(list,i,n) );
            }
        };

    // Create the kokkos execution policy
    using kokkos_policy =
        Kokkos::RangePolicy<typename ExecutionPolicy::execution_space>;
    kokkos_policy k_policy( exec_policy.begin(), exec_policy.end() );

    // Execute the functor.
    Kokkos::parallel_for( str, k_policy, functor_wrapper );

    // Fence.
    Kokkos::fence();
};

//---------------------------------------------------------------------------//
/*!
  \brief Execute \c functor in parallel according to the execution \c policy
  with team parallelism over particle neighbors.

  \tparam ExecutionPolicy The execution policy type over which to execute the
  functor.

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
template<class ExecutionPolicy, class FunctorType, class NeighborListType>
inline void neighbor_parallel_for( const ExecutionPolicy& exec_policy,
                                   const FunctorType& functor,
                                   const NeighborListType& list,
                                   const TeamNeighborOpTag& tag,
                                   const std::string& str = "" )
{
    std::ignore = tag;

    // Create the kokkos execution policy
    using kokkos_policy =
        Kokkos::TeamPolicy<typename ExecutionPolicy::execution_space,
                           Kokkos::IndexType<int>,
                           Kokkos::Schedule<Kokkos::Dynamic> >;
    kokkos_policy k_policy( exec_policy.end() - exec_policy.begin(),
                            Kokkos::AUTO );

    // Create a team operator.
    auto functor_wrapper =
        KOKKOS_LAMBDA( const typename kokkos_policy::member_type& team )
        {
            auto i = team.league_rank();
            auto num_n = NeighborList<NeighborListType>::numNeighbor( list, i );
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team,num_n),
                [&]( const int n ) {
                    functor(
                        i, NeighborList<NeighborListType>::getNeighbor(list,i,n) );
                });

        };

    // Execute the functor.
    Kokkos::parallel_for( str, k_policy, functor_wrapper );

    // Fence.
    Kokkos::fence();
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_NEIGHBORLIST_HPP
