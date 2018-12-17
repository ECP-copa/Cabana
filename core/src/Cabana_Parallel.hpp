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

#ifndef CABANA_PARALLEL_HPP
#define CABANA_PARALLEL_HPP

#include <Cabana_ExecutionPolicy.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
namespace Impl
{

// No work tag was provided so call without a tag argument.
template<class WorkTag, class FunctorType>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<std::is_same<WorkTag,void>::value>::type
simdFunctorDispatch( const FunctorType& functor,
                     const std::size_t s,
                     const int a )
{
    functor(s,a);
}

// The user gave us a tag so call the version using that.
template<class WorkTag, class FunctorType>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<!std::is_same<WorkTag,void>::value>::type
simdFunctorDispatch( const FunctorType& functor,
                     const std::size_t s,
                     const int a )
{
    const WorkTag t{};
    functor(t,s,a);
}

}

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
    using work_tag =
        typename SimdPolicy<VectorLength,ExecParameters...>::work_tag;

    using team_policy =
        typename SimdPolicy<VectorLength,ExecParameters...>::base_type;

   Kokkos::parallel_for(
        str,
        dynamic_cast<const team_policy&>(exec_policy),
        KOKKOS_LAMBDA( const typename team_policy::member_type& team )
        {
            auto s = team.league_rank() + exec_policy.structBegin();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange( team,
                                           exec_policy.arrayBegin(s),
                                           exec_policy.arrayEnd(s)),
                [&]( const int a )
                { Impl::simdFunctorDispatch<work_tag>(functor,s,a);});
        });

    Kokkos::fence();
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_PARALLEL_HPP
