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
#include <Kokkos_Parallel.hpp>

#include <cstdlib>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \brief Execute a functor in parallel with a 1d execution policy.

  \tparam ExecutionSpace The execution space in which to execute the functor.

  \tparam FunctorType The functor type to execute.

  \param exec_policy The 1D range policy over which to execute the functor.

  \param functor The functor to execute in parallel. Must accept a single index.

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
  void operator() ( const int index ) const ;
  };
  \endcode

  In the above example, index is 1D index to a given AoSoA/slice element.  Its
  <tt>operator()</tt> method defines the operation to parallelize, over the
  range of indices <tt>idx=[begin,end]</tt>.
*/
template<class ExecutionSpace, class FunctorType>
inline void parallel_for( const LinearPolicy<ExecutionSpace>& exec_policy,
                          const FunctorType& functor,
                          const std::string& str = "" )
{
    using kokkos_policy = typename LinearPolicy<ExecutionSpace>::base_type;

    Kokkos::parallel_for(
        str, dynamic_cast<const kokkos_policy&>(exec_policy), functor );

    Kokkos::fence();
}

//---------------------------------------------------------------------------//
/*!
  \brief Execute a vectorized functor in parallel with a 2d execution policy.

  \tparam ExecutionSpace The execution space in which to execute the functor.

  \tparam FunctorType The functor type to execute.

  \tparam VectorLength The length of the vector over which to execute the
  vectorized code.

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
*/
template<class ExecutionSpace, class FunctorType, int VectorLength>
inline void parallel_for(
    const SimdPolicy<ExecutionSpace,VectorLength>& exec_policy,
    const FunctorType& functor,
    const std::string& str = "" )
{
    using kokkos_policy =
        typename SimdPolicy<ExecutionSpace,VectorLength>::base_type;

    Kokkos::parallel_for(
        str,
        dynamic_cast<const kokkos_policy&>(exec_policy),
        KOKKOS_LAMBDA( const typename kokkos_policy::member_type& team )
        {
            auto s = team.league_rank() + exec_policy.structBegin();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange( team,
                                           exec_policy.arrayBegin(s),
                                           exec_policy.arrayEnd(s)),
                [&]( const int a ) { functor(s,a);});
        });

    Kokkos::fence();
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_PARALLEL_HPP
