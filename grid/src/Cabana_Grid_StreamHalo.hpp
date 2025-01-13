/****************************************************************************
 * Copyright (c) 2018-2024 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Grid_StreamHalo.hpp
  \brief Stream-triggered multi-node grid scatter/gather
*/
#ifndef CABANA_GRID_STREAMHALO_HPP
#define CABANA_GRID_STREAMHALO_HPP

#include <Cabana_Grid_Halo.hpp>

namespace Cabana
{
namespace Grid
{

//---------------------------------------------------------------------------//
// StreamHalo
// ---------------------------------------------------------------------------//
/*!
  General multiple array halo communication plan for migrating shared data
  between blocks. Arrays may be defined on different entity types and have
  different data types.

  The halo operates on an arbitrary set of arrays. Each of these arrays must
  be defined on the same local grid meaning they that share the same
  communicator and halo size. The arrays must also reside in the same memory
  space. These requirements are checked at construction.
*/
template <class MemorySpace, class CommSpace=MPI>
class StreamHalo : public Halo<MemorySpace, CommSpace>
{

  public:
    //! Memory space.
    using memory_space = MemorySpace;
    using comm_space = CommSpace;

    /*!
      \brief Constructor.
      \tparam The arrays types to construct the halo for.
      \param pattern The halo pattern to use for halo communication.
      \param width Halo cell width. Must be less than or equal to the halo
      width of the block.
      \param arrays The arrays to build the halo for. These arrays must be
      provided in the same order
    */
    template <class Pattern, class... ArrayTypes>
    StreamHalo( const Pattern& pattern, const int width, const ArrayTypes&... arrays ) : Halo<memory_space, comm_space>(pattern, width, arrays...)
  {
  }

    /*!
      \brief Gather data into our ghosts from their owners.

      \param exec_space The execution space to use for pack/unpack.

      \param arrays The arrays to gather. NOTE: These arrays must be given in
      the same order as in the constructor. These could technically be
      different arrays, they just need to have the same layouts and data types
      as the input arrays.
    */
    template <class ExecutionSpace, class... ArrayTypes>
    void enqueueGather( const ExecutionSpace& exec_space,
                 const ArrayTypes&... arrays ) const requires std::is_same_v<comm_space, MPI>
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::enqueueGather" );
	exec_space.fence();
	this->gather(exec_space, arrays...);
    }

    /*!
      \brief Scatter data from our ghosts to their owners using the given type
      of reduce operation.
      \param reduce_op The functor used to reduce the results.
      \param exec_space The execution space to use for pack/unpack.
      \param arrays The arrays to scatter.
    */
    template <class ExecutionSpace, class ReduceOp, class... ArrayTypes>
    void enqueueScatter( const ExecutionSpace& exec_space, const ReduceOp& reduce_op,
                  const ArrayTypes&... arrays ) const requires std::is_same_v<comm_space, MPI>
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::enqueueScatter" );
	exec_space.fence();
	this->scatter(exec_space, reduce_op, arrays...);
    }
};

/*!
  \brief Stream Halo creation function.
  \param TODO add communication space argument
  \param pattern The pattern to build the halo from.
  \param width Must be less than or equal to the width of the array halo.
  \param arrays The arrays over which to build the halo.
  \return Shared pointer to a Halo.
*/
  template <class Pattern, class... ArrayTypes>
auto createStreamHalo( const Pattern& pattern, const int width,
		       const ArrayTypes&... arrays )
{
    using memory_space = typename ArrayPackMemorySpace<ArrayTypes...>::type;
    return std::make_shared<StreamHalo<memory_space>>( pattern, width, arrays... );
}

  
} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_STREAMHALO_HPP
