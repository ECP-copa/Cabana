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
    KOKKOS_INLINE_FUNCTION
    static int numNeighbor( const NeighborListType& list,
                            const std::size_t particle_index );

    // Get the id for a neighbor for a given particle index and the index of
    // the neighbor relative to the particle.
    KOKKOS_INLINE_FUNCTION
    static int getNeighbor( const NeighborListType& list,
                            const std::size_t particle_index,
                            const int neighbor_index );
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_NEIGHBORLIST_HPP
