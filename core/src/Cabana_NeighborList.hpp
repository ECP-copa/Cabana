/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_NeighborList.hpp
  \brief Neighbor list interface
*/
#ifndef CABANA_NEIGHBORLIST_HPP
#define CABANA_NEIGHBORLIST_HPP

#include <Kokkos_Core.hpp>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Neighbor List Interface
//---------------------------------------------------------------------------//
/*!
  \brief Tag for full neighbor lists.

  In this case every particle has its neighbors stored in the list. So, if
  particle "i" neighbors particle "j" then "j" will be in the neighbor list
  for "i" and "i" will bin the the neighbor list for "j".
*/
class FullNeighborTag
{
};

//---------------------------------------------------------------------------//
/*!
  \brief Tag for half neighbor lists.

  In this case only half of the neighbors are stored and the inverse
  relationship is implied. So, if particle "i" neighbors particle "j" then "j"
  will be in the neighbor list for "i" while the fact that "i" is a neighbor
  of "j" is implied.
*/
class HalfNeighborTag
{
};

//---------------------------------------------------------------------------//
// Neighbor List Memory Layout Tag.
//---------------------------------------------------------------------------//
//! CSR (compressed sparse row) neighbor list layout.
struct NeighborLayoutCSR
{
};

//! 2D array neighbor list layout.
struct NeighborLayout2D
{
};

//! CSR (compressed sparse row) neighbor list layout (planned to be deprecated).
struct VerletLayoutCSR
{
};

//! 2D array neighbor list layout (planned to be deprecated).
struct VerletLayout2D
{
};

//---------------------------------------------------------------------------//
// Neighbor List Data.
//---------------------------------------------------------------------------//
template <class MemorySpace, class LayoutTag>
struct NeighborListData;

//! Store the NeighborList compressed sparse row (CSR) neighbor data.
template <class MemorySpace>
struct NeighborListData<MemorySpace, NeighborLayoutCSR>
{
    //! Kokkos memory space.
    using memory_space = MemorySpace;

    //! Number of neighbors per particle.
    Kokkos::View<int*, memory_space> counts;

    //! Offsets into the neighbor list.
    Kokkos::View<int*, memory_space> offsets;

    //! Neighbor list.
    Kokkos::View<int*, memory_space> neighbors;

    //! Add a neighbor to the list.
    KOKKOS_INLINE_FUNCTION
    void addNeighbor( const int pid, const int nid ) const
    {
        neighbors( offsets( pid ) +
                   Kokkos::atomic_fetch_add( &counts( pid ), 1 ) ) = nid;
    }
};

//! Store the NeighborList 2D neighbor data.
template <class MemorySpace>
struct NeighborListData<MemorySpace, NeighborLayout2D>
{
    //! Kokkos memory space.
    using memory_space = MemorySpace;

    //! Number of neighbors per particle.
    Kokkos::View<int*, memory_space> counts;

    //! Neighbor list.
    Kokkos::View<int**, memory_space> neighbors;

    //! Add a neighbor to the list.
    KOKKOS_INLINE_FUNCTION
    void addNeighbor( const int pid, const int nid ) const
    {
        std::size_t count = Kokkos::atomic_fetch_add( &counts( pid ), 1 );
        if ( count < neighbors.extent( 1 ) )
            neighbors( pid, count ) = nid;
    }
};

template <class MemorySpace>
struct NeighborListData<MemorySpace, VerletLayoutCSR>
    : public NeighborListData<MemorySpace, NeighborLayoutCSR>
{
};

template <class MemorySpace>
struct NeighborListData<MemorySpace, VerletLayout2D>
    : public NeighborListData<MemorySpace, NeighborLayout2D>
{
};

//---------------------------------------------------------------------------//
//! Neighborhood discriminator.
template <class Tag>
class NeighborDiscriminator;

//! Full list neighborhood discriminator.
template <>
class NeighborDiscriminator<FullNeighborTag>
{
  public:
    /*!
      \brief Full neighbor lists count and store the neighbors of all particles.

      The only criteria for a potentially valid neighbor is that the particle
      does not neighbor itself (i.e. the particle index "p" is not the same as
      the neighbor index "n").
   */
    KOKKOS_INLINE_FUNCTION
    static bool isValid( const std::size_t p, const double, const double,
                         const double, const std::size_t n, const double,
                         const double, const double )
    {
        return ( p != n );
    }
};

//! Half list neighborhood discriminator.
template <>
class NeighborDiscriminator<HalfNeighborTag>
{
  public:
    /*!
      \brief Half neighbor lists only store half of the neighbors.

      This is done by eliminating duplicate pairs such that the fact that
      particle "p" neighbors particle "n" is stored in the list but "n"
      neighboring "p" is not stored but rather implied. We discriminate by only
      storing neighbors who's coordinates are greater in the x direction. If
      they are the same then the y direction is checked next and finally the z
      direction if the y coordinates are the same.
    */
    KOKKOS_INLINE_FUNCTION
    static bool isValid( const std::size_t p, const double xp, const double yp,
                         const double zp, const std::size_t n, const double xn,
                         const double yn, const double zn )
    {
        return ( ( p != n ) &&
                 ( ( xn > xp ) ||
                   ( ( xn == xp ) &&
                     ( ( yn > yp ) || ( ( yn == yp ) && ( zn > zp ) ) ) ) ) );
    }
};

//---------------------------------------------------------------------------//
/*!
  \brief Neighbor list interface. Provides an interface callable at the functor
  level that gives access to neighbor data for particles.
*/
template <class NeighborListType>
class NeighborList
{
  public:
    //! Kokkos memory space.
    using memory_space = typename NeighborListType::memory_space;

    //! Get the number of neighbors for a given particle index.
    KOKKOS_INLINE_FUNCTION
    static std::size_t numNeighbor( const NeighborListType& list,
                                    const std::size_t particle_index );

    //! Get the id for a neighbor for a given particle index and neighbor index.
    KOKKOS_INLINE_FUNCTION
    static std::size_t getNeighbor( const NeighborListType& list,
                                    const std::size_t particle_index,
                                    const std::size_t neighbor_index );
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_NEIGHBORLIST_HPP
