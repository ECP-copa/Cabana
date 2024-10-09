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
  \file Cabana_NeighborList.hpp
  \brief Neighbor list interface
*/
#ifndef CABANA_NEIGHBORLIST_HPP
#define CABANA_NEIGHBORLIST_HPP

#include <Kokkos_Core.hpp>

#include <Cabana_Sort.hpp>

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
//! Neighborhood discriminator.
template <class Tag>
class NeighborDiscriminator;

//! Full list discriminator specialization.
template <>
class NeighborDiscriminator<FullNeighborTag>
{
  public:
    /*!
      \brief Check whether neighbor pair is valid.

      Full neighbor lists count and store the neighbors of all particles. The
      only criteria for a potentially valid neighbor is that the particle does
      not neighbor itself (i.e. the particle index "p" is not the same as the
      neighbor index "n").
    */
    template <std::size_t NumSpaceDim>
    KOKKOS_INLINE_FUNCTION static bool
    isValid( const std::size_t p, const Kokkos::Array<double, NumSpaceDim>,
             const std::size_t n, const Kokkos::Array<double, NumSpaceDim> )
    {
        return ( p != n );
    }

    /*!
      \brief Check whether neighbor pair is valid.

      Full neighbor lists count and store the neighbors of all particles. The
      only criteria for a potentially valid neighbor is that the particle does
      not neighbor itself (i.e. the particle index "p" is not the same as the
      neighbor index "n").
    */
    KOKKOS_INLINE_FUNCTION
    static bool isValid( const std::size_t p, const double, const double,
                         const double, const std::size_t n, const double,
                         const double, const double )
    {
        return ( p != n );
    }
};

//! Half list discriminator specialization.
template <>
class NeighborDiscriminator<HalfNeighborTag>
{
  public:
    /*!
      \brief Check whether neighbor pair is valid.

      Half neighbor lists only store half of the neighbors be eliminating
      duplicate pairs such that the fact that particle "p" neighbors particle
      "n" is stored in the list but "n" neighboring "p" is not stored but rather
      implied. We discriminate by only storing neighbors whose coordinates are
      greater in the x direction. If they are the same then the y direction is
      checked next and finally the z direction if the y coordinates are the
      same.
    */
    template <std::size_t NumSpaceDim>
    KOKKOS_INLINE_FUNCTION static bool
    isValid( const std::size_t p, const Kokkos::Array<double, NumSpaceDim> xp,
             const std::size_t n, const Kokkos::Array<double, NumSpaceDim> xn )
    {
        return ( ( p != n ) &&
                 ( ( xn[0] > xp[0] ) ||
                   ( ( xn[0] == xp[0] ) &&
                     ( ( xn[1] > xp[1] ) ||
                       ( ( xn[1] == xp[1] ) && ( xn[2] > xp[2] ) ) ) ) ) );
    }

    /*!
      \brief Check whether neighbor pair is valid.

      Half neighbor lists only store half of the neighbors be eliminating
      duplicate pairs such that the fact that particle "p" neighbors particle
      "n" is stored in the list but "n" neighboring "p" is not stored but rather
      implied. We discriminate by only storing neighbors whose coordinates are
      greater in the x direction. If they are the same then the y direction is
      checked next and finally the z direction if the y coordinates are the
      same.
    */
    KOKKOS_INLINE_FUNCTION static bool
    isValid( const std::size_t p, const double xp, const double yp,
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
  \brief Neighbor list interface. Provides an interface callable at the
  functor level that gives access to neighbor data for particles.
*/
template <class NeighborListType>
class NeighborList
{
  public:
    //! Kokkos memory space.
    using memory_space = typename NeighborListType::memory_space;

    //! Get the total number of neighbors across all particles.
    KOKKOS_INLINE_FUNCTION
    static std::size_t totalNeighbor( const NeighborListType& list );

    //! Get the maximum number of neighbors across all particles.
    KOKKOS_INLINE_FUNCTION
    static std::size_t maxNeighbor( const NeighborListType& list );

    //! Get the number of neighbors for a given particle index.
    KOKKOS_INLINE_FUNCTION
    static std::size_t numNeighbor( const NeighborListType& list,
                                    const std::size_t particle_index );

    //! Get the id for a neighbor for a given particle index and neighbor index.
    KOKKOS_INLINE_FUNCTION
    static std::size_t getNeighbor( const NeighborListType& list,
                                    const std::size_t particle_index,
                                    const std::size_t neighbor_index );

    //! Set the id for a neighbor for a given particle index and neighbor index.
    KOKKOS_INLINE_FUNCTION
    std::size_t setNeighbor( NeighborListType& list,
                             const std::size_t particle_index,
                             const std::size_t neighbor_index );
};

//---------------------------------------------------------------------------//

namespace Impl
{
//! Iterate to get the total number of neighbors.
template <class ListType>
KOKKOS_INLINE_FUNCTION std::size_t
totalNeighbor( const ListType& list, const std::size_t num_particles )
{
    std::size_t total_n = 0;
    // Sum neighbors across all particles.
    for ( std::size_t p = 0; p < num_particles; p++ )
        total_n += NeighborList<ListType>::numNeighbor( list, p );
    return total_n;
}

//! Iterate to find the maximum number of neighbors.
template <class ListType>
KOKKOS_INLINE_FUNCTION std::size_t
maxNeighbor( const ListType& list, const std::size_t num_particles )
{
    std::size_t max_n = 0;
    for ( std::size_t p = 0; p < num_particles; p++ )
        if ( NeighborList<ListType>::numNeighbor( list, p ) > max_n )
            max_n = NeighborList<ListType>::numNeighbor( list, p );
    return max_n;
}
} // namespace Impl

/*!
  \brief Construct a histogram of neighbors per particle.
  \param exec_space Kokkos execution space.
  \param num_particles Number of particles.
  \param list Neighbor list with valid NeighborList interface.
  \param num_bin Number of bins for the histogram.
  \return Neighbor list histogram View.
*/
template <class ExecutionSpace, class ListType>
Kokkos::View<int* [2], typename ListType::memory_space>
neighborHistogram( ExecutionSpace exec_space, const std::size_t num_particles,
                   const ListType& list, const int num_bin )
{
    // Allocate View of neighbors per particle
    auto num_neigh = Kokkos::View<int*, typename ListType::memory_space>(
        "particle_count", num_particles );

    // Extract from neighbor list interface.
    auto extract_functor = KOKKOS_LAMBDA( const int p )
    {
        num_neigh( p ) = NeighborList<ListType>::numNeighbor( list, p );
    };
    Kokkos::RangePolicy<ExecutionSpace> particle_policy( exec_space, 0,
                                                         num_particles );
    Kokkos::parallel_for( particle_policy, extract_functor );
    Kokkos::fence();

    auto bin_data = Cabana::binByKey( num_neigh, num_bin );

    auto histogram = Kokkos::View<int* [2], typename ListType::memory_space>(
        "particle_count", num_bin, 2 );
    auto histogram_functor = KOKKOS_LAMBDA( const int b )
    {
        int max_neigh = NeighborList<ListType>::maxNeighbor( list );
        double bin_width =
            static_cast<double>( max_neigh ) / static_cast<double>( num_bin );
        if ( num_bin > max_neigh )
            bin_width = 1;
        // Wait to cast back to int to get the actual bin edge.
        histogram( b, 0 ) = static_cast<int>( ( b + 1 ) * bin_width );
        histogram( b, 1 ) = bin_data.binSize( b );
    };
    Kokkos::RangePolicy<ExecutionSpace> bin_policy( exec_space, 0, num_bin );
    Kokkos::parallel_for( bin_policy, histogram_functor );
    Kokkos::fence();

    return histogram;
}

} // end namespace Cabana

#endif // end CABANA_NEIGHBORLIST_HPP
