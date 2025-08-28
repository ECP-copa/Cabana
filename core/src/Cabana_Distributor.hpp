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
  \file Cabana_Distributor.hpp
  \brief Multi-node particle redistribution
*/
#ifndef CABANA_DISTRIBUTOR_HPP
#define CABANA_DISTRIBUTOR_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_CommunicationPlanBase.hpp>
#include <Cabana_Slice.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

#include <exception>
#include <vector>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \brief A communication plan for migrating data from one uniquely-owned
  decomposition to another uniquely owned decomposition.

  \tparam MemorySpace Kokkos memory space in which data for this class will be
  allocated.

  The Distributor allows data to be migrated to an entirely new
  decomposition. Only uniquely-owned decompositions are handled (i.e. each
  local element in the source rank has a single unique destination rank).

  Some nomenclature:

  Export - the data we uniquely own that we will be sending to other ranks.

  Import - the data we uniquely own that we will be receiving from other
  ranks.

  \note We can migrate data to the same rank. In this case a copy will occur
  instead of communication.

  \note To get the number of elements this rank will be receiving from
  migration in the forward communication plan, call totalNumImport() on the
  distributor. This will be needed when in-place migration is not used and a
  user must allocate their own destination data structure.

*/
template <class MemorySpace, class CommSpaceType = Mpi>
class Distributor : public CommunicationPlan<MemorySpace, CommSpaceType>
{
  public:
    //! Communication space
    using commspace_type = CommSpaceType;

    /*!
      \brief Topology and export rank constructor. Use this when you already
      know which ranks neighbor each other (i.e. every rank already knows who
      they will be sending and receiving from) as it will be more
      efficient. In this case you already know the topology of the
      point-to-point communication but not how much data to send to and
      receive from the neighbors.

      \tparam ViewType The container type for the export element ranks. This
      container type can be either a Kokkos View or a Cabana Slice.

      \param comm The MPI communicator over which the distributor is defined.

      \param element_export_ranks The destination rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique destination to which it
      will be exported. This export rank may be any one of the listed neighbor
      ranks which can include the calling rank. An export rank of -1 will
      signal that this element is *not* to be exported and will be ignored in
      the data migration. The input is expected to be a Kokkos view or Cabana
      slice in the same memory space as the distributor.

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. The elements in this list must be unique.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template <class ViewType>
    Distributor( MPI_Comm comm, const ViewType& element_export_ranks,
                 const std::vector<int>& neighbor_ranks )
        : CommunicationPlan<MemorySpace, CommSpaceType>( comm )
    {
        auto neighbor_ids = this->createWithTopology(
            Export(), element_export_ranks, neighbor_ranks );
        this->createExportSteering( neighbor_ids, element_export_ranks );
    }

    /*!
      \brief Export rank constructor. Use this when you don't know who you
      will be receiving from - only who you are sending to. This is less
      efficient than if we already knew who our neighbors were because we have
      to determine the topology of the point-to-point communication first.

      \tparam ViewType The container type for the export element ranks. This
      container type can be either a Kokkos View or a Cabana Slice.

      \param comm The MPI communicator over which the distributor is defined.

      \param element_export_ranks The destination rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique destination to which it
      will be exported. This export rank may any one of the listed neighbor
      ranks which can include the calling rank. An export rank of -1 will
      signal that this element is *not* to be exported and will be ignored in
      the data migration. The input is expected to be a Kokkos view or Cabana
      slice in the same memory space as the distributor.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template <class ViewType>
    Distributor( MPI_Comm comm, const ViewType& element_export_ranks )
        : CommunicationPlan<MemorySpace, CommSpaceType>( comm )
    {
        auto neighbor_ids =
            this->createWithoutTopology( Export(), element_export_ranks );
        this->createExportSteering( neighbor_ids, element_export_ranks );
    }
};

//---------------------------------------------------------------------------//
//! \cond Impl
template <typename>
struct is_distributor_impl : public std::false_type
{
};

template <typename MemorySpace, typename CommSpaceType>
struct is_distributor_impl<Distributor<MemorySpace, CommSpaceType>>
    : public std::true_type
{
};
//! \endcond

//! Distributor static type checker.
template <class T>
struct is_distributor
    : public is_distributor_impl<typename std::remove_cv<T>::type>::type
{
};

} // end namespace Cabana

// Include communication backends from what is enabled in CMake.
#ifdef Cabana_ENABLE_MPI
#include <impl/Cabana_Migrate_Mpi.hpp>
#endif // Enable MPI

namespace Cabana
{

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Multiple
  AoSoA version.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Distributor_t Distributor type - must be a distributor.
  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param exec_space Kokkos execution space.
  \param distributor The distributor to use for the migration.
  \param src The AoSoA containing the data to be migrated. Must have the same
  number of elements as the inputs used to construct the distributor.
  \param dst The AoSoA to which the migrated data will be written. Must be the
  same size as the number of imports given by the distributor on this
  rank. Call totalNumImport() on the distributor to get this size value.
*/
template <class ExecutionSpace, class Distributor_t, class AoSoA_t>
void migrate( ExecutionSpace exec_space, const Distributor_t& distributor,
              const AoSoA_t& src, AoSoA_t& dst,
              typename std::enable_if<( is_distributor<Distributor_t>::value &&
                                        is_aosoa<AoSoA_t>::value ),
                                      int>::type* = 0 )
{
    // Check that src and dst are the right size.
    if ( src.size() != distributor.exportSize() )
        throw std::runtime_error( "Cabana::migrate: Source is "
                                  "the wrong size for migration! (Label: " +
                                  src.label() + ")" );
    if ( dst.size() != distributor.totalNumImport() )
        throw std::runtime_error( "Cabana::migrate: Destination "
                                  "is the wrong size for migration! (Label: " +
                                  dst.label() + ")" );

    // Move the data.
    Impl::migrateData( typename Distributor_t::commspace_type(), exec_space,
                       distributor, src, dst );
}

/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Multiple AoSoA version.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \tparam Distributor_t Distributor type - must be a distributor.
  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param distributor The distributor to use for the migration.
  \param src The AoSoA containing the data to be migrated. Must have the same
  number of elements as the inputs used to construct the distributor.
  \param dst The AoSoA to which the migrated data will be written. Must be the
  same size as the number of imports given by the distributor on this
  rank. Call totalNumImport() on the distributor to get this size value.
*/
template <class Distributor_t, class AoSoA_t>
void migrate( const Distributor_t& distributor, const AoSoA_t& src,
              AoSoA_t& dst,
              typename std::enable_if<( is_distributor<Distributor_t>::value &&
                                        is_aosoa<AoSoA_t>::value ),
                                      int>::type* = 0 )
{
    migrate( typename Distributor_t::execution_space{}, distributor, src, dst );
}

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Single AoSoA version that will
  resize in-place. Note that resizing does not necessarily allocate more
  memory. The AoSoA memory will only increase if not enough has already been
  reserved/allocated for the needed number of elements.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Distributor_t Distributor type - must be a distributor.
  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param exec_space Kokkos execution space.
  \param distributor The distributor to use for the migration.
  \param aosoa The AoSoA containing the data to be migrated. Upon input, must
  have the same number of elements as the inputs used to construct the
  distributor. At output, it will be the same size as th enumber of import
  elements on this rank provided by the distributor. Before using this
  function, consider reserving enough memory in the data structure so
  reallocating is not necessary.
*/
template <class ExecutionSpace, class Distributor_t, class AoSoA_t>
void migrate( ExecutionSpace exec_space, const Distributor_t& distributor,
              AoSoA_t& aosoa,
              typename std::enable_if<( is_distributor<Distributor_t>::value &&
                                        is_aosoa<AoSoA_t>::value ),
                                      int>::type* = 0 )
{
    // Check that the AoSoA is the right size.
    if ( aosoa.size() != distributor.exportSize() )
        throw std::runtime_error(
            "Cabana::migrate (in-place): "
            "AoSoA is the wrong size for migration! (Label: " +
            aosoa.label() + ")" );

    // Determine if the source of destination decomposition has more data on
    // this rank.
    bool dst_is_bigger =
        ( distributor.totalNumImport() > distributor.exportSize() );

    // If the destination decomposition is bigger than the source
    // decomposition resize now so we have enough space to do the operation.
    if ( dst_is_bigger )
        aosoa.resize( distributor.totalNumImport() );

    // Move the data.
    Impl::migrateData( typename Distributor_t::commspace_type(), exec_space,
                       distributor, aosoa, aosoa );

    // If the destination decomposition is smaller than the source
    // decomposition resize after we have moved the data.
    if ( !dst_is_bigger )
        aosoa.resize( distributor.totalNumImport() );
}

/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Single AoSoA version that will
  resize in-place. Note that resizing does not necessarily allocate more
  memory. The AoSoA memory will only increase if not enough has already been
  reserved/allocated for the needed number of elements.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \tparam Distributor_t Distributor type - must be a distributor.
  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param distributor The distributor to use for the migration.
  \param aosoa The AoSoA containing the data to be migrated. Upon input, must
  have the same number of elements as the inputs used to construct the
  distributor. At output, it will be the same size as th enumber of import
  elements on this rank provided by the distributor. Before using this
  function, consider reserving enough memory in the data structure so
  reallocating is not necessary.
*/
template <class Distributor_t, class AoSoA_t>
void migrate( const Distributor_t& distributor, AoSoA_t& aosoa,
              typename std::enable_if<( is_distributor<Distributor_t>::value &&
                                        is_aosoa<AoSoA_t>::value ),
                                      int>::type* = 0 )
{
    migrate( typename Distributor_t::execution_space{}, distributor, aosoa );
}

/*!
  \brief Synchronously migrate data between two different decompositions using
  the Distributor forward communication plan. Slice version. The user can do
  this in-place with the same slice but they will need to manage the resizing
  themselves as we can't resize slices.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \tparam Distributor_t Distributor type - must be a distributor.
  \tparam Slice_t Slice type - must be an Slice.

  \param distributor The distributor to use for the migration.
  \param src The slice containing the data to be migrated. Must have the same
  number of elements as the inputs used to construct the distributor.
  \param dst The slice to which the migrated data will be written. Must be the
  same size as the number of imports given by the distributor on this
  rank. Call totalNumImport() on the distributor to get this size value.
*/
template <class Distributor_t, class Slice_t>
void migrate( const Distributor_t& distributor, const Slice_t& src,
              Slice_t& dst,
              typename std::enable_if<( is_distributor<Distributor_t>::value &&
                                        is_slice<Slice_t>::value ),
                                      int>::type* = 0 )
{
    // Check that src and dst are the right size.
    if ( src.size() != distributor.exportSize() )
        throw std::runtime_error( "Cabana::migrate: Source Slice is the wrong "
                                  "size for migration! (Label: " +
                                  src.label() + ")" );
    if ( dst.size() != distributor.totalNumImport() )
        throw std::runtime_error( "Cabana::migrate: Destination Slice is the "
                                  "wrong size for migration! (Label: " +
                                  dst.label() + ")" );

    Impl::migrateSlice( typename Distributor_t::commspace_type(),
                        typename Distributor_t::execution_space{}, distributor,
                        src, dst );
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_DISTRIBUTOR_HPP