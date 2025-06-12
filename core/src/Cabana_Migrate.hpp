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
  \file Cabana_Migrate.hpp
  \brief Multi-node particle redistribution
*/
#ifndef CABANA_MIGRATE_HPP
#define CABANA_MIGRATE_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_CommunicationPlan.hpp>
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
  Data is not duplicated - after calling Cabana::migrate
  with a Distributor, the data will exist only on the rank it was exported to;
  it is removed from the rank that called Cabana::migrate. (Unless the data is
  exported to itself)

  Some nomenclature:

  Export - the data we uniquely own that we will be sending to other ranks.

  Import - the data we uniquely own that we will be receiving from other
  ranks.

  \note We can migrate data to the same rank. In this case a copy will occur
  instead of communication.

  \note To get the number of elements this rank will be receiving from
  distribution in the forward communication plan, call totalNumImport() on the
  distributor. This will be needed when in-place distribution is not used and a
  user must allocate their own destination data structure.

*/
template <class MemorySpace>
class Distributor : public CommunicationPlan<MemorySpace>
{
  public:
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
      the data distribution. The input is expected to be a Kokkos view or Cabana
      slice in the same memory space as the distributor.

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. The elements in this list must be unique.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data distribution. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template <class ViewType>
    Distributor( MPI_Comm comm, const ViewType& element_export_ranks,
                 const std::vector<int>& neighbor_ranks )
        : CommunicationPlan<MemorySpace>( comm )
    {
        auto neighbor_ids = this->createFromExportsAndTopology(
            element_export_ranks, neighbor_ranks );
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
      the data distribution. The input is expected to be a Kokkos view or Cabana
      slice in the same memory space as the distributor.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data distribution. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template <class ViewType>
    Distributor( MPI_Comm comm, const ViewType& element_export_ranks )
        : CommunicationPlan<MemorySpace>( comm )
    {
        auto neighbor_ids = this->createFromExportsOnly( element_export_ranks );
        this->createExportSteering( neighbor_ids, element_export_ranks );
    }
};

//---------------------------------------------------------------------------//
//! \cond Impl
template <typename>
struct is_distributor_impl : public std::false_type
{
};

template <typename MemorySpace>
struct is_distributor_impl<Distributor<MemorySpace>> : public std::true_type
{
};
//! \endcond

//! Distributor static type checker.
template <class T>
struct is_distributor
    : public is_distributor_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
/*!
  \brief A communication plan for importing data from one
  decomposition to another decomposition. Imported data is duplicated from its
  source decomposition.

  \tparam MemorySpace Kokkos memory space in which data for this class will be
  allocated.

  The Collector allows data to be imported from remote ranks. Data is duplicated
  - after calling Cabana::migrate with a Collector, the imported data will exist
  on both the remote rank and the rank that called migrate.

  Some nomenclature:

  Import - the data we will be receiving from other ranks.

  Export - the data we uniquely own that we will be sending to other ranks.

  \note We can import data to the same rank. In this case a copy will occur
  instead of communication.

  \note To get the number of elements this rank will be exporting
  in the forward communication plan, call totalNumExport() on the
  collector.
*/
template <class MemorySpace>
class Collector : public CommunicationPlan<MemorySpace>
{
  public:
    /*!
      \brief Topology and export rank constructor. Use this when you already
      know which ranks neighbor each other (i.e. every rank already knows who
      they will be sending and receiving from) as it will be more
      efficient. In this case you already know the topology of the
      point-to-point communication but not how much data to send to and
      receive from the neighbors.

      \tparam ViewType The container type for the export element ranks. This
      container type can be either a Kokkos View or a Cabana Slice.

      \param comm The MPI communicator over which the Collector is defined.

      \param num_owned The number of owned elements on this rank. Used to check
      that AoSoAs are sized properly for importing. For importing in-place, the
      AoSoA size must be num_owned+num_imported.

      \param element_import_ranks The source rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique source from which it
      will be imported. This import rank may be any one of the listed neighbor
      ranks which can include the calling rank. The input is expected to be a
      Kokkos view or Cabana slice in the same memory space as the Collector.

      \param element_import_ids The id that will be imported from each rank
      in element_import_ranks. element_import_ids(i) maps to
      element_import_ranks(i).

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. The elements in this list must be unique.

      \note Unlike for the Distributor, a Collector does not support import
      ranks or ids of -1.
    */
    template <class ViewType>
    Collector( MPI_Comm comm, const std::size_t num_owned,
               const ViewType& element_import_ranks,
               const ViewType& element_import_ids,
               const std::vector<int>& neighbor_ranks )
        : CommunicationPlan<MemorySpace>( comm )
        , _num_owned( num_owned )
    {
        auto neighbor_ids_ranks_indices = this->createFromImportsAndTopology(
            element_import_ranks, element_import_ids, neighbor_ranks );
        this->createExportSteering( std::get<0>( neighbor_ids_ranks_indices ),
                                    std::get<1>( neighbor_ids_ranks_indices ),
                                    std::get<2>( neighbor_ids_ranks_indices ) );
    }

    /*!
      \brief Import rank constructor. Use this when you don't know who you
      will be sending to - only who you are receiving from. This is less
      efficient than if we already knew who our neighbors were because we have
      to determine the topology of the point-to-point communication first.

      \tparam ViewType The container type for the export element ranks. This
      container type can be either a Kokkos View or a Cabana Slice.

      \param comm The MPI communicator over which the collector is defined.

      \param num_owned The number of owned elements on this rank. Used to check
      that AoSoAs are sized properly for importing. For importing in-place, the
      AoSoA size must be num_owned+num_imported.

      \param element_import_ranks The source rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique source from which it
      will be imported. This import rank may be any one of the listed neighbor
      ranks which can include the calling rank. The input is expected to be a
      Kokkos view or Cabana slice in the same memory space as the Collector.

      \param element_import_ids The id that will be imported from each rank
      in element_import_ranks. element_import_ids(i) maps to
      element_import_ranks(i).

      \note Unlike for the Distributor, a Collector does not support import
      ranks or ids of -1.
    */
    template <class ViewType>
    Collector( MPI_Comm comm, const std::size_t num_owned,
               const ViewType& element_import_ranks,
               const ViewType& element_import_ids )
        : CommunicationPlan<MemorySpace>( comm )
        , _num_owned( num_owned )
    {
        auto neighbor_ids_ranks_indices = this->createFromImportsOnly(
            element_import_ranks, element_import_ids );
        this->createExportSteering( std::get<0>( neighbor_ids_ranks_indices ),
                                    std::get<1>( neighbor_ids_ranks_indices ),
                                    std::get<2>( neighbor_ids_ranks_indices ) );
    }

    /*!
      \brief Get the total number of owned elements on this rank
      \return The total number of owned elements on this rank.
    */
    std::size_t numOwned() const { return _num_owned; }

  private:
    std::size_t _num_owned;
};

//---------------------------------------------------------------------------//
//! \cond Impl
template <typename>
struct is_collector_impl : public std::false_type
{
};

template <typename MemorySpace>
struct is_collector_impl<Collector<MemorySpace>> : public std::true_type
{
};
//! \endcond

//! Collector static type checker.
template <class T>
struct is_collector
    : public is_collector_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
namespace Impl
{
//! \cond Impl
//---------------------------------------------------------------------------//
// Synchronously move data between a source and destination AoSoA by executing
// the forward communication plan.
template <class ExecutionSpace, class Migrator_t, class AoSoA_t>
void migrateData(
    ExecutionSpace, const Migrator_t& migrator, const AoSoA_t& src,
    AoSoA_t& dst,
    typename std::enable_if<( ( is_distributor<Migrator_t>::value ||
                                is_collector<Migrator_t>::value ) &&
                              is_aosoa<AoSoA_t>::value ),
                            int>::type* = 0 )
{
    Kokkos::Profiling::ScopedRegion region( "Cabana::migrate" );

    static_assert(
        is_accessible_from<typename Migrator_t::memory_space, ExecutionSpace>{},
        "" );

    int offset = 0;
    if constexpr ( is_collector<Migrator_t>::value )
    {
        // If src and dest are the same, the Collector places
        // collected data at the end of the AoSoA instead of
        // overwriting the existing data like a Distributor.
        if ( src.data() == dst.data() )
            offset = migrator.numOwned();
    }

    // Get the MPI rank we are currently on.
    int my_rank = -1;
    MPI_Comm_rank( migrator.comm(), &my_rank );

    // Get the number of neighbors.
    int num_n = migrator.numNeighbor();

    // Calculate the number of elements that are staying on this rank and
    // therefore can be directly copied. If any of the neighbor ranks are this
    // rank it will be stored in first position (i.e. the first neighbor in
    // the local list is always yourself if you are sending to yourself).
    std::size_t num_stay =
        ( num_n > 0 && migrator.neighborRank( 0 ) == my_rank )
            ? migrator.numExport( 0 )
            : 0;

    // Allocate a send buffer.
    std::size_t num_send = migrator.totalNumExport() - num_stay;
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Migrator_t::memory_space>
        send_buffer(
            Kokkos::ViewAllocateWithoutInitializing( "migrator_send_buffer" ),
            num_send );

    // Allocate a receive buffer.
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Migrator_t::memory_space>
        recv_buffer(
            Kokkos::ViewAllocateWithoutInitializing( "migrator_recv_buffer" ),
            migrator.totalNumImport() );

    // Get the steering vector for the sends.
    auto steering = migrator.getExportSteering();

    // Gather the exports from the source AoSoA into the tuple-contiguous send
    // buffer or the receive buffer if the data is staying. We know that the
    // steering vector is ordered such that the data staying on this rank
    // comes first.
    auto build_send_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        auto tpl = src.getTuple( steering( i ) );
        if ( i < num_stay )
            recv_buffer( i ) = tpl;
        else
            send_buffer( i - num_stay ) = tpl;
    };
    Kokkos::RangePolicy<ExecutionSpace> build_send_buffer_policy(
        0, migrator.totalNumExport() );
    Kokkos::parallel_for( "Cabana::Impl::migrateData::build_send_buffer",
                          build_send_buffer_policy, build_send_buffer_func );
    Kokkos::fence();

    // The migrator has its own communication space so choose any tag.
    const int mpi_tag = 1234;

    // Post non-blocking receives.
    std::vector<MPI_Request> requests;
    requests.reserve( num_n );
    std::pair<std::size_t, std::size_t> recv_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_range.second = recv_range.first + migrator.numImport( n );

        if ( ( migrator.numImport( n ) > 0 ) &&
             ( migrator.neighborRank( n ) != my_rank ) )
        {
            auto recv_subview = Kokkos::subview( recv_buffer, recv_range );

            requests.push_back( MPI_Request() );

            MPI_Irecv( recv_subview.data(),
                       recv_subview.size() *
                           sizeof( typename AoSoA_t::tuple_type ),
                       MPI_BYTE, migrator.neighborRank( n ), mpi_tag,
                       migrator.comm(), &( requests.back() ) );
        }

        recv_range.first = recv_range.second;
    }

    // Do blocking sends.
    std::pair<std::size_t, std::size_t> send_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        if ( ( migrator.numExport( n ) > 0 ) &&
             ( migrator.neighborRank( n ) != my_rank ) )
        {
            send_range.second = send_range.first + migrator.numExport( n );

            auto send_subview = Kokkos::subview( send_buffer, send_range );

            MPI_Send( send_subview.data(),
                      send_subview.size() *
                          sizeof( typename AoSoA_t::tuple_type ),
                      MPI_BYTE, migrator.neighborRank( n ), mpi_tag,
                      migrator.comm() );

            send_range.first = send_range.second;
        }
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( requests.size() );
    const int ec =
        MPI_Waitall( requests.size(), requests.data(), status.data() );
    if ( MPI_SUCCESS != ec )
        throw std::logic_error(
            "Cabana::Impl::migrateData::Failed MPI Communication" );

    // Extract the receive buffer into the destination AoSoA.
    auto extract_recv_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        dst.setTuple( offset + i, recv_buffer( i ) );
    };
    Kokkos::RangePolicy<ExecutionSpace> extract_recv_buffer_policy(
        0, migrator.totalNumImport() );
    Kokkos::parallel_for( "Cabana::Impl::migrateData::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( migrator.comm() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously migrate data between two different decompositions using
  the migrator forward communication plan. Slice version. The user can do
  this in-place with the same slice but they will need to manage the resizing
  themselves as we can't resize slices.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Migrator_t - Migrator type - must be a Distributor or a Collector.
  \tparam Slice_t Slice type - must be a Slice.

  \param migrator The migrator to use for the migration.
  \param src The slice containing the data to be migrated. Must have the same
  number of elements as the inputs used to construct the migrator.
  \param dst The slice to which the migrated data will be written. Must be the
  same size as the number of imports given by the migrator on this
  rank. Call totalNumImport() on the migrator to get this size value.
*/
template <class ExecutionSpace, class Migrator_t, class Slice_t>
void migrateSlice(
    ExecutionSpace, const Migrator_t& migrator, const Slice_t& src,
    Slice_t& dst,
    typename std::enable_if<( ( is_distributor<Migrator_t>::value ||
                                is_collector<Migrator_t>::value ) &&
                              is_slice<Slice_t>::value ),
                            int>::type* = 0 )
{
    // Check that dst is the right size.
    if ( dst.size() != migrator.totalNumImport() )
        throw std::runtime_error( "Cabana::Impl::migrateSlice: Destination is "
                                  "the wrong size for migration!" );

    // Get the number of components in the slices.
    size_t num_comp = 1;
    for ( size_t d = 2; d < src.viewRank(); ++d )
        num_comp *= src.extent( d );

    // Get the raw slice data.
    auto src_data = src.data();
    auto dst_data = dst.data();

    // Get the MPI rank we are currently on.
    int my_rank = -1;
    MPI_Comm_rank( migrator.comm(), &my_rank );

    // Get the number of neighbors.
    int num_n = migrator.numNeighbor();

    // Calculate the number of elements that are staying on this rank and
    // therefore can be directly copied. If any of the neighbor ranks are this
    // rank it will be stored in first position (i.e. the first neighbor in
    // the local list is always yourself if you are sending to yourself).
    std::size_t num_stay =
        ( num_n > 0 && migrator.neighborRank( 0 ) == my_rank )
            ? migrator.numExport( 0 )
            : 0;

    // Allocate a send buffer. Note this one is layout right so the components
    // of each element are consecutive in memory.
    std::size_t num_send = migrator.totalNumExport() - num_stay;
    Kokkos::View<typename Slice_t::value_type**, Kokkos::LayoutRight,
                 typename Migrator_t::memory_space>
        send_buffer(
            Kokkos::ViewAllocateWithoutInitializing( "migrator_send_buffer" ),
            num_send, num_comp );

    // Allocate a receive buffer. Note this one is layout right so the
    // components of each element are consecutive in memory.
    Kokkos::View<typename Slice_t::value_type**, Kokkos::LayoutRight,
                 typename Migrator_t::memory_space>
        recv_buffer(
            Kokkos::ViewAllocateWithoutInitializing( "migrator_recv_buffer" ),
            migrator.totalNumImport(), num_comp );

    // Get the steering vector for the sends.
    auto steering = migrator.getExportSteering();

    // Gather from the source Slice into the contiguous send buffer or,
    // if it is part of the local copy, put it directly in the destination
    // Slice.
    auto build_send_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        auto s_src = Slice_t::index_type::s( steering( i ) );
        auto a_src = Slice_t::index_type::a( steering( i ) );
        std::size_t src_offset = s_src * src.stride( 0 ) + a_src;
        if ( i < num_stay )
            for ( std::size_t n = 0; n < num_comp; ++n )
                recv_buffer( i, n ) =
                    src_data[src_offset + n * Slice_t::vector_length];
        else
            for ( std::size_t n = 0; n < num_comp; ++n )
                send_buffer( i - num_stay, n ) =
                    src_data[src_offset + n * Slice_t::vector_length];
    };
    Kokkos::RangePolicy<ExecutionSpace> build_send_buffer_policy(
        0, migrator.totalNumExport() );
    Kokkos::parallel_for( "Cabana::Impl::migrateSlice::build_send_buffer",
                          build_send_buffer_policy, build_send_buffer_func );
    Kokkos::fence();

    // The migrator has its own communication space so choose any tag.
    const int mpi_tag = 1234;

    // Post non-blocking receives.
    std::vector<MPI_Request> requests;
    requests.reserve( num_n );
    std::pair<std::size_t, std::size_t> recv_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_range.second = recv_range.first + migrator.numImport( n );

        if ( ( migrator.numImport( n ) > 0 ) &&
             ( migrator.neighborRank( n ) != my_rank ) )
        {
            auto recv_subview =
                Kokkos::subview( recv_buffer, recv_range, Kokkos::ALL );

            requests.push_back( MPI_Request() );

            MPI_Irecv( recv_subview.data(),
                       recv_subview.size() *
                           sizeof( typename Slice_t::value_type ),
                       MPI_BYTE, migrator.neighborRank( n ), mpi_tag,
                       migrator.comm(), &( requests.back() ) );
        }

        recv_range.first = recv_range.second;
    }

    // Do blocking sends.
    std::pair<std::size_t, std::size_t> send_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        if ( ( migrator.numExport( n ) > 0 ) &&
             ( migrator.neighborRank( n ) != my_rank ) )
        {
            send_range.second = send_range.first + migrator.numExport( n );

            auto send_subview =
                Kokkos::subview( send_buffer, send_range, Kokkos::ALL );

            MPI_Send( send_subview.data(),
                      send_subview.size() *
                          sizeof( typename Slice_t::value_type ),
                      MPI_BYTE, migrator.neighborRank( n ), mpi_tag,
                      migrator.comm() );

            send_range.first = send_range.second;
        }
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( requests.size() );
    const int ec =
        MPI_Waitall( requests.size(), requests.data(), status.data() );
    if ( MPI_SUCCESS != ec )
        throw std::logic_error(
            "Cabana::Impl::migrateSlice: Failed MPI Communication" );

    // Extract the data from the receive buffer into the destination Slice.
    auto extract_recv_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        auto s = Slice_t::index_type::s( i );
        auto a = Slice_t::index_type::a( i );
        std::size_t dst_offset = s * dst.stride( 0 ) + a;
        for ( std::size_t n = 0; n < num_comp; ++n )
            dst_data[dst_offset + n * Slice_t::vector_length] =
                recv_buffer( i, n );
    };
    Kokkos::RangePolicy<ExecutionSpace> extract_recv_buffer_policy(
        0, migrator.totalNumImport() );
    Kokkos::parallel_for( "Cabana::Impl::migrateSlice::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( migrator.comm() );
}

//---------------------------------------------------------------------------//
//! \endcond
} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Distributor version with multiple
  AoSoAs.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Distributor_t Distributor type - must be a distributor.
  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param exec_space Kokkos execution space.
  \param distributor The distributor to use for the distribution.
  \param src The AoSoA containing the data to be migrated. Must have the same
  number of elements as the inputs used to construct the distributor.
  \param dst The AoSoA to which the migrated data will be written. Must be the
  same size as the number of imports given by the distributor on this
  rank. Call totalNumImport() on the distributor to get this size value.
*/
template <class ExecutionSpace, class MemorySpace, class AoSoA_t>
void migrate(
    ExecutionSpace exec_space, const Distributor<MemorySpace>& distributor,
    const AoSoA_t& src, AoSoA_t& dst,
    typename std::enable_if<( is_distributor<Distributor<MemorySpace>>::value &&
                              is_aosoa<AoSoA_t>::value ),
                            int>::type* = 0 )
{
    // Check that src and dst are the right size.
    if ( src.size() != distributor.exportSize() )
        throw std::runtime_error( "Cabana::migrate (Distributor): Source is "
                                  "the wrong size for distribution!" );
    if ( dst.size() != distributor.totalNumImport() )
        throw std::runtime_error( "Cabana::migrate (Distributor): Destination "
                                  "is the wrong size for distribution!" );

    // Move the data.
    Impl::migrateData( exec_space, distributor, src, dst );
}

/*!
  \brief Synchronously migrate data between two different decompositions using
  the collector forward communication plan. Collector version with multiple
  AoSoAs.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Collector_t Collector type - must be a collector.
  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param exec_space Kokkos execution space.
  \param collector The collector to use for the collection.
  \param src The AoSoA containing the data other ranks will be asking to import.
  Indices in src must match with the import_ids passed by other ranks into the
  Collector.
  \param dst The AoSoA to which the imported data will be written.
  Must be the same size as the number of imports given by the collector on this
  rank. Call totalNumImport() on the collector to get this size value.
*/
template <class ExecutionSpace, class MemorySpace, class AoSoA_t>
void migrate(
    ExecutionSpace exec_space, const Collector<MemorySpace>& collector,
    const AoSoA_t& src, AoSoA_t& dst,
    typename std::enable_if<( is_aosoa<AoSoA_t>::value ), int>::type* = 0 )
{
    // Check that src and dst are the right size.
    if ( src.size() != collector.numOwned() )
        throw std::runtime_error( "Cabana::migrate (Collector): Source is the "
                                  "wrong size for collection!" );
    if ( dst.size() != collector.totalNumImport() )
        throw std::runtime_error( "Cabana::migrate (Collector): Destination is "
                                  "the wrong size for collection!" );

    // Move the data.
    Impl::migrateData( exec_space, collector, src, dst );
}

/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Multiple AoSoA version.

  For Distributors, migrate moves all data to a new distribution that is
  uniquely owned - each element will only have a single destination rank.

  For Collectors, duplicates data on the sending and receiving decompostions.

  \tparam Migrator_t Migrator type - must be a Distributor or a Collector.
  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param migrator The migrator to use for the migration. Either a Distributor or
  a Collector.
  \param src The AoSoA containing the data to be migrated. Must
  have the same number of elements as the inputs used to construct the
  distributor.
  \param dst The AoSoA to which the migrated data will be written.
  Must be the same size as the number of imports given by the distributor on
  this rank. Call totalNumImport() on the distributor to get this size value.
*/
template <class Migrator_t, class AoSoA_t>
void migrate( const Migrator_t& migrator, const AoSoA_t& src, AoSoA_t& dst,
              typename std::enable_if<( ( is_distributor<Migrator_t>::value ||
                                          is_collector<Migrator_t>::value ) &&
                                        is_aosoa<AoSoA_t>::value ),
                                      int>::type* = 0 )
{
    migrate( typename Migrator_t::execution_space{}, migrator, src, dst );
}

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously migrate data between two different decompositions using
  the forward communication plan. Distributor version. Single AoSoA version that
  will resize in-place. Note that resizing does not necessarily allocate more
  memory. The AoSoA memory will only increase if not enough has already been
  reserved/allocated for the needed number of elements.

  Migrate with a Distributor moves all data to a new distribution that is
  uniquely owned - each element will only have a single destination rank.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Distributor_t Distributor type - must be a distributor.
  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param exec_space Kokkos execution space.
  \param distributor The distributor to use for the migration.
  \param aosoa The AoSoA containing the data to be migrated. Upon input, must
  have the same number of elements as the inputs used to construct the
  distributor. At output, it will be the same size as the number of import
  elements on this rank provided by the distributor. Before using this
  function, consider reserving enough memory in the data structure so
  reallocating is not necessary.
*/
template <class ExecutionSpace, class MemorySpace, class AoSoA_t>
void migrate(
    ExecutionSpace exec_space, const Distributor<MemorySpace>& distributor,
    AoSoA_t& aosoa,
    typename std::enable_if<( is_aosoa<AoSoA_t>::value ), int>::type* = 0 )
{
    // Check that the AoSoA is the right size.
    if ( aosoa.size() != distributor.exportSize() )
        throw std::runtime_error( "Cabana::migrate (Distributor): AoSoA is the "
                                  "wrong size for distribution!" );

    // Determine if the source of destination decomposition has more data on
    // this rank.
    bool dst_is_bigger =
        ( distributor.totalNumImport() > distributor.exportSize() );

    // If the destination decomposition is bigger than the source
    // decomposition resize now so we have enough space to do the operation.
    if ( dst_is_bigger )
        aosoa.resize( distributor.totalNumImport() );

    // Move the data.
    Impl::migrateData( exec_space, distributor, aosoa, aosoa );

    // If the destination decomposition is smaller than the source
    // decomposition resize after we have moved the data.
    if ( !dst_is_bigger )
        aosoa.resize( distributor.totalNumImport() );
}

/*!
  \brief Synchronously collect data to different decompositions using
  the forward communication plan. Collector version.

  Migrate with a Collector duplicates data on the sending and receiving
  decompostions.

  \tparam ExecutionSpace Kokkos execution space.
  \tparam Collector_t Collector type - must be a Collector.
  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param exec_space Kokkos execution space.
  \param collector The Collector to use for the collection.
  \param aosoa The AoSoA containing the data other ranks will be asking to
  import. Indices in src must match with the import_ids passed by other ranks
  into the Collector. Must be the same size as the number of
  owned plus imported elements on this rank provided by the collector.
*/
template <class ExecutionSpace, class MemorySpace, class AoSoA_t>
void migrate(
    ExecutionSpace exec_space, const Collector<MemorySpace>& collector,
    AoSoA_t& aosoa,
    typename std::enable_if<( is_aosoa<AoSoA_t>::value ), int>::type* = 0 )
{
    // Check if the aosoa is large enough
    if ( aosoa.size() != ( collector.numOwned() + collector.totalNumImport() ) )
        throw std::runtime_error( "Cabana::migrate (Collector, in-place) "
                                  "Source is the wrong size for collection!" );

    // Move the data.
    Impl::migrateData( exec_space, collector, aosoa, aosoa );
}

/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Single AoSoA version.

  \tparam Migrator_t Migrator type - must be a Distributor or a Collector.
  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param migrator The migrator to use for the migration. Either a Distributor or
  a Collector.
  \param aosoa The AoSoA containing the data to be migrated.
  Behavior varies depending on whether Migrator_t is a Collector or a
  Distributor. See their respective migrate function details.
*/
template <class Migrator_t, class AoSoA_t>
void migrate( const Migrator_t& migrator, AoSoA_t& aosoa,
              typename std::enable_if<( ( is_distributor<Migrator_t>::value ||
                                          is_collector<Migrator_t>::value ) &&
                                        is_aosoa<AoSoA_t>::value ),
                                      int>::type* = 0 )
{
    migrate( typename Migrator_t::execution_space{}, migrator, aosoa );
}

/*!
  \brief Synchronously distribute data between two different decompositions
  using the Distributor forward communication plan. Slice version. The user can
  do this in-place with the same slice but they will need to manage the resizing
  themselves as we can't resize slices.

  Migrate with a Distributor moves all data to a new distribution that is
  uniquely owned - each element will only have a single destination rank.

  \tparam MemorySpace The memory space the Distributor is in.
  \tparam Slice_t Slice type - must be a Slice.

  \param distributor The Distributor to use for the data distribution.
  \param src The slice containing the data to be distributed. Must have the same
  number of elements as the inputs used to construct the Distributor.
  \param dst The slice to which the distributed data will be written. Must be
  the same size as the number of imports given by the distributor on this rank.
  Call totalNumImport() on the distributor to get this size value.
*/
template <class MemorySpace, class Slice_t>
void migrate(
    const Distributor<MemorySpace>& distributor, const Slice_t& src,
    Slice_t& dst,
    typename std::enable_if<( is_slice<Slice_t>::value ), int>::type* = 0 )
{
    if ( src.size() != distributor.exportSize() )
        throw std::runtime_error(
            "Cabana::migrate (Distributor): Source slice is the "
            "wrong size for distribution!" );

    Impl::migrateSlice( typename Distributor<MemorySpace>::execution_space{},
                        distributor, src, dst );
}

/*!
  \brief Synchronously collect data to different decompositions using
  the Collector forward communication plan. Slice version. The user can do
  this in-place with the same slice but they will need to manage the resizing
  themselves as we can't resize slices.

  Migrate with a Collector duplicates data on the sending and receiving
  decompostions.

  \tparam MemorySpace The memory space the Collector is in.
  \tparam Slice_t Slice type - must be a Slice.

  \param collector The Collector to use for the data collection.
  \param src The slice containing the data to be collected. Must have the same
  number of elements as the number of owned elements passed to the constructer
  of the Collector.
  \param dst The slice to which the collected data will be
  written. Must be the same size as the number of imports given by the collector
  on this rank. Call totalNumImport() on the Collector to get this size value.
*/
template <class MemorySpace, class Slice_t>
void migrate(
    const Collector<MemorySpace>& collector, const Slice_t& src, Slice_t& dst,
    typename std::enable_if<( is_slice<Slice_t>::value ), int>::type* = 0 )
{
    if ( src.size() != collector.numOwned() )
        throw std::runtime_error(
            "Cabana::migrate (Collector): Source slice is the "
            "wrong size for collection!" );

    Impl::migrateSlice( typename Collector<MemorySpace>::execution_space{},
                        collector, src, dst );
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_MIGRATE_HPP
