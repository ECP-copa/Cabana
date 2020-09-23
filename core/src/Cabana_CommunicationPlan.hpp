/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_COMMUNICATIONPLAN_HPP
#define CABANA_COMMUNICATIONPLAN_HPP

#include <CabanaCore_config.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <mpi.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace Impl
{
//---------------------------------------------------------------------------//
// Count sends and create steering algorithm tags.
struct CountSendsAndCreateSteeringDuplicated
{
};
struct CountSendsAndCreateSteeringAtomic
{
};

//---------------------------------------------------------------------------//
// Count sends and create steering algorithm selector.
template <class ExecutionSpace>
struct CountSendsAndCreateSteeringAlgorithm;

// CUDA and HIP use atomics.
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct CountSendsAndCreateSteeringAlgorithm<Kokkos::Cuda>
{
    using type = CountSendsAndCreateSteeringAtomic;
};
#endif // end KOKKOS_ENABLE_CUDA
#ifdef KOKKOS_ENABLE_HIP
template <>
struct CountSendsAndCreateSteeringAlgorithm<Kokkos::Experimental::HIP>
{
    using type = CountSendsAndCreateSteeringAtomic;
};
#endif // end KOKKOS_ENABLE_HIP

// The default is to use duplication.
template <class ExecutionSpace>
struct CountSendsAndCreateSteeringAlgorithm
{
    using type = CountSendsAndCreateSteeringDuplicated;
};

//---------------------------------------------------------------------------//
// Count sends and generate the steering vector. Atomic version.
template <class ExportRankView>
auto countSendsAndCreateSteering( const ExportRankView element_export_ranks,
                                  const int comm_size,
                                  CountSendsAndCreateSteeringAtomic )
    -> std::pair<Kokkos::View<int *, typename ExportRankView::device_type>,
                 Kokkos::View<typename ExportRankView::size_type *,
                              typename ExportRankView::device_type>>
{
    using device_type = typename ExportRankView::device_type;
    using execution_space = typename ExportRankView::execution_space;
    using size_type = typename ExportRankView::size_type;

    // Create views.
    Kokkos::View<int *, device_type> neighbor_counts( "neighbor_counts",
                                                      comm_size );
    Kokkos::View<size_type *, device_type> neighbor_ids(
        Kokkos::ViewAllocateWithoutInitializing( "neighbor_ids" ),
        element_export_ranks.size() );

    // Count the sends and create the steering vector.
    Kokkos::parallel_for(
        "Cabana::CommunicationPlan::countSendsAndCreateSteering",
        Kokkos::RangePolicy<execution_space>( 0, element_export_ranks.size() ),
        KOKKOS_LAMBDA( const size_type i ) {
            if ( element_export_ranks( i ) >= 0 )
                neighbor_ids( i ) = Kokkos::atomic_fetch_add(
                    &neighbor_counts( element_export_ranks( i ) ), 1 );
        } );
    Kokkos::fence();

    // Return the counts and ids.
    return std::make_pair( neighbor_counts, neighbor_ids );
}
//---------------------------------------------------------------------------//
// Count sends and generate the steering vector. Duplicated version.
template <class ExportRankView>
auto countSendsAndCreateSteering( const ExportRankView element_export_ranks,
                                  const int comm_size,
                                  CountSendsAndCreateSteeringDuplicated )
    -> std::pair<Kokkos::View<int *, typename ExportRankView::device_type>,
                 Kokkos::View<typename ExportRankView::size_type *,
                              typename ExportRankView::device_type>>
{
    using device_type = typename ExportRankView::device_type;
    using execution_space = typename ExportRankView::execution_space;
    using size_type = typename ExportRankView::size_type;

    // Create a unique thread token.
    Kokkos::Experimental::UniqueToken<
        execution_space, Kokkos::Experimental::UniqueTokenScope::Global>
        unique_token;

    // Create views.
    Kokkos::View<int *, device_type> neighbor_counts(
        Kokkos::ViewAllocateWithoutInitializing( "neighbor_counts" ),
        comm_size );
    Kokkos::View<size_type *, device_type> neighbor_ids(
        Kokkos::ViewAllocateWithoutInitializing( "neighbor_ids" ),
        element_export_ranks.size() );
    Kokkos::View<int **, device_type> neighbor_counts_dup(
        "neighbor_counts", unique_token.size(), comm_size );
    Kokkos::View<size_type **, device_type> neighbor_ids_dup(
        "neighbor_ids", unique_token.size(), element_export_ranks.size() );

    // Compute initial duplicated sends and steering.
    Kokkos::parallel_for(
        "Cabana::CommunicationPlan::intialCount",
        Kokkos::RangePolicy<execution_space>( 0, element_export_ranks.size() ),
        KOKKOS_LAMBDA( const size_type i ) {
            if ( element_export_ranks( i ) >= 0 )
            {
                // Get the thread id.
                auto thread_id = unique_token.acquire();

                // Do the duplicated fetch-add. If this is a valid element id
                // increment the send count for this rank. Add the incremented
                // count as the thread-local neighbor id. This is too big by
                // one (because we use the prefix increment operator) but we
                // want a non-zero value so we can later find which thread
                // this element was located on because we are always
                // guaranteed a non-zero value. We will subtract this value
                // later.
                neighbor_ids_dup( thread_id, i ) = ++neighbor_counts_dup(
                    thread_id, element_export_ranks( i ) );

                // Release the thread id.
                unique_token.release( thread_id );
            }
        } );
    Kokkos::fence();

    // Team policy
    using team_policy =
        Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>;
    using index_type = typename team_policy::index_type;

    // Compute the send counts for each neighbor rank by reducing across
    // the thread duplicates.
    Kokkos::parallel_for(
        "Cabana::CommunicationPlan::finalCount",
        team_policy( neighbor_counts.extent( 0 ), Kokkos::AUTO ),
        KOKKOS_LAMBDA( const typename team_policy::member_type &team ) {
            // Get the element id.
            auto i = team.league_rank();

            // Add the thread results.
            int thread_counts = 0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange( team,
                                         neighbor_counts_dup.extent( 0 ) ),
                [&]( const index_type thread_id, int &result ) {
                    result += neighbor_counts_dup( thread_id, i );
                },
                thread_counts );
            neighbor_counts( i ) = thread_counts;
        } );
    Kokkos::fence();

    // Compute the location of each export element in the send buffer of
    // its destination rank.
    Kokkos::parallel_for(
        "Cabana::CommunicationPlan::createSteering",
        team_policy( element_export_ranks.size(), Kokkos::AUTO ),
        KOKKOS_LAMBDA( const typename team_policy::member_type &team ) {
            // Get the element id.
            auto i = team.league_rank();

            // Only operate on valid elements
            if ( element_export_ranks( i ) >= 0 )
            {
                // Compute the thread id in which we located the element
                // during the count phase. Only the thread in which we
                // located the element will contribute to the reduction.
                index_type dup_thread = 0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange( team,
                                             neighbor_ids_dup.extent( 0 ) ),
                    [&]( const index_type thread_id, index_type &result ) {
                        if ( neighbor_ids_dup( thread_id, i ) > 0 )
                            result += thread_id;
                    },
                    dup_thread );

                // Compute the offset of this element in the steering
                // vector for its destination rank. Loop through the
                // threads up to the thread that found this element in the
                // count stage. All thread counts prior to that thread
                // will contribute to the offset.
                size_type thread_offset = 0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange( team, dup_thread ),
                    [&]( const index_type thread_id, size_type &result ) {
                        result += neighbor_counts_dup(
                            thread_id, element_export_ranks( i ) );
                    },
                    thread_offset );

                // Add the thread-local value to the offset where we subtract
                // the 1 that we added artificially when we were first
                // counting.
                neighbor_ids( i ) =
                    thread_offset + neighbor_ids_dup( dup_thread, i ) - 1;
            }
        } );
    Kokkos::fence();

    // Return the counts and ids.
    return std::make_pair( neighbor_counts, neighbor_ids );
}

//---------------------------------------------------------------------------//

} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \class CommunicationPlan

  \brief Communication plan base class.

  \tparam DeviceType Device type for which the data for this class will be
  allocated and where parallel execution will occur.

  The communication plan computes how to redistribute elements in a parallel
  data structure using MPI. Given a list of data elements on the local MPI
  rank and their destination ranks, the communication plan computes which rank
  each process is sending and receiving from and how many elements we will
  send and receive. In addition, it provides an export steering vector which
  describes how to pack the local data to be exported into contiguous send
  buffers for each destination rank (in the forward communication plan).

  Some nomenclature:

  Export - elements we are sending in the forward communication plan.

  Import - elements we are receiving in the forward communication plan.

  \note If a communication plan does self-sends (i.e. exports and imports data
  from its own ranks) then this data is first in the data structure. What this
  means is that neighbor 0 is the local rank and the data for that rank that
  is being exported will appear first in the steering vector.
*/
template <class DeviceType>
class CommunicationPlan
{
  public:
    // Device type.
    using device_type = DeviceType;

    // Memory space.
    using memory_space = typename device_type::memory_space;

    // Execution space.
    using execution_space = typename device_type::execution_space;

    // Size type.
    using size_type = typename memory_space::size_type;

    /*!
      \brief Constructor.

      \param comm The MPI communicator over which the distributor is defined.

      \return The MPI communicator for this plan.
    */
    CommunicationPlan( MPI_Comm comm )
    {
        _comm_ptr.reset(
            // Duplicate the communicator and store in a std::shared_ptr so that
            // all copies point to the same object
            [comm]() {
                auto p = std::make_unique<MPI_Comm>();
                MPI_Comm_dup( comm, p.get() );
                return p.release();
            }(),
            // Custom deleter to mark the communicator for deallocation
            []( MPI_Comm *p ) {
                MPI_Comm_free( p );
                delete p;
            } );
    }

    /*!
      \brief Get the MPI communicator.
    */
    MPI_Comm comm() const { return *_comm_ptr; }

    /*!
      \brief Get the number of neighbor ranks that this rank will communicate
      with.

      \return The number of MPI ranks that will exchange data with this rank.
    */
    int numNeighbor() const { return _neighbors.size(); }

    /*!
      \brief Given a local neighbor id get its rank in the MPI communicator.

      \param neighbor The local id of the neighbor to get the rank for.

      \return The MPI rank of the neighbor with the given local id.
    */
    int neighborRank( const int neighbor ) const
    {
        return _neighbors[neighbor];
    }

    /*!
      \brief Get the number of elements this rank will export to a given
      neighbor.

      \param neighbor The local id of the neighbor to get the number of
      exports for.

      \return The number of elements this rank will export to the neighbor with
      the given local id.
     */
    std::size_t numExport( const int neighbor ) const
    {
        return _num_export[neighbor];
    }

    /*!
      \brief Get the total number of exports this rank will do.

      \return The total number of elements this rank will export to its
      neighbors.
    */
    std::size_t totalNumExport() const { return _total_num_export; }

    /*!
      \brief Get the number of elements this rank will import from a given
      neighbor.

      \param neighbor The local id of the neighbor to get the number of
      imports for.

      \return The number of elements this rank will import from the neighbor
      with the given local id.
     */
    std::size_t numImport( const int neighbor ) const
    {
        return _num_import[neighbor];
    }

    /*!
      \brief Get the total number of imports this rank will do.

      \return The total number of elements this rank will import from its
      neighhbors.
    */
    std::size_t totalNumImport() const { return _total_num_import; }

    /*!
      \brief Get the number of export elements.

      Whenever the communciation plan is applied, this is the total number of
      elements expected to be input on the sending ranks (in the forward
      communication plan). This will be different than the number returned by
      totalNumExport() if some of the export ranks used in the construction
      are -1 and therefore will not particpate in an export operation.

      \return The number of export elements.
    */
    std::size_t exportSize() const { return _num_export_element; }

    /*!
      \brief Get the steering vector for the exports.

      \return The steering vector for the exports.

      The steering vector places exports in contiguous chunks by destination
      rank. The chunks are in consecutive order based on the local neighbor id
      (i.e. all elements going to neighbor with local id 0 first, then all
      elements going to neighbor with local id 1, etc.).
    */
    Kokkos::View<std::size_t *, device_type> getExportSteering() const
    {
        return _export_steering;
    }

    // The functions in the public block below would normally be protected but
    // we make them public to allow using private class data in CUDA kernels
    // with lambda functions.
  public:
    /*!
      \brief Neighbor and export rank creator. Use this when you already know
      which ranks neighbor each other (i.e. every rank already knows who they
      will be sending and receiving from) as it will be more efficient. In
      this case you already know the topology of the point-to-point
      communication but not how much data to send to and receive from the
      neighbors.

      \param element_export_ranks The destination rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique destination to which it
      will be exported. This export rank may be any one of the listed neighbor
      ranks which can include the calling rank. An export rank of -1 will
      signal that this element is *not* to be exported and will be ignored in
      the data migration. The input is expected to be a Kokkos view or Cabana
      slice in the same memory space as the communication plan.

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. Only the unique elements in this list are used.

      \return The location of each export element in the send buffer for its
      given neighbor.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template <class ViewType>
    Kokkos::View<size_type *, device_type>
    createFromExportsAndTopology( const ViewType &element_export_ranks,
                                  const std::vector<int> &neighbor_ranks )
    {
        // Store the number of export elements.
        _num_export_element = element_export_ranks.size();

        // Store the unique neighbors.
        _neighbors = neighbor_ranks;
        std::sort( _neighbors.begin(), _neighbors.end() );
        auto unique_end = std::unique( _neighbors.begin(), _neighbors.end() );
        _neighbors.resize( std::distance( _neighbors.begin(), unique_end ) );
        int num_n = _neighbors.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( comm(), &my_rank );

        // Pick an mpi tag for communication. This object has it's own
        // communication space so any mpi tag will do.
        const int mpi_tag = 1221;

        // If we are sending to ourself put that one first in the neighbor
        // list.
        for ( auto &n : _neighbors )
            if ( n == my_rank )
            {
                std::swap( n, _neighbors[0] );
                break;
            }

        // Initialize import/export sizes.
        _num_export.assign( num_n, 0 );
        _num_import.assign( num_n, 0 );

        // Count the number of sends this rank will do to other ranks. Keep
        // track of which slot we get in our neighbor's send buffer.
        auto counts_and_ids = Impl::countSendsAndCreateSteering(
            element_export_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                execution_space>::type() );

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), counts_and_ids.first );

        // Get the export counts.
        for ( int n = 0; n < num_n; ++n )
            _num_export[n] = neighbor_counts_host( _neighbors[n] );

        // Post receives for the number of imports we will get.
        std::vector<MPI_Request> requests;
        requests.reserve( num_n );
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != _neighbors[n] )
            {
                requests.push_back( MPI_Request() );
                MPI_Irecv( &_num_import[n], 1, MPI_UNSIGNED_LONG, _neighbors[n],
                           mpi_tag, comm(), &( requests.back() ) );
            }
            else
                _num_import[n] = _num_export[n];

        // Send the number of exports to each of our neighbors.
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != _neighbors[n] )
                MPI_Send( &_num_export[n], 1, MPI_UNSIGNED_LONG, _neighbors[n],
                          mpi_tag, comm() );

        // Wait on receives.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        if ( MPI_SUCCESS != ec )
            throw std::logic_error( "Failed MPI Communication" );

        // Get the total number of imports/exports.
        _total_num_export =
            std::accumulate( _num_export.begin(), _num_export.end(), 0 );
        _total_num_import =
            std::accumulate( _num_import.begin(), _num_import.end(), 0 );

        // Barrier before continuing to ensure synchronization.
        MPI_Barrier( comm() );

        // Return the neighbor ids.
        return counts_and_ids.second;
    }

    /*!
      \brief Export rank creator. Use this when you don't know who you will
      receiving from - only who you are sending to. This is less efficient
      than if we already knew who our neighbors were because we have to
      determine the topology of the point-to-point communication first.

      \param element_export_ranks The destination rank in the target
      decomposition of each locally owned element in the source
      decomposition. Each element will have one unique destination to which it
      will be exported. This export rank may any one of the listed neighbor
      ranks which can include the calling rank. An export rank of -1 will
      signal that this element is *not* to be exported and will be ignored in
      the data migration. The input is expected to be a Kokkos view or Cabana
      slice in the same memory space as the communication plan.

      \return The location of each export element in the send buffer for its
      given neighbor.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template <class ViewType>
    Kokkos::View<size_type *, device_type>
    createFromExportsOnly( const ViewType &element_export_ranks )
    {
        // Store the number of export elements.
        _num_export_element = element_export_ranks.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( comm(), &my_rank );

        // Pick an mpi tag for communication. This object has it's own
        // communication space so any mpi tag will do.
        const int mpi_tag = 1221;

        // Count the number of sends this rank will do to other ranks. Keep
        // track of which slot we get in our neighbor's send buffer.
        auto counts_and_ids = Impl::countSendsAndCreateSteering(
            element_export_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                execution_space>::type() );

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), counts_and_ids.first );

        // Extract the export ranks and number of exports and then flag the
        // send ranks.
        _neighbors.clear();
        _num_export.clear();
        _total_num_export = 0;
        for ( int r = 0; r < comm_size; ++r )
            if ( neighbor_counts_host( r ) > 0 )
            {
                _neighbors.push_back( r );
                _num_export.push_back( neighbor_counts_host( r ) );
                _total_num_export += neighbor_counts_host( r );
                neighbor_counts_host( r ) = 1;
            }

        // Get the number of export ranks and initially allocate the import
        // sizes.
        int num_export_rank = _neighbors.size();
        _num_import.assign( num_export_rank, 0 );

        // If we are sending to ourself put that one first in the neighbor
        // list and assign the number of imports to be the number of exports.
        bool self_send = false;
        for ( int n = 0; n < num_export_rank; ++n )
            if ( _neighbors[n] == my_rank )
            {
                std::swap( _neighbors[n], _neighbors[0] );
                std::swap( _num_export[n], _num_export[0] );
                _num_import[0] = _num_export[0];
                self_send = true;
                break;
            }

        // Determine how many total import ranks each neighbor has.
        int num_import_rank = -1;
        std::vector<int> recv_counts( comm_size, 1 );
        MPI_Reduce_scatter( neighbor_counts_host.data(), &num_import_rank,
                            recv_counts.data(), MPI_INT, MPI_SUM, comm() );
        if ( self_send )
            --num_import_rank;

        // Post the expected number of receives and indicate we might get them
        // from any rank.
        std::vector<std::size_t> import_sizes( num_import_rank );
        std::vector<MPI_Request> requests( num_import_rank );
        for ( int n = 0; n < num_import_rank; ++n )
            MPI_Irecv( &import_sizes[n], 1, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE,
                       mpi_tag, comm(), &requests[n] );

        // Do blocking sends. Dont do any self sends.
        int self_offset = ( self_send ) ? 1 : 0;
        for ( int n = self_offset; n < num_export_rank; ++n )
            MPI_Send( &_num_export[n], 1, MPI_UNSIGNED_LONG, _neighbors[n],
                      mpi_tag, comm() );

        // Wait on non-blocking receives.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        if ( MPI_SUCCESS != ec )
            throw std::logic_error( "Failed MPI Communication" );

        // Compute the total number of imports.
        _total_num_import =
            std::accumulate( import_sizes.begin(), import_sizes.end(),
                             ( self_send ) ? _num_import[0] : 0 );

        // Extract the imports. If we did self sends we already know what
        // imports we got from that.
        for ( int i = 0; i < num_import_rank; ++i )
        {
            // Get the message source.
            const auto source = status[i].MPI_SOURCE;

            // See if the neighbor we received stuff from was someone we also
            // sent stuff to.
            auto found_neighbor =
                std::find( _neighbors.begin(), _neighbors.end(), source );

            // If this is a new neighbor (i.e. someone we didn't send anything
            // to) record this.
            if ( found_neighbor == std::end( _neighbors ) )
            {
                _neighbors.push_back( source );
                _num_import.push_back( import_sizes[i] );
                _num_export.push_back( 0 );
            }

            // Otherwise if we already sent something to this neighbor that
            // means we already have a neighbor/export entry. Just assign the
            // import entry for that neighbor.
            else
            {
                auto n = std::distance( _neighbors.begin(), found_neighbor );
                _num_import[n] = import_sizes[i];
            }
        }

        // Barrier before continuing to ensure synchronization.
        MPI_Barrier( comm() );

        // Return the neighbor ids.
        return counts_and_ids.second;
    }

    /*!
      \brief Create the export steering vector.

      Creates an array describing which export element ids are moved to which
      location in the send buffer of the communcation plan. Ordered such that
      if a rank sends to itself then those values come first.

      \param neighbor_ids The id of each element in the neighbor send buffers.

      \param element_export_ranks The ranks to which we are exporting each
      element. We use this to build the steering vector. The input is expected
      to be a Kokkos view or Cabana slice in the same memory space as the
      communication plan.
    */
    template <class PackViewType, class RankViewType>
    void createExportSteering( const PackViewType &neighbor_ids,
                               const RankViewType &element_export_ranks )
    {
        // passing in element_export_ranks here as a dummy argument.
        createSteering( true, neighbor_ids, element_export_ranks,
                        element_export_ranks );
    }

    /*!
      \brief Create the export steering vector.

      Creates an array describing which export element ids are moved to which
      location in the contiguous send buffer of the communcation plan. Ordered
      such that if a rank sends to itself then those values come first.

      \param neighbor_ids The id of each element in the neighbor send buffers.

      \param element_export_ranks The ranks to which we are exporting each
      element. We use this to build the steering vector. The input is expected
      to be a Kokkos view or Cabana slice in the same memory space as the
      communication plan.

      \param element_export_ids The local ids of the elements to be
      exported. This corresponds with the export ranks vector and must be the
      same length if defined. The input is expected to be a Kokkos view or
      Cabana slice in the same memory space as the communication plan.
    */
    template <class PackViewType, class RankViewType, class IdViewType>
    void createExportSteering( const PackViewType &neighbor_ids,
                               const RankViewType &element_export_ranks,
                               const IdViewType &element_export_ids )
    {
        createSteering( false, neighbor_ids, element_export_ranks,
                        element_export_ids );
    }

    // Create the export steering vector.
    template <class PackViewType, class RankViewType, class IdViewType>
    void createSteering( const bool use_iota, const PackViewType &neighbor_ids,
                         const RankViewType &element_export_ranks,
                         const IdViewType &element_export_ids )
    {
        if ( !use_iota &&
             ( element_export_ids.size() != element_export_ranks.size() ) )
            throw std::runtime_error( "Export ids and ranks different sizes!" );

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( *_comm_ptr, &comm_size );

        // Calculate the steering offsets via exclusive prefix sum for the
        // exports.
        int num_n = _neighbors.size();
        std::vector<std::size_t> offsets( num_n, 0.0 );
        for ( int n = 1; n < num_n; ++n )
            offsets[n] = offsets[n - 1] + _num_export[n - 1];

        // Map the offsets to the device.
        Kokkos::View<std::size_t *, Kokkos::HostSpace> rank_offsets_host(
            Kokkos::ViewAllocateWithoutInitializing( "rank_map" ), comm_size );
        for ( int n = 0; n < num_n; ++n )
            rank_offsets_host( _neighbors[n] ) = offsets[n];
        auto rank_offsets = Kokkos::create_mirror_view_and_copy(
            memory_space(), rank_offsets_host );

        // Create the export steering vector for writing local elements into
        // the send buffer. Note we create a local, shallow copy - this is a
        // CUDA workaround for handling class private data.
        _export_steering = Kokkos::View<std::size_t *, memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "export_steering" ),
            _total_num_export );
        auto steer_vec = _export_steering;
        Kokkos::View<std::size_t *, memory_space> counts( "counts", num_n );
        Kokkos::parallel_for(
            "Cabana::createSteering",
            Kokkos::RangePolicy<execution_space>( 0, _num_export_element ),
            KOKKOS_LAMBDA( const int i ) {
                if ( element_export_ranks( i ) >= 0 )
                    steer_vec( rank_offsets( element_export_ranks( i ) ) +
                               neighbor_ids( i ) ) =
                        ( use_iota ) ? i : element_export_ids( i );
            } );
        Kokkos::fence();
    }

  private:
    std::shared_ptr<MPI_Comm> _comm_ptr;
    std::vector<int> _neighbors;
    std::size_t _total_num_export;
    std::size_t _total_num_import;
    std::vector<std::size_t> _num_export;
    std::vector<std::size_t> _num_import;
    std::size_t _num_export_element;
    Kokkos::View<std::size_t *, device_type> _export_steering;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_COMMUNICATIONPLAN_HPP
