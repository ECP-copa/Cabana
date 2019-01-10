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

#ifndef CABANA_COMMUNICATIONPLAN_HPP
#define CABANA_COMMUNICATIONPLAN_HPP

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <type_traits>
#include <vector>
#include <exception>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class CommunicationPlan

  \brief Communication plan base class.

  \tparam MemorySpace Memory space in which the data for this class will be
  allocated.

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
template<class MemorySpace>
class CommunicationPlan
{
  public:

    // Cabana memory space.
    using memory_space = MemorySpace;

    // Kokkos execution space.
    using execution_space = typename memory_space::execution_space;

    /*!
      \brief Constructor.

      \param comm The MPI communicator over which the distributor is defined.

      \return The MPI communicator for this plan.
    */
    CommunicationPlan( MPI_Comm comm )
        : _comm( comm )
    {}

    /*!
      \brief Get the MPI communicator.
    */
    MPI_Comm comm() const
    { return _comm; }

    /*!
      \brief Get the number of neighbor ranks that this rank will communicate
      with.

      \return The number of MPI ranks that will exchange data with this rank.
    */
    int numNeighbor() const
    { return _neighbors.size(); }

    /*!
      \brief Given a local neighbor id get its rank in the MPI communicator.

      \param neighbor The local id of the neighbor to get the rank for.

      \return The MPI rank of the neighbor with the given local id.
    */
    int neighborRank( const int neighbor ) const
    { return _neighbors[neighbor]; }

    /*!
      \brief Get the number of elements this rank will export to a given neighbor.

      \param neighbor The local id of the neighbor to get the number of
      exports for.

      \return The number of elements this rank will export to the neighbor with the
      given local id.
     */
    std::size_t numExport( const int neighbor ) const
    { return _num_export[neighbor]; }

    /*!
      \brief Get the total number of exports this rank will do.

      \return The total number of elements this rank will export to its
      neighbors.
    */
    std::size_t totalNumExport() const
    { return _total_num_export; }

    /*!
      \brief Get the number of elements this rank will import from a given neighbor.

      \param neighbor The local id of the neighbor to get the number of
      imports for.

      \return The number of elements this rank will import from the neighbor
      with the given local id.
     */
    std::size_t numImport( const int neighbor ) const
    { return _num_import[neighbor]; }

    /*!
      \brief Get the total number of imports this rank will do.

      \return The total number of elements this rank will import from its
      neighhbors.
    */
    std::size_t totalNumImport() const
    { return _total_num_import; }

    /*!
      \brief Get the number of export elements.

      Whenever the communciation plan is applied, this is the total number of
      elements expected to be input on the sending ranks (in the forward
      communication plan). This will be different than the number returned by
      totalNumExport() if some of the export ranks used in the construction
      are -1 and therefore will not particpate in an export operation.

      \return The number of export elements.
    */
    std::size_t exportSize() const
    { return _num_export_element; }

    /*!
      \brief Get the steering vector for the exports.

      \return The steering vector for the exports.

      The steering vector places exports in contiguous chunks by destination
      rank. The chunks are in consecutive order based on the local neighbor id
      (i.e. all elements going to neighbor with local id 0 first, then all
      elements going to neighbor with local id 1, etc.).
    */
    Kokkos::View<std::size_t*,memory_space> getExportSteering() const
    { return _export_steering; }

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

      \param mpi_tag The MPI tag to use for non-blocking communication in the
      communication plan generation.

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. The elements in this list must be unique.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template<class ViewType>
    void createFromExportsAndTopology(
        const ViewType& element_export_ranks,
        const std::vector<int>& neighbor_ranks,
        const int mpi_tag = 1221 )
    {
        // Store the number of export elements.
        _num_export_element = element_export_ranks.size();

        // Store the neighbors.
        _neighbors = neighbor_ranks;
        int num_n = _neighbors.size();

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( _comm, &my_rank );

        // If we are sending to ourself put that one first in the neighbor
        // list.
        for ( auto& n : _neighbors )
            if ( n == my_rank )
            {
                std::swap( n, _neighbors[0] );
                break;
            }

        // Initialize import/export sizes.
        _num_export.assign( num_n, 0 );
        _num_import.assign( num_n, 0 );

        // Copy the topology to the device.
        Kokkos::View<int*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>
            topology_host( _neighbors.data(), num_n );
        auto topology = Kokkos::create_mirror_view_and_copy(
            memory_space(), topology_host );

        // Count the number of sends to each neighbor.
        Kokkos::View<std::size_t*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>
            num_export_host( _num_export.data(), num_n );
        auto export_counts = Kokkos::create_mirror_view_and_copy(
                memory_space(), num_export_host );
        auto count_neighbor_func =
            KOKKOS_LAMBDA( const int i )
            {
                for ( int n = 0; n < num_n; ++n )
                    if ( topology(n) == element_export_ranks(i) )
                        Kokkos::atomic_increment( &export_counts(n) );
            };
        Kokkos::RangePolicy<execution_space> count_neighbor_policy(
            0, _num_export_element );
        Kokkos::parallel_for( "Cabana::CommunicationPlan::count_neighbors",
                              count_neighbor_policy,
                              count_neighbor_func );
        Kokkos::fence();

        // Copy counts back to the host.
        Kokkos::deep_copy( num_export_host, export_counts );

        // Post receives for the number of imports we will get.
        std::vector<MPI_Request> requests;
        requests.reserve( num_n );
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != _neighbors[n] )
            {
                requests.push_back( MPI_Request() );
                MPI_Irecv( &_num_import[n],
                           1,
                           MPI_UNSIGNED_LONG,
                           _neighbors[n],
                           mpi_tag,
                           _comm,
                           &(requests.back()) );
            }
            else
                _num_import[n] = _num_export[n];

        // Send the number of exports to each of our neighbors.
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != _neighbors[n] )
                MPI_Send( &_num_export[n],
                          1,
                          MPI_UNSIGNED_LONG,
                          _neighbors[n],
                          mpi_tag,
                          _comm );

        // Wait on receives.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        assert( MPI_SUCCESS == ec );

        // Get the total number of imports/exports.
        _total_num_export =
            std::accumulate( _num_export.begin(), _num_export.end(), 0 );
        _total_num_import =
            std::accumulate( _num_import.begin(), _num_import.end(), 0 );

        // Barrier before continuing to ensure synchronization.
        MPI_Barrier( _comm );
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

      \param mpi_tag The MPI tag to use for non-blocking communication in the
      communication plan generation.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note For elements that you do not wish to export, use an export rank of
      -1 to signal that this element is *not* to be exported and will be
      ignored in the data migration. In other words, this element will be
      *completely* removed in the new decomposition. If the data is staying on
      this rank, just use this rank as the export destination and the data
      will be efficiently migrated.
    */
    template<class ViewType>
    void createFromExportsOnly( const ViewType& element_export_ranks,
                                const int mpi_tag = 1221 )
    {
        // Store the number of export elements.
        _num_export_element = element_export_ranks.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( _comm, &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( _comm, &my_rank );

        // Count the number of sends this rank will do to other ranks.
        Kokkos::View<int*,memory_space> neighbor_counts(
            "neighbor_counts", comm_size );
        auto count_sends_func =
            KOKKOS_LAMBDA( const int i )
            {
                if ( element_export_ranks(i) >= 0 )
                    Kokkos::atomic_increment(
                        &neighbor_counts(element_export_ranks(i)) );
            };
        Kokkos::RangePolicy<execution_space> count_sends_policy(
            0, _num_export_element );
        Kokkos::parallel_for( "Cabana::CommunicationPlan::count_sends",
                              count_sends_policy,
                              count_sends_func );
        Kokkos::fence();

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), neighbor_counts );

        // Extract the export ranks and number of exports and then flag the
        // send ranks.
        _neighbors.clear();
        _num_export.clear();
        _total_num_export = 0;
        for ( int r = 0; r < comm_size; ++r )
            if ( neighbor_counts_host(r) > 0 )
            {
                _neighbors.push_back( r );
                _num_export.push_back( neighbor_counts_host(r) );
                _total_num_export += neighbor_counts_host(r);
                neighbor_counts_host(r) = 1;
            }

        // Get the number of export ranks and initially allocate the import sizes.
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
        std::vector<int> total_receives( comm_size );
        int root_rank = 0;
        MPI_Reduce( neighbor_counts_host.data(),
                    total_receives.data(),
                    comm_size,
                    MPI_INT,
                    MPI_SUM,
                    root_rank,
                    _comm );
        int num_import_rank = -1;
        MPI_Scatter( total_receives.data(),
                     1,
                     MPI_INT,
                     &num_import_rank,
                     1,
                     MPI_INT,
                     root_rank,
                     _comm );
        if ( self_send ) --num_import_rank;

        // Post the expected number of receives and indicate we might get them
        // from any rank.
        std::vector<std::size_t> import_sizes( num_import_rank );
        std::vector<MPI_Request> requests( num_import_rank );
        for ( int n = 0; n < num_import_rank; ++n )
            MPI_Irecv( &import_sizes[n],
                       1,
                       MPI_UNSIGNED_LONG,
                       MPI_ANY_SOURCE,
                       mpi_tag,
                       _comm,
                       &requests[n] );

        // Do blocking sends. Dont do any self sends.
        int self_offset = (self_send) ? 1 : 0;
        for ( int n = self_offset; n < num_export_rank; ++n )
            MPI_Send( &_num_export[n],
                      1,
                      MPI_UNSIGNED_LONG,
                      _neighbors[n],
                      mpi_tag,
                      _comm );

        // Wait on non-blocking receives.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        assert( MPI_SUCCESS == ec );

        // Compute the total number of imports.
        _total_num_import = std::accumulate(
            import_sizes.begin(), import_sizes.end(),
            (self_send) ? _num_import[0] : 0 );

        // Extract the imports. If we did self sends we already know what
        // imports we got from that.
        for ( int i = 0; i < num_import_rank; ++i )
        {
            // Get the message source.
            const auto source = status[i].MPI_SOURCE;

            // See if the neighbor we received stuff from was someone we also
            // sent stuff to. If it was, just record what they sent us.
            auto found_neighbor =
                std::find( _neighbors.begin(), _neighbors.end(), source );

            // If this is a new neighbor (i.e. someone we didn't send anything
            // to) record this. Otherwise add it to the one we found.
            if ( found_neighbor == std::end(_neighbors) )
            {
                _neighbors.push_back( source );
                _num_import.push_back( import_sizes[i] );
                _num_export.push_back( 0 );
            }
            else
            {
                _num_import[i+self_offset] = import_sizes[i];
            }
        }

        // Barrier before continuing to ensure synchronization.
        MPI_Barrier( _comm );
    }

    /*!
      \brief Create the export steering vector.

      Creates an array describing which export element ids are moved to which
      location in the send buffer of the communcation plan. Ordered such that
      if a rank sends to itself then those values come first.

      \param element_export_ranks The ranks to which we are exporting each
      element. We use this to build the steering vector. The input is expected
      to be a Kokkos view or Cabana slice in the same memory space as the
      communication plan.
    */
    template<class ViewType>
    void createExportSteering( const ViewType& element_export_ranks )
    {
        // passing in element_export_ranks here as a dummy argument.
        createSteering( true, element_export_ranks, element_export_ranks );
    }

    /*!
      \brief Create the export steering vector.

      Creates an array describing which export element ids are moved to which
      location in the contiguous send buffer of the communcation plan. Ordered
      such that if a rank sends to itself then those values come first.

      \param element_export_ranks The ranks to which we are exporting each
      element. We use this to build the steering vector. The input is expected
      to be a Kokkos view or Cabana slice in the same memory space as the
      communication plan.

      \param element_export_ids The local ids of the elements to be
      exported. This corresponds with the export ranks vector and must be the
      same length if defined. The input is expected to be a Kokkos view or
      Cabana slice in the same memory space as the communication plan.
    */
    template<class RankViewType, class IdViewType>
    void createExportSteering(
        const RankViewType& element_export_ranks,
        const IdViewType& element_export_ids )
    {
        createSteering( false, element_export_ranks, element_export_ids );
    }

    // Create the export steering vector.
    template<class RankViewType,class IdViewType>
    void createSteering(
        const bool use_iota,
        const RankViewType& element_export_ranks,
        const IdViewType& element_export_ids )
    {
        if ( !use_iota &&
             (element_export_ids.size() != element_export_ranks.size()) )
            throw std::runtime_error("Export ids and ranks different sizes!");

        // Calculate the steering offsets via exclusive prefix sum for the
        // exports.
        int num_n = _neighbors.size();
        Kokkos::View<std::size_t*,Kokkos::HostSpace>
            offsets_host( "offsets", num_n );
        for ( int n = 1; n < num_n; ++n )
            offsets_host(n) = offsets_host(n-1) + _num_export[n-1];

        // Copy the offsets to the device.
        auto offsets = Kokkos::create_mirror_view_and_copy(
            memory_space(), offsets_host );

        // Copy the neighbors to the device.
        Kokkos::View<int*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>
            neighbor_ranks_host( _neighbors.data(), num_n );
        auto neighbor_ranks = Kokkos::create_mirror_view_and_copy(
            memory_space(), neighbor_ranks_host );

        // Create the export steering vector for writing local elements into
        // the send buffer. Note we create a local, shallow copy - this is a
        // CUDA workaround.
        _export_steering = Kokkos::View<std::size_t*,memory_space>(
            Kokkos::ViewAllocateWithoutInitializing("export_steering"),
            _total_num_export );
        auto steer_vec = _export_steering;
        Kokkos::View<std::size_t*,memory_space> counts( "counts", num_n );
        auto steer_func =
            KOKKOS_LAMBDA( const int i )
            {
                for ( int n = 0; n < num_n; ++n )
                    if ( element_export_ranks(i) == neighbor_ranks(n) )
                    {
                        auto c = Kokkos::atomic_fetch_add( &counts(n), 1 );
                        steer_vec( offsets(n) + c ) =
                            (use_iota) ? i : element_export_ids(i);
                        break;
                    }
            };
        Kokkos::RangePolicy<execution_space> steer_policy(
            0, element_export_ranks.size() );
        Kokkos::parallel_for( "Cabana::createSteering",
                              steer_policy,
                              steer_func );
        Kokkos::fence();
    }

  private:

    MPI_Comm _comm;
    std::vector<int> _neighbors;
    std::size_t _total_num_export;
    std::size_t _total_num_import;
    std::vector<std::size_t> _num_export;
    std::vector<std::size_t> _num_import;
    std::size_t _num_export_element;
    Kokkos::View<std::size_t*,memory_space> _export_steering;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_COMMUNICATIONPLAN_HPP
