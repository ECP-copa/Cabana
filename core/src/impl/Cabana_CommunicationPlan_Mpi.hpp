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
  \file Cabana_CommunicationPlan_Mpi.hpp
  \brief Multi-node communication patterns.
  Uses vanilla MPI as the communication backend.
*/
#ifndef CABANA_COMMUNICATIONPLAN_MPI_HPP
#define CABANA_COMMUNICATIONPLAN_MPI_HPP

#include <Cabana_Utils.hpp>

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

//---------------------------------------------------------------------------//
/*!
  \brief Communication plan class. Uses vanilla MPI as the communication
  backend.
*/

template <class MemorySpace>
class CommunicationPlan<MemorySpace, Mpi>
    : public CommunicationPlanBase<MemorySpace>
{
  public:
    using typename CommunicationPlanBase<MemorySpace>::memory_space;
    using typename CommunicationPlanBase<MemorySpace>::execution_space;
    using typename CommunicationPlanBase<MemorySpace>::size_type;

  protected:
    /*!
      \brief Constructor.

      \param comm The MPI communicator over which the CommunicationPlan is
      defined.
    */
    CommunicationPlan( MPI_Comm comm )
        : CommunicationPlanBase<MemorySpace>( comm )
    {
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

      \param exec_space Kokkos execution space.

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
    template <class ExecutionSpace, class RankViewType>
    Kokkos::View<size_type*, memory_space>
    createWithTopology( ExecutionSpace exec_space, Export,
                        const RankViewType& element_export_ranks,
                        const std::vector<int>& neighbor_ranks )
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        // Store the number of export elements.
        this->_num_export_element = element_export_ranks.size();

        // Store the unique neighbors (this rank first).
        this->_neighbors = getUniqueTopology( this->comm(), neighbor_ranks );
        int num_n = this->_neighbors.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( this->comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( this->comm(), &my_rank );

        // Pick an mpi tag for communication. This object has it's own
        // communication space so any mpi tag will do.
        const int mpi_tag = 1221;

        // Initialize import/export sizes.
        this->_num_export.assign( num_n, 0 );
        this->_num_import.assign( num_n, 0 );

        // Count the number of sends this rank will do to other ranks. Keep
        // track of which slot we get in our neighbor's send buffer.
        auto counts_and_ids = Impl::countSendsAndCreateSteering(
            exec_space, element_export_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), counts_and_ids.first );

        // Get the export counts.
        for ( int n = 0; n < num_n; ++n )
            this->_num_export[n] = neighbor_counts_host( this->_neighbors[n] );

        // Post receives for the number of imports we will get.
        std::vector<MPI_Request> requests;
        requests.reserve( num_n );
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != this->_neighbors[n] )
            {
                requests.push_back( MPI_Request() );
                MPI_Irecv( &this->_num_import[n], 1, MPI_UNSIGNED_LONG,
                           this->_neighbors[n], mpi_tag, this->comm(),
                           &( requests.back() ) );
            }
            else
                this->_num_import[n] = this->_num_export[n];

        // Send the number of exports to each of our neighbors.
        for ( int n = 0; n < num_n; ++n )
            if ( my_rank != this->_neighbors[n] )
                MPI_Send( &this->_num_export[n], 1, MPI_UNSIGNED_LONG,
                          this->_neighbors[n], mpi_tag, this->comm() );

        // Wait on receives.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        if ( MPI_SUCCESS != ec )
            throw std::logic_error(
                "Cabana::CommunicationPlan::createFromExportsAndTopology: "
                "Failed MPI Communication" );

        // Get the total number of imports/exports.
        this->_total_num_export =
            std::accumulate( this->_num_export.begin(), this->_num_export.end(),
                             std::size_t{ 0u } );
        this->_total_num_import =
            std::accumulate( this->_num_import.begin(), this->_num_import.end(),
                             std::size_t{ 0u } );

        // No barrier is needed because all ranks know who they are receiving
        // and sending to.

        // Return the neighbor ids.
        return counts_and_ids.second;
    }

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
    template <class RankViewType>
    Kokkos::View<size_type*, memory_space>
    createWithTopology( Export, const RankViewType& element_export_ranks,
                        const std::vector<int>& neighbor_ranks )
    {
        // Use the default execution space.
        return createWithTopology( execution_space{}, Export(),
                                   element_export_ranks, neighbor_ranks );
    }

    /*!
      \brief Export rank creator. Use this when you don't know who you will
      receiving from - only who you are sending to. This is less efficient
      than if we already knew who our neighbors were because we have to
      determine the topology of the point-to-point communication first.

      \param exec_space Kokkos execution space.

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
    template <class ExecutionSpace, class RankViewType>
    Kokkos::View<size_type*, memory_space>
    createWithoutTopology( ExecutionSpace exec_space, Export,
                           const RankViewType& element_export_ranks )
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        // Store the number of export elements.
        this->_num_export_element = element_export_ranks.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( this->comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( this->comm(), &my_rank );

        // Pick an mpi tag for communication. This object has it's own
        // communication space so any mpi tag will do.
        const int mpi_tag = 1221;

        // Count the number of sends this rank will do to other ranks. Keep
        // track of which slot we get in our neighbor's send buffer.
        auto counts_and_ids = Impl::countSendsAndCreateSteering(
            exec_space, element_export_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), counts_and_ids.first );

        // Extract the export ranks and number of exports and then flag the
        // send ranks.
        this->_neighbors.clear();
        this->_num_export.clear();
        this->_total_num_export = 0;
        for ( int r = 0; r < comm_size; ++r )
            if ( neighbor_counts_host( r ) > 0 )
            {
                this->_neighbors.push_back( r );
                this->_num_export.push_back( neighbor_counts_host( r ) );
                this->_total_num_export += neighbor_counts_host( r );
                neighbor_counts_host( r ) = 1;
            }

        // Get the number of export ranks and initially allocate the import
        // sizes.
        int num_export_rank = this->_neighbors.size();
        this->_num_import.assign( num_export_rank, 0 );

        // If we are sending to ourself put that one first in the neighbor
        // list and assign the number of imports to be the number of exports.
        bool self_send = false;
        for ( int n = 0; n < num_export_rank; ++n )
            if ( this->_neighbors[n] == my_rank )
            {
                std::swap( this->_neighbors[n], this->_neighbors[0] );
                std::swap( this->_num_export[n], this->_num_export[0] );
                this->_num_import[0] = this->_num_export[0];
                self_send = true;
                break;
            }

        // Determine how many total import ranks each neighbor has.
        int num_import_rank = -1;
        std::vector<int> recv_counts( comm_size, 1 );
        MPI_Reduce_scatter( neighbor_counts_host.data(), &num_import_rank,
                            recv_counts.data(), MPI_INT, MPI_SUM,
                            this->comm() );
        if ( self_send )
            --num_import_rank;

        // Post the expected number of receives and indicate we might get them
        // from any rank.
        std::vector<std::size_t> import_sizes( num_import_rank );
        std::vector<MPI_Request> requests( num_import_rank );
        for ( int n = 0; n < num_import_rank; ++n )
            MPI_Irecv( &import_sizes[n], 1, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE,
                       mpi_tag, this->comm(), &requests[n] );

        // Do blocking sends. Dont do any self sends.
        int self_offset = ( self_send ) ? 1 : 0;
        for ( int n = self_offset; n < num_export_rank; ++n )
            MPI_Send( &this->_num_export[n], 1, MPI_UNSIGNED_LONG,
                      this->_neighbors[n], mpi_tag, this->comm() );

        // Wait on non-blocking receives.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        if ( MPI_SUCCESS != ec )
            throw std::logic_error(
                "Cabana::CommunicationPlan::createFromExportsOnly: Failed MPI "
                "Communication" );

        // Compute the total number of imports.
        this->_total_num_import =
            std::accumulate( import_sizes.begin(), import_sizes.end(),
                             ( self_send ) ? this->_num_import[0] : 0 );

        // Extract the imports. If we did self sends we already know what
        // imports we got from that.
        for ( int i = 0; i < num_import_rank; ++i )
        {
            // Get the message source.
            const auto source = status[i].MPI_SOURCE;

            // See if the neighbor we received stuff from was someone we also
            // sent stuff to.
            auto found_neighbor = std::find( this->_neighbors.begin(),
                                             this->_neighbors.end(), source );

            // If this is a new neighbor (i.e. someone we didn't send anything
            // to) record this.
            if ( found_neighbor == std::end( this->_neighbors ) )
            {
                this->_neighbors.push_back( source );
                this->_num_import.push_back( import_sizes[i] );
                this->_num_export.push_back( 0 );
            }

            // Otherwise if we already sent something to this neighbor that
            // means we already have a neighbor/export entry. Just assign the
            // import entry for that neighbor.
            else
            {
                auto n =
                    std::distance( this->_neighbors.begin(), found_neighbor );
                this->_num_import[n] = import_sizes[i];
            }
        }

        // A barrier is needed because of the use of wildcard receives. This
        // avoids successive calls interfering with each other.
        MPI_Barrier( this->comm() );

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
    template <class RankViewType>
    Kokkos::View<size_type*, memory_space>
    createWithoutTopology( Export, const RankViewType& element_export_ranks )
    {
        // Use the default execution space.
        return createWithoutTopology( execution_space{}, Export(),
                                      element_export_ranks );
    }

    /*!
      \brief Neighbor and import rank creator. Use this when you already know
      which ranks neighbor each other (i.e. every rank already knows who they
      will be sending and receiving from) as it will be more efficient. In
      this case you already know the topology of the point-to-point
      communication but not how much data to send to and receive from the
      neighbors.

      \param exec_space Kokkos execution space.

      \param element_import_ranks The source rank in the target
      decomposition of each remotely owned element in element_import_ids.
      This import rank may be any one of the listed neighbor
      ranks which can include the calling rank. The input is expected
      to be a Kokkos view in the same memory space as the communication plan.

      \param element_import_ids The local IDs of remotely owned elements that
      are to be imported. These are local IDs on the remote rank.
      element_import_ids is mapped such that element_import_ids(i) lives on
      remote rank element_import_ranks(i).

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. Only the unique elements in this list are used.

      \return A tuple of Kokkos views, where:
      Element 1: The location of each export element in the send buffer for its
      given neighbor.
      Element 2: The remote ranks this rank will export to
      Element 3: The local IDs this rank will export
      Elements 2 and 3 are mapped in the same way as element_import_ranks
      and element_import_ids

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note Unlike creating from exports, an import rank of -1 is not supported.
    */
    template <class ExecutionSpace, class RankViewType, class IdViewType>
    auto createWithTopology( ExecutionSpace exec_space, Import,
                             const RankViewType& element_import_ranks,
                             const IdViewType& element_import_ids,
                             const std::vector<int>& neighbor_ranks )
        -> std::tuple<Kokkos::View<typename RankViewType::size_type*,
                                   typename RankViewType::memory_space>,
                      Kokkos::View<int*, typename RankViewType::memory_space>,
                      Kokkos::View<int*, typename IdViewType::memory_space>>
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        if ( element_import_ids.size() != element_import_ranks.size() )
            throw std::runtime_error( "Export ids and ranks different sizes!" );

        // Store the unique neighbors (this rank first).
        this->_neighbors = getUniqueTopology( this->comm(), neighbor_ranks );
        std::size_t num_n = this->_neighbors.size();

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( this->comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int my_rank = -1;
        MPI_Comm_rank( this->comm(), &my_rank );

        // Pick an mpi tag for communication. This object has it's own
        // communication space so any mpi tag will do.
        const int mpi_tag = 1221;

        // Initialize import/export sizes.
        this->_num_export.assign( num_n, 0 );
        this->_num_import.assign( num_n, 0 );

        // Count the number of imports this rank needs from other ranks. Keep
        // track of which slot we get in our neighbor's send buffer?
        auto counts_and_ids = Impl::countSendsAndCreateSteering(
            exec_space, element_import_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), counts_and_ids.first );

        // Get the import counts.
        for ( std::size_t n = 0; n < num_n; ++n )
            this->_num_import[n] = neighbor_counts_host( this->_neighbors[n] );

        // Post receives to get the number of indices I will send to each rank.
        // Post that many wildcard recieves to get the number of indices I will
        // send to each rank
        std::vector<MPI_Request> requests;
        requests.reserve( num_n * 2 );
        for ( std::size_t n = 0; n < num_n; ++n )
            if ( my_rank != this->_neighbors[n] )
            {
                requests.push_back( MPI_Request() );
                MPI_Irecv( &this->_num_export[n], 1, MPI_UNSIGNED_LONG,
                           this->_neighbors[n], mpi_tag, this->comm(),
                           &( requests.back() ) );
            }
            else // Self import
            {
                this->_num_export[n] = this->_num_import[n];
            }

        // Send the number of imports to each of our neighbors.
        for ( std::size_t n = 0; n < num_n; ++n )
            if ( my_rank != this->_neighbors[n] )
            {
                requests.push_back( MPI_Request() );
                MPI_Isend( &this->_num_import[n], 1, MPI_UNSIGNED_LONG,
                           this->_neighbors[n], mpi_tag, this->comm(),
                           &( requests.back() ) );
            }

        // Wait on messages.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        if ( MPI_SUCCESS != ec )
            throw std::logic_error( "Failed MPI Communication" );

        // Get the total number of imports/exports.
        this->_total_num_export = std::accumulate( this->_num_export.begin(),
                                                   this->_num_export.end(), 0 );
        this->_total_num_import = std::accumulate( this->_num_import.begin(),
                                                   this->_num_import.end(), 0 );
        this->_num_export_element = this->_total_num_export;

        // Post receives to get the indices other processes are requesting
        // i.e. our export indices
        Kokkos::View<int*, memory_space> export_indices(
            "export_indices", this->_total_num_export );
        std::size_t idx = 0;
        int num_messages =
            this->_total_num_export + element_import_ranks.extent( 0 );
        std::vector<MPI_Request> mpi_requests( num_messages );
        std::vector<MPI_Status> mpi_statuses( num_messages );

        // Increment the mpi_tag for this round of messages to ensure messages
        // are processed in the correct order from the previous round of Isends
        // and Irecvs.
        for ( std::size_t i = 0; i < num_n; i++ )
        {
            for ( std::size_t j = 0; j < this->_num_export[i]; j++ )
            {
                MPI_Irecv( export_indices.data() + idx, 1, MPI_INT,
                           this->_neighbors[i], mpi_tag + 1, this->comm(),
                           &mpi_requests[idx] );
                idx++;
            }
        }

        // Send the indices we need
        for ( std::size_t i = 0; i < element_import_ranks.extent( 0 ); i++ )
        {
            MPI_Isend( element_import_ids.data() + i, 1, MPI_INT,
                       *( element_import_ranks.data() + i ), mpi_tag + 1,
                       this->comm(), &mpi_requests[idx++] );
        }

        // Wait for all count exchanges to complete
        const int ec1 = MPI_Waitall( num_messages, mpi_requests.data(),
                                     mpi_statuses.data() );
        if ( MPI_SUCCESS != ec1 )
            throw std::logic_error( "Failed MPI Communication" );

        // Now, build the export steering
        // Export rank in mpi_statuses[i].MPI_SOURCE
        // Export ID in export_indices(i)
        Kokkos::View<int*, Kokkos::HostSpace> element_export_ranks_h(
            "element_export_ranks_h", this->_total_num_export );
        for ( std::size_t i = 0; i < this->_total_num_export; i++ )
        {
            element_export_ranks_h[i] = mpi_statuses[i].MPI_SOURCE;
        }
        auto element_export_ranks = Kokkos::create_mirror_view_and_copy(
            memory_space(), element_export_ranks_h );

        auto counts_and_ids2 = Impl::countSendsAndCreateSteering(
            exec_space, element_export_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // No barrier is needed because all ranks know who they are receiving
        // from and sending to.

        // Return the neighbor ids, export ranks, and export indices
        return std::tuple{ counts_and_ids2.second, element_export_ranks,
                           export_indices };
    }

    /*!
      \brief Neighbor and import rank creator. Use this when you already know
      which ranks neighbor each other (i.e. every rank already knows who they
      will be sending and receiving from) as it will be more efficient. In
      this case you already know the topology of the point-to-point
      communication but not how much data to send to and receive from the
      neighbors.

      \param element_import_ranks The source rank in the target
      decomposition of each remotely owned element in element_import_ids.
      This import rank may be any one of the listed neighbor
      ranks which can include the calling rank. The input is expected
      to be a Kokkos view in the same memory space as the communication plan.

      \param element_import_ids The local IDs of remotely owned elements that
      are to be imported. These are local IDs on the remote rank.
      element_import_ids is mapped such that element_import_ids(i) lives on
      remote rank element_import_ranks(i).

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. Only the unique elements in this list are used.

      \return A tuple of Kokkos views, where:
      Element 1: The location of each export element in the send buffer for its
      given neighbor.
      Element 2: The remote ranks this rank will export to
      Element 3: The local IDs this rank will export
      Elements 2 and 3 are mapped in the same way as element_import_ranks
      and element_import_ids

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note Unlike creating from exports, an import rank of -1 is not supported.
    */
    template <class RankViewType, class IdViewType>
    auto createWithTopology( Import, const RankViewType& element_import_ranks,
                             const IdViewType& element_import_ids,
                             const std::vector<int>& neighbor_ranks )
    {
        // Use the default execution space.
        return createWithTopology( execution_space{}, Import(),
                                   element_import_ranks, element_import_ids,
                                   neighbor_ranks );
    }

    /*!
      \brief Import rank creator. Use this when you don't know who you will
      be receiving from - only who you are importing from. This is less
      efficient than if we already knew who our neighbors were because we have
      to determine the topology of the point-to-point communication first.

      \param exec_space Kokkos execution space.

      \param element_import_ranks The source rank in the target
      decomposition of each remotely owned element in element_import_ids.
      This import rank may be any one of the listed neighbor
      ranks which can include the calling rank. The input is expected
      to be a Kokkos view in the same memory space as the communication plan.

      \param element_import_ids The local IDs of remotely owned elements that
      are to be imported. These are local IDs on the remote rank.
      element_import_ids is mapped such that element_import_ids(i) lives on
      remote rank element_import_ranks(i).

      \return A tuple of Kokkos views, where:
      Element 1: The location of each export element in the send buffer for its
      given neighbor.
      Element 2: The remote ranks this rank will export to
      Element 3: The local IDs this rank will export
      Elements 2 and 3 are mapped in the same way as element_import_ranks
      and element_import_ids

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note Unlike creating from exports, an import rank of -1 is not supported.
    */
    template <class ExecutionSpace, class RankViewType, class IdViewType>
    auto createWithoutTopology( ExecutionSpace exec_space, Import,
                                const RankViewType& element_import_ranks,
                                const IdViewType& element_import_ids )
        -> std::tuple<Kokkos::View<typename RankViewType::size_type*,
                                   typename RankViewType::memory_space>,
                      Kokkos::View<int*, typename RankViewType::memory_space>,
                      Kokkos::View<int*, typename IdViewType::memory_space>>
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        if ( element_import_ids.size() != element_import_ranks.size() )
            throw std::runtime_error( "Export ids and ranks different sizes!" );

        // Get the size of this communicator.
        int comm_size = -1;
        MPI_Comm_size( this->comm(), &comm_size );

        // Get the MPI rank we are currently on.
        int rank = -1;
        MPI_Comm_rank( this->comm(), &rank );

        // Pick an mpi tag for communication. This object has it's own
        // communication space so any mpi tag will do.
        const int mpi_tag = 1221;

        // Store which ranks I need to recieve from (i.e. send data to me)
        Kokkos::View<int*, memory_space> importing_ranks( "importing_ranks",
                                                          comm_size );
        Kokkos::deep_copy( importing_ranks, 0 );
        Kokkos::parallel_for(
            "Cabana::storeImportRanks",
            Kokkos::RangePolicy<ExecutionSpace>(
                0, element_import_ranks.extent( 0 ) ),
            KOKKOS_LAMBDA( const int i ) {
                int import_rank = element_import_ranks( i );
                Kokkos::atomic_store( &importing_ranks( import_rank ), 1 );
            } );
        Kokkos::fence();
        auto importing_ranks_h = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), importing_ranks );

        // Allreduce to count number of ranks I am communicating with
        Kokkos::View<int*, Kokkos::HostSpace> num_ranks_communicate(
            "num_ranks_communicate", comm_size );
        MPI_Allreduce( importing_ranks_h.data(), num_ranks_communicate.data(),
                       comm_size, MPI_INT, MPI_SUM, this->comm() );

        // Post that many wildcard recieves to get the number of indices I will
        // send to each rank Allocate buffers based on num_ranks_communicate
        int num_recvs = num_ranks_communicate( rank );
        Kokkos::View<int*, Kokkos::HostSpace> send_counts( "send_counts",
                                                           num_recvs );
        Kokkos::View<int*, Kokkos::HostSpace> send_to( "send_to", num_recvs );

        std::vector<MPI_Request> mpi_requests( num_recvs );
        std::vector<MPI_Status> mpi_statuses( num_recvs );

        // Receive counts for indices this process will send
        for ( int i = 0; i < num_recvs; i++ )
        {
            MPI_Irecv( &send_counts( i ), 1, MPI_INT, MPI_ANY_SOURCE, mpi_tag,
                       this->comm(), &mpi_requests[i] );
        }

        // Count the number of imports this rank needs from other ranks. Keep
        // track of which slot we get in our neighbor's send buffer?
        auto counts_and_ids = Impl::countSendsAndCreateSteering(
            exec_space, element_import_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // Copy the counts to the host.
        auto neighbor_counts_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), counts_and_ids.first );

        // Clear vectors before we use them
        this->_neighbors.clear();
        this->_num_export.clear();
        this->_num_import.clear();

        for ( std::size_t i = 0; i < neighbor_counts_host.extent( 0 ); i++ )
        {
            if ( neighbor_counts_host( i ) != 0 )
            {
                // Send counts of needed indices
                MPI_Send( &neighbor_counts_host( i ), 1, MPI_INT, i, mpi_tag,
                          this->comm() );

                // Store we are importing this count from this rank
                this->_neighbors.push_back( i );
                this->_num_import.push_back( neighbor_counts_host( i ) );
            }
        }
        // Assign all exports to zero
        this->_num_export.assign( this->_num_import.size(), 0 );

        // Wait for all count exchanges to complete
        const int ec0 =
            MPI_Waitall( num_recvs, mpi_requests.data(), mpi_statuses.data() );
        if ( MPI_SUCCESS != ec0 )
            throw std::logic_error( "Failed MPI Communication" );

        // Save ranks we got messages from and track total messages to size
        // buffers
        this->_total_num_export = 0;
        for ( int i = 0; i < num_recvs; i++ )
        {
            send_to( i ) = mpi_statuses[i].MPI_SOURCE;
            this->_total_num_export += send_counts( i );
        }

        // Extract the export ranks and number of exports and then flag the
        // send ranks.
        for ( int r = 0; r < num_recvs; ++r )
        {
            int export_to = send_to( r );
            if ( export_to > -1 )
            {
                // See if the neighbor we are exporting to is someone we are
                // also importing from
                auto found_neighbor =
                    std::find( this->_neighbors.begin(), this->_neighbors.end(),
                               export_to );

                // If this is a new neighbor (i.e. someone we are not importing
                // from) record this.
                if ( found_neighbor == std::end( this->_neighbors ) )
                {
                    this->_neighbors.push_back( export_to );
                    this->_num_import.push_back( 0 );
                    this->_num_export.push_back( send_counts( r ) );
                }

                // Otherwise if we are already importing from this neighbor that
                // means we already have a neighbor/import entry. Just assign
                // the export entry for that neighbor.
                else
                {
                    auto n = std::distance( this->_neighbors.begin(),
                                            found_neighbor );
                    this->_num_export[n] = send_counts( r );
                }
            }
            else
            {
                // This block should never be reached as
                // mpi_statuses[i].MPI_SOURCE will never be less than 0.
                throw std::runtime_error(
                    "CommunicationPlan::createFromImportsOnly: "
                    "mpi_statuses[i].MPI_SOURCE returned a value >= -1" );
            }
        }
        // If we are sending to ourself put that one first in the neighbor
        // list and assign the number of exports to be the number of imports.
        for ( std::size_t n = 0; n < this->_neighbors.size(); ++n )
            if ( this->_neighbors[n] == rank )
            {
                std::swap( this->_neighbors[n], this->_neighbors[0] );
                std::swap( this->_num_export[n], this->_num_export[0] );
                std::swap( this->_num_import[n], this->_num_import[0] );
                this->_num_export[0] = this->_num_import[0];
                break;
            }

        // Total number of imports and exports are now known
        this->_total_num_import = element_import_ranks.extent( 0 );
        this->_num_export_element = this->_total_num_export;

        // Post receives to get the indices other processes are requesting
        // i.e. our export indices
        Kokkos::View<int*, memory_space> export_indices(
            "export_indices", this->_total_num_export );
        std::size_t idx = 0;
        mpi_requests.clear();
        mpi_statuses.clear();
        int num_messages =
            this->_total_num_export + element_import_ranks.extent( 0 );
        mpi_requests.resize( num_messages );
        mpi_statuses.resize( num_messages );

        // Increment the mpi_tag for this round of messages to ensure messages
        // are processed in the correct order from the previous round of Isends
        // and Irecvs.
        for ( int i = 0; i < num_recvs; i++ )
        {
            for ( int j = 0; j < send_counts( i ); j++ )
            {
                MPI_Irecv( export_indices.data() + idx, 1, MPI_INT,
                           send_to( i ), mpi_tag + 1, this->comm(),
                           &mpi_requests[idx] );
                idx++;
            }
        }

        // Send the indices we need
        for ( std::size_t i = 0; i < element_import_ranks.extent( 0 ); i++ )
        {
            MPI_Isend( element_import_ids.data() + i, 1, MPI_INT,
                       *( element_import_ranks.data() + i ), mpi_tag + 1,
                       this->comm(), &mpi_requests[idx++] );
        }

        // Wait for all count exchanges to complete
        const int ec1 = MPI_Waitall( num_messages, mpi_requests.data(),
                                     mpi_statuses.data() );
        if ( MPI_SUCCESS != ec1 )
            throw std::logic_error( "Failed MPI Communication" );

        // Now, build the export steering
        // Export rank in mpi_statuses[i].MPI_SOURCE
        // Export ID in export_indices(i)
        Kokkos::View<int*, Kokkos::HostSpace> element_export_ranks_h(
            "element_export_ranks_h", this->_total_num_export );
        for ( std::size_t i = 0; i < this->_total_num_export; i++ )
        {
            element_export_ranks_h[i] = mpi_statuses[i].MPI_SOURCE;
        }
        auto element_export_ranks = Kokkos::create_mirror_view_and_copy(
            memory_space(), element_export_ranks_h );

        auto counts_and_ids2 = Impl::countSendsAndCreateSteering(
            exec_space, element_export_ranks, comm_size,
            typename Impl::CountSendsAndCreateSteeringAlgorithm<
                ExecutionSpace>::type() );

        // A barrier is needed because of the use of wildcard receives. This
        // avoids successive calls interfering with each other.
        MPI_Barrier( this->comm() );

        return std::tuple{ counts_and_ids2.second, element_export_ranks,
                           export_indices };
    }

    /*!
      \brief Import rank creator. Use this when you don't know who you will
      be receiving from - only who you are importing from. This is less
      efficient than if we already knew who our neighbors were because we have
      to determine the topology of the point-to-point communication first.

      \param element_import_ranks The source rank in the target
      decomposition of each remotely owned element in element_import_ids.
      This import rank may be any one of the listed neighbor
      ranks which can include the calling rank. The input is expected
      to be a Kokkos view in the same memory space as the communication plan.

      \param element_import_ids The local IDs of remotely owned elements that
      are to be imported. These are local IDs on the remote rank.
      element_import_ids is mapped such that element_import_ids(i) lives on
      remote rank element_import_ranks(i).

      \return A tuple of Kokkos views, where:
      Element 1: The location of each export element in the send buffer for its
      given neighbor.
      Element 2: The remote ranks this rank will export to
      Element 3: The local IDs this rank will export
      Elements 2 and 3 are mapped in the same way as element_import_ranks
      and element_import_ids

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.

      \note Unlike creating from exports, an import rank of -1 is not supported.
    */
    template <class RankViewType, class IdViewType>
    auto createWithoutTopology( Import,
                                const RankViewType& element_import_ranks,
                                const IdViewType& element_import_ids )
    {
        // Use the default execution space.
        return createWithoutTopology( execution_space{}, Import(),
                                      element_import_ranks,
                                      element_import_ids );
    }
};

/*!
  \brief Store AoSoA send/receive buffers. Mpi variant.
*/
template <class CommPlanType, class CommDataType>
class CommunicationData<CommPlanType, CommDataType, Mpi>
    : public CommunicationDataBase<CommPlanType, CommDataType>
{
  protected:
    using typename CommunicationDataBase<CommPlanType,
                                         CommDataType>::particle_data_type;
    /*!
      \param comm_plan The communication plan.
      \param particles The particle data (either AoSoA or slice).
      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    CommunicationData( const CommPlanType& comm_plan,
                       const particle_data_type& particles,
                       const double overallocation = 1.0 )
        : CommunicationDataBase<CommPlanType, CommDataType>(
              comm_plan, particles, overallocation )
    {
    }
};

} // end namespace Cabana

#endif // end CABANA_COMMUNICATIONPLAN_MPI_HPP
