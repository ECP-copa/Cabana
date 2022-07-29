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
  \file Cabana_Distributor.hpp
  \brief Multi-node particle redistribution
*/
#ifndef CABANA_DISTRIBUTOR_HPP
#define CABANA_DISTRIBUTOR_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_CommunicationPlan.hpp>
#include <Cabana_Slice.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <exception>
#include <vector>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \brief A communication plan for migrating data from one uniquely-owned
  decomposition to another uniquely owned decomposition.

  \tparam DeviceType Device type for which the data for this class will be
  allocated and where parallel compuations will be executed.

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
template <class DeviceType>
class Distributor : public CommunicationPlan<DeviceType>
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
        : CommunicationPlan<DeviceType>( comm )
    {
        setRank();
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
        : CommunicationPlan<DeviceType>( comm )
    {
        setRank();
        auto neighbor_ids = this->createFromExportsOnly( element_export_ranks );
        this->createExportSteering( neighbor_ids, element_export_ranks );
    }

    //! Total migrate send size for this rank.
    auto totalSend() const { return this->totalNumExport() - numStay(); }
    //! Total migrate receive size for this rank.
    auto totalReceive() const { return this->totalNumImport(); }

    //! Total migrate size staying on the current rank.
    auto numStay() const
    {
        // Calculate the number of elements that are staying on this rank and
        // therefore can be directly copied. If any of the neighbor ranks are
        // this rank it will be stored in first position (i.e. the first
        // neighbor in the local list is always yourself if you are sending to
        // yourself).
        return ( this->numNeighbor() > 0 &&
                 this->neighborRank( 0 ) == _my_rank )
                   ? this->numExport( 0 )
                   : 0;
    }

    //! Set the current MPI rank.
    void setRank()
    {
        _my_rank = -1;
        MPI_Comm_rank( this->comm(), &_my_rank );
    }
    //! Get the current MPI rank.
    auto getRank() { return _my_rank; }

  private:
    int _my_rank;
};

//---------------------------------------------------------------------------//
//! \cond Impl
template <typename>
struct is_distributor_impl : public std::false_type
{
};

template <typename DeviceType>
struct is_distributor_impl<Distributor<DeviceType>> : public std::true_type
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
  \brief Ensure the particle size matches the distributor size.

  \param distributor The distributor that will be used for the migrate. Used to
  query import and export sizes.

  \param particles The particle data (either AoSoA or slice). Used to query the
  total size.
*/
template <class DistributorType, class ParticleData>
bool distributorCheckValidSize(
    const DistributorType& distributor, const ParticleData& particles,
    typename std::enable_if<( is_distributor<DistributorType>::value ),
                            int>::type* = 0 )
{
    return ( particles.size() == distributor.exportSize() );
}
/*!
  \brief Ensure the particle size matches the distributor size.

  \param distributor The distributor that will be used for the migrate. Used to
  query import and export sizes.

  \param particles The particle data (either AoSoA or slice). Used to query the
  total size.
*/
template <class DistributorType, class ParticleData>
bool distributorCheckValidSize(
    const DistributorType& distributor, const ParticleData& src,
    const ParticleData& dst,
    typename std::enable_if<( is_distributor<DistributorType>::value ),
                            int>::type* = 0 )
{
    return ( distributorCheckValidSize( distributor, src ) &&
             dst.size() == distributor.totalNumImport() );
}

//---------------------------------------------------------------------------//
template <class DistributorType, class AoSoAType, class SFINAE = void>
class Migrate;

/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. AoSoA version.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.
*/
template <class DistributorType, class AoSoAType>
class Migrate<DistributorType, AoSoAType,
              typename std::enable_if<is_aosoa<AoSoAType>::value>::type>
    : public CommunicationData<DistributorType,
                               CommunicationDataAoSoA<AoSoAType>>
{
  public:
    static_assert( is_distributor<DistributorType>::value, "" );

    //! Base type.
    using base_type =
        CommunicationData<DistributorType, CommunicationDataAoSoA<AoSoAType>>;
    //! Communication plan type (Distributor)
    using plan_type = typename base_type::plan_type;
    //! Kokkos execution space.
    using execution_space = typename base_type::execution_space;
    //! Kokkos memory space.
    using memory_space = typename base_type::memory_space;
    //! Communication data type.
    using data_type = typename base_type::data_type;
    //! Communication buffer type.
    using buffer_type = typename base_type::buffer_type;

    /*!
      \param distributor The Distributor to be used for the migrate.

      \param aosoa Upon input, must have the same number of elements as the
      inputs used to construct the distributor. At output, it will be the same
      size as the number of import elements on this rank provided by the
      distributor. Before using this function, consider reserving enough memory
      in the data structure so reallocating is not necessary.

      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    Migrate( DistributorType distributor, AoSoAType aosoa,
             const double overallocation = 1.0 )
        : base_type( distributor, overallocation )
    {
        update( _distributor, aosoa );
    }

    /*!
      \param distributor The Distributor to be used for the migrate.

      \param src The AoSoA containing the data to be migrated. Must have the
      same number of elements as the inputs used to construct the distributor.

      \param dst The AoSoA to which the migrated data will be written. Must be
      the same size as the number of imports given by the distributor on this
      rank. Call totalNumImport() on the distributor to get this size value.

      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    Migrate( DistributorType distributor, AoSoAType src, AoSoAType dst,
             const double overallocation = 1.0 )
        : base_type( distributor, overallocation )
    {
        update( _distributor, src, dst );
    }

    /*!
      \brief Perform the migrate operation. In-place AoSoA version.

      Synchronously migrate data between two different decompositions using the
      distributor forward communication plan. Single AoSoA version that will
      resize in-place. Note that resizing does not necessarily allocate more
      memory. The AoSoA memory will only increase if not enough has already been
      reserved/allocated for the needed number of elements.

      \param aosoa The AoSoA containing the data to be migrated. Upon input,
      must have the same number of elements as the inputs used to construct the
      distributor. At output, it will be the same size as the number of import
      elements on this rank provided by the distributor. Before using this
      function, consider reserving enough memory in the data structure so
      reallocating is not necessary.
    */
    void apply( AoSoAType& aosoa ) override
    {
        migrate( aosoa, aosoa );

        // If the destination decomposition is smaller than the source
        // decomposition resize after we have moved the data.
        bool dst_is_bigger =
            ( _distributor.totalNumImport() > _distributor.exportSize() );
        if ( !dst_is_bigger )
            aosoa.resize( _distributor.totalNumImport() );
    }

    /*!
      \brief Perform the migrate operation. Multiple AoSoA version.

      \param src The AoSoA containing the data to be migrated. Must have the
      same number of elements as the inputs used to construct the distributor.

      \param dst The AoSoA to which the migrated data will be written. Must be
      the same size as the number of imports given by the distributor on this
      rank. Call totalNumImport() on the distributor to get this size value.
    */
    void apply( const AoSoAType& src, AoSoAType& dst ) override
    {
        migrate( src, dst );
    }

    /*!
      \brief Update the distributor and AoSoA data for migration.

      \param distributor The Distributor to be used for the migrate.
      \param aosoa The AoSoA on which to perform the migrate.
      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    void update( const DistributorType& distributor, AoSoAType& aosoa,
                 const double overallocation )
    {
        // Check that the AoSoA is the right size.
        if ( !distributorCheckValidSize( distributor, aosoa ) )
            throw std::runtime_error( "AoSoA is the wrong size for migrate!" );

        // Determine if the source of destination decomposition has more data on
        // this rank.
        bool dst_is_bigger =
            ( distributor.totalNumImport() > distributor.exportSize() );

        // If the destination decomposition is bigger than the source
        // decomposition resize now so we have enough space to do the operation.
        if ( dst_is_bigger )
            aosoa.resize( distributor.totalNumImport() );

        this->updateImpl( distributor, aosoa, distributor.totalSend(),
                          distributor.totalReceive(), overallocation );
    }
    /*!
      \brief Update the distributor and AoSoA data for migration.

      \param distributor The Distributor to be used for the migrate.
      \param aosoa The AoSoA on which to perform the migrate.
    */
    void update( const DistributorType& distributor, AoSoAType& aosoa )
    {
        // Check that the AoSoA is the right size.
        if ( !distributorCheckValidSize( distributor, aosoa ) )
            throw std::runtime_error( "AoSoA is the wrong size for migrate!" );

        // Determine if the source of destination decomposition has more data on
        // this rank.
        bool dst_is_bigger =
            ( distributor.totalNumImport() > distributor.exportSize() );

        // If the destination decomposition is bigger than the source
        // decomposition resize now so we have enough space to do the operation.
        if ( dst_is_bigger )
            aosoa.resize( distributor.totalNumImport() );

        this->updateImpl( distributor, aosoa, distributor.totalSend(),
                          distributor.totalReceive() );
    }

    /*!
      \brief Update the distributor and AoSoA data for migration.

      \param distributor The Distributor to be used for the migrate.
      \param src The AoSoA containing the data to be migrated. Must have the
      same number of elements as the inputs used to construct the distributor.
      \param dst The AoSoA to which the migrated data will be written. Must be
      the same size as the number of imports given by the distributor on this
      rank.
      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    void update( const DistributorType& distributor, const AoSoAType& src,
                 AoSoAType& dst, const double overallocation )
    {
        // Check that src and dst are the right size.
        if ( !distributorCheckValidSize( distributor, src, dst ) )
            throw std::runtime_error( "AoSoA is the wrong size for migrate!" );

        this->updateImpl( distributor, src, distributor.totalSend(),
                          distributor.totalReceive(), overallocation );
    }
    /*!
      \brief Update the distributor and AoSoA data for migration.

      \param distributor The Distributor to be used for the migrate.
      \param src The AoSoA containing the data to be migrated. Must have the
      same number of elements as the inputs used to construct the distributor.
      \param dst The AoSoA to which the migrated data will be written. Must be
      the same size as the number of imports given by the distributor on this
      rank.
    */
    void update( const DistributorType& distributor, const AoSoAType& src,
                 AoSoAType& dst )
    {
        // Check that src and dst are the right size.
        if ( !distributorCheckValidSize( distributor, src, dst ) )
            throw std::runtime_error( "AoSoA is the wrong size for migrate!" );

        this->updateImpl( distributor, src, distributor.totalSend(),
                          distributor.totalReceive() );
    }

  private:
    // Implementation of the migration.
    void migrate( const AoSoAType& src, AoSoAType& dst )
    {
        // Get the buffers (local copies for lambdas below).
        auto send_buffer = this->getSendBuffer();
        auto recv_buffer = this->getReceiveBuffer();

        // Get the number of neighbors.
        int num_n = _distributor.numNeighbor();

        // Number of elements that are staying on this rank.
        auto num_stay = _distributor.numStay();

        // Get the steering vector for the sends.
        auto steering = _distributor.getExportSteering();

        // Gather the exports from the source AoSoA into the tuple-contiguous
        // send buffer or the receive buffer if the data is staying. We know
        // that the steering vector is ordered such that the data staying on
        // this rank comes first.
        auto build_send_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
        {
            auto tpl = src.getTuple( steering( i ) );
            if ( i < num_stay )
                recv_buffer( i ) = tpl;
            else
                send_buffer( i - num_stay ) = tpl;
        };
        Kokkos::parallel_for( "Cabana::migrate::build_send_buffer",
                              _send_policy, build_send_buffer_func );
        Kokkos::fence();

        // The distributor has its own communication space so choose any tag.
        const int mpi_tag = 1234;

        // Post non-blocking receives.
        std::vector<MPI_Request> requests;
        requests.reserve( num_n );
        std::pair<std::size_t, std::size_t> recv_range = { 0, 0 };
        for ( int n = 0; n < num_n; ++n )
        {
            recv_range.second = recv_range.first + _distributor.numImport( n );

            if ( ( _distributor.numImport( n ) > 0 ) &&
                 ( _distributor.neighborRank( n ) != _my_rank ) )
            {
                auto recv_subview = Kokkos::subview( recv_buffer, recv_range );

                requests.push_back( MPI_Request() );

                MPI_Irecv( recv_subview.data(),
                           recv_subview.size() * sizeof( data_type ), MPI_BYTE,
                           _distributor.neighborRank( n ), mpi_tag,
                           _distributor.comm(), &( requests.back() ) );
            }

            recv_range.first = recv_range.second;
        }

        // Do blocking sends.
        std::pair<std::size_t, std::size_t> send_range = { 0, 0 };
        for ( int n = 0; n < num_n; ++n )
        {
            if ( ( _distributor.numExport( n ) > 0 ) &&
                 ( _distributor.neighborRank( n ) != _my_rank ) )
            {
                send_range.second =
                    send_range.first + _distributor.numExport( n );

                auto send_subview = Kokkos::subview( send_buffer, send_range );

                MPI_Send( send_subview.data(),
                          send_subview.size() * sizeof( data_type ), MPI_BYTE,
                          _distributor.neighborRank( n ), mpi_tag,
                          _distributor.comm() );

                send_range.first = send_range.second;
            }
        }

        // Wait on non-blocking receives.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        if ( MPI_SUCCESS != ec )
            throw std::logic_error( "Failed MPI Communication" );

        // Extract the receive buffer into the destination AoSoA.
        auto extract_recv_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
        {
            dst.setTuple( i, recv_buffer( i ) );
        };
        Kokkos::parallel_for( "Cabana::migrate::extract_recv_buffer",
                              _recv_policy, extract_recv_buffer_func );
        Kokkos::fence();

        // Barrier before completing to ensure synchronization.
        MPI_Barrier( _distributor.comm() );
    }

    plan_type _distributor = base_type::_comm_plan;
    using base_type::_recv_policy;
    using base_type::_send_policy;

    int _my_rank;
};

/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Slice version.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.
*/
template <class DistributorType, class SliceType>
class Migrate<DistributorType, SliceType,
              typename std::enable_if<is_slice<SliceType>::value>::type>
    : CommunicationData<DistributorType, CommunicationDataSlice<SliceType>>
{
  public:
    static_assert( is_distributor<DistributorType>::value, "" );

    //! Base type.
    using base_type =
        CommunicationData<DistributorType, CommunicationDataSlice<SliceType>>;
    //! Communication plan type (Distributor)
    using plan_type = typename base_type::plan_type;
    //! Kokkos execution space.
    using execution_space = typename base_type::execution_space;
    //! Kokkos memory space.
    using memory_space = typename base_type::memory_space;
    //! Communication data type.
    using data_type = typename base_type::data_type;
    //! Communication buffer type.
    using buffer_type = typename base_type::buffer_type;

    /*!
      \param distributor The Distributor to be used for the migrate.
      \param src The slice containing the data to be migrated.
      \param dst The slice to which the migrated data will be written.
      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    Migrate( const DistributorType& distributor, const SliceType& src,
             SliceType& dst, const double overallocation = 1.0 )
        : base_type( distributor, overallocation )
    {
        _my_rank = _distributor.getRank();
        update( _distributor, src, dst );
    }

    /*!
      \brief Perform the migrate operation.

      \param src The slice containing the data to be migrated.
      \param dst The slice to which the migrated data will be written.
    */
    void apply( const SliceType& src, SliceType& dst ) override
    {
        // Get the buffers (local copies for lambdas below).
        auto send_buffer = this->getSendBuffer();
        auto recv_buffer = this->getReceiveBuffer();

        // Get the number of components in the slices.
        auto num_comp = this->getSliceComponents( src );

        // Get the raw slice data.
        auto src_data = src.data();
        auto dst_data = dst.data();

        // Get the number of neighbors.
        int num_n = _distributor.numNeighbor();

        // Number of elements that are staying on this rank.
        auto num_stay = _distributor.numStay();

        // Get the steering vector for the sends.
        auto steering = _distributor.getExportSteering();

        // Gather from the source Slice into the contiguous send buffer or,
        // if it is part of the local copy, put it directly in the destination
        // Slice.
        auto build_send_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
        {
            auto s_src = SliceType::index_type::s( steering( i ) );
            auto a_src = SliceType::index_type::a( steering( i ) );
            std::size_t src_offset = s_src * src.stride( 0 ) + a_src;
            if ( i < num_stay )
                for ( std::size_t n = 0; n < num_comp; ++n )
                    recv_buffer( i, n ) =
                        src_data[src_offset + n * SliceType::vector_length];
            else
                for ( std::size_t n = 0; n < num_comp; ++n )
                    send_buffer( i - num_stay, n ) =
                        src_data[src_offset + n * SliceType::vector_length];
        };
        Kokkos::parallel_for( "Cabana::migrate::build_send_buffer",
                              _send_policy, build_send_buffer_func );
        Kokkos::fence();

        // The distributor has its own communication space so choose any tag.
        const int mpi_tag = 1234;

        // Post non-blocking receives.
        std::vector<MPI_Request> requests;
        requests.reserve( num_n );
        std::pair<std::size_t, std::size_t> recv_range = { 0, 0 };
        for ( int n = 0; n < num_n; ++n )
        {
            recv_range.second = recv_range.first + _distributor.numImport( n );

            if ( ( _distributor.numImport( n ) > 0 ) &&
                 ( _distributor.neighborRank( n ) != _my_rank ) )
            {
                auto recv_subview =
                    Kokkos::subview( recv_buffer, recv_range, Kokkos::ALL );

                requests.push_back( MPI_Request() );

                MPI_Irecv( recv_subview.data(),
                           recv_subview.size() * sizeof( data_type ), MPI_BYTE,
                           _distributor.neighborRank( n ), mpi_tag,
                           _distributor.comm(), &( requests.back() ) );
            }

            recv_range.first = recv_range.second;
        }

        // Do blocking sends.
        std::pair<std::size_t, std::size_t> send_range = { 0, 0 };
        for ( int n = 0; n < num_n; ++n )
        {
            if ( ( _distributor.numExport( n ) > 0 ) &&
                 ( _distributor.neighborRank( n ) != _my_rank ) )
            {
                send_range.second =
                    send_range.first + _distributor.numExport( n );

                auto send_subview =
                    Kokkos::subview( send_buffer, send_range, Kokkos::ALL );

                MPI_Send( send_subview.data(),
                          send_subview.size() * sizeof( data_type ), MPI_BYTE,
                          _distributor.neighborRank( n ), mpi_tag,
                          _distributor.comm() );

                send_range.first = send_range.second;
            }
        }

        // Wait on non-blocking receives.
        std::vector<MPI_Status> status( requests.size() );
        const int ec =
            MPI_Waitall( requests.size(), requests.data(), status.data() );
        if ( MPI_SUCCESS != ec )
            throw std::logic_error( "Failed MPI Communication" );

        // Extract the data from the receive buffer into the destination Slice.
        auto extract_recv_buffer_func = KOKKOS_LAMBDA( const std::size_t i )
        {
            auto s = SliceType::index_type::s( i );
            auto a = SliceType::index_type::a( i );
            std::size_t dst_offset = s * dst.stride( 0 ) + a;
            for ( std::size_t n = 0; n < num_comp; ++n )
                dst_data[dst_offset + n * SliceType::vector_length] =
                    recv_buffer( i, n );
        };
        Kokkos::parallel_for( "Cabana::migrate::extract_recv_buffer",
                              _recv_policy, extract_recv_buffer_func );
        Kokkos::fence();

        // Barrier before completing to ensure synchronization.
        MPI_Barrier( _distributor.comm() );
    }

    //! \cond Impl
    void apply( SliceType& ) override
    {
        // This should never be called. It exists to override the base.
        throw std::runtime_error(
            "In-place slice migrate is not implemented!" );
    }
    //! \endcond

    /*!
      \brief Update the distributor and slice data for migrate.

      \param distributor The Distributor to be used for the migrate.
      \param src The slice containing the data to be migrated.
      \param dst The slice to which the migrated data will be written.
      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    void update( const DistributorType& distributor, const SliceType& src,
                 SliceType& dst, const double overallocation )
    {
        // Check that src and dst are the right size.
        if ( !distributorCheckValidSize( distributor, src, dst ) )
            throw std::runtime_error( "AoSoA is the wrong size for migrate!" );

        this->updateImpl( distributor, src, distributor.totalSend(),
                          distributor.totalReceive(), overallocation );
    }
    /*!
      \brief Update the distributor and slice data for migrate.

      \param distributor The Distributor to be used for the migrate.
      \param src The slice containing the data to be migrated.
      \param dst The slice to which the migrated data will be written.
    */
    void update( const DistributorType& distributor, const SliceType& src,
                 SliceType& dst )
    {
        // Check that src and dst are the right size.
        if ( !distributorCheckValidSize( distributor, src, dst ) )
            throw std::runtime_error( "AoSoA is the wrong size for migrate!" );

        this->updateImpl( distributor, src, distributor.totalSend(),
                          distributor.totalReceive() );
    }

  private:
    plan_type _distributor = base_type::_comm_plan;
    using base_type::_recv_policy;
    using base_type::_send_policy;

    int _my_rank;
};

//---------------------------------------------------------------------------//
/*!
  \brief Create the migrate.

  \param distributor The distributor to use for the migrate.
  \param data The data on which to perform the migrate.
  \param overallocation An optional factor to keep extra space in the buffers to
  avoid frequent resizing.
*/
template <class DistributorType, class ParticleDataType>
auto createMigrate( DistributorType distributor, ParticleDataType data,
                    const double overallocation = 1.0 )
{
    return Migrate<DistributorType, ParticleDataType>( distributor, data,
                                                       overallocation );
}

/*!
  \brief Create the migrate.

  \param distributor The distributor to use for the migrate.
  \param src The AoSoA containing the data to be migrated.
  \param dst The AoSoA to which the migrated data will be written.
  \param overallocation An optional factor to keep extra space in the buffers to
  avoid frequent resizing.
*/
template <class DistributorType, class ParticleDataType>
auto createMigrate( const DistributorType& distributor,
                    const ParticleDataType& src, ParticleDataType& dst,
                    const double overallocation = 1.0 )
{
    return Migrate<DistributorType, ParticleDataType>( distributor, src, dst,
                                                       overallocation );
}

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Multiple AoSoA version.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \note This routine allocates send and receive buffers internally. This is
  often not performant due to frequent buffer reallocations - consider creating
  and reusing Migrate instead.

  \param distributor The distributor to use for the migration.

  \param src The AoSoA containing the data to be migrated. Must have the same
  number of elements as the inputs used to construct the distributor.

  \param dst The AoSoA to which the migrated data will be written. Must be the
  same size as the number of imports given by the distributor on this
  rank. Call totalNumImport() on the distributor to get this size value.
*/
template <class DistributorType, class AoSoAType>
void migrate(
    const DistributorType& distributor, const AoSoAType& src, AoSoAType& dst,
    typename std::enable_if<( is_distributor<DistributorType>::value &&
                              is_aosoa<AoSoAType>::value ),
                            int>::type* = 0 )
{
    auto migrate = createMigrate( distributor, src, dst );
    migrate.apply( src, dst );
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

  \note This routine allocates send and receive buffers internally. This is
  often not performant due to frequent buffer reallocations - consider creating
  and reusing Migrate instead.

  \param distributor The distributor to use for the migration.

  \param aosoa The AoSoA containing the data to be migrated. Upon input, must
  have the same number of elements as the inputs used to construct the
  distributor. At output, it will be the same size as the number of import
  elements on this rank provided by the distributor. Before using this
  function, consider reserving enough memory in the data structure so
  reallocating is not necessary.
*/
template <class DistributorType, class AoSoAType>
void migrate(
    const DistributorType& distributor, AoSoAType& aosoa,
    typename std::enable_if<( is_distributor<DistributorType>::value &&
                              is_aosoa<AoSoAType>::value ),
                            int>::type* = 0 )
{
    auto migrate = createMigrate( distributor, aosoa );
    migrate.apply( aosoa );
}

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously migrate data between two different decompositions using
  the distributor forward communication plan. Slice version. The user can do
  this in-place with the same slice but they will need to manage the resizing
  themselves as we can't resize slices.

  Migrate moves all data to a new distribution that is uniquely owned - each
  element will only have a single destination rank.

  \param distributor The distributor to use for the migration.

  \param src The slice containing the data to be migrated. Must have the same
  number of elements as the inputs used to construct the distributor.

  \param dst The slice to which the migrated data will be written. Must be the
  same size as the number of imports given by the distributor on this
  rank. Call totalNumImport() on the distributor to get this size value.
*/
template <class DistributorType, class SliceType>
void migrate(
    const DistributorType& distributor, const SliceType& src, SliceType& dst,
    typename std::enable_if<( is_distributor<DistributorType>::value &&
                              is_slice<SliceType>::value ),
                            int>::type* = 0 )
{
    auto migrate = createMigrate( distributor, src, dst );
    migrate.apply( src, dst );
}
//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_DISTRIBUTOR_HPP
