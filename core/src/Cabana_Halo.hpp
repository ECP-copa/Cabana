/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_HALO_HPP
#define CABANA_HALO_HPP

#include <Cabana_CommunicationPlan.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <vector>
#include <exception>
#include <cassert>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class Halo

  \brief Halo communication plan for scattering and gathering of ghosted
  data.

  \tparam DeviceType Device type for which the data for this class will be
  allocated and where parallel execution occurs.

  The halo allows for scatter and gather operations between locally-owned and
  ghosted data. All data in the Halo (e.g. export and import data) is from the
  point of view of the forward *GATHER* operation such that, for example, the
  number of exports is the number of exports in the gather and the number of
  imports is the number of imports in the gather. The reverse *SCATTER*
  operation sends the ghosted data back the the uniquely-owned decomposition
  and resolves collisions. Based on input for the forward communication plan
  (where local data will be sent) the local number of ghosts is computed. Some
  nomenclature:

  Export - the local data we uniquely own that we will send to other ranks for
  those ranks to be used as ghosts. Export is used in the context of the
  forward communication plan (the gather).

  Import - the ghost data that we get from other ranks. The rank we get a
  ghost from is the unique owner of that data. Import is used in the context
  of the forward communication plan (the gather).
*/
template<class DeviceType>
class Halo : public CommunicationPlan<DeviceType>
{
  public:

    /*!
      \brief Neighbor and export rank constructor. Use this when you already
      know which ranks neighbor each other (i.e. every rank already knows who
      they will be exporting to and receiving from) as it will be more
      efficient. In this case you already know the topology of the
      point-to-point communication but not how much data to send and receive
      from the neighbors.

      \tparam IdViewType The container type for the export element ids. This
      container type can be either a Kokkos View or a Cabana Slice.

      \tparam RankViewType The container type for the export element
      ranks. This container type can be either a Kokkos View or a Cabana
      Slice.

      \param comm The MPI communicator over which the halo is defined.

      \param num_local The number of locally-owned elements on this rank.

      \param element_export_ids The local ids of the elements that will be
      exported to other ranks to be used as ghosts. Element ids may be
      repeated in this list if they are sent to multiple destinations. Must be
      the same length as element_export_ranks. The input is expected to be a
      Kokkos view or Cabana slice in the same memory space as the
      communication plan.

      \param element_export_ranks The ranks to which we will send each element
      in element_export_ids. In this case each rank must be one of the
      neighbor ranks. Must be the same length as element_export_ids. A rank is
      allowed to send to itself. The input is expected to be a Kokkos view or
      Cabana slice in the same memory space as the communication plan.

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. The elements in this list must be unique.

      \param mpi_tag The MPI tag to use for non-blocking communication in the
      communication plan generation.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.
    */
    template<class IdViewType, class RankViewType>
    Halo( MPI_Comm comm,
          const std::size_t num_local,
          const IdViewType& element_export_ids,
          const RankViewType& element_export_ranks,
          const std::vector<int>& neighbor_ranks,
          const int mpi_tag = 1221 )
        : CommunicationPlan<DeviceType>( comm )
        , _num_local( num_local )
    {
        if ( element_export_ids.size() != element_export_ranks.size() )
            throw std::runtime_error("Export ids and ranks different sizes!");

        auto neighbor_ids = this->createFromExportsAndTopology(
            element_export_ranks, neighbor_ranks, mpi_tag );
        this->createExportSteering(
            neighbor_ids, element_export_ranks, element_export_ids );
    }

    /*!
      \brief Export rank constructor. Use this when you don't know who you
      will receiving from - only who you are sending to. This is less
      efficient than if we already knew who our neighbors were because we have
      to determine the topology of the point-to-point communication first.

      \tparam IdViewType The container type for the export element ids. This
      container type can be either a Kokkos View or a Cabana Slice.

      \tparam RankViewType The container type for the export element
      ranks. This container type can be either a Kokkos View or a Cabana
      Slice.

      \param comm The MPI communicator over which the halo is defined.

      \param num_local The number of locally-owned elements on this rank.

      \param element_export_ids The local ids of the elements that will be
      sent to other ranks to be used as ghosts. Element ids may be repeated in
      this list if they are sent to multiple destinations. Must be the same
      length as element_export_ranks. The input is expected to be a Kokkos
      view or Cabana slice in the same memory space as the communication plan.

      \param element_export_ranks The ranks to which we will export each element
      in element_export_ids. Must be the same length as
      element_export_ids. The neighbor ranks will be determined from this
      list. A rank is allowed to send to itself. The input is expected to be a
      Kokkos view or Cabana slice in the same memory space as the
      communication plan.

      \param mpi_tag The MPI tag to use for non-blocking communication in the
      communication plan generation.

      \note Calling this function completely updates the state of this object
      and invalidates the previous state.
    */
    template<class IdViewType, class RankViewType>
    Halo( MPI_Comm comm,
          const std::size_t num_local,
          const IdViewType& element_export_ids,
          const RankViewType& element_export_ranks,
          const int mpi_tag = 1221 )
        : CommunicationPlan<DeviceType>( comm )
        , _num_local( num_local )
    {
        if ( element_export_ids.size() != element_export_ranks.size() )
            throw std::runtime_error("Export ids and ranks different sizes!");

        auto neighbor_ids =
            this->createFromExportsOnly( element_export_ranks, mpi_tag );
        this->createExportSteering(
            neighbor_ids, element_export_ranks, element_export_ids );
    }

    /*!
      \brief Get the number of elements locally owned by this rank.

      \return THe number of elements locally owned by this rank.
    */
    std::size_t numLocal() const
    { return _num_local; }

    /*!
      \brief Get the number of ghost elements this rank. Use this to resize a
      data structure for scatter/gather operations. For use with scatter
      gather, a data structure should be of size numLocal() + numGhost().

      \return The number of ghosted elements on this rank.
    */
    std::size_t numGhost() const
    { return this->totalNumImport(); }

  private:

    std::size_t _num_local;
};

//---------------------------------------------------------------------------//
// Static type checker.
template<typename >
struct is_halo : public std::false_type {};

template<typename DeviceType>
struct is_halo<Halo<DeviceType> >
    : public std::true_type {};

template<typename DeviceType>
struct is_halo<const Halo<DeviceType> >
    : public std::true_type {};

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously gather data from the local decomposition to the ghosts
  using the halo forward communication plan. AoSoA version. This is a
  uniquely-owned to multiply-owned communication.

  A gather sends data from a locally owned elements to one or many ranks on
  which they exist as ghosts. A locally owned element may be sent to as many
  ranks as desired to be used as a ghost on those ranks. The value of the
  element in the locally owned decomposition will be the value assigned to the
  element in the ghosted decomposition.

  \tparam Halo_t Halo type - must be a Halo.

  \tparam AoSoA_t AoSoA type - must be an AoSoA.

  \param halo The halo to use for the gather.

  \param aosoa The AoSoA on which to perform the gather. The AoSoA should have
  a size equivalent to halo.numGhost() + halo.numLocal(). The locally owned
  elements are expected to appear first (i.e. in the first halo.numLocal()
  elements) and the ghosted elements are expected to appear second (i.e. in
  the next halo.numGhost() elements()).

  \param mpi_tag The MPI tag to use for non-blocking communication in the
  gather. Note here that if multiple instances of this function are being
  called at once then different tags should be used in each function call to
  avoid any communication conflicts.
*/
template<class Halo_t, class AoSoA_t>
void gather( const Halo_t& halo,
             AoSoA_t& aosoa,
             int mpi_tag = 1002,
             typename std::enable_if<(is_halo<Halo_t>::value &&
                                      is_aosoa<AoSoA_t>::value),
             int>::type * = 0 )
{
    // Check that the AoSoA is the right size.
    if ( aosoa.size() != halo.numLocal() + halo.numGhost() )
        throw std::runtime_error("AoSoA is the wrong size for scatter!");

    // Allocate a send buffer.
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Halo_t::memory_space>
        send_buffer(
            Kokkos::ViewAllocateWithoutInitializing("halo_send_buffer"),
            halo.totalNumExport() );

    // Get the steering vector for the sends.
    auto steering = halo.getExportSteering();

    // Gather from the local data into a tuple-contiguous send buffer.
    auto gather_send_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            send_buffer( i ) = aosoa.getTuple( steering(i) );
        };
    Kokkos::RangePolicy<typename Halo_t::execution_space>
        gather_send_buffer_policy( 0, halo.totalNumExport() );
    Kokkos::parallel_for( "Cabana::gather::gather_send_buffer",
                          gather_send_buffer_policy,
                          gather_send_buffer_func );
    Kokkos::fence();

    // Allocate a receive buffer.
    Kokkos::View<typename AoSoA_t::tuple_type*,
                 typename Halo_t::memory_space>
        recv_buffer(
            Kokkos::ViewAllocateWithoutInitializing("halo_recv_buffer"),
            halo.totalNumImport() );

    // Post non-blocking receives.
    int num_n = halo.numNeighbor();
    std::vector<MPI_Request> requests( num_n );
    std::pair<std::size_t,std::size_t> recv_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_range.second =
            recv_range.first + halo.numImport(n);

        auto recv_subview = Kokkos::subview( recv_buffer, recv_range );

        MPI_Irecv( recv_subview.data(),
                   recv_subview.size() * sizeof(typename AoSoA_t::tuple_type),
                   MPI_BYTE,
                   halo.neighborRank(n),
                   mpi_tag,
                   halo.comm(),
                   &(requests[n]) );

        recv_range.first = recv_range.second;
    }

    // Do blocking sends.
    std::pair<std::size_t,std::size_t> send_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        send_range.second =
            send_range.first + halo.numExport(n);

        auto send_subview = Kokkos::subview( send_buffer, send_range );

        MPI_Send( send_subview.data(),
                  send_subview.size() * sizeof(typename AoSoA_t::tuple_type),
                  MPI_BYTE,
                  halo.neighborRank(n),
                  mpi_tag,
                  halo.comm() );

        send_range.first = send_range.second;
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( num_n );
    const int ec =
        MPI_Waitall( requests.size(), requests.data(), status.data() );
    assert( MPI_SUCCESS == ec );

    // Extract the receive buffer into the ghosted elements.
    std::size_t num_local = halo.numLocal();
    auto extract_recv_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            std::size_t ghost_idx = i + num_local;
            aosoa.setTuple( ghost_idx, recv_buffer(i) );
        };
    Kokkos::RangePolicy<typename Halo_t::execution_space>
        extract_recv_buffer_policy( 0, halo.totalNumImport() );
    Kokkos::parallel_for( "Cabana::gather::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( halo.comm() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously gather data from the local decomposition to the ghosts
  using the halo forward communication plan. Slice version. This is a
  uniquely-owned to multiply-owned communication.

  A gather sends data from a locally owned elements to one or many ranks on
  which they exist as ghosts. A locally owned element may be sent to as many
  ranks as desired to be used as a ghost on those ranks. The value of the
  element in the locally owned decomposition will be the value assigned to the
  element in the ghosted decomposition.

  \tparam Halo_t Halo type - must be a Halo.

  \tparam Slice_t Slice type - must be a Slice.

  \param halo The halo to use for the gather.

  \param slice The Slice on which to perform the gather. The Slice should have
  a size equivalent to halo.numGhost() + halo.numLocal(). The locally owned
  elements are expected to appear first (i.e. in the first halo.numLocal()
  elements) and the ghosted elements are expected to appear second (i.e. in
  the next halo.numGhost() elements()).

  \param mpi_tag The MPI tag to use for non-blocking communication in the
  gather. Note here that if multiple instances of this function are being
  called at once then different tags should be used in each function call to
  avoid any communication conflicts.
*/
template<class Halo_t, class Slice_t>
void gather( const Halo_t& halo,
             Slice_t& slice,
             int mpi_tag = 1002,
             typename std::enable_if<(is_halo<Halo_t>::value &&
                                      is_slice<Slice_t>::value),
             int>::type * = 0 )
{
    // Check that the Slice is the right size.
    if ( slice.size() != halo.numLocal() + halo.numGhost() )
        throw std::runtime_error("Slice is the wrong size for scatter!");

    // Get the number of components in the slice.
    std::size_t num_comp = 1;
    for ( std::size_t d = 2; d < slice.rank(); ++d )
        num_comp *= slice.extent(d);

    // Get the raw slice data.
    auto slice_data = slice.data();

    // Allocate a send buffer. Note this one is layout right so the components
    // are consecutive.
    Kokkos::View<typename Slice_t::value_type**,
                 Kokkos::LayoutRight,
                 typename Halo_t::memory_space>
        send_buffer(
            Kokkos::ViewAllocateWithoutInitializing("halo_send_buffer"),
            halo.totalNumExport(), num_comp );

    // Get the steering vector for the sends.
    auto steering = halo.getExportSteering();

    // Gather from the local data into a tuple-contiguous send buffer.
    auto gather_send_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            auto s = Slice_t::index_type::s( steering(i) );
            auto a = Slice_t::index_type::a( steering(i) );
            std::size_t slice_offset = s*slice.stride(0) + a;
            for ( std::size_t n = 0; n < num_comp; ++n )
                send_buffer( i, n ) =
                    slice_data[ slice_offset + n * Slice_t::vector_length ];
        };
    Kokkos::RangePolicy<typename Halo_t::execution_space>
        gather_send_buffer_policy( 0, halo.totalNumExport() );
    Kokkos::parallel_for( "Cabana::gather::gather_send_buffer",
                          gather_send_buffer_policy,
                          gather_send_buffer_func );
    Kokkos::fence();

    // Allocate a receive buffer. Note this one is layout right so the components
    // are consecutive.
    Kokkos::View<typename Slice_t::value_type**,
                 Kokkos::LayoutRight,
                 typename Halo_t::memory_space>
        recv_buffer(
            Kokkos::ViewAllocateWithoutInitializing("halo_recv_buffer"),
            halo.totalNumImport(), num_comp );

    // Post non-blocking receives.
    int num_n = halo.numNeighbor();
    std::vector<MPI_Request> requests( num_n );
    std::pair<std::size_t,std::size_t> recv_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_range.second = recv_range.first + halo.numImport(n);

        auto recv_subview =
            Kokkos::subview( recv_buffer, recv_range, Kokkos::ALL );

        MPI_Irecv( recv_subview.data(),
                   recv_subview.size() * sizeof(typename Slice_t::value_type),
                   MPI_BYTE,
                   halo.neighborRank(n),
                   mpi_tag,
                   halo.comm(),
                   &(requests[n]) );

        recv_range.first = recv_range.second;
    }

    // Do blocking sends.
    std::pair<std::size_t,std::size_t> send_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        send_range.second = send_range.first + halo.numExport(n);

        auto send_subview =
            Kokkos::subview( send_buffer, send_range, Kokkos::ALL );

        MPI_Send( send_subview.data(),
                  send_subview.size() * sizeof(typename Slice_t::value_type),
                  MPI_BYTE,
                  halo.neighborRank(n),
                  mpi_tag,
                  halo.comm() );

        send_range.first = send_range.second;
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( num_n );
    const int ec =
        MPI_Waitall( requests.size(), requests.data(), status.data() );
    assert( MPI_SUCCESS == ec );

    // Extract the receive buffer into the ghosted elements.
    std::size_t num_local = halo.numLocal();
    auto extract_recv_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            std::size_t ghost_idx = i + num_local;
            auto s = Slice_t::index_type::s( ghost_idx );
            auto a = Slice_t::index_type::a( ghost_idx );
            std::size_t slice_offset = s*slice.stride(0) + a;
            for ( std::size_t n = 0; n < num_comp; ++n )
                slice_data[ slice_offset + Slice_t::vector_length * n ] =
                    recv_buffer( i, n );
        };
    Kokkos::RangePolicy<typename Halo_t::execution_space>
        extract_recv_buffer_policy( 0, halo.totalNumImport() );
    Kokkos::parallel_for( "Cabana::gather::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( halo.comm() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously scatter data from the ghosts to the local decomposition
  of a slice using the halo reverse communication plan. This is a
  multiply-owned to uniquely owned communication.

  In a scatter operation results from ghosted values on other processors are
  scattered back to the owning processor of the ghost and the value associated
  with the ghost is summed into the locally owned value the ghost
  represents. If a locally owned element is ghosted on multiple ranks, then
  multiple contributions will be made to the sum, one for each rank.

  \tparam Halo_t Halo type - must be a Halo.

  \tparam Slice_t Slice type - must be a Slice.

  \param halo The halo to use for the scatter.

  \param slice The Slice on which to perform the scatter. The Slice should have
  a size equivalent to halo.numGhost() + halo.numLocal(). The locally owned
  elements are expected to appear first (i.e. in the first halo.numLocal()
  elements) and the ghosted elements are expected to appear second (i.e. in
  the next halo.numGhost() elements()).

  \param mpi_tag The MPI tag to use for non-blocking communication in the
  scatter. Note here that if multiple instances of this function are being
  called at once then different tags should be used in each function call to
  avoid any communication conflicts.
*/
template<class Halo_t, class Slice_t>
void scatter( const Halo_t& halo,
              Slice_t& slice,
              int mpi_tag = 1003,
              typename std::enable_if<(is_halo<Halo_t>::value &&
                                       is_slice<Slice_t>::value),
              int>::type * = 0 )
{
    // Check that the Slice is the right size.
    if ( slice.size() != halo.numLocal() + halo.numGhost() )
        throw std::runtime_error("Slice is the wrong size for scatter!");

    // Get the number of components in the slice.
    std::size_t num_comp = 1;
    for ( std::size_t d = 2; d < slice.rank(); ++d )
        num_comp *= slice.extent(d);

    // Get the raw slice data. Wrap in a 1D Kokkos View so we can unroll the
    // components of each slice element.
    Kokkos::View<typename Slice_t::value_type*,
                 typename Slice_t::memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        slice_data( slice.data(), slice.numSoA() * slice.stride(0) );

    // Allocate a send buffer. Note this one is layout right so the components
    // are consecutive.
    Kokkos::View<typename Slice_t::value_type**,
                 Kokkos::LayoutRight,
                 typename Halo_t::memory_space>
        send_buffer(
            Kokkos::ViewAllocateWithoutInitializing("halo_send_buffer"),
            halo.totalNumImport(), num_comp );

    // Extract the send buffer from the ghosted elements.
    std::size_t num_local = halo.numLocal();
    auto extract_send_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            std::size_t ghost_idx = i + num_local;
            auto s = Slice_t::index_type::s( ghost_idx );
            auto a = Slice_t::index_type::a( ghost_idx );
            std::size_t slice_offset = s*slice.stride(0) + a;
            for ( std::size_t n = 0; n < num_comp; ++n )
                send_buffer( i, n ) =
                    slice_data( slice_offset + Slice_t::vector_length * n );
        };
    Kokkos::RangePolicy<typename Halo_t::execution_space>
        extract_send_buffer_policy( 0, halo.totalNumImport() );
    Kokkos::parallel_for( "Cabana::scatter::extract_send_buffer",
                          extract_send_buffer_policy,
                          extract_send_buffer_func );
    Kokkos::fence();

    // Allocate a receive buffer. Note this one is layout right so the components
    // are consecutive.
    Kokkos::View<typename Slice_t::value_type**,
                 Kokkos::LayoutRight,
                 typename Halo_t::memory_space>
        recv_buffer(
            Kokkos::ViewAllocateWithoutInitializing("halo_recv_buffer"),
            halo.totalNumExport(), num_comp );

    // Post non-blocking receives.
    int num_n = halo.numNeighbor();
    std::vector<MPI_Request> requests( num_n );
    std::pair<std::size_t,std::size_t> recv_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_range.second = recv_range.first + halo.numExport(n);

        auto recv_subview =
            Kokkos::subview( recv_buffer, recv_range, Kokkos::ALL );

        MPI_Irecv( recv_subview.data(),
                   recv_subview.size() * sizeof(typename Slice_t::value_type),
                   MPI_BYTE,
                   halo.neighborRank(n),
                   mpi_tag,
                   halo.comm(),
                   &(requests[n]) );

        recv_range.first = recv_range.second;
    }

    // Do blocking sends.
    std::pair<std::size_t,std::size_t> send_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        send_range.second = send_range.first + halo.numImport(n);

        auto send_subview =
            Kokkos::subview( send_buffer, send_range, Kokkos::ALL );

        MPI_Send( send_subview.data(),
                  send_subview.size() * sizeof(typename Slice_t::value_type),
                  MPI_BYTE,
                  halo.neighborRank(n),
                  mpi_tag,
                  halo.comm() );

        send_range.first = send_range.second;
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( num_n );
    const int ec =
        MPI_Waitall( requests.size(), requests.data(), status.data() );
    assert( MPI_SUCCESS == ec );

    // Get the steering vector for the sends.
    auto steering = halo.getExportSteering();

    // Scatter the ghosts in the receive buffer into the local values.
    auto scatter_recv_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            auto s = Slice_t::index_type::s( steering(i) );
            auto a = Slice_t::index_type::a( steering(i) );
            std::size_t slice_offset = s*slice.stride(0) + a;
            for ( std::size_t n = 0; n < num_comp; ++n )
                Kokkos::atomic_add(
                    &slice_data(slice_offset + Slice_t::vector_length * n),
                    recv_buffer(i,n) );
        };
    Kokkos::RangePolicy<typename Halo_t::execution_space>
        scatter_recv_buffer_policy( 0, halo.totalNumExport() );
    Kokkos::parallel_for( "Cabana::scatter::scatter_recv_buffer",
                          scatter_recv_buffer_policy,
                          scatter_recv_buffer_func );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( halo.comm() );
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_HALO_HPP
