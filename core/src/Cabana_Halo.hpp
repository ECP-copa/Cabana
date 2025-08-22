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
  \file Cabana_Halo.hpp
  \brief Multi-node particle scatter/gather
*/
#ifndef CABANA_HALO_HPP
#define CABANA_HALO_HPP

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
  \brief A communication plan for scattering and gathering of ghosted data.

  \tparam MemorySpace Kokkos memory space in which data for this class will be
  allocated.

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
template <class MemorySpace, class BuildType = Export,
          class CommSpaceType = CommSpace::Mpi>
class Halo : public CommunicationPlan<MemorySpace, CommSpaceType>
{
  public:
    using commspace_type = CommSpaceType;

    /*!
      \brief Neighbor and export rank constructor. Use this when you don't know
      who you will receiving from - only who you are sending to, but you already
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

      \tparam BuildType must be Cabana::Export

      \param comm The MPI communicator over which the halo is defined.

      \param num_local The number of locally-owned elements on this rank.

      \param element_ids The local ids of the elements that will be
      exported to other ranks to be used as ghosts. Element ids may be
      repeated in this list if they are sent to multiple destinations. Must be
      the same length as element_ranks. The input is expected to be a
      Kokkos view or Cabana slice in the same memory space as the
      communication plan.

      \param element_ranks The ranks to which we will send each element
      in element_ids. In this case each rank must be one of the
      neighbor ranks. Must be the same length as element_ids. A rank is
      allowed to send to itself. The input is expected to be a Kokkos view or
      Cabana slice in the same memory space as the communication plan.

      \param neighbor_ranks List of ranks this rank will send to and receive
      from. This list can include the calling rank. This is effectively a
      description of the topology of the point-to-point communication
      plan. The elements in this list must be unique.
    */
    template <class IdViewType, class RankViewType, typename T = BuildType,
              std::enable_if_t<std::is_same<T, Export>::value, int> = 0>
    Halo( MPI_Comm comm, const std::size_t num_local,
          const IdViewType& element_ids, const RankViewType& element_ranks,
          const std::vector<int>& neighbor_ranks )
        : CommunicationPlan<MemorySpace, CommSpaceType>( comm )
        , _num_local( num_local )
    {
        if ( element_ids.size() != element_ranks.size() )
            throw std::runtime_error( "Cabana::Halo (export): ids and ranks "
                                      "views are different sizes!" );

        auto neighbor_ids = this->createWithTopology(
            BuildType(), element_ranks, neighbor_ranks );
        this->createExportSteering( neighbor_ids, element_ranks, element_ids );
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

      \tparam BuildType must be Cabana::Export

      \param comm The MPI communicator over which the halo is defined.

      \param num_local The number of locally-owned elements on this rank.

      \param element_ids The local ids of the elements that will be
      sent to other ranks to be used as ghosts. Element ids may be repeated in
      this list if they are sent to multiple destinations. Must be the same
      length as element_ranks. The input is expected to be a Kokkos
      view or Cabana slice in the same memory space as the communication plan.

      \param element_ranks The ranks to which we will export each element
      in element_ids. Must be the same length as
      element_ids. The neighbor ranks will be determined from this
      list. A rank is allowed to send to itself. The input is expected to be a
      Kokkos view or Cabana slice in the same memory space as the
      communication plan.
    */
    template <class IdViewType, class RankViewType, typename T = BuildType,
              std::enable_if_t<std::is_same<T, Export>::value, int> = 0>
    Halo( MPI_Comm comm, const std::size_t num_local,
          const IdViewType& element_ids, const RankViewType& element_ranks )
        : CommunicationPlan<MemorySpace, CommSpaceType>( comm )
        , _num_local( num_local )
    {
        if ( element_ids.size() != element_ranks.size() )
            throw std::runtime_error( "Cabana::Halo (import): ids and ranks "
                                      "views are different sizes!" );

        auto neighbor_ids =
            this->createWithoutTopology( BuildType(), element_ranks );
        this->createExportSteering( neighbor_ids, element_ranks, element_ids );
    }

    /*!
     \brief Neighbor and import rank constructor. Use this when you don't know
     who you will sending to - only who you are receiving from, but you already
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

     \tparam BuildType must be Cabana::Import

     \param comm The MPI communicator over which the halo is defined.

     \param num_local The number of locally-owned elements on this rank.

     \param element_ids The local ids of the elements that will be
     imported from other ranks to be used as ghosts. Element ids may be
     repeated in this list if they are sent to multiple destinations. Must be
     the same length as element_ranks. The input is expected to be a
     Kokkos view or Cabana slice in the same memory space as the
     communication plan.

     \param element_ranks The ranks from which we will import each element
     in element_ids. In this case each rank must be one of the
     neighbor ranks. Must be the same length as element_ids. A rank is
     allowed to send to itself. The input is expected to be a Kokkos view or
     Cabana slice in the same memory space as the communication plan.

     \param neighbor_ranks List of ranks this rank will send to and receive
     from. This list can include the calling rank. This is effectively a
     description of the topology of the point-to-point communication
     plan. The elements in this list must be unique.
   */
    template <class IdViewType, class RankViewType, typename T = BuildType,
              std::enable_if_t<std::is_same<T, Import>::value, int> = 0>
    Halo( MPI_Comm comm, const std::size_t num_local,
          const IdViewType& element_ids, const RankViewType& element_ranks,
          const std::vector<int>& neighbor_ranks )
        : CommunicationPlan<MemorySpace, CommSpaceType>( comm )
        , _num_local( num_local )
    {
        if ( element_ids.size() != element_ranks.size() )
            throw std::runtime_error( "Cabana::Halo (import): ids and ranks "
                                      "views are different sizes!" );

        auto neighbor_ids_ranks_indices = this->createWithTopology(
            BuildType(), element_ranks, element_ids, neighbor_ranks );
        this->createExportSteering( std::get<0>( neighbor_ids_ranks_indices ),
                                    std::get<1>( neighbor_ids_ranks_indices ),
                                    std::get<2>( neighbor_ids_ranks_indices ) );
    }

    /*!
  \brief Import rank constructor. Use this when you don't know which ranks
  neighbor each other. (i.e. every rank does not already know who they will
  be exporting to and receiving from)

  \tparam IdViewType The container type for the export element ids. This
  container type can be either a Kokkos View or a Cabana Slice.

  \tparam RankViewType The container type for the export element
  ranks. This container type can be either a Kokkos View or a Cabana
  Slice.

  \tparam BuildType must be Cabana::Import

  \param comm The MPI communicator over which the halo is defined.

  \param num_local The number of locally-owned elements on this rank.

  \param element_ids The local ids of the elements that will be
  imported from other ranks to be used as ghosts. Element ids may be
  repeated in this list if they are sent to multiple destinations. Must be
  the same length as element_ranks. The input is expected to be a
  Kokkos view or Cabana slice in the same memory space as the
  communication plan.

  \param element_ranks The ranks from which we will import each element
  in element_ids. In this case each rank must be one of the
  neighbor ranks. Must be the same length as element_ids. A rank is
  allowed to send to itself. The input is expected to be a Kokkos view or
  Cabana slice in the same memory space as the communication plan.
*/
    template <class IdViewType, class RankViewType, typename T = BuildType,
              std::enable_if_t<std::is_same<T, Import>::value, int> = 0>
    Halo( MPI_Comm comm, const std::size_t num_local,
          const IdViewType& element_ids, const RankViewType& element_ranks )
        : CommunicationPlan<MemorySpace, CommSpaceType>( comm )
        , _num_local( num_local )
    {
        if ( element_ids.size() != element_ranks.size() )
            throw std::runtime_error( "Cabana::Halo (import): ids and ranks "
                                      "views are different sizes!" );

        auto neighbor_ids_ranks_indices = this->createWithoutTopology(
            BuildType(), element_ranks, element_ids );
        this->createExportSteering( std::get<0>( neighbor_ids_ranks_indices ),
                                    std::get<1>( neighbor_ids_ranks_indices ),
                                    std::get<2>( neighbor_ids_ranks_indices ) );
    }

    /*!
      \brief Get the number of elements locally owned by this rank.

      \return THe number of elements locally owned by this rank.
    */
    std::size_t numLocal() const { return _num_local; }

    /*!
      \brief Get the number of ghost elements this rank. Use this to resize a
      data structure for scatter/gather operations. For use with scatter
      gather, a data structure should be of size numLocal() + numGhost().

      \return The number of ghosted elements on this rank.
    */
    std::size_t numGhost() const { return this->totalNumImport(); }

  private:
    std::size_t _num_local;
};

//---------------------------------------------------------------------------//
//! \cond Impl
template <typename>
struct is_halo_impl : public std::false_type
{
};

template <typename MemorySpace, typename BuildType, typename CommSpaceType>
struct is_halo_impl<Halo<MemorySpace, BuildType, CommSpaceType>>
    : public std::true_type
{
};
//! \endcond

//! Halo static type checker.
template <class T>
struct is_halo : public is_halo_impl<typename std::remove_cv<T>::type>::type
{
};

/*!
  \brief Ensure the particle size matches the total halo (local and ghost) size.

  \param halo The halo that will be used for the gather. Used to query import
  and export sizes.

  \param particles The particle data (either AoSoA or slice). Used to query the
  total size.
*/
template <class Halo, class ParticleData>
bool haloCheckValidSize(
    const Halo& halo, const ParticleData& particles,
    typename std::enable_if<( is_halo<Halo>::value ), int>::type* = 0 )
{
    // Check that the data is the right size.
    return ( particles.size() == halo.numLocal() + halo.numGhost() );
}

template <class HaloType, class AoSoAType, class SFINAE = void>
class Gather;

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
*/
template <class HaloType, class AoSoAType>
class Gather<HaloType, AoSoAType,
             typename std::enable_if<is_aosoa<AoSoAType>::value>::type>
    : public CommunicationData<HaloType, CommunicationDataAoSoA<AoSoAType>>
{
  public:
    static_assert( is_halo<HaloType>::value, "" );

    //! Communication space type.
    using commspace_type = typename HaloType::commspace_type;
    //! Base type.
    using base_type =
        CommunicationData<HaloType, CommunicationDataAoSoA<AoSoAType>>;
    //! Communication plan type (Halo)
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
      \param halo The Halo to be used for the gather.

      \param aosoa The AoSoA on which to perform the gather. The AoSoA should
      have a size equivalent to halo.numGhost() + halo.numLocal(). The locally
      owned elements are expected to appear first (i.e. in the first
      halo.numLocal() elements) and the ghosted elements are expected to appear
      second (i.e. in the next halo.numGhost() elements()).

      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    Gather( HaloType halo, AoSoAType aosoa, const double overallocation = 1.0 )
        : base_type( halo, aosoa, overallocation )
    {
        reserve( _halo, aosoa );
    }

    //! Total gather send size for this rank.
    auto totalSend() { return _halo.totalNumExport(); }
    //! Total gather receive size for this rank.
    auto totalReceive() { return _halo.totalNumImport(); }

    /*!
    \brief Perform the gather operation.
    */
    void apply() override { applyImpl( execution_space{}, commspace_type{} ); }

    template <class ExecutionSpace, class CommSpaceTypeType>
    std::enable_if_t<std::is_same<CommSpaceTypeType, CommSpace::Mpi>::value,
                     void>
        applyImpl( ExecutionSpace, CommSpaceTypeType );

    // Future: Add applyImpl that is enabled for other CommSpaceTypeType types.

    /*!
      \brief Reserve new buffers as needed and update the halo and AoSoA data.

      \param halo The Halo to be used for the gather.
      \param aosoa The AoSoA on which to perform the gather.
    */
    void reserve( const HaloType& halo, AoSoAType& aosoa )
    {
        if ( !haloCheckValidSize( halo, aosoa ) )
            throw std::runtime_error(
                "Cabana::Gather:reserve: "
                "AoSoA is the wrong size for gather! (Label: " +
                aosoa.label() + ")" );

        this->reserveImpl( halo, aosoa, totalSend(), totalReceive() );
    }
    /*!
      \brief Reserve new buffers as needed and update the halo and AoSoA data.

      \param halo The Halo to be used for the gather.
      \param aosoa The AoSoA on which to perform the gather.
      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    void reserve( const HaloType& halo, AoSoAType& aosoa,
                  const double overallocation )
    {
        if ( !haloCheckValidSize( halo, aosoa ) )
            throw std::runtime_error(
                "Cabana::Gather:reserve: "
                "AoSoA is the wrong size for gather! (Label: " +
                aosoa.label() + ")" );

        this->reserveImpl( halo, aosoa, totalSend(), totalReceive(),
                           overallocation );
    }

  private:
    plan_type _halo = base_type::_comm_plan;
    using base_type::_recv_size;
    using base_type::_send_size;
};

/*!
  \brief Synchronously gather data from the local decomposition to the ghosts
  using the halo forward communication plan. AoSoA version. This is a
  uniquely-owned to multiply-owned communication.

  A gather sends data from a locally owned elements to one or many ranks on
  which they exist as ghosts. A locally owned element may be sent to as many
  ranks as desired to be used as a ghost on those ranks. The value of the
  element in the locally owned decomposition will be the value assigned to the
  element in the ghosted decomposition.
*/
template <class HaloType, class SliceType>
class Gather<HaloType, SliceType,
             typename std::enable_if<is_slice<SliceType>::value>::type>
    : public CommunicationData<HaloType, CommunicationDataSlice<SliceType>>
{
  public:
    static_assert( is_halo<HaloType>::value, "" );

    //! Communication space type.
    using commspace_type = typename HaloType::commspace_type;
    //! Base type.
    using base_type =
        CommunicationData<HaloType, CommunicationDataSlice<SliceType>>;
    //! Communication plan type (Halo)
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
      \param halo The Halo to be used for the gather.

      \param slice The slice on which to perform the gather. The slice should
      have a size equivalent to halo.numGhost() + halo.numLocal(). The locally
      owned elements are expected to appear first (i.e. in the first
      halo.numLocal() elements) and the ghosted elements are expected to appear
      second (i.e. in the next halo.numGhost() elements()).

      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    Gather( HaloType halo, SliceType slice, const double overallocation = 1.0 )
        : base_type( halo, slice, overallocation )
    {
        reserve( _halo, slice );
    }

    //! Total gather send size for this rank.
    auto totalSend() { return _halo.totalNumExport(); }
    //! Total gather receive size for this rank.
    auto totalReceive() { return _halo.totalNumImport(); }

    /*!
    \brief Perform the gather operation.
    */
    void apply() override { applyImpl( execution_space{}, commspace_type{} ); }

    template <class ExecutionSpace, class CommSpaceTypeType>
    std::enable_if_t<std::is_same<CommSpaceTypeType, CommSpace::Mpi>::value,
                     void>
        applyImpl( ExecutionSpace, CommSpaceTypeType );

    // Future: Add applyImpl that is enabled for other CommSpaceTypeType types.

    /*!
      \brief Reserve new buffers as needed and update the halo and slice data.

      \param halo The Halo to be used for the gather.
      \param slice The slice on which to perform the gather.
      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    void reserve( const HaloType& halo, const SliceType& slice,
                  const double overallocation )
    {
        if ( !haloCheckValidSize( halo, slice ) )
            throw std::runtime_error(
                "Cabana::Gather:reserve: "
                "Slice is the wrong size for gather! (Label: " +
                slice.label() + ")" );

        this->reserveImpl( halo, slice, totalSend(), totalReceive(),
                           overallocation );
    }
    /*!
      \brief Reserve new buffers as needed and update the halo and slice data.

      \param halo The Halo to be used for the gather.
      \param slice The slice on which to perform the gather.
    */
    void reserve( const HaloType& halo, const SliceType& slice )
    {
        if ( !haloCheckValidSize( halo, slice ) )
            throw std::runtime_error(
                "Cabana::Gather:reserve: "
                "Slice is the wrong size for gather! (Label: " +
                slice.label() + ")" );

        this->reserveImpl( halo, slice, totalSend(), totalReceive() );
    }

  private:
    plan_type _halo = base_type::_comm_plan;
    using base_type::_recv_size;
    using base_type::_send_size;
};

//---------------------------------------------------------------------------//
/*!
  \brief Create the gather.

  \param halo The halo to use for the gather.
  \param data The data on which to perform the gather. The slice should have a
  size equivalent to halo.numGhost() + halo.numLocal(). The locally owned
  elements are expected to appear first (i.e. in the first halo.numLocal()
  elements) and the ghosted elements are expected to appear second (i.e. in the
  next halo.numGhost() elements()).
  \param overallocation An optional factor to keep extra space in the buffers to
  avoid frequent resizing.
  \return Gather
*/
template <class HaloType, class ParticleDataType>
auto createGather( const HaloType& halo, const ParticleDataType& data,
                   const double overallocation = 1.0 )
{
    return Gather<HaloType, ParticleDataType>( halo, data, overallocation );
}

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously gather data from the local decomposition to the
  ghosts using the halo forward communication plan. Slice version. This is a
  uniquely-owned to multiply-owned communication.

  \note This routine allocates send and receive buffers internally. This is
  often not performant due to frequent buffer reallocations - consider creating
  and reusing Gather instead.

  \param halo The halo to use for the gather.

  \param data The data on which to perform the gather. The slice should
  have a size equivalent to halo.numGhost() + halo.numLocal(). The locally
  owned elements are expected to appear first (i.e. in the first
  halo.numLocal() elements) and the ghosted elements are expected to appear
  second (i.e. in the next halo.numGhost() elements()).
*/
template <class HaloType, class ParticleDataType>
void gather( const HaloType& halo, ParticleDataType& data )
{
    auto gather = createGather( halo, data );
    gather.apply();
}

/**********
 * SCATTER *
 **********/

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously scatter data from the ghosts to the local decomposition
  of a slice using the halo reverse communication plan. This is a multiply-owned
  to uniquely owned communication.

  In a scatter operation results from ghosted values on other processors are
  scattered back to the owning processor of the ghost and the value associated
  with the ghost is summed into the locally owned value the ghost represents.
  If a locally owned element is ghosted on multiple ranks, then multiple
  contributions will be made to the sum, one for each rank.
*/
template <class HaloType, class SliceType>
class Scatter
    : public CommunicationData<HaloType, CommunicationDataSlice<SliceType>>
{
  public:
    static_assert( is_halo<HaloType>::value, "" );

    //! Communication space type.
    using commspace_type = typename HaloType::commspace_type;
    //! Base type.
    using base_type =
        CommunicationData<HaloType, CommunicationDataSlice<SliceType>>;
    //! Communication plan type (Halo).
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
      \param halo The Halo to be used for the gather.

      \param slice The slice on which to perform the gather. The slice should
      have a size equivalent to halo.numGhost() + halo.numLocal(). The locally
      owned elements are expected to appear first (i.e. in the first
      halo.numLocal() elements) and the ghosted elements are expected to appear
      second (i.e. in the next halo.numGhost() elements()).

      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    Scatter( HaloType halo, SliceType slice, const double overallocation = 1.0 )
        : base_type( halo, slice, overallocation )
    {
        reserve( _halo, slice );
    }

    //! Total scatter send size for this rank.
    auto totalSend() { return _halo.totalNumImport(); }
    //! Total scatter receive size for this rank.
    auto totalReceive() { return _halo.totalNumExport(); }

    /*!
      \brief Perform the scatter operation.
    */
    void apply() override { applyImpl( execution_space{}, commspace_type{} ); }

    template <class ExecutionSpace, class CommSpaceTypeType>
    std::enable_if_t<std::is_same<CommSpaceTypeType, CommSpace::Mpi>::value,
                     void>
        applyImpl( ExecutionSpace, CommSpaceTypeType );

    // Future: Add applyImpl that is enabled for other CommSpaceTypeType types.

    /*!
      \brief Reserve new buffers as needed and update the halo and slice data.
      Reallocation only occurs if there is not enough space in the buffers.

      \param halo The Halo to be used for the scatter.
      \param slice The slice on which to perform the scatter.
      \param overallocation An optional factor to keep extra space in the
      buffers to avoid frequent resizing.
    */
    void reserve( const HaloType& halo, const SliceType& slice,
                  const double overallocation )
    {
        if ( !haloCheckValidSize( halo, slice ) )
            throw std::runtime_error(
                "Cabana::Scatter::reserve: "
                "Slice is the wrong size for scatter! (Label: " +
                slice.label() + ")" );

        this->reserveImpl( halo, slice, totalSend(), totalReceive(),
                           overallocation );
    }
    /*!
      \brief Reserve new buffers as needed and update the halo and slice data.

      \param halo The Halo to be used for the scatter.
      \param slice The slice on which to perform the scatter.
    */
    void reserve( const HaloType& halo, const SliceType& slice )
    {
        if ( !haloCheckValidSize( halo, slice ) )
            throw std::runtime_error(
                "Cabana::Scatter::reserve: "
                "Slice is the wrong size for scatter! (Label: " +
                slice.label() + ")" );

        this->reserveImpl( halo, slice, totalSend(), totalReceive() );
    }

  private:
    plan_type _halo = base_type::_comm_plan;
    using base_type::_recv_size;
    using base_type::_send_size;
};

} // end namespace Cabana

// Include communication backends from what is enabled in CMake.
#ifdef Cabana_ENABLE_MPI
#include <impl/Cabana_Halo_Mpi.hpp>
#endif // Enable MPI

namespace Cabana
{

/*!
  \brief Create the scatter.

  \param halo The halo to use for the scatter.
  \param slice The slice on which to perform the scatter. The slice should have
  a size equivalent to halo.numGhost() + halo.numLocal(). The locally owned
  elements are expected to appear first (i.e. in the first halo.numLocal()
  elements) and the ghosted elements are expected to appear second (i.e. in the
  next halo.numGhost() elements()).
  \param overallocation An optional factor to keep extra space in the buffers to
  avoid frequent resizing.
  \return Scatter
*/
template <class HaloType, class SliceType>
auto createScatter( const HaloType& halo, const SliceType& slice,
                    const double overallocation = 1.0,
                    typename std::enable_if<( is_halo<HaloType>::value &&
                                              is_slice<SliceType>::value ),
                                            int>::type* = 0 )
{
    return Scatter<HaloType, SliceType>( halo, slice, overallocation );
}

//---------------------------------------------------------------------------//
/*!
  \brief Synchronously scatter data from the ghosts to the local decomposition
  of a slice using the halo reverse communication plan. This is a
  multiply-owned to uniquely owned communication.

  \note This routine allocates send and receive buffers internally. This is
  often not performant due to frequent buffer reallocations - consider creating
  and reusing Gather instead.

  \param halo The halo to use for the scatter.

  \param slice The Slice on which to perform the scatter. The Slice should have
  a size equivalent to halo.numGhost() + halo.numLocal(). The locally owned
  elements are expected to appear first (i.e. in the first halo.numLocal()
  elements) and the ghosted elements are expected to appear second (i.e. in
  the next halo.numGhost() elements()).
*/
template <class HaloType, class SliceType>
void scatter( const HaloType& halo, SliceType& slice,
              typename std::enable_if<( is_halo<HaloType>::value &&
                                        is_slice<SliceType>::value ),
                                      int>::type* = 0 )
{
    auto scatter = createScatter( halo, slice );
    scatter.apply();
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_HALO_HPP
