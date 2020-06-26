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

#ifndef CABANA_GRIDCOMM_HPP
#define CABANA_GRIDCOMM_HPP

#include <Cabana_Distributor.hpp>
#include <Cabana_Halo.hpp>
#include <Cabana_Tuple.hpp>

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>

#include <Kokkos_Core.hpp>

#include <vector>

namespace Cabana
{

namespace Impl
{

//---------------------------------------------------------------------------//
// Of the 27 potential local grids figure out which are in our topology.
// Some of the ranks in this list may be invalid. This needs to be updated
// after computing destination ranks to only contain valid ranks.
template <class LocalGridType>
auto getTopology( const LocalGridType& local_grid )
{
    std::vector<int> topology( 27, -1 );
    int nr = 0;
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i, ++nr )
                topology[nr] = local_grid.neighborRank( i, j, k );
    return topology;
}

} // namespace Impl

//---------------------------------------------------------------------------//
// Grid Distributor/migrate
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
\brief Wrap particles through periodic bounds according to Cajita grid global
bounds.

\tparam LocalGridType Cajita LocalGrid type.

\tparam PositionSliceType Particle position type.

\param local_grid The local grid containing periodicity and system bound
information.

\param positions The particle position container, either Slice or View.
*/
template <class LocalGridType, class PositionSliceType>
void periodicWrap( const LocalGridType& local_grid,
                   PositionSliceType& positions )
{
    using execution_space = typename PositionSliceType::execution_space;

    const auto& global_grid = local_grid.globalGrid();
    const auto& global_mesh = global_grid.globalMesh();
    const Kokkos::Array<bool, 3> periodic = {
        global_grid.isPeriodic( Cajita::Dim::I ),
        global_grid.isPeriodic( Cajita::Dim::J ),
        global_grid.isPeriodic( Cajita::Dim::K ) };
    const Kokkos::Array<double, 3> global_low = {
        global_mesh.lowCorner( Cajita::Dim::I ),
        global_mesh.lowCorner( Cajita::Dim::J ),
        global_mesh.lowCorner( Cajita::Dim::K ) };
    const Kokkos::Array<double, 3> global_high = {
        global_mesh.highCorner( Cajita::Dim::I ),
        global_mesh.highCorner( Cajita::Dim::J ),
        global_mesh.highCorner( Cajita::Dim::K ) };
    const Kokkos::Array<double, 3> global_extent = {
        global_mesh.extent( Cajita::Dim::I ),
        global_mesh.extent( Cajita::Dim::J ),
        global_mesh.extent( Cajita::Dim::K ) };
    Kokkos::parallel_for(
        "periodic_wrap",
        Kokkos::RangePolicy<execution_space>( 0, positions.size() ),
        KOKKOS_LAMBDA( const int p ) {
            for ( int d = 0; d < 3; ++d )
            {
                if ( periodic[d] )
                {
                    if ( positions( p, d ) > global_high[d] )
                        positions( p, d ) -= global_extent[d];
                    else if ( positions( p, d ) < global_low[d] )
                        positions( p, d ) += global_extent[d];
                }
            }
        } );
}

//---------------------------------------------------------------------------//
/*!
\brief Check for the number of particles that must be communicated

\tparam LocalGridType Cajita LocalGrid type.

\tparam PositionSliceType Particle position type.

\param local_grid The local grid containing periodicity and system bound
information.

\param positions The particle position container, either Slice or View.

\param minimum_halo_width Number of halo mesh widths to include for ghosting.
*/
template <class LocalGridType, class PositionSliceType>
int migrateCount( const LocalGridType& local_grid,
                  const PositionSliceType& positions,
                  const int minimum_halo_width )
{
    using execution_space = typename PositionSliceType::execution_space;

    // Check within the halo width, within the ghosted domain.
    const auto& local_mesh =
        Cajita::createLocalMesh<Kokkos::HostSpace>( local_grid );
    auto dx = local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::I );
    auto dy = local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::J );
    auto dz = local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::K );
    const Kokkos::Array<double, 3> local_low = {
        local_mesh.lowCorner( Cajita::Ghost(), Cajita::Dim::I ) +
            minimum_halo_width * dx,
        local_mesh.lowCorner( Cajita::Ghost(), Cajita::Dim::J ) +
            minimum_halo_width * dy,
        local_mesh.lowCorner( Cajita::Ghost(), Cajita::Dim::K ) +
            minimum_halo_width * dz };
    const Kokkos::Array<double, 3> local_high = {
        local_mesh.highCorner( Cajita::Ghost(), Cajita::Dim::I ) -
            minimum_halo_width * dx,
        local_mesh.highCorner( Cajita::Ghost(), Cajita::Dim::J ) -
            minimum_halo_width * dy,
        local_mesh.highCorner( Cajita::Ghost(), Cajita::Dim::K ) -
            minimum_halo_width * dz };
    int comm_count = 0;
    Kokkos::parallel_reduce(
        "redistribute_count",
        Kokkos::RangePolicy<execution_space>( 0, positions.size() ),
        KOKKOS_LAMBDA( const int p, int& result ) {
            if ( positions( p, Cajita::Dim::I ) < local_low[Cajita::Dim::I] ||
                 positions( p, Cajita::Dim::I ) > local_high[Cajita::Dim::I] ||
                 positions( p, Cajita::Dim::J ) < local_low[Cajita::Dim::J] ||
                 positions( p, Cajita::Dim::J ) > local_high[Cajita::Dim::J] ||
                 positions( p, Cajita::Dim::K ) < local_low[Cajita::Dim::K] ||
                 positions( p, Cajita::Dim::K ) > local_high[Cajita::Dim::K] )
                result += 1;
        },
        comm_count );

    MPI_Allreduce( MPI_IN_PLACE, &comm_count, 1, MPI_INT, MPI_SUM,
                   local_grid.globalGrid().comm() );

    return comm_count;
}

namespace Impl
{
//---------------------------------------------------------------------------//
// Locate the particles in the local grid and get their destination rank.
// Particles are assumed to only migrate to a location in the 26 neighbor halo
// or stay on this rank. If the particle crosses a global periodic boundary,
// wrap it's coordinates back into the domain.
template <class LocalGridType, class PositionSliceType, class NeighborRankView,
          class DestinationRankView>
void getMigrateDestinations( const LocalGridType& local_grid,
                             const NeighborRankView& neighbor_ranks,
                             DestinationRankView& destinations,
                             const PositionSliceType& positions )
{
    using execution_space = typename PositionSliceType::execution_space;

    const auto& local_mesh =
        Cajita::createLocalMesh<Kokkos::HostSpace>( local_grid );

    // Check within the local domain.
    const Kokkos::Array<double, 3> local_low = {
        local_mesh.lowCorner( Cajita::Own(), Cajita::Dim::I ),
        local_mesh.lowCorner( Cajita::Own(), Cajita::Dim::J ),
        local_mesh.lowCorner( Cajita::Own(), Cajita::Dim::K ) };
    const Kokkos::Array<double, 3> local_high = {
        local_mesh.highCorner( Cajita::Own(), Cajita::Dim::I ),
        local_mesh.highCorner( Cajita::Own(), Cajita::Dim::J ),
        local_mesh.highCorner( Cajita::Own(), Cajita::Dim::K ) };

    Kokkos::parallel_for(
        "get_migrate_destinations",
        Kokkos::RangePolicy<execution_space>( 0, positions.size() ),
        KOKKOS_LAMBDA( const int p ) {
            // Compute the logical index of the neighbor we are sending to.
            int nid[3] = { 1, 1, 1 };
            for ( int d = 0; d < 3; ++d )
            {
                if ( positions( p, d ) < local_low[d] )
                    nid[d] = 0;
                else if ( positions( p, d ) > local_high[d] )
                    nid[d] = 2;
            }

            // Compute the destination MPI rank.
            destinations( p ) = neighbor_ranks(
                nid[Cajita::Dim::I] +
                3 * ( nid[Cajita::Dim::J] + 3 * nid[Cajita::Dim::K] ) );
        } );
    // TODO: fuse kernels
    periodicWrap( local_grid, positions );
}

} // namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Determine which data should be migrated from one
  uniquely-owned decomposition to another uniquely-owned decomposition, using
  bounds of a Cajita grid and taking periodic boundaries into account.

  \tparam LocalGridType Cajita LocalGrid type.

  \tparam PositionContainer AoSoA type.

  \param local_grid The local grid containing periodicity and system bound
  information.

  \param positions The particle positions.

  \return Distributor for later migration.
*/
template <class LocalGridType, class PositionSliceType>
Distributor<typename PositionSliceType::device_type>
gridDistributor( const LocalGridType& local_grid, PositionSliceType& positions )
{
    using device_type = typename PositionSliceType::device_type;

    // Get all 26 neighbor ranks.
    auto topology = Impl::getTopology( local_grid );

    Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        topology_host( topology.data(), topology.size() );
    auto topology_mirror =
        Kokkos::create_mirror_view_and_copy( device_type(), topology_host );
    Kokkos::View<int*, device_type> destinations(
        Kokkos::ViewAllocateWithoutInitializing( "destinations" ),
        positions.size() );

    // Determine destination ranks for all particles and wrap positions across
    // periodic boundaries.
    Impl::getMigrateDestinations( local_grid, topology_mirror, destinations,
                                  positions );

    // Create the Cabana distributor.
    Distributor<device_type> distributor( local_grid.globalGrid().comm(),
                                          destinations, topology );
    return distributor;
}

//---------------------------------------------------------------------------//
/*!
  \brief Migrate data from one uniquely-owned decomposition to another
  uniquely-owned decomposition, using the bounds and periodic boundaries of a
  Cajita grid to determine which particles should be moved. In-place variant.

  \tparam LocalGridType Cajita LocalGrid type.

  \tparam PositionContainer AoSoA type.

  \tparam PositionIndex Particle position index within the AoSoA.

  \param local_grid The local grid containing periodicity and system bound
  information.

  \param particles The particle AoSoA, containing positions.

  \param PositionIndex Particle position index within the AoSoA.

  \param min_halo_width Number of halo mesh widths to allow particles before
  migrating.

  \param force_migrate Migrate particles outside the local domain regardless of
  ghosted halo.
*/
template <class LocalGridType, class ParticleContainer,
          std::size_t PositionIndex>
void gridMigrate( const LocalGridType& local_grid, ParticleContainer& particles,
                  std::integral_constant<std::size_t, PositionIndex>,
                  const int min_halo_width, const bool force_migrate = false )
{
    // Get the positions.
    auto positions = slice<PositionIndex>( particles );

    // When false, this option checks that any particles are nearly outside the
    // ghosted halo region (outside the  min_halo_width) before initiating
    // migration. Otherwise, anything outside the local domain is migrated
    // regardless of position in the halo.
    if ( !force_migrate )
    {
        // Check to see if we need to communicate.
        auto comm_count = migrateCount( local_grid, positions, min_halo_width );

        // If we have no particles near the ghosted boundary, then exit.
        if ( 0 == comm_count )
            return;
    }

    auto distributor = gridDistributor( local_grid, positions );

    // Redistribute the particles.
    migrate( distributor, particles );
}

//---------------------------------------------------------------------------//
/*!
  \brief Migrate data from one uniquely-owned decomposition to another
  uniquely-owned decomposition, using the bounds and periodic boundaries of a
  Cajita grid to determine which particles should be moved. Separate AoSoA
  variant.

  \tparam LocalGridType Cajita LocalGrid type.

  \tparam ParticleContainer AoSoA type.

  \tparam PositionIndex Particle position index within the AoSoA.

  \param local_grid The local grid containing periodicity and system bound
  information.

  \param src_particles The source particle AoSoA, containing positions.

  \param PositionIndex Particle position index within the AoSoA.

  \param src_particles The destination particle AoSoA, containing positions.

  \param min_halo_width Number of halo mesh widths to allow particles before
  migrating.

  \param force_migrate Migrate particles outside the local domain regardless of
  ghosted halo.
*/
template <class LocalGridType, class ParticleContainer,
          std::size_t PositionIndex>
void gridMigrate( const LocalGridType& local_grid,
                  ParticleContainer& src_particles,
                  std::integral_constant<std::size_t, PositionIndex>,
                  ParticleContainer& dst_particles, const int min_halo_width,
                  const bool force_migrate = false )
{
    // Get the positions.
    auto positions = slice<PositionIndex>( src_particles );

    // When false, this option checks that any particles are nearly outside the
    // ghosted halo region (outside the  min_halo_width) before initiating
    // migration. Otherwise, anything outside the local domain is migrated
    // regardless of position in the halo.
    if ( !force_migrate )
    {
        // Check to see if we need to communicate.
        auto comm_count = migrateCount( local_grid, positions, min_halo_width );

        // If we have no particles near the ghosted boundary, copy, then exit.
        if ( 0 == comm_count )
        {
            Cabana::deep_copy( dst_particles, src_particles );
            return;
        }
    }

    auto distributor = gridDistributor( local_grid, positions );

    // Resize as needed.
    dst_particles.resize( distributor.totalNumImport() );

    // Redistribute the particles.
    migrate( distributor, src_particles, dst_particles );
}

//---------------------------------------------------------------------------//
// Grid Halo/gather
//---------------------------------------------------------------------------//

namespace Impl
{

template <class LocalGridType, class PositionSliceType, class CountView,
          class DestinationRankView, class ShiftViewType>
struct HaloIds
{
    Kokkos::Array<bool, 3> _periodic;
    Kokkos::Array<double, 3> _global_low;
    Kokkos::Array<double, 3> _global_high;
    Kokkos::Array<double, 3> _global_extent;

    int _min_halo;
    int _neighbor_rank;

    CountView _send_count;
    DestinationRankView _destinations;
    DestinationRankView _ids;
    ShiftViewType _shifts;
    PositionSliceType _positions;

    Kokkos::Array<int, 3> _ijk;
    Kokkos::Array<double, 3> _min_coord;
    Kokkos::Array<double, 3> _max_coord;

    HaloIds( const LocalGridType& local_grid,
             const PositionSliceType& positions, CountView& send_count,
             DestinationRankView& destinations, DestinationRankView& ids,
             ShiftViewType& shifts, const int minimum_halo_width )
    {
        _send_count = send_count;
        _destinations = destinations;
        _ids = ids;
        _shifts = shifts;
        _positions = positions;

        // Check within the halo width, within the local domain.
        const auto& global_grid = local_grid.globalGrid();
        _periodic = { global_grid.isPeriodic( Cajita::Dim::I ),
                      global_grid.isPeriodic( Cajita::Dim::J ),
                      global_grid.isPeriodic( Cajita::Dim::K ) };
        auto dx =
            local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::I );
        auto dy =
            local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::J );
        auto dz =
            local_grid.globalGrid().globalMesh().cellSize( Cajita::Dim::K );
        const auto& global_mesh = global_grid.globalMesh();
        _global_low = {
            global_mesh.lowCorner( Cajita::Dim::I ) + minimum_halo_width * dx,
            global_mesh.lowCorner( Cajita::Dim::J ) + minimum_halo_width * dy,
            global_mesh.lowCorner( Cajita::Dim::K ) + minimum_halo_width * dz };
        _global_high = {
            global_mesh.highCorner( Cajita::Dim::I ) - minimum_halo_width * dx,
            global_mesh.highCorner( Cajita::Dim::J ) - minimum_halo_width * dy,
            global_mesh.highCorner( Cajita::Dim::K ) -
                minimum_halo_width * dz };
        _global_extent = { global_mesh.extent( Cajita::Dim::I ),
                           global_mesh.extent( Cajita::Dim::J ),
                           global_mesh.extent( Cajita::Dim::K ) };

        _min_halo = minimum_halo_width;

        build( local_grid );
    }

    KOKKOS_INLINE_FUNCTION void operator()( const int p ) const
    {
        Kokkos::Array<double, 3> pos = { _positions( p, Cajita::Dim::I ),
                                         _positions( p, Cajita::Dim::J ),
                                         _positions( p, Cajita::Dim::K ) };

        // Check the if particle is both in the owned space
        // and the ghosted space of this neighbor (ignore
        // the current cell).
        if ( ( pos[Cajita::Dim::I] > _min_coord[Cajita::Dim::I] &&
               pos[Cajita::Dim::I] < _max_coord[Cajita::Dim::I] ) &&
             ( pos[Cajita::Dim::J] > _min_coord[Cajita::Dim::J] &&
               pos[Cajita::Dim::J] < _max_coord[Cajita::Dim::J] ) &&
             ( pos[Cajita::Dim::K] > _min_coord[Cajita::Dim::K] &&
               pos[Cajita::Dim::K] < _max_coord[Cajita::Dim::K] ) )
        {
            const std::size_t sc = _send_count()++;
            // If the size of the arrays is exceeded, keep
            // counting to resize and fill next.
            if ( sc < _destinations.extent( 0 ) )
            {
                // Keep the destination MPI rank.
                _destinations( sc ) = _neighbor_rank;
                // Keep the particle ID.
                _ids( sc ) = p;
                // Determine if this ghost particle needs to
                // be shifted through the periodic boundary.
                for ( int d = 0; d < 3; ++d )
                {
                    _shifts( sc, d ) = 0.0;
                    if ( _periodic[d] && _ijk[d] )
                    {
                        if ( pos[d] > _global_high[d] )
                            _shifts( sc, d ) = -_global_extent[d];
                        else if ( pos[d] < _global_low[d] )
                            _shifts( sc, d ) = _global_extent[d];
                    }
                }
            }
        }
    }

    //---------------------------------------------------------------------------//
    // Locate particles within the local grid and determine if any from this
    // rank need to be ghosted to one (or more) of the 26 neighbor ranks,
    // keeping track of destination rank, index in the container, and periodic
    // shift needed (but not yet applied).
    void build( const LocalGridType& local_grid )
    {
        using execution_space = typename PositionSliceType::execution_space;
        const auto& local_mesh =
            Cajita::createLocalMesh<Kokkos::HostSpace>( local_grid );

        auto policy =
            Kokkos::RangePolicy<execution_space>( 0, _positions.size() );

        // Add a ghost if this particle is near the local boundary, potentially
        // for each of the 26 neighbors cells. Do this one neighbor rank at a
        // time so that sends are contiguous.
        auto topology = getTopology( local_grid );
        auto unique_topology = getUniqueTopology( topology );
        for ( std::size_t ar = 0; ar < unique_topology.size(); ar++ )
        {
            int nr = 0;
            for ( int k = -1; k < 2; ++k )
            {
                for ( int j = -1; j < 2; ++j )
                {
                    for ( int i = -1; i < 2; ++i, ++nr )
                    {
                        if ( i != 0 || j != 0 || k != 0 )
                        {
                            const int _neighbor_rank = topology[nr];
                            if ( _neighbor_rank == unique_topology[ar] )
                            {
                                auto sis = local_grid.sharedIndexSpace(
                                    Cajita::Own(), Cajita::Cell(), i, j, k,
                                    _min_halo );
                                const int min_ind_i = sis.min( Cajita::Dim::I );
                                const int min_ind_j = sis.min( Cajita::Dim::J );
                                const int min_ind_k = sis.min( Cajita::Dim::K );
                                Kokkos::Array<int, 3> min_ind = {
                                    min_ind_i, min_ind_j, min_ind_k };
                                const int max_ind_i =
                                    sis.max( Cajita::Dim::I ) + 1;
                                const int max_ind_j =
                                    sis.max( Cajita::Dim::J ) + 1;
                                const int max_ind_k =
                                    sis.max( Cajita::Dim::K ) + 1;
                                Kokkos::Array<int, 3> max_ind = {
                                    max_ind_i, max_ind_j, max_ind_k };

                                local_mesh.coordinates( Cajita::Node(),
                                                        min_ind.data(),
                                                        _min_coord.data() );
                                local_mesh.coordinates( Cajita::Node(),
                                                        max_ind.data(),
                                                        _max_coord.data() );
                                _ijk = { i, j, k };

                                Kokkos::parallel_for( "get_halo_ids", policy,
                                                      *this );
                                Kokkos::fence();
                            }
                        }
                    }
                }
            }
            // Shift periodic coordinates in send buffers.
        }
    }

    void rebuild( const LocalGridType& local_grid )
    {
        // Resize views to actual send sizes.
        int dest_size = _destinations.extent( 0 );
        int dest_count = 0;
        Kokkos::deep_copy( dest_count, _send_count );
        if ( dest_count != dest_size )
        {
            Kokkos::resize( _destinations, dest_count );
            Kokkos::resize( _ids, dest_count );
            Kokkos::resize( _shifts, dest_count, 3 );
        }

        // If original view sizes were exceeded, only counting was done so
        // we need to rerun.
        if ( dest_count > dest_size )
        {
            Kokkos::deep_copy( _send_count, 0 );
            build( local_grid );
        }
    }
};
} // namespace Impl

//---------------------------------------------------------------------------//
/*!
  \class PeriodicShift

  \brief Store and apply periodic shifts for halo communication.

  \tparam DeviceType Device type for which the data for this class will be
  allocated and where parallel execution occurs.

  \tparam PositionIndex Particle position index within the AoSoA.

  Ideally this would inherit from Halo (PeriodicHalo), combining the periodic
  shift and halo together. This is not currently done because the
  CommunicationPlan contains std member variables that would be captured on the
  device (warnings with NVCC).
*/
template <class DeviceType, std::size_t PositionIndex>
struct PeriodicShift
{
    Kokkos::View<double**, DeviceType> _shifts;

    /*!
      \brief Constructor

      \tparam ShiftViewType The periodic shift Kokkos View type.

      \param shifts The periodic shifts for each element being sent.
    */
    template <class ShiftViewType>
    PeriodicShift( const ShiftViewType shifts )
        : _shifts( shifts )
    {
    }

    /*!
      \brief Modify the send buffer with periodic shifts.

      \tparam ViewType The container type for the send buffer.

      \param send_buffer Send buffer of positions being ghosted.

      \param i Particle index.
     */
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION void operator()( ViewType& send_buffer,
                                            const int i ) const
    {
        for ( int d = 0; d < 3; ++d )
            get<PositionIndex>( send_buffer( i ), d ) += _shifts( i, d );
    }
};

//---------------------------------------------------------------------------//
/*!
  \brief Determine which data should be ghosted on another decomposition, using
  bounds of a Cajita grid and taking periodic boundaries into account. Slice
  variant.

  \tparam LocalGridType Cajita LocalGrid type.

  \tparam PositionSliceType Slice/View type.

  \param local_grid The local grid containing periodicity and system bound
  information.

  \param positions The particle positions.

  \param PositionIndex Particle position index within the AoSoA.

  \param min_halo_width Number of halo mesh widths to include for ghosting.

  \param max_export_guess The allocation size for halo export ranks, IDs, and
  periodic shifts

  \return Pair containing the Halo and PeriodicShift.
*/
template <class LocalGridType, class PositionSliceType,
          std::size_t PositionIndex>
auto gridHalo(
    const LocalGridType& local_grid, const PositionSliceType& positions,
    std::integral_constant<std::size_t, PositionIndex>,
    const int min_halo_width, const int max_export_guess = 0,
    typename std::enable_if<is_slice<PositionSliceType>::value, int>::type* =
        0 )
{
    using device_type = typename PositionSliceType::device_type;
    using pos_value = typename PositionSliceType::value_type;

    // Get all 26 neighbor ranks.
    auto topology = Impl::getTopology( local_grid );

    using DestinationRankView = typename Kokkos::View<int*, device_type>;
    using ShiftViewType = typename Kokkos::View<pos_value**, device_type>;
    using CountView =
        typename Kokkos::View<int, Kokkos::LayoutRight, device_type,
                              Kokkos::MemoryTraits<Kokkos::Atomic>>;
    DestinationRankView destinations(
        Kokkos::ViewAllocateWithoutInitializing( "destinations" ),
        max_export_guess );
    DestinationRankView ids( Kokkos::ViewAllocateWithoutInitializing( "ids" ),
                             max_export_guess );
    ShiftViewType shifts( Kokkos::ViewAllocateWithoutInitializing( "shifts" ),
                          max_export_guess, 3 );
    CountView send_count( "halo_send_count" );

    // Determine which particles need to be ghosted to neighbors.
    auto halo_ids = Impl::HaloIds<LocalGridType, PositionSliceType, CountView,
                                  DestinationRankView, ShiftViewType>(
        local_grid, positions, send_count, destinations, ids, shifts,
        min_halo_width );
    // Rebuild if needed.
    halo_ids.rebuild( local_grid );

    // Create the Cabana Halo.
    auto halo =
        Halo<device_type>( local_grid.globalGrid().comm(), positions.size(),
                           ids, destinations, topology );

    // Create the Shifts.
    auto periodic_shift = PeriodicShift<device_type, PositionIndex>( shifts );

    return std::make_pair( halo, periodic_shift );
}

//---------------------------------------------------------------------------//
/*!
  \brief Determine which data should be ghosted on another decomposition, using
  bounds of a Cajita grid and taking periodic boundaries into account. AoSoA
  variant.

  \tparam LocalGridType Cajita LocalGrid type.

  \tparam ParticleContainer AoSoA type.
  \param local_grid The local grid for creating halo and periodicity.

  \param particles The particle AoSoA, containing positions.

  \param PositionIndex Particle position index within the AoSoA.

  \param min_halo_width Number of halo mesh widths to include for ghosting.

  \param max_export_guess The allocation size for halo export ranks, IDs, and
  periodic shifts.

  \return Pair containing the Halo and PeriodicShift.
*/
template <class LocalGridType, class ParticleContainer,
          std::size_t PositionIndex>
auto gridHalo(
    const LocalGridType& local_grid, const ParticleContainer& particles,
    std::integral_constant<std::size_t, PositionIndex>,
    const int min_halo_width, const int max_export_guess = 0,
    typename std::enable_if<is_aosoa<ParticleContainer>::value, int>::type* =
        0 )
{
    auto positions = slice<PositionIndex>( particles );
    return gridHalo( local_grid, positions,
                     std::integral_constant<std::size_t, PositionIndex>(),
                     min_halo_width, max_export_guess );
}

//---------------------------------------------------------------------------//
/*!
  \brief Gather data from one decomposition and ghosts on another decomposition,
  using the bounds and periodicity of a Cajita grid to determine which particles
  should be copied. AoSoA variant.

  \tparam HaloType Halo type.

  \tparam PeriodicShiftType Periodic shift type.

  \tparam ParticleContainer AoSoA type.

  \param halo The communication halo.

  \param shift Periodic shift functor.

  \param particles The particle AoSoA, containing positions.
*/
template <class HaloType, class PeriodicShiftType, class ParticleContainer>
void gridGather( const HaloType& halo, const PeriodicShiftType& shift,
                 ParticleContainer& particles )
{
    particles.resize( halo.numLocal() + halo.numGhost() );

    gather( halo, particles, shift );
}

// TODO: slice version

} // namespace Cabana

#endif // end CABANA_GRIDCOMM_HPP
