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

#ifndef CABANA_PARTICLEDISTRIBUTOR_HPP
#define CABANA_PARTICLEDISTRIBUTOR_HPP

#include <Cabana_Distributor.hpp>

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>

#include <Kokkos_Core.hpp>

#include <vector>

namespace Cajita
{

//---------------------------------------------------------------------------//
// Particle Grid Distributor/migrate
//---------------------------------------------------------------------------//

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

    // Use global domain for periodicity.
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

            // Shift periodic coordinates if needed.
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
Cabana::Distributor<typename PositionSliceType::device_type>
createGridDistributor( const LocalGridType& local_grid,
                       PositionSliceType& positions )
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
    Cabana::Distributor<device_type> distributor(
        local_grid.globalGrid().comm(), destinations, topology );
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
    auto positions = Cabana::slice<PositionIndex>( particles );

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

    auto distributor = createGridDistributor( local_grid, positions );

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
    auto positions = Cabana::slice<PositionIndex>( src_particles );

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

    auto distributor = createGridDistributor( local_grid, positions );

    // Resize as needed.
    dst_particles.resize( distributor.totalNumImport() );

    // Redistribute the particles.
    migrate( distributor, src_particles, dst_particles );
}

} // namespace Cajita

#endif // end CABANA_PARTICLEDISTRIBUTOR_HPP
