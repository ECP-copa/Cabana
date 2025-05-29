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
  \file Cabana_Grid_ParticleDistributor.hpp
  \brief Multi-node particle redistribution using the grid halo.
*/
#ifndef CABANA_PARTICLEGRIDDISTRIBUTOR_HPP
#define CABANA_PARTICLEGRIDDISTRIBUTOR_HPP

#include <Cabana_DeepCopy.hpp>
#include <Cabana_Migrate.hpp>

#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Grid_LocalMesh.hpp>

#include <Kokkos_Core.hpp>

#include <vector>

namespace Cabana
{
namespace Grid
{

//---------------------------------------------------------------------------//
// Particle Grid Distributor
//---------------------------------------------------------------------------//

/*!
  \brief Build neighbor topology of 27 nearest 3D neighbors. Some of the ranks
  in this list may be invalid.
  \param local_grid Local grid from which MPI neighbors will be extracted.
  \return MPI neighbor ranks in k,j,i order.
*/
template <class LocalGridType, std::size_t NSD = LocalGridType::num_space_dim>
std::enable_if_t<3 == NSD, std::vector<int>>
getTopology( const LocalGridType& local_grid )
{
    std::vector<int> topology( 27, -1 );
    int nr = 0;
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i, ++nr )
                topology[nr] = local_grid.neighborRank( i, j, k );
    return topology;
}

/*!
  \brief Build neighbor topology of 8 nearest 2D neighbors. Some of the ranks
  in this list may be invalid.
  \param local_grid Local grid from which MPI neighbors will be extracted.
  \return MPI neighbor ranks in k,j,i order.
*/
template <class LocalGridType, std::size_t NSD = LocalGridType::num_space_dim>
std::enable_if_t<2 == NSD, std::vector<int>>
getTopology( const LocalGridType& local_grid )
{
    std::vector<int> topology( 9, -1 );
    int nr = 0;
    for ( int j = -1; j < 2; ++j )
        for ( int i = -1; i < 2; ++i, ++nr )
            topology[nr] = local_grid.neighborRank( i, j );
    return topology;
}

namespace Impl
{
//! \cond Impl

// Locate the particles in the local grid and get their destination rank.
// Particles are assumed to only migrate to a location in the nearest
// neighbor halo or stay on this rank. If the particle crosses a global
// periodic boundary, wrap it's coordinates back into the domain.
template <class LocalGridType, class PositionSliceType, class NeighborRankView,
          class DestinationRankView>
void getMigrateDestinations( const LocalGridType& local_grid,
                             const NeighborRankView& neighbor_ranks,
                             DestinationRankView& destinations,
                             PositionSliceType& positions )
{
    static constexpr std::size_t num_space_dim = LocalGridType::num_space_dim;
    using execution_space = typename PositionSliceType::execution_space;

    // Check within the local domain.
    const auto& local_mesh = createLocalMesh<Kokkos::HostSpace>( local_grid );

    // Use global domain for periodicity.
    const auto& global_grid = local_grid.globalGrid();
    const auto& global_mesh = global_grid.globalMesh();

    Kokkos::Array<double, num_space_dim> local_low{};
    Kokkos::Array<double, num_space_dim> local_high{};
    Kokkos::Array<bool, num_space_dim> periodic{};
    Kokkos::Array<double, num_space_dim> global_low{};
    Kokkos::Array<double, num_space_dim> global_high{};
    Kokkos::Array<double, num_space_dim> global_extent{};

    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        local_low[d] = local_mesh.lowCorner( Own(), d );
        local_high[d] = local_mesh.highCorner( Own(), d );
        periodic[d] = global_grid.isPeriodic( d );
        global_low[d] = global_mesh.lowCorner( d );
        global_high[d] = global_mesh.highCorner( d );
        global_extent[d] = global_mesh.extent( d );
    }

    Kokkos::parallel_for(
        "Cabana::Grid::ParticleGridMigrate::get_destinations",
        Kokkos::RangePolicy<execution_space>( 0, positions.size() ),
        KOKKOS_LAMBDA( const int p ) {
            // Compute the logical index of the neighbor we are sending to.
            int nid[num_space_dim];
            for ( std::size_t d = 0; d < num_space_dim; ++d )
            {
                nid[d] = 1;
                if ( positions( p, d ) < local_low[d] )
                    nid[d] = 0;
                else if ( positions( p, d ) > local_high[d] )
                    nid[d] = 2;
            }

            // Compute the destination MPI rank [ni + 3*(nj + 3*nk) in 3d].
            int neighbor_index = nid[0];
            for ( std::size_t d = 1; d < num_space_dim; ++d )
            {
                int npower = 1;
                for ( std::size_t dp = 1; dp <= d; ++dp )
                    npower *= 3;
                neighbor_index += npower * nid[d];
            }
            destinations( p ) = neighbor_ranks( neighbor_index );

            // Shift particles through periodic boundaries.
            for ( std::size_t d = 0; d < num_space_dim; ++d )
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
//! \endcond
} // namespace Impl

//-----------------------------------------------------------------------//
/*!
  \brief Check for the number of particles that must be communicated

  \tparam LocalGridType LocalGrid type.
  \tparam PositionSliceType Particle position type.

  \param local_grid The local grid containing periodicity and system bound
  information.
  \param positions The particle position container, either Slice or View.
  \param minimum_halo_width Number of halo mesh widths to include for
  ghosting.
*/
template <class LocalGridType, class PositionSliceType>
int migrateCount( const LocalGridType& local_grid,
                  const PositionSliceType& positions,
                  const int minimum_halo_width )
{
    using grid_type = LocalGridType;
    static constexpr std::size_t num_space_dim = grid_type::num_space_dim;
    using mesh_type = typename grid_type::mesh_type;
    using scalar_type = typename mesh_type::scalar_type;
    using uniform_type = UniformMesh<scalar_type, num_space_dim>;
    static_assert( std::is_same<mesh_type, uniform_type>::value,
                   "Migrate count requires a uniform mesh." );

    using execution_space = typename PositionSliceType::execution_space;

    // Check within the halo width, within the ghosted domain.
    const auto& local_mesh = createLocalMesh<Kokkos::HostSpace>( local_grid );

    Kokkos::Array<double, num_space_dim> local_low{};
    Kokkos::Array<double, num_space_dim> local_high{};
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        auto dx = local_grid.globalGrid().globalMesh().cellSize( d );
        local_low[d] =
            local_mesh.lowCorner( Ghost(), d ) + minimum_halo_width * dx;
        local_high[d] =
            local_mesh.highCorner( Ghost(), d ) - minimum_halo_width * dx;
    }

    int comm_count = 0;
    Kokkos::parallel_reduce(
        "Cabana::Grid::ParticleGridMigrate::count",
        Kokkos::RangePolicy<execution_space>( 0, positions.size() ),
        KOKKOS_LAMBDA( const int p, int& result ) {
            for ( std::size_t d = 0; d < num_space_dim; ++d )
                if ( positions( p, d ) < local_low[d] ||
                     positions( p, d ) > local_high[d] )
                {
                    result += 1;
                    break;
                }
        },
        comm_count );

    MPI_Allreduce( MPI_IN_PLACE, &comm_count, 1, MPI_INT, MPI_SUM,
                   local_grid.globalGrid().comm() );

    return comm_count;
}

//---------------------------------------------------------------------------//
/*!
  \brief Determine which data should be migrated from one uniquely-owned
  decomposition to another uniquely-owned decomposition, using bounds of the
  grid and taking periodic boundaries into account.

  \tparam LocalGridType LocalGrid type.
  \tparam PositionSliceType Position type.

  \param local_grid The local grid containing periodicity and system bound
  information.
  \param positions The particle positions.

  \return Distributor for later migration.
*/
template <class LocalGridType, class PositionSliceType>
Cabana::Distributor<typename PositionSliceType::memory_space>
createParticleDistributor( const LocalGridType& local_grid,
                           PositionSliceType& positions )
{
    using memory_space = typename PositionSliceType::memory_space;

    // Get all 26 neighbor ranks.
    auto topology = getTopology( local_grid );

    Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
        topology_host( topology.data(), topology.size() );
    auto topology_mirror =
        Kokkos::create_mirror_view_and_copy( memory_space(), topology_host );
    Kokkos::View<int*, memory_space> destinations(
        Kokkos::ViewAllocateWithoutInitializing( "destinations" ),
        positions.size() );

    // Determine destination ranks for all particles and wrap positions across
    // periodic boundaries.
    Impl::getMigrateDestinations( local_grid, topology_mirror, destinations,
                                  positions );

    // Create the Cabana distributor.
    Cabana::Distributor<memory_space> distributor(
        local_grid.globalGrid().comm(), destinations, topology );
    return distributor;
}

//---------------------------------------------------------------------------//
/*!
  \brief Migrate data from one uniquely-owned decomposition to another
  uniquely-owned decomposition, using the bounds and periodic boundaries of the
  grid to determine which particles should be moved. In-place variant.

  \tparam LocalGridType LocalGrid type.
  \tparam ParticlePositions Particle position type.
  \tparam PositionContainer AoSoA type.

  \param local_grid The local grid containing periodicity and system bounds.
  \param positions Particle positions.
  \param particles The particle AoSoA.
  \param min_halo_width Number of halo mesh widths to allow particles before
  migrating.
  \param force_migrate Migrate particles outside the local domain regardless of
  ghosted halo.
  \return Whether any particle migration occurred.
*/
template <class LocalGridType, class ParticlePositions, class ParticleContainer>
bool particleMigrate( const LocalGridType& local_grid,
                      const ParticlePositions& positions,
                      ParticleContainer& particles, const int min_halo_width,
                      const bool force_migrate = false )
{
    // When false, this option checks that any particles are nearly outside the
    // ghosted halo region (outside the min_halo_width) before initiating
    // migration. Otherwise, anything outside the local domain is migrated
    // regardless of position in the halo.
    if ( !force_migrate )
    {
        // Check to see if we need to communicate.
        auto comm_count = migrateCount( local_grid, positions, min_halo_width );

        // If we have no particles near the ghosted boundary, then exit.
        if ( 0 == comm_count )
            return false;
    }

    auto distributor = createParticleDistributor( local_grid, positions );

    // Redistribute the particles.
    migrate( distributor, particles );
    return true;
}

//---------------------------------------------------------------------------//
/*!
  \brief Migrate data from one uniquely-owned decomposition to another
  uniquely-owned decomposition, using the bounds and periodic boundaries of the
  grid to determine which particles should be moved. Separate AoSoA
  variant.

  \tparam LocalGridType LocalGrid type.
  \tparam ParticlePositions Particle position type.
  \tparam ParticleContainer AoSoA type.

  \param local_grid The local grid containing periodicity and system bounds.
  \param positions Particle positions.
  \param src_particles The source particle AoSoA.
  \param dst_particles The destination particle AoSoA.
  \param min_halo_width Number of halo mesh widths to allow particles before
  migrating.
  \param force_migrate Migrate particles outside the local domain regardless of
  ghosted halo.
  \return Whether any particle migration occurred.
*/
template <class LocalGridType, class ParticlePositions, class ParticleContainer>
bool particleMigrate( const LocalGridType& local_grid,
                      const ParticlePositions& positions,
                      const ParticleContainer& src_particles,
                      ParticleContainer& dst_particles,
                      const int min_halo_width,
                      const bool force_migrate = false )
{
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
            return false;
        }
    }

    auto distributor = createParticleDistributor( local_grid, positions );

    // Resize as needed.
    dst_particles.resize( distributor.totalNumImport() );

    // Redistribute the particles.
    migrate( distributor, src_particles, dst_particles );
    return true;
}

//! \cond Deprecated
template <class... Args>
[[deprecated( "Cabana::Grid::particleGridMigrate is now "
              "Cabana::Grid::particleMigrate. This function wrapper will be "
              "removed in a future release." )]] auto
particleGridMigrate( Args&&... args )
{
    return Cabana::Grid::particleMigrate( std::forward<Args>( args )... );
}

template <class... Args>
[[deprecated(
    "Cabana::Grid::createParticleGridDistributor is now "
    "Cabana::Grid::createParticleDistributor. This function wrapper will be "
    "removed in a future release." )]] auto
createParticleGridDistributor( Args&&... args )
{
    return Cabana::Grid::createParticleDistributor(
        std::forward<Args>( args )... );
}
//! \endcond

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_PARTICLEGRIDDISTRIBUTOR_HPP
