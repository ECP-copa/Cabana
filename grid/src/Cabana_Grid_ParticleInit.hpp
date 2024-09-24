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

/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Grid_ParticleInit.hpp
  \brief Particle creation utilities based on uniform grids.
*/
#ifndef CAJITA_PARTICLEINIT_HPP
#define CAJITA_PARTICLEINIT_HPP

#include <Cabana_Grid_Parallel.hpp>
#include <Cabana_Grid_ParticleList.hpp>

#include <Cabana_ParticleInit.hpp>
#include <Cabana_Slice.hpp>
#include <Cabana_Utils.hpp> // FIXME: remove after next release.

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <exception>

namespace Cabana
{
namespace Grid
{
// Filter out empty particles that weren't created.
template <class CreationView, class ParticleAoSoA, class ExecutionSpace>
void filterEmpties( const ExecutionSpace& exec_space,
                    const int local_num_create,
                    const int previous_num_particles,
                    const CreationView& particle_created,
                    ParticleAoSoA& particles, const bool shrink_to_fit = true )
{
    using memory_space = typename CreationView::memory_space;

    // Determine the empty particle positions in the compaction zone.
    int num_particles = particles.size();
    // This View is either empty indices to be filled or the created particle
    // indices, depending on the ratio of allocated space to the number
    // created.
    Kokkos::View<int*, memory_space> indices(
        Kokkos::ViewAllocateWithoutInitializing( "empty_or_filled" ),
        std::min( num_particles - previous_num_particles - local_num_create,
                  local_num_create ) );

    Kokkos::parallel_scan(
        "Picasso::ParticleInit::FindEmpty",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, local_num_create ),
        KOKKOS_LAMBDA( const int i, int& count, const bool final_pass ) {
            if ( !particle_created( i ) )
            {
                if ( final_pass )
                {
                    indices( count ) = i + previous_num_particles;
                }
                ++count;
            }
        } );
    Kokkos::fence();

    // Compact the list so the it only has real particles.
    int new_num_particles = previous_num_particles + local_num_create;
    Kokkos::parallel_scan(
        "Picasso::ParticleInit::RemoveEmpty",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, new_num_particles,
                                             num_particles ),
        KOKKOS_LAMBDA( const int i, int& count, const bool final_pass ) {
            if ( particle_created( i - previous_num_particles ) )
            {
                if ( final_pass )
                {
                    particles.setTuple( indices( count ),
                                        particles.getTuple( i ) );
                }
                ++count;
            }
        } );
    particles.resize( new_num_particles );
    if ( shrink_to_fit )
        particles.shrinkToFit();
}

//---------------------------------------------------------------------------//
/*!
  \brief Initialize a random number of particles in each cell given an
  initialization functor.

  \param exec_space Kokkos execution space.
  \param create_functor A functor which populates a particle given the logical
  position of a particle. This functor returns true if a particle was created
  and false if it was not giving the signature:

      bool createFunctor( const double pid, const double px[3], const double pv,
                          typename ParticleAoSoA::tuple_type& particle );
  \param particle_list The ParticleList to populate. This will be filled with
  particles and resized to a size equal to the number of particles created.
  \param particles_per_cell The number of particles to sample each cell with.
  \param local_grid The LocalGrid over which particles will be created.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param shrink_to_fit Optionally remove unused allocated space after creation.
  \param seed Optional random seed for generating particles.
*/
template <class ExecutionSpace, class InitFunctor, class ParticleListType,
          class LocalGridType>
int createParticles(
    Cabana::InitRandom, const ExecutionSpace& exec_space,
    const InitFunctor& create_functor, ParticleListType& particle_list,
    const int particles_per_cell, LocalGridType& local_grid,
    const std::size_t previous_num_particles = 0,
    const bool shrink_to_fit = true, const uint64_t seed = 123456,
    typename std::enable_if<is_particle_list<ParticleListType>::value,
                            int>::type* = 0 )
{
    // Memory space.
    using memory_space = typename ParticleListType::memory_space;

    // Create a local mesh.
    auto local_mesh = createLocalMesh<ExecutionSpace>( local_grid );

    // Get the global grid.
    const auto& global_grid = local_grid.globalGrid();

    // Get the local set of owned cell indices.
    auto owned_cells = local_grid.indexSpace( Own(), Cell(), Local() );

    // Create a random number generator.
    const auto local_seed =
        global_grid.blockId() + ( seed % ( global_grid.blockId() + 1 ) );
    using rnd_type = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    rnd_type pool;
    pool.init( local_seed, owned_cells.size() );

    // Allocate enough space for the case the particles consume the entire
    // local grid.
    int num_particles = particles_per_cell * owned_cells.size();
    particle_list.resize( previous_num_particles + num_particles );

    // Creation status.
    Kokkos::View<bool*, memory_space> particle_created(
        Kokkos::ViewAllocateWithoutInitializing( "particle_created" ),
        num_particles );

    // Initialize particles.
    int local_num_create = 0;
    grid_parallel_reduce(
        "Cabana::Grid::ParticleInit::Random", exec_space, owned_cells,
        KOKKOS_LAMBDA( const int i, const int j, const int k, int& count ) {
            // Compute the owned local cell id.
            int i_own = i - owned_cells.min( Dim::I );
            int j_own = j - owned_cells.min( Dim::J );
            int k_own = k - owned_cells.min( Dim::K );
            int cell_id =
                i_own + owned_cells.extent( Dim::I ) *
                            ( j_own + k_own * owned_cells.extent( Dim::J ) );

            // Get the coordinates of the low cell node.
            int low_node[3] = { i, j, k };
            double low_coords[3];
            local_mesh.coordinates( Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Node(), high_node, high_coords );

            // Random number generator.
            auto rand = pool.get_state( cell_id );

            // Particle coordinate.
            double px[3];

            // Particle volume.
            double pv =
                local_mesh.measure( Cell(), low_node ) / particles_per_cell;

            // Create particles.
            for ( int p = 0; p < particles_per_cell; ++p )
            {
                // Select a random point in the cell for the particle
                // location. These coordinates are logical.
                for ( int d = 0; d < 3; ++d )
                {
                    px[d] = Kokkos::rand<decltype( rand ), double>::draw(
                        rand, low_coords[d], high_coords[d] );
                }

                // Local particle id.
                int pid = cell_id * particles_per_cell + p;
                int new_pid = previous_num_particles + pid;

                // Create a new particle with the given logical coordinates.
                auto particle = particle_list.getParticle( new_pid );
                particle_created( pid ) =
                    create_functor( new_pid, px, pv, particle );

                // If we created a new particle insert it into the list.
                if ( particle_created( pid ) )
                {
                    particle_list.setParticle( particle, new_pid );
                    count++;
                }
            }
        },
        local_num_create );
    Kokkos::fence();

    // Filter empties.
    auto& aosoa = particle_list.aosoa();
    filterEmpties( exec_space, local_num_create, previous_num_particles,
                   particle_created, aosoa, shrink_to_fit );

    return particle_list.size();
}

/*!
  \brief Initialize random particles per cell given an initialization functor.

  \param tag Initialization type tag.
  \param create_functor A functor which populates a particle given the logical
  position of a particle. This functor returns true if a particle was created
  and false if it was not giving the signature:

      bool createFunctor( const double pid, const double px[3], const double pv,
                          typename ParticleAoSoA::tuple_type& particle );
  \param particle_list The ParticleList to populate. This will be filled with
  particles and resized to a size equal to the number of particles created.
  \param particles_per_cell The number of particles to sample each cell with.
  \param local_grid The LocalGrid over which particles will be created.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param shrink_to_fit Optionally remove unused allocated space after creation.
  \param seed Optional random seed for generating particles.
*/
template <class InitFunctor, class ParticleListType, class LocalGridType>
int createParticles(
    Cabana::InitRandom tag, const InitFunctor& create_functor,
    ParticleListType& particle_list, const int particles_per_cell,
    LocalGridType& local_grid, const std::size_t previous_num_particles = 0,
    const bool shrink_to_fit = true, const uint64_t seed = 123456,
    typename std::enable_if<is_particle_list<ParticleListType>::value,
                            int>::type* = 0 )
{
    using exec_space = typename ParticleListType::memory_space::execution_space;
    return createParticles( tag, exec_space{}, create_functor, particle_list,
                            particles_per_cell, local_grid,
                            previous_num_particles, shrink_to_fit, seed );
}

//---------------------------------------------------------------------------//
/*!
  \brief Initialize a random number of particles in each cell.

  \param exec_space Kokkos execution space.
  \param positions Particle positions slice. This should be already the size of
  the number of grid cells times particles_per_cell.s
  \param particles_per_cell The number of particles to sample each cell with.
  \param local_grid The LocalGrid over which particles will be created.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param seed Optional random seed for generating particles.
*/
template <class ExecutionSpace, class PositionType, class LocalGridType>
void createParticles(
    Cabana::InitRandom, const ExecutionSpace& exec_space,
    PositionType& positions, const int particles_per_cell,
    LocalGridType& local_grid, const std::size_t previous_num_particles = 0,
    const uint64_t seed = 123456,
    typename std::enable_if<( Cabana::is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    // Create a local mesh.
    auto local_mesh = createLocalMesh<ExecutionSpace>( local_grid );

    // Get the global grid.
    const auto& global_grid = local_grid.globalGrid();

    // Get the local set of owned cell indices.
    auto owned_cells = local_grid.indexSpace( Own(), Cell(), Local() );

    // Create a random number generator.
    const auto local_seed =
        global_grid.blockId() + ( seed % ( global_grid.blockId() + 1 ) );
    using rnd_type = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    rnd_type pool;
    pool.init( local_seed, owned_cells.size() );

    // Ensure correct space for the particles.
    assert( positions.size() == static_cast<std::size_t>( particles_per_cell *
                                                          owned_cells.size() ) +
                                    previous_num_particles );

    // Initialize particles.
    grid_parallel_for(
        "Cabana::Grid::ParticleInit::Random", exec_space, owned_cells,
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            // Compute the owned local cell id.
            int i_own = i - owned_cells.min( Dim::I );
            int j_own = j - owned_cells.min( Dim::J );
            int k_own = k - owned_cells.min( Dim::K );
            int cell_id =
                i_own + owned_cells.extent( Dim::I ) *
                            ( j_own + k_own * owned_cells.extent( Dim::J ) );

            // Get the coordinates of the low cell node.
            int low_node[3] = { i, j, k };
            double low_coords[3];
            local_mesh.coordinates( Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Node(), high_node, high_coords );

            // Random number generator.
            auto rand = pool.get_state( cell_id );

            // Create particles.
            for ( int p = 0; p < particles_per_cell; ++p )
            {
                // Local particle id.
                int pid =
                    previous_num_particles + cell_id * particles_per_cell + p;

                // Select a random point in the cell for the particle
                // location. These coordinates are logical.
                for ( int d = 0; d < 3; ++d )
                {
                    positions( pid, d ) =
                        Kokkos::rand<decltype( rand ), double>::draw(
                            rand, low_coords[d], high_coords[d] );
                }
            }
        } );
}

/*!
  \brief Initialize a random number of particles in each cell.

  \param tag Initialization type tag.
  \param positions Particle positions slice. This should be already the size of
  the number of grid cells times particles_per_cell.s
  \param particles_per_cell The number of particles to sample each cell with.
  \param local_grid The LocalGrid over which particles will be created.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param seed Optional random seed for generating particles.
*/
template <class PositionType, class LocalGridType>
void createParticles(
    Cabana::InitRandom tag, PositionType& positions,
    const int particles_per_cell, LocalGridType& local_grid,
    const std::size_t previous_num_particles = 0, const uint64_t seed = 123456,
    typename std::enable_if<( Cabana::is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    using exec_space = typename PositionType::execution_space;
    createParticles( tag, exec_space{}, positions, particles_per_cell,
                     local_grid, previous_num_particles, seed );
}

//---------------------------------------------------------------------------//
/*!
  \brief Initialize uniform particles per cell given an initialization
  functor.

  \param exec_space Kokkos execution space.
  \param create_functor A functor which populates a particle given the
  logical position of a particle. This functor returns true if a particle
  was created and false if it was not giving the signature:

      bool createFunctor( const double px[3],
                          typename ParticleAoSoA::tuple_type& particle );
  \param particle_list The ParticleList to populate. This will be filled
  with particles and resized to a size equal to the number of particles
  created.
  \param particles_per_cell_dim The number of particles to populate each cell
  dimension with.
  \param local_grid The LocalGrid over which particles will be created.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param shrink_to_fit Optionally remove unused allocated space after creation.
*/
template <class ExecutionSpace, class InitFunctor, class ParticleListType,
          class LocalGridType>
int createParticles(
    Cabana::InitUniform, const ExecutionSpace& exec_space,
    const InitFunctor& create_functor, ParticleListType& particle_list,
    const int particles_per_cell_dim, LocalGridType& local_grid,
    const std::size_t previous_num_particles = 0,
    const bool shrink_to_fit = true,
    typename std::enable_if<is_particle_list<ParticleListType>::value,
                            int>::type* = 0 )
{
    // Memory space.
    using memory_space = typename ParticleListType::memory_space;

    // Create a local mesh.
    auto local_mesh = createLocalMesh<ExecutionSpace>( local_grid );

    // Get the local set of owned cell indices.
    auto owned_cells = local_grid.indexSpace( Own(), Cell(), Local() );

    // Allocate enough space for particles fill the entire local grid.
    int particles_per_cell = particles_per_cell_dim * particles_per_cell_dim *
                             particles_per_cell_dim;
    int num_particles = particles_per_cell * owned_cells.size();
    particle_list.resize( previous_num_particles + num_particles );

    // Creation status.
    Kokkos::View<bool*, memory_space> particle_created(
        Kokkos::ViewAllocateWithoutInitializing( "particle_created" ),
        num_particles );

    // Initialize particles.
    int local_num_create = 0;
    grid_parallel_reduce(
        "Cabana::Grid::ParticleInit::Uniform", exec_space, owned_cells,
        KOKKOS_LAMBDA( const int i, const int j, const int k, int& count ) {
            // Compute the owned local cell id.
            int i_own = i - owned_cells.min( Dim::I );
            int j_own = j - owned_cells.min( Dim::J );
            int k_own = k - owned_cells.min( Dim::K );
            int cell_id =
                i_own + owned_cells.extent( Dim::I ) *
                            ( j_own + k_own * owned_cells.extent( Dim::J ) );

            // Get the coordinates of the low cell node.
            int low_node[3] = { i, j, k };
            double low_coords[3];
            local_mesh.coordinates( Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Node(), high_node, high_coords );

            // Compute the particle spacing in each dimension.
            double spacing[3] = { ( high_coords[Dim::I] - low_coords[Dim::I] ) /
                                      particles_per_cell_dim,
                                  ( high_coords[Dim::J] - low_coords[Dim::J] ) /
                                      particles_per_cell_dim,
                                  ( high_coords[Dim::K] - low_coords[Dim::K] ) /
                                      particles_per_cell_dim };

            // Particle coordinate.
            double px[3];

            // Particle volume.
            double pv =
                local_mesh.measure( Cell(), low_node ) / particles_per_cell;

            // Create particles.
            for ( int ip = 0; ip < particles_per_cell_dim; ++ip )
                for ( int jp = 0; jp < particles_per_cell_dim; ++jp )
                    for ( int kp = 0; kp < particles_per_cell_dim; ++kp )
                    {
                        // Set the particle position in logical coordinates.
                        px[Dim::I] = 0.5 * spacing[Dim::I] +
                                     ip * spacing[Dim::I] + low_coords[Dim::I];
                        px[Dim::J] = 0.5 * spacing[Dim::J] +
                                     jp * spacing[Dim::J] + low_coords[Dim::J];
                        px[Dim::K] = 0.5 * spacing[Dim::K] +
                                     kp * spacing[Dim::K] + low_coords[Dim::K];

                        // Local particle id.
                        int pid = cell_id * particles_per_cell + ip +
                                  particles_per_cell_dim *
                                      ( jp + particles_per_cell_dim * kp );
                        int new_pid = previous_num_particles + pid;

                        // Create a new particle with the given logical
                        // coordinates.
                        auto particle = particle_list.getParticle( new_pid );
                        particle_created( pid ) =
                            create_functor( new_pid, px, pv, particle );

                        // If we created a new particle insert it into the list.
                        if ( particle_created( pid ) )
                        {
                            particle_list.setParticle( particle, new_pid );
                            count++;
                        }
                    }
        },
        local_num_create );
    Kokkos::fence();

    // Filter empties.
    auto& aosoa = particle_list.aosoa();
    filterEmpties( exec_space, local_num_create, previous_num_particles,
                   particle_created, aosoa, shrink_to_fit );

    return particle_list.size();
}

/*!
  \brief Initialize uniform particles per cell given an initialization functor.

  \param tag Initialization type tag.
  \param create_functor A functor which populates a particle given the logical
  position of a particle. This functor returns true if a particle was created
  and false if it was not giving the signature:

      bool createFunctor( const double pid, const double px[3], const double pv,
                          typename ParticleAoSoA::tuple_type& particle );
  \param particle_list The ParticleList to populate. This will be filled with
  particles and resized to a size equal to the number of particles created.
  \param particles_per_cell_dim The number of particles to populate each cell
  dimension with.
  \param local_grid The LocalGrid over which particles will be created.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
   \param shrink_to_fit Optionally remove unused allocated space after creation.
*/
template <class InitFunctor, class ParticleListType, class LocalGridType>
int createParticles(
    Cabana::InitUniform tag, const InitFunctor& create_functor,
    ParticleListType& particle_list, const int particles_per_cell_dim,
    LocalGridType& local_grid, const std::size_t previous_num_particles = 0,
    const bool shrink_to_fit = true,
    typename std::enable_if<is_particle_list<ParticleListType>::value,
                            int>::type* = 0 )
{
    using exec_space = typename ParticleListType::memory_space::execution_space;
    return createParticles( tag, exec_space{}, create_functor, particle_list,
                            particles_per_cell_dim, local_grid,
                            previous_num_particles, shrink_to_fit );
}

//---------------------------------------------------------------------------//
/*!
  \brief Initialize a uniform number of particles in each cell.

  \param exec_space Kokkos execution space.
  \param positions Particle positions slice. This should be already the size of
  the number of grid cells times particles_per_cell.s
  \param particles_per_cell_dim The number of particles to populate each cell
  dimension with.
  \param local_grid The LocalGrid over which particles will be created.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
*/
template <class ExecutionSpace, class PositionType, class LocalGridType>
void createParticles(
    Cabana::InitUniform, const ExecutionSpace& exec_space,
    PositionType& positions, const int particles_per_cell_dim,
    LocalGridType& local_grid, const std::size_t previous_num_particles = 0,
    typename std::enable_if<( Cabana::is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    // Create a local mesh.
    auto local_mesh = createLocalMesh<ExecutionSpace>( local_grid );

    // Get the local set of owned cell indices.
    auto owned_cells = local_grid.indexSpace( Own(), Cell(), Local() );

    int particles_per_cell = particles_per_cell_dim * particles_per_cell_dim *
                             particles_per_cell_dim;

    // Ensure correct space for the particles.
    assert( positions.size() == static_cast<std::size_t>( particles_per_cell *
                                                          owned_cells.size() ) +
                                    previous_num_particles );

    // Initialize particles.
    grid_parallel_for(
        "Cabana::Grid::ParticleInit::Uniform", exec_space, owned_cells,
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            // Compute the owned local cell id.
            int i_own = i - owned_cells.min( Dim::I );
            int j_own = j - owned_cells.min( Dim::J );
            int k_own = k - owned_cells.min( Dim::K );
            int cell_id =
                i_own + owned_cells.extent( Dim::I ) *
                            ( j_own + k_own * owned_cells.extent( Dim::J ) );

            // Get the coordinates of the low cell node.
            int low_node[3] = { i, j, k };
            double low_coords[3];
            local_mesh.coordinates( Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Node(), high_node, high_coords );

            // Compute the particle spacing in each dimension.
            double spacing[3] = { ( high_coords[Dim::I] - low_coords[Dim::I] ) /
                                      particles_per_cell_dim,
                                  ( high_coords[Dim::J] - low_coords[Dim::J] ) /
                                      particles_per_cell_dim,
                                  ( high_coords[Dim::K] - low_coords[Dim::K] ) /
                                      particles_per_cell_dim };

            // Create particles.
            for ( int ip = 0; ip < particles_per_cell_dim; ++ip )
                for ( int jp = 0; jp < particles_per_cell_dim; ++jp )
                    for ( int kp = 0; kp < particles_per_cell_dim; ++kp )
                    {
                        // Local particle id.
                        int pid = previous_num_particles +
                                  cell_id * particles_per_cell + ip +
                                  particles_per_cell_dim *
                                      ( jp + particles_per_cell_dim * kp );

                        // Set the particle position in logical coordinates.
                        positions( pid, 0 ) = 0.5 * spacing[Dim::I] +
                                              ip * spacing[Dim::I] +
                                              low_coords[Dim::I];
                        positions( pid, 1 ) = 0.5 * spacing[Dim::J] +
                                              jp * spacing[Dim::J] +
                                              low_coords[Dim::J];
                        positions( pid, 2 ) = 0.5 * spacing[Dim::K] +
                                              kp * spacing[Dim::K] +
                                              low_coords[Dim::K];
                    }
        } );
}

/*!
  \brief Initialize a uniform number of particles in each cell.

  \param tag Initialization type tag.
  \param positions Particle positions slice. This should be already the size of
  the number of grid cells times particles_per_cell.s
  \param particles_per_cell_dim The number of particles to populate each cell
  dimension with.
  \param local_grid The LocalGrid over which particles will be created.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
*/
template <class PositionType, class LocalGridType>
void createParticles(
    Cabana::InitUniform tag, PositionType& positions,
    const int particles_per_cell_dim, LocalGridType& local_grid,
    const std::size_t previous_num_particles = 0,
    typename std::enable_if<( Cabana::is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    using exec_space = typename PositionType::execution_space;
    createParticles( tag, exec_space{}, positions, particles_per_cell_dim,
                     local_grid, previous_num_particles );
}
} // namespace Grid
} // namespace Cabana

namespace Cajita
{
//! \cond Deprecated
template <class... Args>
CAJITA_DEPRECATED auto createParticles( Args&&... args )
{
    return Cabana::Grid::createParticles( std::forward<Args>( args )... );
}
//! \endcond
} // namespace Cajita

#endif
