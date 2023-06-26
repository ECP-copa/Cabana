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
  \file Cajita_ParticleInit.hpp
  \brief Particle creation utilities based on uniform grids.
*/
#ifndef CAJITA_PARTICLEINIT_HPP
#define CAJITA_PARTICLEINIT_HPP

#include <Cajita_Parallel.hpp>
#include <Cajita_ParticleList.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <exception>

namespace Cajita
{

//---------------------------------------------------------------------------//
/*!
  \brief Filter out empty particles that weren't created.

  \param exec_space Kokkos execution space.
  \param local_num_create The number of particles created.
  \param aosoa The particle AoSoA.
  \param particle_created Whether to remove unused allocated space.
  \param shrink_to_fit Whether to remove unused allocated space.
*/
template <class CreationView, class ParticleAoSoA, class ExecutionSpace>
void filterEmpties( const ExecutionSpace& exec_space,
                    const int local_num_create,
                    const CreationView& particle_created, ParticleAoSoA& aosoa,
                    const bool shrink_to_fit )
{
    using memory_space = typename CreationView::memory_space;

    // Determine the empty particle positions in the compaction zone.
    int num_particles = aosoa.size();
    Kokkos::View<int*, memory_space> empties(
        Kokkos::ViewAllocateWithoutInitializing( "empties" ),
        std::min( num_particles - local_num_create, local_num_create ) );
    Kokkos::parallel_scan(
        "Cabana::ParticleInit::FindEmpty",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, local_num_create ),
        KOKKOS_LAMBDA( const int i, int& count, const bool final_pass ) {
            if ( !particle_created( i ) )
            {
                if ( final_pass )
                {
                    empties( count ) = i;
                }
                ++count;
            }
        } );

    // Compact the list so the it only has real particles.
    Kokkos::parallel_scan(
        "Cabana::ParticleInit::RemoveEmpty",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, local_num_create,
                                             num_particles ),
        KOKKOS_LAMBDA( const int i, int& count, const bool final_pass ) {
            if ( particle_created( i ) )
            {
                if ( final_pass )
                {
                    aosoa.setTuple( empties( count ), aosoa.getTuple( i ) );
                }
                ++count;
            }
        } );
    aosoa.resize( local_num_create );
    if ( shrink_to_fit )
        aosoa.shrinkToFit();
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
  \param shrink_to_fit Optionally remove unused allocated space after creation.
  \param seed Optional random seed for generating particles.
*/
template <class ExecutionSpace, class InitFunctor, class ParticleListType,
          class LocalGridType>
void createParticles(
    Cabana::InitRandom, const ExecutionSpace& exec_space,
    const InitFunctor& create_functor, ParticleListType& particle_list,
    const int particles_per_cell, LocalGridType& local_grid,
    const bool shrink_to_fit = true, const uint64_t seed = 123456,
    typename std::enable_if<Cajita::is_particle_list<ParticleListType>::value,
                            int>::type* = 0 )
{
    // Memory space.
    using memory_space = typename ParticleListType::memory_space;

    // Create a local mesh.
    auto local_mesh = Cajita::createLocalMesh<ExecutionSpace>( local_grid );

    // Get the global grid.
    const auto& global_grid = local_grid.globalGrid();

    // Get the local set of owned cell indices.
    auto owned_cells =
        local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

    // Create a random number generator.
    const auto local_seed =
        global_grid.blockId() + ( seed % ( global_grid.blockId() + 1 ) );
    using rnd_type = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    rnd_type pool;
    pool.init( local_seed, owned_cells.size() );

    // Get the aosoa.
    auto& aosoa = particle_list.aosoa();

    // Allocate enough space for the case the particles consume the entire
    // local grid.
    int num_particles = particles_per_cell * owned_cells.size();
    aosoa.resize( num_particles );

    // Creation status.
    auto particle_created = Kokkos::View<bool*, memory_space>(
        Kokkos::ViewAllocateWithoutInitializing( "particle_created" ),
        num_particles );

    // Initialize particles.
    int local_num_create = 0;
    Cajita::grid_parallel_reduce(
        "Cajita::ParticleInit::Random", exec_space, owned_cells,
        KOKKOS_LAMBDA( const int i, const int j, const int k,
                       int& create_count ) {
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
            local_mesh.coordinates( Cajita::Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Cajita::Node(), high_node, high_coords );

            // Random number generator.
            auto rand = pool.get_state( cell_id );

            // Particle coordinate.
            double px[3];

            // Particle volume.
            double pv = local_mesh.measure( Cajita::Cell(), low_node ) /
                        particles_per_cell;

            // Create particles.
            for ( int p = 0; p < particles_per_cell; ++p )
            {
                // Local particle id.
                int pid = cell_id * particles_per_cell + p;

                // Select a random point in the cell for the particle
                // location. These coordinates are logical.
                for ( int d = 0; d < 3; ++d )
                {
                    px[d] = Kokkos::rand<decltype( rand ), double>::draw(
                        rand, low_coords[d], high_coords[d] );
                }

                // Create a new particle with the given logical coordinates.
                auto particle = particle_list.getParticle( pid );
                particle_created( pid ) =
                    create_functor( pid, px, pv, particle );

                // If we created a new particle insert it into the list.
                if ( particle_created( pid ) )
                {
                    particle_list.setParticle( particle, pid );
                    ++create_count;
                }
            }
        },
        local_num_create );

    // Filter empties.
    filterEmpties( exec_space, local_num_create, particle_created, aosoa,
                   shrink_to_fit );
}

/*!
  \brief Initialize random particles per cell given an initialization functor.

  \tparam ParticleListType The type of particle list to initialize.
  \tparam InitFunctor Initialization functor type. See the documentation below
  for the create_functor parameter on the signature of this functor.

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
  \param shrink_to_fit Optionally remove unused allocated space after creation.
  \param seed Optional random seed for generating particles.
*/
template <class InitFunctor, class ParticleListType, class LocalGridType>
void createParticles(
    Cabana::InitRandom tag, const InitFunctor& create_functor,
    ParticleListType& particle_list, const int particles_per_cell,
    LocalGridType& local_grid, const bool shrink_to_fit = true,
    const uint64_t seed = 123456,
    typename std::enable_if<Cajita::is_particle_list<ParticleListType>::value,
                            int>::type* = 0 )
{
    using exec_space = typename ParticleListType::memory_space::execution_space;
    createParticles( tag, exec_space{}, create_functor, particle_list,
                     particles_per_cell, local_grid, shrink_to_fit, seed );
}

//---------------------------------------------------------------------------//
/*!
  \brief Initialize a random number of particles in each cell.

  \param exec_space Kokkos execution space.
  \param positions Particle positions slice. This should be already the size of
  the number of grid cells times particles_per_cell.s
  \param particles_per_cell The number of particles to sample each cell with.
  \param local_grid The LocalGrid over which particles will be created.
  \param seed Optional random seed for generating particles.
*/
template <class ExecutionSpace, class PositionType, class LocalGridType>
void createParticles(
    Cabana::InitRandom, const ExecutionSpace& exec_space,
    PositionType& positions, const int particles_per_cell,
    LocalGridType& local_grid, const uint64_t seed = 123456,
    typename std::enable_if<( Cabana::is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    // Create a local mesh.
    auto local_mesh = Cajita::createLocalMesh<ExecutionSpace>( local_grid );

    // Get the global grid.
    const auto& global_grid = local_grid.globalGrid();

    // Get the local set of owned cell indices.
    auto owned_cells =
        local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

    // Create a random number generator.
    const auto local_seed =
        global_grid.blockId() + ( seed % ( global_grid.blockId() + 1 ) );
    using rnd_type = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    rnd_type pool;
    pool.init( local_seed, owned_cells.size() );

    // Ensure correct space for the particles.
    assert( positions.size() == static_cast<std::size_t>(
                                    particles_per_cell * owned_cells.size() ) );

    // Initialize particles.
    Cajita::grid_parallel_for(
        "Cajita::ParticleInit::Random", exec_space, owned_cells,
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
            local_mesh.coordinates( Cajita::Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Cajita::Node(), high_node, high_coords );

            // Random number generator.
            auto rand = pool.get_state( cell_id );

            // Create particles.
            for ( int p = 0; p < particles_per_cell; ++p )
            {
                // Local particle id.
                int pid = cell_id * particles_per_cell + p;

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
  \param seed Optional random seed for generating particles.
*/
template <class PositionType, class LocalGridType>
void createParticles(
    Cabana::InitRandom tag, PositionType& positions,
    const int particles_per_cell, LocalGridType& local_grid,
    const uint64_t seed = 123456,
    typename std::enable_if<( Cabana::is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    using exec_space = typename PositionType::execution_space;
    createParticles( tag, exec_space{}, positions, particles_per_cell,
                     local_grid, seed );
}

//---------------------------------------------------------------------------//
/*!
  \brief Initialize uniform particles per cell given an initialization
  functor.

  \tparam ParticleListType The type of particle list to initialize.
  \tparam InitFunctor Initialization functor type. See the documentation
  below for the create_functor parameter on the signature of this functor.

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
  \param shrink_to_fit Optionally remove unused allocated space after creation.
*/
template <class ExecutionSpace, class InitFunctor, class ParticleListType,
          class LocalGridType>
void createParticles(
    Cabana::InitUniform, const ExecutionSpace& exec_space,
    const InitFunctor& create_functor, ParticleListType& particle_list,
    const int particles_per_cell_dim, LocalGridType& local_grid,
    const bool shrink_to_fit = true,
    typename std::enable_if<Cajita::is_particle_list<ParticleListType>::value,
                            int>::type* = 0 )
{
    // Memory space.
    using memory_space = typename ParticleListType::memory_space;

    // Create a local mesh.
    auto local_mesh = Cajita::createLocalMesh<ExecutionSpace>( local_grid );

    // Get the local set of owned cell indices.
    auto owned_cells =
        local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

    // Get the aosoa.
    auto& aosoa = particle_list.aosoa();

    // Allocate enough space for particles fill the entire local grid.
    int particles_per_cell = particles_per_cell_dim * particles_per_cell_dim *
                             particles_per_cell_dim;
    int num_particles = particles_per_cell * owned_cells.size();
    aosoa.resize( num_particles );

    // Creation status.
    auto particle_created = Kokkos::View<bool*, memory_space>(
        Kokkos::ViewAllocateWithoutInitializing( "particle_created" ),
        num_particles );

    // Initialize particles.
    int local_num_create = 0;
    Cajita::grid_parallel_reduce(
        "Cajita::ParticleInit::Uniform", exec_space, owned_cells,
        KOKKOS_LAMBDA( const int i, const int j, const int k,
                       int& create_count ) {
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
            local_mesh.coordinates( Cajita::Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Cajita::Node(), high_node, high_coords );

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
            double pv = local_mesh.measure( Cajita::Cell(), low_node ) /
                        particles_per_cell;

            // Create particles.
            for ( int ip = 0; ip < particles_per_cell_dim; ++ip )
                for ( int jp = 0; jp < particles_per_cell_dim; ++jp )
                    for ( int kp = 0; kp < particles_per_cell_dim; ++kp )
                    {
                        // Local particle id.
                        int pid = cell_id * particles_per_cell + ip +
                                  particles_per_cell_dim *
                                      ( jp + particles_per_cell_dim * kp );

                        // Set the particle position in logical coordinates.
                        px[Dim::I] = 0.5 * spacing[Dim::I] +
                                     ip * spacing[Dim::I] + low_coords[Dim::I];
                        px[Dim::J] = 0.5 * spacing[Dim::J] +
                                     jp * spacing[Dim::J] + low_coords[Dim::J];
                        px[Dim::K] = 0.5 * spacing[Dim::K] +
                                     kp * spacing[Dim::K] + low_coords[Dim::K];

                        // Create a new particle with the given logical
                        // coordinates.
                        auto particle = particle_list.getParticle( pid );
                        particle_created( pid ) =
                            create_functor( pid, px, pv, particle );

                        // If we created a new particle insert it into the list.
                        if ( particle_created( pid ) )
                        {
                            particle_list.setParticle( particle, pid );
                            ++create_count;
                        }
                    }
        },
        local_num_create );

    // Filter empties.
    filterEmpties( exec_space, local_num_create, particle_created, aosoa,
                   shrink_to_fit );
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
  \param shrink_to_fit Optionally remove unused allocated space after creation.
*/
template <class InitFunctor, class ParticleListType, class LocalGridType>
void createParticles(
    Cabana::InitUniform tag, const InitFunctor& create_functor,
    ParticleListType& particle_list, const int particles_per_cell_dim,
    LocalGridType& local_grid, const bool shrink_to_fit = true,
    typename std::enable_if<Cajita::is_particle_list<ParticleListType>::value,
                            int>::type* = 0 )
{
    using exec_space = typename ParticleListType::memory_space::execution_space;
    createParticles( tag, exec_space{}, create_functor, particle_list,
                     particles_per_cell_dim, local_grid, shrink_to_fit );
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
*/
template <class ExecutionSpace, class PositionType, class LocalGridType>
void createParticles(
    Cabana::InitUniform, const ExecutionSpace& exec_space,
    PositionType& positions, const int particles_per_cell_dim,
    LocalGridType& local_grid,
    typename std::enable_if<( Cabana::is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    // Create a local mesh.
    auto local_mesh = Cajita::createLocalMesh<ExecutionSpace>( local_grid );

    // Get the local set of owned cell indices.
    auto owned_cells =
        local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

    int particles_per_cell = particles_per_cell_dim * particles_per_cell_dim *
                             particles_per_cell_dim;

    // Ensure correct space for the particles.
    assert( positions.size() == static_cast<std::size_t>(
                                    particles_per_cell * owned_cells.size() ) );

    // Initialize particles.
    Cajita::grid_parallel_for(
        "Cajita::ParticleInit::Uniform", exec_space, owned_cells,
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
            local_mesh.coordinates( Cajita::Node(), low_node, low_coords );

            // Get the coordinates of the high cell node.
            int high_node[3] = { i + 1, j + 1, k + 1 };
            double high_coords[3];
            local_mesh.coordinates( Cajita::Node(), high_node, high_coords );

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
                        int pid = cell_id * particles_per_cell + ip +
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

//---------------------------------------------------------------------------//
/*!
  \brief Initialize a uniform number of particles in each cell.

  \param tag Initialization type tag.
  \param positions Particle positions slice. This should be already the size of
  the number of grid cells times particles_per_cell.s
  \param particles_per_cell_dim The number of particles to populate each cell
  dimension with.
  \param local_grid The LocalGrid over which particles will be created.
*/
template <class PositionType, class LocalGridType>
void createParticles(
    Cabana::InitUniform tag, PositionType& positions,
    const int particles_per_cell_dim, LocalGridType& local_grid,
    typename std::enable_if<( Cabana::is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    using exec_space = typename PositionType::execution_space;
    createParticles( tag, exec_space{}, positions, particles_per_cell_dim,
                     local_grid );
}
} // namespace Cajita

#endif
