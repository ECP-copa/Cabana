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
  \file Cabana_ParticleInit.hpp
  \brief Particle creation utilities.
*/
#ifndef CABANA_PARTICLEINIT_HPP
#define CABANA_PARTICLEINIT_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <Cabana_ParticleList.hpp>
#include <Cabana_Slice.hpp>
#include <Cabana_Utils.hpp>

#include <random>
#include <type_traits>

namespace Cabana
{

//---------------------------------------------------------------------------//
// Initialization type tags.

//! Uniform particle initialization type tag.
struct InitUniform
{
};
//! Random particle initialization type tag.
struct InitRandom
{
};

//---------------------------------------------------------------------------//
// Fully random
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
  \brief Initialize random particles given an initialization functor.

  \param exec_space Kokkos execution space.
  \param create_functor A functor which populates a particle given the logical
  position of a particle. This functor returns true if a particle was created
  and false if it was not giving the signature:

      bool createFunctor( const double pid, const double px[d], const double pv,
                          typename ParticleAoSoA::tuple_type& particle );
  \param particle_list The ParticleList to populate. This will be filled with
  particles and resized to a size equal to the number of particles created.
  \param num_particles The number of particles to create.
  \param box_min Array specifying lower corner to create particles within.
  \param box_max Array specifying upper corner to create particles within.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param shrink_to_fit Optionally remove unused allocated space after creation.
  \param seed Optional random seed for generating particles.

  \return Number of particles created.
*/
template <class ExecutionSpace, class InitFunctor, class ParticleListType,
          template <class, std::size_t, class...> class ArrayType, class Scalar,
          std::size_t NumSpaceDim, class... Args>
int createParticles(
    InitRandom, ExecutionSpace exec_space, const InitFunctor& create_functor,
    ParticleListType& particle_list, const std::size_t num_particles,
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
    const ArrayType<Scalar, NumSpaceDim, Args...> box_min,
    const ArrayType<Scalar, NumSpaceDim, Args...> box_max,
#else
    const ArrayType<Scalar, NumSpaceDim> box_min,
    const ArrayType<Scalar, NumSpaceDim> box_max,
#endif
    const std::size_t previous_num_particles = 0,
    const bool shrink_to_fit = true, const uint64_t seed = 342343901,
    typename std::enable_if<is_particle_list<ParticleListType>::value,
                            int>::type* = 0 )
{
    // Memory space.
    using memory_space = typename ParticleListType::memory_space;

    using PoolType = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    using RandomType = Kokkos::Random_XorShift64<ExecutionSpace>;
    PoolType pool( seed );

    // Resize the aosoa prior to lambda capture.
    auto& aosoa = particle_list.aosoa();
    aosoa.resize( previous_num_particles + num_particles );

    // Creation count.
    auto count = Kokkos::View<int*, memory_space>( "particle_count", 1 );
    Kokkos::deep_copy( count, previous_num_particles );

    // Copy corners to device accessible arrays.
    auto kokkos_min = copyArray( box_min );
    auto kokkos_max = copyArray( box_max );

    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        // Particle coordinate.
        double px[NumSpaceDim];

        auto gen = pool.get_state();
        auto particle = particle_list.getParticle( p );
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            px[d] = Kokkos::rand<RandomType, double>::draw( gen, kokkos_min[d],
                                                            kokkos_max[d] );
        pool.free_state( gen );

        // No volume information, so pass zero.
        int create = create_functor( count( 0 ), px, 0.0, particle );

        // If we created a new particle insert it into the list.
        if ( create )
        {
            auto c = Kokkos::atomic_fetch_add( &count( 0 ), 1 );
            particle_list.setParticle( particle, c );
        }
    };

    Kokkos::RangePolicy<ExecutionSpace> exec_policy(
        exec_space, previous_num_particles,
        previous_num_particles + num_particles );
    Kokkos::parallel_for( "Cabana::createParticles", exec_policy,
                          random_coord_op );
    Kokkos::fence();

    auto count_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), count );
    aosoa.resize( count_host( 0 ) );
    if ( shrink_to_fit )
        aosoa.shrinkToFit();
    return count_host( 0 );
}

//---------------------------------------------------------------------------//
/*!
  \brief Initialize random particles given an initialization functor.

  \param tag Initialization type tag.
  \param create_functor A functor which populates a particle given the logical
  position of a particle. This functor returns true if a particle was created
  and false if it was not giving the signature:

      bool createFunctor( const double pid, const double px[d], const double pv,
                          typename ParticleAoSoA::tuple_type& particle );
  \param particle_list The ParticleList to populate. This will be filled with
  particles and resized to a size equal to the number of particles created.
  \param num_particles The number of particles to create.
  \param box_min Array specifying lower corner to create particles within.
  \param box_max Array specifying upper corner to create particles within.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param shrink_to_fit Optionally remove unused allocated space after creation.
  \param seed Optional random seed for generating particles.

  \return Number of particles created.
*/
template <class InitFunctor, class ParticleListType,
          template <class, std::size_t, class...> class ArrayType, class Scalar,
          std::size_t NumSpaceDim, class... Args>
int createParticles( InitRandom tag, const InitFunctor& create_functor,
                     ParticleListType& particle_list,
                     const std::size_t num_particles,
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
                     const ArrayType<Scalar, NumSpaceDim, Args...> box_min,
                     const ArrayType<Scalar, NumSpaceDim, Args...> box_max,
#else
                     const ArrayType<Scalar, NumSpaceDim> box_min,
                     const ArrayType<Scalar, NumSpaceDim> box_max,
#endif
                     const std::size_t previous_num_particles = 0,
                     const bool shrink_to_fit = true,
                     const uint64_t seed = 342343901 )
{
    using exec_space = typename ParticleListType::memory_space::execution_space;
    return createParticles( tag, exec_space{}, create_functor, particle_list,
                            num_particles, box_min, box_max,
                            previous_num_particles, shrink_to_fit, seed );
}

/*!
  \brief Initialize random particles.

  \param exec_space Kokkos execution space.
  \param positions Particle positions slice.
  \param num_particles The number of particles to create.
  \param box_min Array specifying lower corner to create particles within.
  \param box_max Array specifying upper corner to create particles within.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param seed Optional random seed for generating particles.
*/
template <class ExecutionSpace, class PositionType,
          template <class, std::size_t, class...> class ArrayType, class Scalar,
          std::size_t NumSpaceDim, class... Args>
void createParticles(
    InitRandom, ExecutionSpace exec_space, PositionType& positions,
    const std::size_t num_particles,
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
    const ArrayType<Scalar, NumSpaceDim, Args...> box_min,
    const ArrayType<Scalar, NumSpaceDim, Args...> box_max,
#else
    const ArrayType<Scalar, NumSpaceDim> box_min,
    const ArrayType<Scalar, NumSpaceDim> box_max,
#endif
    const std::size_t previous_num_particles = 0,
    const uint64_t seed = 342343901,
    typename std::enable_if<( is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    // Ensure correct space for the particles (View or Slice).
    checkSize( positions, num_particles + previous_num_particles );

    // Copy corners to device accessible arrays.
    auto kokkos_min = copyArray( box_min );
    auto kokkos_max = copyArray( box_max );

    using PoolType = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    using RandomType = Kokkos::Random_XorShift64<ExecutionSpace>;
    PoolType pool( seed );
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            positions( p, d ) = Kokkos::rand<RandomType, double>::draw(
                gen, kokkos_min[d], kokkos_max[d] );
        pool.free_state( gen );
    };

    Kokkos::RangePolicy<ExecutionSpace> exec_policy(
        exec_space, previous_num_particles,
        previous_num_particles + num_particles );
    Kokkos::parallel_for( "Cabana::createParticles", exec_policy,
                          random_coord_op );
    Kokkos::fence();
}

/*!
  \brief Initialize random particles.

  \param tag Initialization type tag.
  \param positions Particle positions slice.
  \param num_particles The number of particles to create.
  \param box_min Array specifying lower corner to create particles within.
  \param box_max Array specifying upper corner to create particles within.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param seed Optional random seed for generating particles.
*/
template <class PositionType,
          template <class, std::size_t, class...> class ArrayType, class Scalar,
          std::size_t NumSpaceDim, class... Args>
void createParticles(
    InitRandom tag, PositionType& positions, const std::size_t num_particles,
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
    const ArrayType<Scalar, NumSpaceDim, Args...> box_min,
    const ArrayType<Scalar, NumSpaceDim, Args...> box_max,
#else
    const ArrayType<Scalar, NumSpaceDim> box_min,
    const ArrayType<Scalar, NumSpaceDim> box_max,
#endif
    const std::size_t previous_num_particles = 0,
    const uint64_t seed = 342343901,
    typename std::enable_if<( is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    using exec_space = typename PositionType::execution_space;
    createParticles( tag, exec_space{}, positions, num_particles, box_min,
                     box_max, previous_num_particles, seed );
}

/*!
  \brief Initialize random particles.

  \param tag Initialization type tag.
  \param exec_space Kokkos execution space.
  \param positions Particle positions slice.
  \param num_particles The number of particles to create.
  \param box_min Array specifying lower corner to create particles within.
  \param box_max Array specifying upper corner to create particles within.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param seed Optional random seed for generating particles.
*/
template <class ExecutionSpace, class PositionType, class Scalar>
void createParticles( InitRandom tag, ExecutionSpace exec_space,
                      PositionType& positions, const std::size_t num_particles,
                      const Scalar box_min[3], const Scalar box_max[3],
                      const std::size_t previous_num_particles = 0,
                      const uint64_t seed = 342343901 )
{
    auto kokkos_min = copyArray<Scalar, 3>( box_min );
    auto kokkos_max = copyArray<Scalar, 3>( box_max );
    createParticles( tag, exec_space, positions, num_particles, kokkos_min,
                     kokkos_max, previous_num_particles, seed );
}

/*!
  \brief Initialize random particles.

  \param tag Initialization type tag.
  \param positions Particle positions slice.
  \param num_particles The number of particles to create.
  \param box_min C-array specifying lower corner to create particles within.
  \param box_max C-array specifying upper corner to create particles within.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param seed Optional random seed for generating particles.
*/
template <class PositionType, class Scalar>
void createParticles( InitRandom tag, PositionType& positions,
                      const std::size_t num_particles, const Scalar box_min[3],
                      const Scalar box_max[3],
                      const std::size_t previous_num_particles = 0,
                      const uint64_t seed = 342343901 )
{
    using exec_space = typename PositionType::memory_space::execution_space;
    createParticles( tag, exec_space{}, positions, num_particles, box_min,
                     box_max, previous_num_particles, seed );
}

//---------------------------------------------------------------------------//
// Random with minimum separation
//---------------------------------------------------------------------------//

/*!
  \brief Initialize random particles with minimum separation.

  \param tag Initialization type tag.
  \param exec_space Kokkos execution space.
  \param create_functor A functor which populates a particle given the logical
  position of a particle. This functor returns true if a particle was created
  and false if it was not giving the signature:

      bool createFunctor( const double pid, const double px[d], const double pv,
                          typename ParticleAoSoA::tuple_type& particle );
  \param particle_list The ParticleList to populate. This will be filled with
  particles and resized to a size equal to the number of particles created.
  \param position_tag Position particle list type tag.
  \param num_particles The number of particles to create.
  \param min_dist Minimum separation distance between particles. Potential
  particles created within this distance of an existing particle are rejected.
  \param box_min Array specifying lower corner to create particles within.
  \param box_max Array specifying upper corner to create particles within.
  \param shrink_to_fit Optionally remove unused allocated space after creation.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param seed Optional random seed for generating particles.

  \return Number of particles created.

  \note This approximates many physical scenarios, e.g. atomic simulations.
*/
template <class ExecutionSpace, class InitFunctor, class ParticleListType,
          class PositionTag,
          template <class, std::size_t, class...> class ArrayType, class Scalar,
          std::size_t NumSpaceDim, class... Args>
int createParticles(
    InitRandom tag, ExecutionSpace exec_space,
    const InitFunctor& create_functor, ParticleListType& particle_list,
    PositionTag position_tag, const std::size_t num_particles,
    const double min_dist,
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
    const ArrayType<Scalar, NumSpaceDim, Args...> box_min,
    const ArrayType<Scalar, NumSpaceDim, Args...> box_max,
#else
    const ArrayType<Scalar, NumSpaceDim> box_min,
    const ArrayType<Scalar, NumSpaceDim> box_max,
#endif
    const std::size_t previous_num_particles = 0,
    const bool shrink_to_fit = true, const uint64_t seed = 342343901,
    typename std::enable_if<is_particle_list<ParticleListType>::value,
                            int>::type* = 0 )
{
    double min_dist_sqr = min_dist * min_dist;

    // Resize the aosoa prior to lambda capture.
    auto& aosoa = particle_list.aosoa();
    aosoa.resize( previous_num_particles + num_particles );

    auto positions = particle_list.slice( position_tag );

    // Create the functor which ignores particles within the radius of another.
    auto min_distance_op =
        KOKKOS_LAMBDA( const int id, const double px[NumSpaceDim], const double,
                       typename ParticleListType::particle_type& particle )
    {
        // Ensure this particle is not within the minimum distance of any other
        // existing particle.
        for ( int n = 0; n < id; n++ )
        {
            double dist = 0.0;
            for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            {
                double dx = positions( n, d ) - px[d];
                dist += dx * dx;
            }
            if ( dist < min_dist_sqr )
                return false;
        }

        bool create = create_functor( id, px, 0.0, particle );
        return create;
    };

    // Pass the functor to the general case.
    return createParticles( tag, exec_space, min_distance_op, particle_list,
                            num_particles, box_min, box_max,
                            previous_num_particles, shrink_to_fit, seed );
}

/*!
  \brief Initialize random particles with minimum separation.

  \param tag Initialization type tag.
  \param create_functor A functor which populates a particle given the logical
  position of a particle. This functor returns true if a particle was created
  and false if it was not giving the signature:

      bool createFunctor( const double pid, const double px[d], const double pv,
                          typename ParticleAoSoA::tuple_type& particle );
  \param particle_list The ParticleList to populate. This will be filled with
  particles and resized to a size equal to the number of particles created.
  \param position_tag Position particle list type tag.
  \param num_particles The number of particles to create.
  \param min_dist Minimum separation distance between particles. Potential
  particles created within this distance of an existing particle are rejected.
  \param box_min Array specifying lower corner to create particles within.
  \param box_max Array specifying upper corner to create particles within.
  \param shrink_to_fit Optionally remove unused allocated space after creation.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param seed Optional random seed for generating particles.

  \return Number of particles created.

  \note This approximates many physical scenarios, e.g. atomic simulations.
*/
template <class InitFunctor, class ParticleListType, class PositionTag,
          template <class, std::size_t, class...> class ArrayType, class Scalar,
          std::size_t NumSpaceDim, class... Args>
int createParticles( InitRandom tag, const InitFunctor& create_functor,
                     ParticleListType& particle_list, PositionTag position_tag,
                     const std::size_t num_particles, const double min_dist,
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
                     const ArrayType<Scalar, NumSpaceDim, Args...> box_min,
                     const ArrayType<Scalar, NumSpaceDim, Args...> box_max,
#else
                     const ArrayType<Scalar, NumSpaceDim> box_min,
                     const ArrayType<Scalar, NumSpaceDim> box_max,
#endif
                     const std::size_t previous_num_particles = 0,
                     const bool shrink_to_fit = true,
                     const uint64_t seed = 342343901 )
{
    using exec_space = typename ParticleListType::memory_space::execution_space;
    return createParticles( tag, exec_space{}, create_functor, particle_list,
                            position_tag, num_particles, min_dist, box_min,
                            box_max, previous_num_particles, shrink_to_fit,
                            seed );
}

/*!
  \brief Initialize random particles.

  \param tag Initialization type tag.
  \param exec_space Kokkos execution space.
  \param positions Particle positions slice.
  \param num_particles The number of particles to create.
  \param min_dist Minimum separation distance between particles. Potential
  particles created within this distance of an existing particle are rejected.
  \param box_min C-array specifying lower corner to create particles within.
  \param box_max C-array specifying upper corner to create particles within.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param seed Optional random seed for generating particles.
*/
template <class ExecutionSpace, class PositionType, class Scalar>
void createParticles( InitRandom tag, ExecutionSpace exec_space,
                      PositionType& positions, const std::size_t num_particles,
                      const double min_dist, const Scalar box_min[3],
                      const Scalar box_max[3],
                      const std::size_t previous_num_particles = 0,
                      const uint64_t seed = 342343901 )
{
    auto kokkos_min = copyArray<Scalar, 3>( box_min );
    auto kokkos_max = copyArray<Scalar, 3>( box_max );
    createParticles( tag, exec_space, positions, num_particles, min_dist,
                     kokkos_min, kokkos_max, previous_num_particles, seed );
}

/*!
  \brief Initialize random particles.

  \param tag Initialization type tag.
  \param positions Particle positions slice.
  \param num_particles The number of particles to create.
  \param min_dist Minimum separation distance between particles. Potential
  particles created within this distance of an existing particle are rejected.
  \param box_min C-array specifying lower corner to create particles within.
  \param box_max C-array specifying upper corner to create particles within.
  \param previous_num_particles Optionally specify how many particles are
  already in the container (and should be unchanged).
  \param seed Optional random seed for generating particles.
*/
template <class PositionType, class Scalar>
void createParticles( InitRandom tag, PositionType& positions,
                      const std::size_t num_particles, const double min_dist,
                      const Scalar box_min[3], const Scalar box_max[3],
                      const std::size_t previous_num_particles = 0,
                      const uint64_t seed = 342343901 )
{
    using exec_space = typename PositionType::memory_space::execution_space;
    createParticles( tag, exec_space{}, positions, num_particles, min_dist,
                     box_min, box_max, previous_num_particles, seed );
}

} // namespace Cabana

#endif
