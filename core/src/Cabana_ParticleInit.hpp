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
#include <Cabana_GaussianMixtureModel.hpp>
#include <impl/Cabana_Erfinv.hpp>
#include <impl/Cabana_Hammersley.hpp>
#include <impl/Cabana_Matrix2d.hpp>

#include <Cabana_ParticleList.hpp>
#include <Cabana_Slice.hpp>

#include <random>
#include <type_traits>

namespace Cabana
{

namespace Impl
{
//! Copy array (std, c-array) into Kokkos::Array for potential device use.
template <class ArrayType>
auto copyArray( ArrayType corner )
{
    using value_type = std::remove_reference_t<decltype( corner[0] )>;
    Kokkos::Array<value_type, 3> kokkos_corner;
    for ( std::size_t d = 0; d < 3; ++d )
        kokkos_corner[d] = corner[d];

    return kokkos_corner;
}

} // namespace Impl

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

      bool createFunctor( const double pid, const double px[3], const double pv,
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
          class ArrayType>
int createParticles(
    InitRandom, ExecutionSpace exec_space, const InitFunctor& create_functor,
    ParticleListType& particle_list, const std::size_t num_particles,
    const ArrayType box_min, const ArrayType box_max,
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
    auto kokkos_min = Impl::copyArray( box_min );
    auto kokkos_max = Impl::copyArray( box_max );

    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        // Particle coordinate.
        double px[3];

        auto gen = pool.get_state();
        auto particle = particle_list.getParticle( p );
        for ( int d = 0; d < 3; ++d )
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
    Kokkos::parallel_for( exec_policy, random_coord_op );
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

      bool createFunctor( const double pid, const double px[3], const double pv,
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
template <class InitFunctor, class ParticleListType, class ArrayType>
int createParticles( InitRandom tag, const InitFunctor& create_functor,
                     ParticleListType& particle_list,
                     const std::size_t num_particles, const ArrayType box_min,
                     const ArrayType box_max,
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
template <class ExecutionSpace, class PositionType, class ArrayType>
void createParticles(
    InitRandom, ExecutionSpace exec_space, PositionType& positions,
    const std::size_t num_particles, const ArrayType box_min,
    const ArrayType box_max, const std::size_t previous_num_particles = 0,
    const uint64_t seed = 342343901,
    typename std::enable_if<( is_slice<PositionType>::value ||
                              Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    // Ensure correct space for the particles (View or Slice).
    checkSize( positions, num_particles + previous_num_particles );

    // Copy corners to device accessible arrays.
    auto kokkos_min = Impl::copyArray( box_min );
    auto kokkos_max = Impl::copyArray( box_max );

    using PoolType = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    using RandomType = Kokkos::Random_XorShift64<ExecutionSpace>;
    PoolType pool( seed );
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( int d = 0; d < 3; ++d )
            positions( p, d ) = Kokkos::rand<RandomType, double>::draw(
                gen, kokkos_min[d], kokkos_max[d] );
        pool.free_state( gen );
    };

    Kokkos::RangePolicy<ExecutionSpace> exec_policy(
        exec_space, previous_num_particles,
        previous_num_particles + num_particles );
    Kokkos::parallel_for( exec_policy, random_coord_op );
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
template <class PositionType, class ArrayType>
void createParticles(
    InitRandom tag, PositionType& positions, const std::size_t num_particles,
    const ArrayType box_min, const ArrayType box_max,
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

//! Generate random particles.
template <class ExecutionSpace, class PositionType>
[[deprecated( "Use createParticles instead." )]] void
createRandomParticles( ExecutionSpace exec_space, PositionType& positions,
                       const std::size_t num_particles, const double box_min,
                       const double box_max )
{
    std::array<double, 3> array_min = { box_min, box_min, box_min };
    std::array<double, 3> array_max = { box_max, box_max, box_max };
    createParticles( InitRandom{}, exec_space, positions, num_particles,
                     array_min, array_max );
}

/*!
  Generate random particles. Default execution space.
*/
template <class PositionType>
[[deprecated( "Use createParticles instead." )]] void
createRandomParticles( PositionType& positions, const std::size_t num_particles,
                       const double box_min, const double box_max )
{
    using exec_space = typename PositionType::execution_space;
    createRandomParticles( exec_space{}, positions, num_particles, box_min,
                           box_max );
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

      bool createFunctor( const double pid, const double px[3], const double pv,
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
          class PositionTag, class ArrayType>
int createParticles(
    InitRandom tag, ExecutionSpace exec_space,
    const InitFunctor& create_functor, ParticleListType& particle_list,
    PositionTag position_tag, const std::size_t num_particles,
    const double min_dist, const ArrayType box_min, const ArrayType box_max,
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
        KOKKOS_LAMBDA( const int id, const double px[3], const double,
                       typename ParticleListType::particle_type& particle )
    {
        // Ensure this particle is not within the minimum distance of any other
        // existing particle.
        for ( int n = 0; n < id; n++ )
        {
            double dx = positions( n, 0 ) - px[0];
            double dy = positions( n, 1 ) - px[1];
            double dz = positions( n, 2 ) - px[2];
            double dist = dx * dx + dy * dy + dz * dz;
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

      bool createFunctor( const double pid, const double px[3], const double pv,
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
          class ArrayType>
int createParticles( InitRandom tag, const InitFunctor& create_functor,
                     ParticleListType& particle_list, PositionTag position_tag,
                     const std::size_t num_particles, const double min_dist,
                     const ArrayType box_min, const ArrayType box_max,
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
  \brief Generate random particles with minimum distance between neighbors.
  \note This approximates many physical scenarios, e.g. atomic simulations.

  \note This version will continue sampling until the number of selected
  particles is created. This can be extremely slow based on the requested box
  size and minimum separation distance.
*/
template <class PositionType>
[[deprecated( "Use createParticles instead." )]] void
createRandomParticlesMinDistance( PositionType& positions,
                                  const std::size_t num_particles,
                                  const double box_min, const double box_max,
                                  const double min_dist )
{
    using exec_space = typename PositionType::execution_space;
    createRandomParticlesMinDistance( exec_space{}, positions, num_particles,
                                      box_min, box_max, min_dist );
}

/*!
  \brief Generate random particles with minimum distance between neighbors.
  \note This approximates many physical scenarios, e.g. atomic simulations.

  This version will continue sampling until the number of selected particles is
  created. This can be extremely slow based on the requested box size and
  minimum separation distance.
*/
template <class ExecutionSpace, class PositionType>
[[deprecated( "Use createParticles instead." )]] void
createRandomParticlesMinDistance( ExecutionSpace exec_space,
                                  PositionType& positions,
                                  const std::size_t num_particles,
                                  const double box_min, const double box_max,
                                  const double min_dist )
{
    double min_dist_sqr = min_dist * min_dist;

    using PoolType = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    using RandomType = Kokkos::Random_XorShift64<ExecutionSpace>;
    PoolType pool( 342343901 );

    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();

        // Create particles. Only add particles that are outside a minimum
        // distance from other particles.
        bool found_neighbor = true;

        // Keep trying new random coordinates until we insert one that is
        // not within the minimum distance of any other particle.
        while ( found_neighbor )
        {
            found_neighbor = false;

            for ( int d = 0; d < 3; ++d )
                positions( p, d ) = Kokkos::rand<RandomType, double>::draw(
                    gen, box_min, box_max );

            for ( int n = 0; n < p; n++ )
            {
                double dx = positions( n, 0 ) - positions( p, 0 );
                double dy = positions( n, 1 ) - positions( p, 1 );
                double dz = positions( n, 2 ) - positions( p, 2 );
                double dist = dx * dx + dy * dy + dz * dz;
                if ( dist < min_dist_sqr )
                {
                    found_neighbor = true;
                    break;
                }
            }
        }
        pool.free_state( gen );
    };

    Kokkos::RangePolicy<ExecutionSpace> exec_policy( exec_space, 0,
                                                     num_particles );
    Kokkos::parallel_for( exec_policy, random_coord_op );
    Kokkos::fence();
}

/*!
  Populate the particles based on the description of the distribution function
  in gaussians. Particles are places in 1d1v phase space.
*/
template <typename CellSliceType, typename WeightSliceType,
          typename PositionSliceType, typename VelocitySliceType,
          typename GaussianType>
size_t initializeRandomParticles( CellSliceType& cell, WeightSliceType& macro,
                                  PositionSliceType& position_x,
                                  VelocitySliceType& velocity_x,
                                  const GaussianType& gaussians,
                                  const int seed )
{
    using gmm_float_type = typename GaussianType::value_type;
    using particle_float_type = typename WeightSliceType::value_type;
    Kokkos::Random_XorShift64_Pool<> random_pool( seed );

    auto g_dev = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), gaussians );
    Kokkos::deep_copy( g_dev, gaussians );

    // Define how to create ONE particle
    auto _init = KOKKOS_LAMBDA( const int s, const int i )
    {
        // acquire the state of the random number generator engine
        auto generator = random_pool.get_state();

        // Figure out which Cell this particle is in
        const int c = generator.drand( 0., g_dev.extent( 0 ) );

        // Pick a random location in that cell
        const particle_float_type x = generator.drand( c, c + 1. );

        // Figure out which Gaussian to draw this particle from
        const gmm_float_type r = generator.drand( 0., 1. );
        unsigned int j = 0;
        gmm_float_type sum = g_dev( c, j, Weight );
        while ( ( sum < r ) && ( j < g_dev.extent( 1 ) - 1 ) )
        {
            j++;
            sum += g_dev( c, j, Weight );
        }

        // Generate standard normal random variables
        const gmm_float_type rx = generator.normal( 0., 1. );

        velocity_x.access( s, i ) =
            g_dev( c, j, MuX ) + Kokkos::sqrt( g_dev( c, j, Cxx ) ) * rx;

        // Store the cell index
        cell.access( s, i ) = c;

        // Assign uniform particle weight
        macro.access( s, i ) = 1.;

        // Store particle position;
        position_x.access( s, i ) = x;

        // do not forget to release the state of the engine
        random_pool.free_state( generator );
    };

    // Define an execution policy
    Cabana::SimdPolicy<cell.vector_length, Kokkos::DefaultExecutionSpace>
        vec_policy( 0, cell.size() );

    // Execute
    Cabana::simd_parallel_for( vec_policy, _init, "init()" );

    return cell.size();
}

/*!
  Populate the particles based on the description of the distribution function
  in gaussians. Particles are places in 1d2v phase space with cylindrical
  velocity space.
*/
template <typename CellSliceType, typename WeightSliceType,
          typename PositionSliceType, typename VelocitySliceType,
          typename GaussianType>
size_t initializeRandomParticles( CellSliceType& cell, WeightSliceType& macro,
                                  PositionSliceType& position_x,
                                  VelocitySliceType& velocity_par,
                                  VelocitySliceType& velocity_per,
                                  const GaussianType& gaussians,
                                  const int seed )
{
    using gmm_float_type = typename GaussianType::value_type;
    using particle_float_type = typename WeightSliceType::value_type;
    Kokkos::Random_XorShift64_Pool<> random_pool( seed );

    auto g_dev = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), gaussians );
    Kokkos::deep_copy( g_dev, gaussians );

    // Define how to create ONE particle
    auto _init = KOKKOS_LAMBDA( const int s, const int i )
    {
        // acquire the state of the random number generator engine
        auto generator = random_pool.get_state();

        // Figure out which Cell this particle is in
        const int c = generator.drand( 0., g_dev.extent( 0 ) );

        // Pick a random location in that cell
        const particle_float_type x = generator.drand( c, c + 1. );

        // Figure out which Gaussian to draw this particle from
        const gmm_float_type r = generator.drand( 0., 1. );
        unsigned int j = 0;
        gmm_float_type sum = g_dev( c, j, Weight );
        while ( ( sum < r ) && ( j < g_dev.extent( 1 ) - 1 ) )
        {
            j++;
            sum += g_dev( c, j, Weight );
        }

        // TODO: Extend this to ring distributions that allow for correlations
        // between parallel and perpendicular velocity components
        //
        //  Put Covariance matrix of the jth Gaussian into 3x3 Matrix
        const gmm_float_type C[3][3] = { { g_dev( c, j, Cparpar ), 0, 0 },
                                         { 0, g_dev( c, j, Cperper ), 0 },
                                         { 0, 0, g_dev( c, j, Cperper ) } };
        gmm_float_type B[3][3];
        // Get Cholesky decomposition
        Cabana::Impl::Matrix2d<gmm_float_type, 3>::cholesky( B, C );

        // Generate standard normal random variables
        const gmm_float_type rx = generator.normal( 0., 1. );
        const gmm_float_type ry = generator.normal( 0., 1. );
        const gmm_float_type rz = generator.normal( 0., 1. );

        const particle_float_type vx =
            g_dev( c, j, MuPar ) + B[0][0] * rx + B[0][1] * ry + B[0][2] * rz;
        const particle_float_type vy =
            g_dev( c, j, MuPer ) + B[1][0] * rx + B[1][1] * ry + B[1][2] * rz;
        const particle_float_type vz =
            B[2][0] * rx + B[2][1] * ry + B[2][2] * rz;

        velocity_par.access( s, i ) = vx;
        velocity_per.access( s, i ) = Kokkos::sqrt( vy * vy + vz * vz );

        // Store the cell index
        cell.access( s, i ) = c;

        // Assign uniform particle weight
        macro.access( s, i ) = 1.;

        // Store particle position;
        position_x.access( s, i ) = x;

        // do not forget to release the state of the engine
        random_pool.free_state( generator );
    };

    // Define an execution policy
    Cabana::SimdPolicy<cell.vector_length, Kokkos::DefaultExecutionSpace>
        vec_policy( 0, cell.size() );

    // Execute
    Cabana::simd_parallel_for( vec_policy, _init, "init()" );

    return cell.size();
}

/*!
  Populate the particles based on the description of the distribution function
  in gaussians. Particles are places in 1d3v phase space.
*/
template <typename CellSliceType, typename WeightSliceType,
          typename PositionSliceType, typename VelocitySliceType,
          typename GaussianType>
size_t initializeRandomParticles( CellSliceType& cell, WeightSliceType& macro,
                                  PositionSliceType& position_x,
                                  VelocitySliceType& velocity_x,
                                  VelocitySliceType& velocity_y,
                                  VelocitySliceType& velocity_z,
                                  const GaussianType& gaussians,
                                  const int seed )
{
    using gmm_float_type = typename GaussianType::value_type;
    using particle_float_type = typename WeightSliceType::value_type;
    Kokkos::Random_XorShift64_Pool<> random_pool( seed );

    auto g_dev = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), gaussians );
    Kokkos::deep_copy( g_dev, gaussians );

    // Define how to create ONE particle
    auto _init = KOKKOS_LAMBDA( const int s, const int i )
    {
        // acquire the state of the random number generator engine
        auto generator = random_pool.get_state();

        // Figure out which Cell this particle is in
        const int c = generator.drand( 0., g_dev.extent( 0 ) );

        // Pick a random location in that cell
        const particle_float_type x = generator.drand( c, c + 1. );

        // Figure out which Gaussian to draw this particle from
        const gmm_float_type r = generator.drand( 0., 1. );
        unsigned int j = 0;
        gmm_float_type sum = g_dev( c, j, Weight );
        while ( ( sum < r ) && ( j < g_dev.extent( 1 ) - 1 ) )
        {
            j++;
            sum += g_dev( c, j, Weight );
        }

        // Put Covariance matrix of the jth Gaussian into 3x3 Matrix
        const gmm_float_type C[3][3] = {
            { g_dev( c, j, Cxx ), g_dev( c, j, Cxy ), g_dev( c, j, Cxz ) },
            { g_dev( c, j, Cyx ), g_dev( c, j, Cyy ), g_dev( c, j, Cyz ) },
            { g_dev( c, j, Czx ), g_dev( c, j, Czy ), g_dev( c, j, Czz ) } };
        gmm_float_type B[3][3];
        // Get Cholesky decomposition
        Cabana::Impl::Matrix2d<gmm_float_type, 3>::cholesky( B, C );

        // Generate standard normal random variables
        const particle_float_type rx = generator.normal( 0., 1. );
        const particle_float_type ry = generator.normal( 0., 1. );
        const particle_float_type rz = generator.normal( 0., 1. );

        velocity_x.access( s, i ) =
            g_dev( c, j, MuX ) + B[0][0] * rx + B[0][1] * ry + B[0][2] * rz;
        velocity_y.access( s, i ) =
            g_dev( c, j, MuY ) + B[1][0] * rx + B[1][1] * ry + B[1][2] * rz;
        velocity_z.access( s, i ) =
            g_dev( c, j, MuZ ) + B[2][0] * rx + B[2][1] * ry + B[2][2] * rz;

        // Store the cell index
        cell.access( s, i ) = c;

        // Assign uniform particle weight
        macro.access( s, i ) = 1.;

        // Store particle position;
        position_x.access( s, i ) = x;

        // do not forget to release the state of the engine
        random_pool.free_state( generator );
    };

    // Define an execution policy
    Cabana::SimdPolicy<cell.vector_length, Kokkos::DefaultExecutionSpace>
        vec_policy( 0, cell.size() );

    // Execute
    Cabana::simd_parallel_for( vec_policy, _init, "init()" );

    return cell.size();
}

/*!
 Populate the particles based on the description of the distribution function in
 gaussians
*/
template <typename CellSliceType, typename WeightSliceType,
          typename PositionSliceType, typename VelocitySliceType,
          typename GaussianType>
size_t initializeParticlesFromCDF( CellSliceType& cell, WeightSliceType& macro,
                                   PositionSliceType& position_x,
                                   VelocitySliceType& velocity_x,
                                   const GaussianType& gaussians,
                                   const int seed )
{
    using gmm_float_type = typename WeightSliceType::value_type;
    auto g_dev = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), gaussians );
    Kokkos::deep_copy( g_dev, gaussians );
    Kokkos::Random_XorShift64_Pool<> random_pool( seed );

    const int N_cells = gaussians.extent( 0 );
    const int N_gauss = gaussians.extent( 1 );
    const int N_particles = cell.size();

    int start = 0;
    for ( int c = 0; c < N_cells; c++ )
    {
        for ( int m = 0; m < N_gauss; m++ )
        {
            int Np = int(
                N_particles * gaussians( c, m, Weight ) /
                gmm_float_type(
                    N_cells ) ); // number of particles to add for that gaussian

            // Define how to create ONE particle in cell c, gaussian m
            auto _init = KOKKOS_LAMBDA( const int s, const int i )
            {
                // acquire the state of the random number generator engine
                auto generator = random_pool.get_state();

                int id = (s)*cell.vector_length + i;

                gmm_float_type cdf =
                    ( ( id - start ) + 1 ) / gmm_float_type( Np + 1 );

                // solution to 1/2 * (1+erf((x-mu)/(sqrt(2)*sigma))) == cdf
                gmm_float_type vx =
                    g_dev( c, m, MuX ) +
                    Kokkos::sqrt( g_dev( c, m, Cxx ) ) * Cabana::Impl::ppnd7( cdf );
                velocity_x.access( s, i ) = vx;

                // Store the cell index
                cell.access( s, i ) = c;

                // Assign uniform particle weight
                macro.access( s, i ) = 1.;

                // Generate and store a random position in that cell
                position_x.access( s, i ) = generator.drand( c, c + 1. );

                // do not forget to release the state of the engine
                random_pool.free_state( generator );
            };

            // Define an execution policy
            Cabana::SimdPolicy<cell.vector_length,
                               Kokkos::DefaultExecutionSpace>
                vec_policy( start, start + Np );

            // printf("Generating %d particles from %d to %d from Gaussian %d in
            // cell %d\n", Np, start, start+Np, m, c);

            // Execute
            Cabana::simd_parallel_for( vec_policy, _init, "init()" );

            start += Np;
        }
    }

    // TODO: we may not have created exactly N_particles particles. Do we want
    // to bother creating a few more to fix that?
    return ( start );
}

/*!
 Populate the particles based on the description of the distribution function in
 gaussians
*/
template <typename CellSliceType, typename WeightSliceType,
          typename PositionSliceType, typename VelocitySliceType,
          typename GaussianType>
size_t initializeEqualDensityParticlesWithHammersley(
    CellSliceType& cell, WeightSliceType& macro, PositionSliceType& position_x,
    VelocitySliceType& velocity_x, const GaussianType& gaussians )
{
    using gmm_float_type = typename WeightSliceType::value_type;
    auto g_dev = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), gaussians );
    Kokkos::deep_copy( g_dev, gaussians );

    const int N_cells = gaussians.extent( 0 ); // Number of cells
    const int N_gauss =
        gaussians.extent( 1 ); // Maximum number of Gaussians per cell
    const int N_particles_in_total =
        cell.size(); // Total number of particles the weight off all particles
                     // _also_ needs to add to this number

    int start = 0;
    for ( int c = 0; c < N_cells; c++ )
    {
        // number of physical particles in this cell
        // We assume that these get evenly distributed to all cells. We might
        // change that in the future to be proportional to the plasma density
        // in each cell.
        int N_particles_in_cell = N_particles_in_total / N_cells;

        // number of computational particles to add for each Gaussian
        Kokkos::View<double*> Ncd( "Nc double", N_gauss );
        Kokkos::View<int*> Nc( "Nc int", N_gauss );
        auto Ncd_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), Ncd );
        auto Nc_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), Nc );

        double sum = 0.;
        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) > 0. )
            {
                Ncd_host( m ) = Kokkos::pow( gaussians( c, m, Weight ), 1.0 );
                sum += Ncd_host( m );
            }
            else
            {
                Ncd_host( m ) = 0.;
            }
        }

        double norm = N_particles_in_cell / sum;

        N_particles_in_cell = 0;
        for ( int m = 0; m < N_gauss; m++ )
        {
            Nc_host( m ) = int( Ncd_host( m ) * norm );
            N_particles_in_cell += Nc_host( m );
            // if(Nc_host(m) > 0) {
            //    printf("Nc(%d,%d) = %d\n", c,m, Nc_host(m));
            // }
        }
        Kokkos::deep_copy( Nc, Nc_host );

        // number of physical particles for each Gaussian
        Kokkos::View<double*> Npd( "Np double", N_gauss );
        auto Npd_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), Npd );
        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) > 0. )
            {
                Npd_host( m ) = gaussians( c, m, Weight ) * N_particles_in_cell;
                // printf("Npd(%d,%d) = %f\n", c,m, Npd_host(m));
            }
            else
            {
                Npd_host( m ) = 0.;
            }
        }
        Kokkos::deep_copy( Npd, Npd_host );

        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) <= 0. )
            {
                continue;
            }
            // printf("\n");

            // Define how to create ONE particle in cell c, gaussian m
            auto _init = KOKKOS_LAMBDA( const int s, const int i )
            {
                int id = (s)*cell.vector_length + i;

                const gmm_float_type vx =
                    g_dev( c, m, MuX ) +
                    5. * Kokkos::sqrt( g_dev( c, m, Cxx ) ) *
                        ( 2. * Cabana::Impl::hammersley( 1, id - start, Nc( m ) ) -
                          1. );
                const gmm_float_type dv = 10. * Kokkos::sqrt( g_dev( c, m, Cxx ) );
                const gmm_float_type v[1] = { vx };
                const gmm_float_type Mu[1] = { g_dev( c, m, MuX ) };
                const gmm_float_type C[1][1] = { { g_dev( c, m, Cxx ) } };
                const gmm_float_type p =
                    Cabana::Impl::GaussianWeight<gmm_float_type>::weight_1d(
                        v, Mu, C );

                velocity_x.access( s, i ) = vx;

                // Store the cell index
                cell.access( s, i ) = c;

                // Assign uniform particle weight
                macro.access( s, i ) =
                    p * Npd( m ) / (gmm_float_type)Nc( m ) * dv;

                // Store particle position
                position_x.access( s, i ) =
                    c + Cabana::Impl::hammersley( 0, id - start + 1, Nc( m ) + 1 );
            };

            // Define an execution policy
            Cabana::SimdPolicy<cell.vector_length,
                               Kokkos::DefaultExecutionSpace>
                vec_policy( start, start + Nc_host( m ) );

            // printf("Generating %d particles from %d to %d from Gaussian %d in
            // cell %d\n", Nc_host(m), start, start+Nc_host(m), m, c);

            // Execute
            Cabana::simd_parallel_for( vec_policy, _init, "init()" );

            // Count weight of generated particles
            double count_of_gaussian = 0.;
            double weight_of_gaussian = 0.;
            auto _sum =
                KOKKOS_LAMBDA( const int i, double& lcount, double& lsum )
            {
                if ( ( start <= i ) && ( i < start + Nc( m ) ) )
                {
                    lcount += 1.;
                    lsum += macro( i );
                }
            };
            // printf("Checking particles %d to %d\n", start, start+Nc_host(m));
            Kokkos::parallel_reduce( "Particle Scan", N_particles_in_total,
                                     _sum, count_of_gaussian,
                                     weight_of_gaussian );
            // printf("Generated %f particles have weight %f\n",
            // count_of_gaussian, weight_of_gaussian);

            start += Nc_host( m );
        }
        // printf("\n");
    }

    // TODO: we may not have created exactly N_particles_in_total particles. Do
    // we want to bother creating a few more to fix that?
    return ( start );
}

/*!
  Populate the particles based on the description of the distribution function
  in gaussians. Particles are places in 1d2v phase space.
*/
template <typename CellSliceType, typename WeightSliceType,
          typename PositionSliceType, typename VelocitySliceType,
          typename GaussianType>
size_t initializeEqualDensityParticlesWithHammersley(
    CellSliceType& cell, WeightSliceType& macro, PositionSliceType& position_x,
    VelocitySliceType& velocity_par, VelocitySliceType& velocity_per,
    const GaussianType& gaussians )
{
    using gmm_float_type = typename WeightSliceType::value_type;
    auto g_dev = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), gaussians );
    Kokkos::deep_copy( g_dev, gaussians );

    const int N_cells = gaussians.extent( 0 ); // Number of cells
    const int N_gauss =
        gaussians.extent( 1 ); // Maximum number of Gaussians per cell
    const int N_particles_in_total =
        cell.size(); // Total number of particles the weight off all particles
                     // _also_ needs to add to this number

    int start = 0;
    for ( int c = 0; c < N_cells; c++ )
    {
        // number of physical particles in this cell
        // We assume that these get evenly distributed to all cells. We might
        // change that in the future to be proportional to the plasma density
        // in each cell.
        int N_particles_in_cell = N_particles_in_total / N_cells;

        // number of computational particles to add for each Gaussian
        Kokkos::View<double*> Ncd( "Nc double", N_gauss );
        Kokkos::View<int*> Nc( "Nc int", N_gauss );
        auto Ncd_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), Ncd );
        auto Nc_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), Nc );

        double sum = 0.;
        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) > 0. )
            {
                Ncd_host( m ) = Kokkos::pow( gaussians( c, m, Weight ), 1.0 );
                sum += Ncd_host( m );
            }
            else
            {
                Ncd_host( m ) = 0.;
            }
        }

        double norm = N_particles_in_cell / sum;

        N_particles_in_cell = 0;
        for ( int m = 0; m < N_gauss; m++ )
        {
            Nc_host( m ) = int( Ncd_host( m ) * norm );
            N_particles_in_cell += Nc_host( m );
            // if(Nc_host(m) > 0) {
            //    printf("Nc(%d,%d) = %d\n", c,m, Nc_host(m));
            // }
        }
        Kokkos::deep_copy( Nc, Nc_host );

        // number of physical particles for each Gaussian
        Kokkos::View<double*> Npd( "Np double", N_gauss );
        auto Npd_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), Npd );
        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) > 0. )
            {
                Npd_host( m ) = gaussians( c, m, Weight ) * N_particles_in_cell;
                // printf("Npd(%d,%d) = %f\n", c,m, Npd_host(m));
            }
            else
            {
                Npd_host( m ) = 0.;
            }
        }
        Kokkos::deep_copy( Npd, Npd_host );

        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) <= 0. )
            {
                continue;
            }
            // printf("\n");

            // Define how to create ONE particle in cell c, gaussian m
            auto _init = KOKKOS_LAMBDA( const int s, const int i )
            {
                int id = (s)*cell.vector_length + i;

                const gmm_float_type vpar =
                    g_dev( c, m, MuPar ) +
                    5. * Kokkos::sqrt( g_dev( c, m, Cparpar ) ) *
                        ( 2. * Cabana::Impl::hammersley( 1, id - start + 1,
                                                   Nc( m ) + 1 ) -
                          1. );
                const gmm_float_type vpermin =
                    Kokkos::max( 0., g_dev( c, m, MuPer ) -
                                         5. * Kokkos::sqrt( g_dev( c, m, Cperper ) ) );
                const gmm_float_type vpermax =
                    g_dev( c, m, MuPer ) + 5. * Kokkos::sqrt( g_dev( c, m, Cperper ) );
                const gmm_float_type vper =
                    vpermin +
                    ( vpermax - vpermin ) *
                        Cabana::Impl::hammersley( 2, id - start + 1, Nc( m ) + 1 );
                const gmm_float_type dvpar =
                    10. * Kokkos::sqrt( g_dev( c, m, Cparpar ) );
                const gmm_float_type dvper = ( vpermax - vpermin );
                const gmm_float_type v[2] = { vpar, vper };
                const gmm_float_type Mu[2] = { g_dev( c, m, MuPar ),
                                               g_dev( c, m, MuPer ) };
                const gmm_float_type C[2][2] = {
                    { g_dev( c, m, Cparpar ), g_dev( c, m, Cparper ) },
                    { g_dev( c, m, Cperpar ), g_dev( c, m, Cperper ) } };
                const gmm_float_type p =
                    Cabana::Impl::GaussianWeight<gmm_float_type>::weight_2d(
                        v, Mu, C );

                velocity_par.access( s, i ) = vpar;
                velocity_per.access( s, i ) = vper;

                // Store the cell index
                cell.access( s, i ) = c;

                // Assign uniform particle weight
                macro.access( s, i ) =
                    p * Npd( m ) / (gmm_float_type)Nc( m ) * dvpar * dvper;

                // Store particle position
                position_x.access( s, i ) =
                    c + Cabana::Impl::hammersley( 0, id - start + 1, Nc( m ) + 1 );
            };

            // Define an execution policy
            Cabana::SimdPolicy<cell.vector_length,
                               Kokkos::DefaultExecutionSpace>
                vec_policy( start, start + Nc_host( m ) );

            // printf("Generating %d particles from %d to %d from Gaussian %d in
            // cell %d\n", Nc_host(m), start, start+Nc_host(m), m, c);

            // Execute
            Cabana::simd_parallel_for( vec_policy, _init, "init()" );

            // Count weight of generated particles
            double count_of_gaussian = 0.;
            double weight_of_gaussian = 0.;
            auto _sum =
                KOKKOS_LAMBDA( const int i, double& lcount, double& lsum )
            {
                if ( ( start <= i ) && ( i < start + Nc( m ) ) )
                {
                    lcount += 1.;
                    lsum += macro( i );
                }
            };
            // printf("Checking particles %d to %d\n", start, start+Nc_host(m));
            Kokkos::parallel_reduce( "Particle Scan", N_particles_in_total,
                                     _sum, count_of_gaussian,
                                     weight_of_gaussian );
            // printf("Generated %f particles have weight %f\n",
            // count_of_gaussian, weight_of_gaussian);

            start += Nc_host( m );
        }
        // printf("\n");
    }

    // TODO: we may not have created exactly N_particles_in_total particles. Do
    // we want to bother creating a few more to fix that?
    return ( start );
}

/*!
  Populate the particles based on the description of the distribution function
  in gaussians. Particles are places in 1d3v phase space.
*/
template <typename CellSliceType, typename WeightSliceType,
          typename PositionSliceType, typename VelocitySliceType,
          typename GaussianType>
size_t initializeEqualDensityParticlesWithHammersley(
    CellSliceType& cell, WeightSliceType& macro, PositionSliceType& position_x,
    VelocitySliceType& velocity_x, VelocitySliceType& velocity_y,
    VelocitySliceType& velocity_z, const GaussianType& gaussians )
{
    using gmm_float_type = typename WeightSliceType::value_type;
    auto g_dev = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), gaussians );
    Kokkos::deep_copy( g_dev, gaussians );

    const int N_cells = gaussians.extent( 0 ); // Number of cells
    const int N_gauss =
        gaussians.extent( 1 ); // Maximum number of Gaussians per cell
    const int N_particles_in_total =
        cell.size(); // Total number of particles the weight off all particles
                     // _also_ needs to add to this number

    int start = 0;
    for ( int c = 0; c < N_cells; c++ )
    {
        // number of physical particles in this cell
        // We assume that these get evenly distributed to all cells. We might
        // change that in the future to be proportional to the plasma density
        // in each cell.
        int N_particles_in_cell = N_particles_in_total / N_cells;

        // number of computational particles to add for each Gaussian
        Kokkos::View<double*> Ncd( "Nc double", N_gauss );
        Kokkos::View<int*> Nc( "Nc int", N_gauss );
        auto Ncd_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), Ncd );
        auto Nc_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), Nc );

        double sum = 0.;
        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) > 0. )
            {
                Ncd_host( m ) = Kokkos::pow( gaussians( c, m, Weight ), 1.0 );
                sum += Ncd_host( m );
            }
            else
            {
                Ncd_host( m ) = 0.;
            }
        }

        double norm = N_particles_in_cell / sum;

        N_particles_in_cell = 0;
        for ( int m = 0; m < N_gauss; m++ )
        {
            Nc_host( m ) = int( Ncd_host( m ) * norm );
            N_particles_in_cell += Nc_host( m );
            // if(Nc_host(m) > 0) {
            //    printf("Nc(%d,%d) = %d\n", c,m, Nc_host(m));
            // }
        }
        Kokkos::deep_copy( Nc, Nc_host );

        // number of physical particles for each Gaussian
        Kokkos::View<double*> Npd( "Np double", N_gauss );
        auto Npd_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), Npd );
        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) > 0. )
            {
                Npd_host( m ) = gaussians( c, m, Weight ) * N_particles_in_cell;
                // printf("Npd(%d,%d) = %f\n", c,m, Npd_host(m));
            }
            else
            {
                Npd_host( m ) = 0.;
            }
        }
        Kokkos::deep_copy( Npd, Npd_host );

        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) <= 0. )
            {
                continue;
            }
            // printf("\n");

            // Define how to create ONE particle in cell c, gaussian m
            auto _init = KOKKOS_LAMBDA( const int s, const int i )
            {
                int id = (s)*cell.vector_length + i;

                const gmm_float_type vx =
                    g_dev( c, m, MuX ) +
                    3. * Kokkos::sqrt( g_dev( c, m, Cxx ) ) *
                        ( 2. * Cabana::Impl::hammersley( 1, id - start + 1,
                                                   Nc( m ) + 1 ) -
                          1. );
                const gmm_float_type vy =
                    g_dev( c, m, MuY ) +
                    3. * Kokkos::sqrt( g_dev( c, m, Cyy ) ) *
                        ( 2. * Cabana::Impl::hammersley( 2, id - start + 1,
                                                   Nc( m ) + 1 ) -
                          1. );
                const gmm_float_type vz =
                    g_dev( c, m, MuZ ) +
                    3. * Kokkos::sqrt( g_dev( c, m, Czz ) ) *
                        ( 2. * Cabana::Impl::hammersley( 3, id - start + 1,
                                                   Nc( m ) + 1 ) -
                          1. );
                const gmm_float_type dvx = 6. * Kokkos::sqrt( g_dev( c, m, Cxx ) );
                const gmm_float_type dvy = 6. * Kokkos::sqrt( g_dev( c, m, Cyy ) );
                const gmm_float_type dvz = 6. * Kokkos::sqrt( g_dev( c, m, Czz ) );

                const gmm_float_type v[3] = { vx, vy, vz };
                const gmm_float_type Mu[3] = { g_dev( c, m, MuX ),
                                               g_dev( c, m, MuY ),
                                               g_dev( c, m, MuZ ) };
                const gmm_float_type C[3][3] = {
                    { g_dev( c, m, Cxx ), g_dev( c, m, Cxy ),
                      g_dev( c, m, Cxz ) },
                    { g_dev( c, m, Cyx ), g_dev( c, m, Cyy ),
                      g_dev( c, m, Cyz ) },
                    { g_dev( c, m, Czx ), g_dev( c, m, Czy ),
                      g_dev( c, m, Czz ) } };
                const gmm_float_type p =
                    Cabana::Impl::GaussianWeight<gmm_float_type>::weight_3d(
                        v, Mu, C );

                velocity_x.access( s, i ) = vx;
                velocity_y.access( s, i ) = vy;
                velocity_z.access( s, i ) = vz;

                // Store the cell index
                cell.access( s, i ) = c;

                // Assign particle weight
                macro.access( s, i ) =
                    p * Npd( m ) / (gmm_float_type)Nc( m ) * dvx * dvy * dvz;

                // Store particle position
                position_x.access( s, i ) =
                    c + Cabana::Impl::hammersley( 0, id - start + 1, Nc( m ) + 1 );
            };

            // Define an execution policy
            Cabana::SimdPolicy<cell.vector_length,
                               Kokkos::DefaultExecutionSpace>
                vec_policy( start, start + Nc_host( m ) );

            // printf("Generating %d particles from %d to %d from Gaussian %d in
            // cell %d\n", Nc_host(m), start, start+Nc_host(m), m, c);

            // Execute
            Cabana::simd_parallel_for( vec_policy, _init, "init()" );

            // Count weight of generated particles
            double count_of_gaussian = 0.;
            double weight_of_gaussian = 0.;
            auto _sum =
                KOKKOS_LAMBDA( const int i, double& lcount, double& lsum )
            {
                if ( ( start <= i ) && ( i < start + Nc( m ) ) )
                {
                    lcount += 1.;
                    lsum += macro( i );
                }
            };
            // printf("Checking particles %d to %d\n", start, start+Nc_host(m));
            Kokkos::parallel_reduce( "Particle Scan", N_particles_in_total,
                                     _sum, count_of_gaussian,
                                     weight_of_gaussian );
            // printf("Generated %f particles have weight %f\n",
            // count_of_gaussian, weight_of_gaussian);

            start += Nc_host( m );
        }
        // printf("\n");
    }

    // TODO: we may not have created exactly N_particles_in_total particles. Do
    // we want to bother creating a few more to fix that?
    return ( start );
}

/*!
  Populate the particles based on the description of the distribution function
  in gaussians
*/
template <typename CellSliceType, typename WeightSliceType,
          typename PositionSliceType, typename VelocitySliceType,
          typename GaussianType>
size_t initializeEqualWeightParticlesWithHammersley(
    CellSliceType& cell, WeightSliceType& macro, PositionSliceType& position_x,
    VelocitySliceType& velocity_x, const GaussianType& gaussians )
{
    using gmm_float_type = typename WeightSliceType::value_type;
    auto g_dev = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), gaussians );
    Kokkos::deep_copy( g_dev, gaussians );

    const int N_cells = gaussians.extent( 0 );
    const int N_gauss = gaussians.extent( 1 );
    const int N_particles = cell.size();

    int start = 0;
    for ( int c = 0; c < N_cells; c++ )
    {
        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) <= 0. )
            {
                continue;
            }

            int Np = int(
                N_particles * gaussians( c, m, Weight ) /
                gmm_float_type(
                    N_cells ) ); // number of particles to add for that gaussian

            // Define how to create ONE particle in cell c, gaussian m
            auto _init = KOKKOS_LAMBDA( const int s, const int i )
            {
                int id = (s)*cell.vector_length + i;

                // Generate uniform pseudo-radom variables
                const gmm_float_type w =
                    Cabana::Impl::hammersley( 1, id - start + 1, Np + 1 );
                const gmm_float_type x =
                    Cabana::Impl::hammersley( 2, id - start + 1, Np + 1 );

                // Generate standard normal random variables
                const gmm_float_type rx =
                    Kokkos::sqrt( -2. * Kokkos::log( w ) ) * Kokkos::sin( 2. * M_PI * x );

                velocity_x.access( s, i ) =
                    g_dev( c, m, MuX ) + Kokkos::sqrt( g_dev( c, m, Cxx ) ) * rx;

                // Store the cell index
                cell.access( s, i ) = c;

                // Assign uniform particle weight
                macro.access( s, i ) = 1.;

                // Store particle position
                position_x.access( s, i ) =
                    c + Cabana::Impl::hammersley( 0, id - start, Np );
            };

            // Define an execution policy
            Cabana::SimdPolicy<cell.vector_length,
                               Kokkos::DefaultExecutionSpace>
                vec_policy( start, start + Np );

            // printf("Generating %d particles from %d to %d from Gaussian %d in
            // cell %d\n", Np, start, start+Np, m, c);

            // Execute
            Cabana::simd_parallel_for( vec_policy, _init, "init()" );

            start += Np;
        }
    }

    // TODO: we may not have created exactly N_particles particles. Do we want
    // to bother creating a few more to fix that?
    return ( start );
}

/*!
  Populate the particles based on the description of the distribution function
  in gaussians. Particles are places in 1d2v phase space.
*/
template <typename CellSliceType, typename WeightSliceType,
          typename PositionSliceType, typename VelocitySliceType,
          typename GaussianType>
size_t initializeEqualWeightParticlesWithHammersley(
    CellSliceType& cell, WeightSliceType& macro, PositionSliceType& position_x,
    VelocitySliceType& velocity_par, VelocitySliceType& velocity_per,
    const GaussianType& gaussians )
{
    using gmm_float_type = typename WeightSliceType::value_type;
    auto g_dev = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), gaussians );
    Kokkos::deep_copy( g_dev, gaussians );

    const int N_cells = gaussians.extent( 0 );
    const int N_gauss = gaussians.extent( 1 );
    const int N_particles = cell.size();

    int start = 0;
    for ( int c = 0; c < N_cells; c++ )
    {
        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) <= 0. )
            {
                continue;
            }

            int Np = int(
                N_particles * gaussians( c, m, Weight ) /
                gmm_float_type(
                    N_cells ) ); // number of particles to add for that gaussian

            // Define how to create ONE particle in cell c, gaussian m
            auto _init = KOKKOS_LAMBDA( const int s, const int i )
            {
                int id = (s)*cell.vector_length + i;

                // TODO: Extend this to ring distributions that allow for correlations
                // between parallel and perpendicular velocity components
                //
                //  Put Covariance matrix of the jth Gaussian into 3x3 Matrix
                const gmm_float_type C[3][3] = {
                    { g_dev( c, m, Cparpar ), 0, 0 },
                    { 0, g_dev( c, m, Cperper ), 0 },
                    { 0, 0, g_dev( c, m, Cperper ) } };
                gmm_float_type B[3][3];
                // Get Cholesky decomposition
                Cabana::Impl::Matrix2d<gmm_float_type, 3>::cholesky( B, C );

                // Generate uniform pseudo-radom variables
                const gmm_float_type w =
                    Cabana::Impl::hammersley( 1, id - start + 1, Np + 1 );
                const gmm_float_type x =
                    Cabana::Impl::hammersley( 2, id - start + 1, Np + 1 );
                const gmm_float_type y =
                    Cabana::Impl::hammersley( 3, id - start + 1, Np + 1 );
                const gmm_float_type z =
                    Cabana::Impl::hammersley( 4, id - start + 1, Np + 1 );

                // Generate standard normal random variables
                const gmm_float_type rx =
                    Kokkos::sqrt( -2. * Kokkos::log( w ) ) * Kokkos::sin( 2. * M_PI * x );
                const gmm_float_type ry =
                    Kokkos::sqrt( -2. * Kokkos::log( w ) ) * Kokkos::cos( 2. * M_PI * x );
                const gmm_float_type rz =
                    Kokkos::sqrt( -2. * Kokkos::log( y ) ) * Kokkos::sin( 2. * M_PI * z );

                const gmm_float_type vx = g_dev( c, m, MuPar ) + B[0][0] * rx +
                                          B[0][1] * ry + B[0][2] * rz;
                const gmm_float_type vy = g_dev( c, m, MuPer ) + B[1][0] * rx +
                                          B[1][1] * ry + B[1][2] * rz;
                const gmm_float_type vz =
                    B[2][0] * rx + B[2][1] * ry + B[2][2] * rz;

                velocity_par.access( s, i ) = vx;
                velocity_per.access( s, i ) = Kokkos::sqrt( vy * vy + vz * vz );

                // Store the cell index
                cell.access( s, i ) = c;

                // Assign uniform particle weight
                macro.access( s, i ) = 1.;

                // Store particle position
                position_x.access( s, i ) =
                    c + Cabana::Impl::hammersley( 0, id - start, Np );
            };

            // Define an execution policy
            Cabana::SimdPolicy<cell.vector_length,
                               Kokkos::DefaultExecutionSpace>
                vec_policy( start, start + Np );

            // printf("Generating %d particles from %d to %d from Gaussian %d in
            // cell %d\n", Np, start, start+Np, m, c);

            // Execute
            Cabana::simd_parallel_for( vec_policy, _init, "init()" );

            start += Np;
        }
    }

    // TODO: we may not have created exactly N_particles particles. Do we want
    // to bother creating a few more to fix that?
    return ( start );
}

/*!
  Populate the particles based on the description of the distribution function
  in gaussians. Particles are places in 1d3v phase space.
*/
template <typename CellSliceType, typename WeightSliceType,
          typename PositionSliceType, typename VelocitySliceType,
          typename GaussianType>
size_t initializeEqualWeightParticlesWithHammersley(
    CellSliceType& cell, WeightSliceType& macro, PositionSliceType& position_x,
    VelocitySliceType& velocity_x, VelocitySliceType& velocity_y,
    VelocitySliceType& velocity_z, const GaussianType& gaussians )
{
    using gmm_float_type = typename WeightSliceType::value_type;
    auto g_dev = Kokkos::create_mirror_view(
        Kokkos::DefaultExecutionSpace::memory_space(), gaussians );
    Kokkos::deep_copy( g_dev, gaussians );

    const int N_cells = gaussians.extent( 0 );
    const int N_gauss = gaussians.extent( 1 );
    const int N_particles = cell.size();

    int start = 0;
    for ( int c = 0; c < N_cells; c++ )
    {
        for ( int m = 0; m < N_gauss; m++ )
        {
            if ( gaussians( c, m, Weight ) <= 0. )
            {
                continue;
            }

            int Np = int(
                N_particles * gaussians( c, m, Weight ) /
                gmm_float_type(
                    N_cells ) ); // number of particles to add for that gaussian

            // Define how to create ONE particle in cell c, gaussian m
            auto _init = KOKKOS_LAMBDA( const int s, const int i )
            {
                int id = (s)*cell.vector_length + i;

                // Put Covariance matrix of the mth Gaussian into 3x3 Matrix
                const gmm_float_type C[3][3] = {
                    { g_dev( c, m, Cxx ), g_dev( c, m, Cxy ),
                      g_dev( c, m, Cxz ) },
                    { g_dev( c, m, Cyx ), g_dev( c, m, Cyy ),
                      g_dev( c, m, Cyz ) },
                    { g_dev( c, m, Czx ), g_dev( c, m, Czy ),
                      g_dev( c, m, Czz ) } };
                gmm_float_type B[3][3];
                // Get Cholesky decomposition
                Cabana::Impl::Matrix2d<gmm_float_type, 3>::cholesky( B, C );

                // Generate uniform pseudo-radom variables
                const gmm_float_type w =
                    Cabana::Impl::hammersley( 1, id - start + 1, Np + 1 );
                const gmm_float_type x =
                    Cabana::Impl::hammersley( 2, id - start + 1, Np + 1 );
                const gmm_float_type y =
                    Cabana::Impl::hammersley( 3, id - start + 1, Np + 1 );
                const gmm_float_type z =
                    Cabana::Impl::hammersley( 4, id - start + 1, Np + 1 );

                // Generate standard normal random variables
                const gmm_float_type rx =
                    Kokkos::sqrt( -2 * Kokkos::log( w ) ) * Kokkos::sin( 2. * M_PI * x );
                const gmm_float_type ry =
                    Kokkos::sqrt( -2 * Kokkos::log( w ) ) * Kokkos::cos( 2. * M_PI * x );
                const gmm_float_type rz =
                    Kokkos::sqrt( -2 * Kokkos::log( y ) ) * Kokkos::sin( 2. * M_PI * z );

                const gmm_float_type vx = g_dev( c, m, MuX ) + B[0][0] * rx +
                                          B[0][1] * ry + B[0][2] * rz;
                const gmm_float_type vy = g_dev( c, m, MuY ) + B[1][0] * rx +
                                          B[1][1] * ry + B[1][2] * rz;
                const gmm_float_type vz = g_dev( c, m, MuZ ) + B[2][0] * rx +
                                          B[2][1] * ry + B[2][2] * rz;

                velocity_x.access( s, i ) = vx;
                velocity_y.access( s, i ) = vy;
                velocity_z.access( s, i ) = vz;

                // Store the cell index
                cell.access( s, i ) = c;

                // Assign uniform particle weight
                macro.access( s, i ) = 1.;

                // Store particle position
                position_x.access( s, i ) =
                    c + Cabana::Impl::hammersley( 0, id - start, Np );
            };

            // Define an execution policy
            Cabana::SimdPolicy<cell.vector_length,
                               Kokkos::DefaultExecutionSpace>
                vec_policy( start, start + Np );

            // printf("Generating %d particles from %d to %d from Gaussian %d in
            // cell %d\n", Np, start, start+Np, m, c);

            // Execute
            Cabana::simd_parallel_for( vec_policy, _init, "init()" );

            start += Np;
        }
    }

    // TODO: we may not have created exactly N_particles particles. Do we want
    // to bother creating a few more to fix that?
    return ( start );
}



} // namespace Cabana

#endif
