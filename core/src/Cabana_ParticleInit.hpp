/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
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

#include <Kokkos_Random.hpp>

#include <random>
#include <type_traits>

namespace Cabana
{

//! Generate random particles with minimum distance between neighbors. This
// approximates many physical scenarios, e.g. atomic simulations. Kokkos
// device version.
template <class PositionType>
void createRandomParticlesMinDistance( PositionType& positions,
                                       const std::size_t num_particles,
                                       const double box_min,
                                       const double box_max,
                                       const double min_dist )
{
    using exec_space = typename PositionType::execution_space;

    double min_dist_sqr = min_dist * min_dist;

    using PoolType = Kokkos::Random_XorShift64_Pool<exec_space>;
    using RandomType = Kokkos::Random_XorShift64<exec_space>;
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
            pool.free_state( gen );
        }
    };
    Kokkos::RangePolicy<exec_space> exec_policy( 0, num_particles );
    Kokkos::parallel_for( exec_policy, random_coord_op );
    Kokkos::fence();
}

//! Generate random particles with minimum distance between neighbors. This
// approximates many physical scenarios, e.g. atomic simulations. Non-Kokkos
// host version.
template <class PositionType>
void createRandomParticlesMinDistanceHost( PositionType& positions,
                                           const std::size_t num_particles,
                                           const double box_min,
                                           const double box_max,
                                           const double min_dist )
{
    using memory_space = typename PositionType::memory_space;
    static_assert( std::is_same<memory_space, Kokkos::HostSpace>::value,
                   "Must use host space." );

    double min_dist_sqr = min_dist * min_dist;
    std::default_random_engine rand_gen;
    std::uniform_real_distribution<double> rand_dist( box_min, box_max );

    // Seed the distribution with a particle at the origin.
    positions( 0, 0 ) = ( box_max - box_min ) / 2.0;
    positions( 0, 1 ) = ( box_max - box_min ) / 2.0;
    positions( 0, 2 ) = ( box_max - box_min ) / 2.0;

    // Create particles. Only add particles that are outside a minimum
    // distance from other particles.
    for ( std::size_t p = 0; p < num_particles; ++p )
    {
        bool found_neighbor = true;

        // Keep trying new random coordinates until we insert one that is
        // not within the minimum distance of any other particle.
        while ( found_neighbor )
        {
            found_neighbor = false;

            // Create particle coordinates.
            positions( p, 0 ) = rand_dist( rand_gen );
            positions( p, 1 ) = rand_dist( rand_gen );
            positions( p, 2 ) = rand_dist( rand_gen );

            for ( std::size_t n = 0; n < p; n++ )
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
    }
}

//! Generate random particles. Kokkos device version.
template <class PositionType>
void createRandomParticles( PositionType& positions,
                            const std::size_t num_particles,
                            const double box_min, const double box_max )
{
    using exec_space = typename PositionType::execution_space;

    using PoolType = Kokkos::Random_XorShift64_Pool<exec_space>;
    using RandomType = Kokkos::Random_XorShift64<exec_space>;
    PoolType pool( 342343901 );
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( int d = 0; d < 3; ++d )
            positions( p, d ) =
                Kokkos::rand<RandomType, double>::draw( gen, box_min, box_max );
        pool.free_state( gen );
    };
    Kokkos::RangePolicy<exec_space> exec_policy( 0, num_particles );
    Kokkos::parallel_for( exec_policy, random_coord_op );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
//! Generate random particles. Non-Kokkos host version.
template <typename PositionType>
void createRandomParticlesHost( PositionType& positions,
                                const std::size_t num_particles,
                                const double box_min, const double box_max )
{
    using memory_space = typename PositionType::memory_space;
    static_assert( std::is_same<memory_space, Kokkos::HostSpace>::value,
                   "Must use host space." );

    using value_type = typename PositionType::value_type;

    // compute position range
    value_type box_range = box_max - box_min;

    // compute generator range
    std::minstd_rand0 generator( 3439203991 );
    value_type generator_range =
        static_cast<value_type>( generator.max() - generator.min() );

    // generate random particles
    for ( std::size_t n = 0; n < num_particles; ++n )
        for ( std::size_t d = 0; d < 3; ++d )
            positions( n, d ) =
                ( static_cast<value_type>( ( generator() - generator.min() ) ) *
                  box_range / generator_range ) +
                box_min;
}

} // namespace Cabana

#endif
