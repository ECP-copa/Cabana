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

/*!
  Generate random particles with minimum distance between neighbors. This
  approximates many physical scenarios, e.g. atomic simulations. Kokkos
  device version.
*/
template <class ExecutionSpace, class PositionType>
void createRandomParticlesMinDistance( ExecutionSpace, PositionType& positions,
                                       const std::size_t num_particles,
                                       const double box_min,
                                       const double box_max,
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
            pool.free_state( gen );
        }
    };
    Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, num_particles );
    Kokkos::parallel_for( exec_policy, random_coord_op );
    Kokkos::fence();
}

/*!
  Generate random particles with minimum distance between neighbors. This
  approximates many physical scenarios, e.g. atomic simulations. Kokkos
  device version with default execution space.
*/
template <class PositionType>
void createRandomParticlesMinDistance( PositionType& positions,
                                       const std::size_t num_particles,
                                       const double box_min,
                                       const double box_max,
                                       const double min_dist )
{
    using exec_space = typename PositionType::execution_space;
    createRandomParticlesMinDistance( exec_space{}, positions, num_particles,
                                      box_min, box_max, min_dist );
}

//! Generate random particles. Kokkos device version.
template <class ExecutionSpace, class PositionType>
void createRandomParticles( ExecutionSpace, PositionType& positions,
                            const std::size_t num_particles,
                            const double box_min, const double box_max )
{
    using PoolType = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    using RandomType = Kokkos::Random_XorShift64<ExecutionSpace>;
    PoolType pool( 342343901 );
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( int d = 0; d < 3; ++d )
            positions( p, d ) =
                Kokkos::rand<RandomType, double>::draw( gen, box_min, box_max );
        pool.free_state( gen );
    };
    Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, num_particles );
    Kokkos::parallel_for( exec_policy, random_coord_op );
    Kokkos::fence();
}

/*!
  Generate random particles. Kokkos device version with default execution
  space.
*/
template <class PositionType>
void createRandomParticles( PositionType& positions,
                            const std::size_t num_particles,
                            const double box_min, const double box_max )
{
    using exec_space = typename PositionType::execution_space;
    createRandomParticles( exec_space{}, positions, num_particles, box_min,
                           box_max );
}

} // namespace Cabana

#endif
