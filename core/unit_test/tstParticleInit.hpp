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

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_ParticleInit.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

template <class PositionType>
void checkRandomParticles( const int num_particle, const double box_min,
                           const double box_max,
                           const PositionType host_positions )
{
    // Check that we got as many particles as we should have.
    EXPECT_EQ( host_positions.size(), num_particle );

    // Check that all of the particles are in the domain.
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_GE( host_positions( p, d ), box_min );
            EXPECT_LE( host_positions( p, d ), box_max );
        }
}

template <class PositionType>
void checkRandomDistances( const int min_distance,
                           const PositionType host_positions )
{
    std::size_t num_particle = host_positions.size();

    // Check that none of the particles are are too close.
    for ( std::size_t i = 0; i < num_particle; ++i )
        for ( std::size_t j = 0; j < num_particle; ++j )
        {
            double dsqr = 0.0;
            for ( int d = 0; d < 3; ++d )
            {
                double diff = host_positions( i, d ) - host_positions( j, d );
                dsqr += diff * diff;
            }
            EXPECT_GE( dsqr, min_distance );
        }
}

void testRandomCreationMinDistance()
{
    int num_particle = 200;
    Cabana::AoSoA<Cabana::MemberTypes<double[3]>, TEST_MEMSPACE> aosoa(
        "random", num_particle );
    auto positions = Cabana::slice<0>( aosoa );

    double min_dist = 0.47;
    double box_min = -9.5;
    double box_max = 7.6;
    Cabana::createRandomParticlesMinDistance( positions, positions.size(),
                                              box_min, box_max, min_dist );
    auto host_aosoa =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto host_positions = Cabana::slice<0>( host_aosoa );

    checkRandomParticles( num_particle, box_min, box_max, host_positions );
    checkRandomDistances( min_dist, host_positions );
}

void testRandomCreationMinDistanceHost()
{
    int num_particle = 2000;
    Cabana::AoSoA<Cabana::MemberTypes<double[3]>, Kokkos::HostSpace> host_aosoa(
        "random", num_particle );
    auto host_positions = Cabana::slice<0>( host_aosoa );

    double min_dist = 0.47;
    double box_min = -9.5;
    double box_max = 7.6;
    Cabana::createRandomParticlesMinDistanceHost(
        host_positions, host_positions.size(), box_min, box_max, min_dist );

    checkRandomParticles( num_particle, box_min, box_max, host_positions );
    checkRandomDistances( min_dist, host_positions );
}

void testRandomCreation()
{
    int num_particle = 200;
    Cabana::AoSoA<Cabana::MemberTypes<double[3]>, TEST_MEMSPACE> aosoa(
        "random", num_particle );
    auto positions = Cabana::slice<0>( aosoa );

    double box_min = -9.5;
    double box_max = 7.6;
    Cabana::createRandomParticles( positions, positions.size(), box_min,
                                   box_max );
    auto host_aosoa =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto host_positions = Cabana::slice<0>( host_aosoa );

    checkRandomParticles( num_particle, box_min, box_max, host_positions );
}

void testRandomCreationHost()
{
    int num_particle = 200;
    Cabana::AoSoA<Cabana::MemberTypes<double[3]>, Kokkos::HostSpace> host_aosoa(
        "random", num_particle );
    auto host_positions = Cabana::slice<0>( host_aosoa );

    double box_min = -9.5;
    double box_max = 7.6;
    Cabana::createRandomParticles( host_positions, host_positions.size(),
                                   box_min, box_max );

    checkRandomParticles( num_particle, box_min, box_max, host_positions );
}

TEST( TEST_CATEGORY, random_particle_creation_test )
{
    testRandomCreationMinDistance();
    testRandomCreationMinDistanceHost();
    testRandomCreation();
    testRandomCreationHost();
}
