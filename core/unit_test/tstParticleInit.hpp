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

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_ParticleInit.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{

struct Foo : Cabana::Field::Scalar<double>
{
    static std::string label() { return "foo"; }
};

struct Bar : Cabana::Field::Matrix<double, 3, 3>
{
    static std::string label() { return "bar"; }
};

template <class PositionType>
void checkRandomParticles( const int num_particle,
                           const Kokkos::Array<double, 3> box_min,
                           const Kokkos::Array<double, 3> box_max,
                           const PositionType host_positions )
{
    // Check that we got as many particles as we should have.
    EXPECT_EQ( host_positions.size(), num_particle );

    // Check that all of the particles are in the domain.
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_GE( host_positions( p, d ), box_min[d] );
            EXPECT_LE( host_positions( p, d ), box_max[d] );
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
            EXPECT_GE( dsqr, min_distance * min_distance );
        }
}

void testRandomCreationSlice( const int multiplier = 1 )
{
    int num_particle = 200;
    int prev_particle = 0;
    Cabana::AoSoA<Cabana::MemberTypes<double[3]>, TEST_MEMSPACE> aosoa(
        "random", num_particle );
    auto positions = Cabana::slice<0>( aosoa );

    Kokkos::Array<double, 3> box_min = { -9.5, -4.7, 0.5 };
    Kokkos::Array<double, 3> box_max = { 7.6, -1.5, 5.5 };

    for ( int m = 0; m < multiplier; ++m )
    {
        aosoa.resize( prev_particle + num_particle );
        positions = Cabana::slice<0>( aosoa );
        Cabana::createParticles( Cabana::InitRandom(), positions, num_particle,
                                 box_min, box_max, prev_particle );
        prev_particle += num_particle;
    }

    auto host_aosoa =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto host_positions = Cabana::slice<0>( host_aosoa );

    checkRandomParticles( multiplier * num_particle, box_min, box_max,
                          host_positions );
}

void testRandomCreationParticleListMinDistance( const int multiplier = 1 )
{
    int num_particle = 200;
    int prev_particle = 0;
    Cabana::ParticleTraits<Cabana::Field::Position<3>, Foo, Bar> fields;
    auto particle_list =
        Cabana::createParticleList<TEST_MEMSPACE>( "random_particles", fields );
    using plist_type = decltype( particle_list );

    double min_dist = 0.47;
    Kokkos::Array<double, 3> box_min = { -9.5, -4.7, 0.5 };
    Kokkos::Array<double, 3> box_max = { 7.6, -1.5, 5.5 };
    // Create all particles.
    auto init_func =
        KOKKOS_LAMBDA( const int, const double x[3], const double,
                       typename plist_type::particle_type& particle )
    {
        // Set positions.
        for ( int d = 0; d < 3; ++d )
            Cabana::get( particle, Cabana::Field::Position<3>(), d ) = x[d];

        return true;
    };

    int created_particles = 0;
    for ( int m = 0; m < multiplier; ++m )
    {
        created_particles = Cabana::createParticles(
            Cabana::InitRandom(), init_func, particle_list,
            Cabana::Field::Position<3>(), num_particle, min_dist, box_min,
            box_max, prev_particle );
        prev_particle = created_particles;
    }

    auto host_particle_list = Cabana::create_mirror_view_and_copy(
        Kokkos::HostSpace(), particle_list );
    auto host_positions =
        host_particle_list.slice( Cabana::Field::Position<3>() );

    // All particles may not have been created in this case (some skipped by the
    // minimum distance criterion).
    EXPECT_LE( created_particles, multiplier * num_particle );
    EXPECT_GE( host_positions.size(), 0 );
    checkRandomParticles( created_particles, box_min, box_max, host_positions );
    checkRandomDistances( min_dist, host_positions );
}

void testRandomCreationParticleList( const int multiplier = 1 )
{
    int num_particle = 200;
    int prev_particle = 0;
    Cabana::ParticleTraits<Cabana::Field::Position<3>, Foo, Bar> fields;
    auto particle_list =
        Cabana::createParticleList<TEST_MEMSPACE>( "random_particles", fields );
    using plist_type = decltype( particle_list );

    Kokkos::Array<double, 3> box_min = { -9.5, -4.7, 0.5 };
    Kokkos::Array<double, 3> box_max = { 7.6, -1.5, 5.5 };
    // Create all particles.
    auto init_func =
        KOKKOS_LAMBDA( const int, const double x[3], const double,
                       typename plist_type::particle_type& particle )
    {
        for ( int d = 0; d < 3; ++d )
            Cabana::get( particle, Cabana::Field::Position<3>(), d ) = x[d];

        return true;
    };
    int created_particles = 0;
    for ( int m = 0; m < multiplier; ++m )
    {
        created_particles = Cabana::createParticles(
            Cabana::InitRandom(), init_func, particle_list, num_particle,
            box_min, box_max, prev_particle );
        prev_particle = created_particles;
    }
    EXPECT_EQ( multiplier * num_particle, created_particles );

    auto host_particle_list = Cabana::create_mirror_view_and_copy(
        Kokkos::HostSpace(), particle_list );
    auto host_positions =
        host_particle_list.slice( Cabana::Field::Position<3>() );

    checkRandomParticles( multiplier * num_particle, box_min, box_max,
                          host_positions );
}

TEST( TEST_CATEGORY, random_particle_creation_slice_test )
{
    testRandomCreationSlice();
}
TEST( TEST_CATEGORY, random_particle_creation_particlelist_test )
{
    testRandomCreationParticleListMinDistance();
    testRandomCreationParticleList();
}
TEST( TEST_CATEGORY, multiple_random_particle_creation_slice_test )
{
    testRandomCreationSlice( 3 );
}
TEST( TEST_CATEGORY, multiple_random_particle_creation_particlelist_test )
{
    testRandomCreationParticleListMinDistance( 3 );
    testRandomCreationParticleList( 3 );
}
} // namespace Test
