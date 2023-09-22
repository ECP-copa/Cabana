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

#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Grid_LocalMesh.hpp>
#include <Cabana_Grid_ParticleInit.hpp>
#include <Cabana_Grid_ParticleList.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Cabana_DeepCopy.hpp>
#include <Cabana_Fields.hpp>
#include <Cabana_ParticleInit.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using Cabana::Grid::Dim;

namespace Test
{
//---------------------------------------------------------------------------//
int totalParticlesPerCell( Cabana::InitUniform, int ppc )
{
    return ppc * ppc * ppc;
}

int totalParticlesPerCell( Cabana::InitRandom, int ppc ) { return ppc; }

//---------------------------------------------------------------------------//
// Field tags.
struct Foo : public Cabana::Field::Vector<double, 3>
{
    static std::string label() { return "foo"; }
};

struct Bar : public Cabana::Field::Scalar<double>
{
    static std::string label() { return "bar"; }
};

//---------------------------------------------------------------------------//
template <class InitType>
void initParticleListTest( InitType init_type, int ppc,
                           const int multiplier = 1 )
{
    // Global bounding box.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 17, 10, 23 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    std::array<bool, 3> is_dim_periodic = { true, true, true };
    Cabana::Grid::DimBlockPartitioner<3> partitioner;
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, 0 );

    // Make a particle list.
    auto fields = Cabana::ParticleTraits<Foo, Bar>();
    auto particles = Cabana::Grid::createParticleList<TEST_MEMSPACE>(
        "test_particles", fields );
    using plist_type = decltype( particles );

    // Particle initialization functor.
    const Kokkos::Array<double, 6> box = {
        global_low_corner[Dim::I] + cell_size,
        global_high_corner[Dim::I] - cell_size,
        global_low_corner[Dim::J] + cell_size,
        global_high_corner[Dim::J] - cell_size,
        global_low_corner[Dim::K] + cell_size,
        global_high_corner[Dim::K] - cell_size };
    auto init_func =
        KOKKOS_LAMBDA( const int, const double x[3], const double v,
                       typename plist_type::particle_type& p )
    {
        // Put particles in a box that is one cell smaller than the global
        // mesh. This will give us a layer of empty cells.
        if ( x[Dim::I] > box[0] && x[Dim::I] < box[1] && x[Dim::J] > box[2] &&
             x[Dim::J] < box[3] && x[Dim::K] > box[4] && x[Dim::K] < box[5] )
        {
            for ( int d = 0; d < 3; ++d )
                get( p, Foo(), d ) = x[d];

            get( p, Bar() ) = v;
            return true;
        }
        else
        {
            return false;
        }
    };

    // Initialize particles (potentially multiple times).
    int created_particles = 0;
    int prev_particle = 0;
    for ( int m = 0; m < multiplier; ++m )
    {
        created_particles = Cabana::Grid::createParticles(
            init_type, TEST_EXECSPACE(), init_func, particles, ppc, *local_grid,
            prev_particle );
        prev_particle = created_particles;
    }
    // Check that we made particles.
    EXPECT_TRUE( created_particles > 0 );

    // Compute the global number of particles.
    int global_num_particle = particles.size();
    MPI_Allreduce( MPI_IN_PLACE, &global_num_particle, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    int expect_num_particle =
        multiplier * totalParticlesPerCell( init_type, ppc ) *
        ( global_grid->globalNumEntity( Cabana::Grid::Cell(), Dim::I ) - 2 ) *
        ( global_grid->globalNumEntity( Cabana::Grid::Cell(), Dim::J ) - 2 ) *
        ( global_grid->globalNumEntity( Cabana::Grid::Cell(), Dim::K ) - 2 );
    EXPECT_EQ( global_num_particle, expect_num_particle );

    // Particle volume.
    double cell_volume = global_mesh->cellSize( 0 ) *
                         global_mesh->cellSize( 1 ) *
                         global_mesh->cellSize( 2 );
    double volume = cell_volume / totalParticlesPerCell( init_type, ppc );

    // Check that all particles are in the box and got initialized correctly.
    auto host_particles =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), particles );
    for ( int p = 0; p < created_particles; ++p )
    {
        auto particle = host_particles.getParticle( p );

        for ( int d = 0; d < 3; ++d )
        {
            auto px = get( particle, Foo(), d );
            EXPECT_TRUE( px > box[2 * d] );
            EXPECT_TRUE( px < box[2 * d + 1] );
        }
        auto pv = get( particle, Bar() );
        EXPECT_DOUBLE_EQ( pv, volume );
    }
}

//---------------------------------------------------------------------------//
template <class InitType>
void initSliceTest( InitType init_type, int ppc, const int multiplier = 1 )
{
    // Global bounding box.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 21, 19, 7 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    std::array<bool, 3> is_dim_periodic = { true, true, true };
    Cabana::Grid::DimBlockPartitioner<3> partitioner;
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, 0 );
    auto owned_cells = local_grid->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

    int num_particle =
        owned_cells.size() * totalParticlesPerCell( init_type, ppc );
    Cabana::AoSoA<Cabana::MemberTypes<double[3]>, TEST_MEMSPACE> aosoa(
        "random", num_particle );
    auto positions = Cabana::slice<0>( aosoa );

    // Particle initialization functor.
    const Kokkos::Array<double, 6> box = {
        global_low_corner[Dim::I], global_high_corner[Dim::I],
        global_low_corner[Dim::J], global_high_corner[Dim::J],
        global_low_corner[Dim::K], global_high_corner[Dim::K] };

    // Initialize all particles (potentially multiple times).
    int prev_particle = 0;
    for ( int m = 0; m < multiplier; ++m )
    {
        aosoa.resize( prev_particle + num_particle );
        positions = Cabana::slice<0>( aosoa );
        Cabana::Grid::createParticles( init_type, TEST_EXECSPACE(), positions,
                                       ppc, *local_grid, prev_particle );
        prev_particle += num_particle;
    }

    // Check that we created all particles.
    int global_num_particle = positions.size();
    MPI_Allreduce( MPI_IN_PLACE, &global_num_particle, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    int expect_num_particle =
        multiplier * totalParticlesPerCell( init_type, ppc ) *
        global_grid->globalNumEntity( Cabana::Grid::Cell(), Dim::I ) *
        global_grid->globalNumEntity( Cabana::Grid::Cell(), Dim::J ) *
        global_grid->globalNumEntity( Cabana::Grid::Cell(), Dim::K );
    EXPECT_EQ( global_num_particle, expect_num_particle );

    // Check that all particles are in the box.
    auto host_aosoa =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto host_positions = Cabana::slice<0>( host_aosoa );
    for ( std::size_t p = 0; p < host_positions.size(); ++p )
    {
        for ( std::size_t d = 0; d < 3; ++d )
        {
            EXPECT_TRUE( host_positions( p, d ) > box[2 * d] );
            EXPECT_TRUE( host_positions( p, d ) < box[2 * d + 1] );
        }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, random_init_test )
{
    initParticleListTest( Cabana::InitRandom(), 17 );
    initSliceTest( Cabana::InitRandom(), 12 );
}

TEST( TEST_CATEGORY, uniform_init_test )
{
    initParticleListTest( Cabana::InitUniform(), 3 );
    initSliceTest( Cabana::InitUniform(), 2 );
}

TEST( TEST_CATEGORY, multiple_random_init_test )
{
    initParticleListTest( Cabana::InitRandom(), 5, 4 );
    initSliceTest( Cabana::InitRandom(), 9, 3 );
}

TEST( TEST_CATEGORY, multiple_uniform_init_test )
{
    initParticleListTest( Cabana::InitUniform(), 2, 5 );
    initSliceTest( Cabana::InitRandom(), 2, 3 );
}
//---------------------------------------------------------------------------//

} // end namespace Test
