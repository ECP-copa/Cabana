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

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_Distributor.hpp>
#include <Cabana_Slice.hpp>

#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_GlobalParticleComm.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Grid_LocalMesh.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_Types.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

namespace Test
{

//---------------------------------------------------------------------------//
void testMigrate3d()
{
    std::array<bool, 3> is_dim_periodic = { true, true, true };

    // Create the global mesh.
    std::array<double, 3> global_low = { -1.2, 0.1, 1.1 };
    std::array<double, 3> global_high = { -0.3, 9.5, 2.3 };
    double cell_size = 0.05;
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low, global_high, cell_size );
    int num_particles = 200;

    // Create the global grid.
    Cabana::Grid::DimBlockPartitioner<3> partitioner;
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create a local grid
    int halo_width = 1;
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

    // Create the communication object.
    auto global_comm =
        Cabana::Grid::createGlobalParticleComm<TEST_MEMSPACE>( *local_grid );

    // Create random particles.
    using DataTypes = Cabana::MemberTypes<int, double[3]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t particles( "random", num_particles );
    auto position = Cabana::slice<1>( particles );

    using PoolType = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    using RandomType = Kokkos::Random_XorShift64<TEST_EXECSPACE>;
    PoolType pool( 174748 );

    // Copy box bounds to device array.
    Kokkos::Array<double, 3> global_low_kokkos;
    Kokkos::Array<double, 3> global_high_kokkos;
    for ( int d = 0; d < 3; ++d )
    {
        global_low_kokkos[d] = global_low[d];
        global_high_kokkos[d] = global_high[d];
    }

    // Create particles randomly in the global domain.
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( int d = 0; d < 3; ++d )
        {
            position( p, d ) = Kokkos::rand<RandomType, double>::draw(
                gen, global_low_kokkos[d], global_high_kokkos[d] );
        }
        pool.free_state( gen );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particles );
    Kokkos::parallel_for( policy, random_coord_op );
    Kokkos::fence();

    // Plan the communication.
    global_comm->build( position );

    // Move particles to the correct rank.
    global_comm->migrate( global_grid->comm(), particles );

    // Get the local domain bounds to check particles.
    auto local_mesh =
        Cabana::Grid::createLocalMesh<TEST_MEMSPACE>( *local_grid );
    std::array<double, 3> local_low = {
        local_mesh.lowCorner( Cabana::Grid::Own(), Cabana::Grid::Dim::I ),
        local_mesh.lowCorner( Cabana::Grid::Own(), Cabana::Grid::Dim::J ),
        local_mesh.lowCorner( Cabana::Grid::Own(), Cabana::Grid::Dim::K ) };
    std::array<double, 3> local_high = {
        local_mesh.highCorner( Cabana::Grid::Own(), Cabana::Grid::Dim::I ),
        local_mesh.highCorner( Cabana::Grid::Own(), Cabana::Grid::Dim::J ),
        local_mesh.highCorner( Cabana::Grid::Own(), Cabana::Grid::Dim::K ) };

    // Copy particles to the host.
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> particles_host(
        "migrated", particles.size() );
    Cabana::deep_copy( particles_host, particles );
    auto position_host = Cabana::slice<1>( particles_host );

    // Make sure the total particles were conserved.
    int global_particles;
    int local_particles = static_cast<int>( particles.size() );
    MPI_Reduce( &local_particles, &global_particles, 1, MPI_INT, MPI_SUM, 0,
                MPI_COMM_WORLD );
    if ( global_grid->blockId() == 0 )
    {
        EXPECT_EQ( global_particles,
                   num_particles * global_grid->totalNumBlock() );
    }

    for ( std::size_t p = 0; p < particles.size(); ++p )
    {
        // Check that all of the particles were moved to the correct local rank.
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_GE( position_host( p, d ), local_low[d] );
            EXPECT_LE( position_host( p, d ), local_high[d] );
        }
    }
}

void testMigrate2d()
{
    std::array<bool, 2> is_dim_periodic = { true, true };

    // Create the global mesh.
    std::array<double, 2> global_low = { -1.2, 0.1 };
    std::array<double, 2> global_high = { -0.3, 9.5 };
    double cell_size = 0.05;
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low, global_high, cell_size );
    int num_particles = 200;

    // Create the global grid.
    Cabana::Grid::DimBlockPartitioner<2> partitioner;
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create a local grid
    int halo_width = 1;
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

    // Create the communication object.
    auto global_comm =
        Cabana::Grid::createGlobalParticleComm<TEST_MEMSPACE>( *local_grid );

    // Create random particles.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t particles( "random", num_particles );
    auto position = Cabana::slice<1>( particles );

    using PoolType = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    using RandomType = Kokkos::Random_XorShift64<TEST_EXECSPACE>;
    PoolType pool( 174748 );

    // Copy box bounds to device array.
    Kokkos::Array<double, 2> global_low_kokkos;
    Kokkos::Array<double, 2> global_high_kokkos;
    for ( int d = 0; d < 2; ++d )
    {
        global_low_kokkos[d] = global_low[d];
        global_high_kokkos[d] = global_high[d];
    }

    // Create particles randomly in the global domain.
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( int d = 0; d < 2; ++d )
        {
            position( p, d ) = Kokkos::rand<RandomType, double>::draw(
                gen, global_low_kokkos[d], global_high_kokkos[d] );
        }
        pool.free_state( gen );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particles );
    Kokkos::parallel_for( policy, random_coord_op );
    Kokkos::fence();

    // Plan the communication.
    global_comm->build( position );

    // Move particles to the correct rank.
    global_comm->migrate( global_grid->comm(), particles );

    // Get the local domain bounds to check particles.
    auto local_mesh =
        Cabana::Grid::createLocalMesh<TEST_MEMSPACE>( *local_grid );
    std::array<double, 2> local_low = {
        local_mesh.lowCorner( Cabana::Grid::Own(), Cabana::Grid::Dim::I ),
        local_mesh.lowCorner( Cabana::Grid::Own(), Cabana::Grid::Dim::J ) };
    std::array<double, 2> local_high = {
        local_mesh.highCorner( Cabana::Grid::Own(), Cabana::Grid::Dim::I ),
        local_mesh.highCorner( Cabana::Grid::Own(), Cabana::Grid::Dim::J ) };

    // Copy particles to the host.
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> particles_host(
        "migrated", particles.size() );
    Cabana::deep_copy( particles_host, particles );
    auto position_host = Cabana::slice<1>( particles_host );

    // Make sure the total particles were conserved.
    int global_particles;
    int local_particles = static_cast<int>( particles.size() );
    MPI_Reduce( &local_particles, &global_particles, 1, MPI_INT, MPI_SUM, 0,
                MPI_COMM_WORLD );
    if ( global_grid->blockId() == 0 )
    {
        EXPECT_EQ( global_particles,
                   num_particles * global_grid->totalNumBlock() );
    }

    for ( std::size_t p = 0; p < particles.size(); ++p )
    {
        // Check that all of the particles were moved to the correct local rank.
        for ( int d = 0; d < 2; ++d )
        {
            EXPECT_GE( position_host( p, d ), local_low[d] );
            EXPECT_LE( position_host( p, d ), local_high[d] );
        }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( GlobalParticleComm, Migrate3d ) { testMigrate3d(); }

TEST( GlobalParticleComm, Migrate2d ) { testMigrate2d(); }

} // namespace Test
