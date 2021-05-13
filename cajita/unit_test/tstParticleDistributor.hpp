/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
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
#include <Cabana_Distributor.hpp>

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>
#include <Cajita_ManualPartitioner.hpp>
#include <Cajita_ParticleGridDistributor.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace Test
{

using Cajita::Dim;

//---------------------------------------------------------------------------//
// Shared test settings.
struct PGCommTestData
{
    using DataTypes = Cabana::MemberTypes<int, double[3]>;
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_host;

    std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<double>>> local_grid;

    int num_data = 27;

    double lo_x, lo_y, lo_z, hi_x, hi_y, hi_z;
    double ghost_x, ghost_y, ghost_z;
    double global_lo_x, global_lo_y, global_lo_z;
    double global_hi_x, global_hi_y, global_hi_z;

    PGCommTestData( bool ghost, int halo_width )
    {
        // Create the MPI partitions.
        Cajita::UniformDimPartitioner partitioner;

        // Create the global MPI decomposition mesh.
        std::array<double, 3> lo_corner = { 0, 0, 0 };
        std::array<double, 3> hi_corner = { 1, 1, 1 };
        std::array<int, 3> num_cell = { 10, 10, 10 };
        auto global_mesh =
            Cajita::createUniformGlobalMesh( lo_corner, hi_corner, num_cell );

        // Create the global grid.
        std::array<bool, 3> is_periodic = { true, true, true };
        auto global_grid = Cajita::createGlobalGrid(
            MPI_COMM_WORLD, global_mesh, is_periodic, partitioner );

        // Create a grid local_grid with large enough halo for test loops below.
        local_grid = Cajita::createLocalGrid( global_grid, 4 );
        auto local_mesh = Cajita::createLocalMesh<TEST_DEVICE>( *local_grid );

        // Make some data to migrate, one for each neighbor (and one in the
        // center).
        data_host =
            Cabana::AoSoA<DataTypes, Kokkos::HostSpace>( "host", num_data );
        auto pos_host = Cabana::slice<1>( data_host );

        // Get mesh info.
        auto dx =
            local_grid->globalGrid().globalMesh().cellSize( Cajita::Dim::I );
        auto dy =
            local_grid->globalGrid().globalMesh().cellSize( Cajita::Dim::J );
        auto dz =
            local_grid->globalGrid().globalMesh().cellSize( Cajita::Dim::K );
        double width_x, width_y, width_z;
        // Create data near the boundary, either at the edge of the local domain
        // or at the edge of the ghost region.
        if ( ghost )
        {
            hi_x = local_mesh.highCorner( Cajita::Ghost(), Cajita::Dim::I );
            hi_y = local_mesh.highCorner( Cajita::Ghost(), Cajita::Dim::J );
            hi_z = local_mesh.highCorner( Cajita::Ghost(), Cajita::Dim::K );
            lo_x = local_mesh.lowCorner( Cajita::Ghost(), Cajita::Dim::I );
            lo_y = local_mesh.lowCorner( Cajita::Ghost(), Cajita::Dim::J );
            lo_z = local_mesh.lowCorner( Cajita::Ghost(), Cajita::Dim::K );
            width_x =
                local_mesh.extent( Cajita::Ghost(), Cajita::Dim::I ) / 2.0;
            width_y =
                local_mesh.extent( Cajita::Ghost(), Cajita::Dim::J ) / 2.0;
            width_z =
                local_mesh.extent( Cajita::Ghost(), Cajita::Dim::K ) / 2.0;
        }
        else
        {
            hi_x = local_mesh.highCorner( Cajita::Own(), Cajita::Dim::I );
            hi_y = local_mesh.highCorner( Cajita::Own(), Cajita::Dim::J );
            hi_z = local_mesh.highCorner( Cajita::Own(), Cajita::Dim::K );
            lo_x = local_mesh.lowCorner( Cajita::Own(), Cajita::Dim::I );
            lo_y = local_mesh.lowCorner( Cajita::Own(), Cajita::Dim::J );
            lo_z = local_mesh.lowCorner( Cajita::Own(), Cajita::Dim::K );
            width_x = local_mesh.extent( Cajita::Own(), Cajita::Dim::I ) / 2.0;
            width_y = local_mesh.extent( Cajita::Own(), Cajita::Dim::J ) / 2.0;
            width_z = local_mesh.extent( Cajita::Own(), Cajita::Dim::K ) / 2.0;
        }
        auto center_x = width_x + lo_x;
        auto center_y = width_y + lo_y;
        auto center_z = width_z + lo_z;
        auto shift_x = width_x - ( halo_width - 0.1 ) * dx;
        auto shift_y = width_y - ( halo_width - 0.1 ) * dy;
        auto shift_z = width_z - ( halo_width - 0.1 ) * dz;

        // Fill the data. Add particles near the local domain in each direction
        // and one in the center (that should never move).
        int nr = 0;
        for ( int k = -1; k < 2; ++k )
            for ( int j = -1; j < 2; ++j )
                for ( int i = -1; i < 2; ++i, ++nr )
                {
                    pos_host( nr, 0 ) = center_x + i * shift_x;
                    pos_host( nr, 1 ) = center_y + j * shift_y;
                    pos_host( nr, 2 ) = center_z + k * shift_z;
                }

        // Add a particle on rank zero to force some resizing for sends.
        int my_rank = -1;
        MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
        if ( my_rank == 0 )
        {
            num_data++;
            data_host.resize( num_data );
            auto pos = Cabana::slice<1>( data_host );
            pos_host( num_data - 1, 0 ) = center_x + shift_x;
            pos_host( num_data - 1, 1 ) = center_y + shift_y;
            pos_host( num_data - 1, 2 ) = center_z + shift_z;
        }

        ghost_x = halo_width * dx;
        ghost_y = halo_width * dy;
        ghost_z = halo_width * dz;

        global_hi_x = global_mesh->highCorner( Cajita::Dim::I );
        global_hi_y = global_mesh->highCorner( Cajita::Dim::J );
        global_hi_z = global_mesh->highCorner( Cajita::Dim::K );
        global_lo_x = global_mesh->lowCorner( Cajita::Dim::I );
        global_lo_y = global_mesh->lowCorner( Cajita::Dim::J );
        global_lo_z = global_mesh->lowCorner( Cajita::Dim::K );
    }
};

//---------------------------------------------------------------------------//
void testMigrate( const int halo_width, const int test_halo_width,
                  const bool force_comm, const int test_type )
{
    PGCommTestData test_data( true, halo_width );
    auto data_host = test_data.data_host;
    auto local_grid = *( test_data.local_grid );
    int num_data = test_data.num_data;

    using DataTypes = Cabana::MemberTypes<int, double[3]>;
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> initial( "initial", num_data );
    Cabana::deep_copy( initial, data_host );
    auto pos_initial = Cabana::slice<1>( initial );

    // Copy to the device.
    Cabana::AoSoA<DataTypes, TEST_MEMSPACE> data_src( "data_src", num_data );
    Cabana::deep_copy( data_src, data_host );

    // Do the migration in-place.
    if ( test_type == 0 )
    {
        auto pos = Cabana::slice<1>( data_src );
        Cajita::particleGridMigrate( local_grid, pos, data_src, test_halo_width,
                                     force_comm );

        data_host.resize( data_src.size() );
        Cabana::deep_copy( data_host, data_src );
    }
    // Do the migration with a separate destination AoSoA.
    else if ( test_type == 1 )
    {
        Cabana::AoSoA<DataTypes, TEST_MEMSPACE> data_dst( "data_dst",
                                                          num_data );
        auto pos = Cabana::slice<1>( data_src );
        Cajita::particleGridMigrate( local_grid, pos, data_src, data_dst,
                                     test_halo_width, force_comm );

        data_host.resize( data_dst.size() );
        Cabana::deep_copy( data_host, data_dst );
    }
    // Do the migration with separate slices (need to use createGridDistributor
    // directly since slices can't be resized).
    else if ( test_type == 2 )
    {
        auto pos_src = Cabana::slice<1>( data_src );
        int comm_count = 0;
        if ( !force_comm )
        {
            // Check to see if we need to communicate.
            comm_count = migrateCount( local_grid, pos_src, test_halo_width );
        }

        if ( force_comm || comm_count > 0 )
        {
            auto distributor =
                Cajita::createParticleGridDistributor( local_grid, pos_src );
            Cabana::AoSoA<DataTypes, TEST_MEMSPACE> data_dst(
                "data_dst", distributor.totalNumImport() );
            auto pos_dst = Cabana::slice<1>( data_dst );
            Cabana::migrate( distributor, pos_src, pos_dst );

            data_host.resize( data_dst.size() );
            Cabana::deep_copy( data_host, data_dst );
        }
        else
        {
            data_host.resize( data_src.size() );
            Cabana::deep_copy( data_host, data_src );
        }
    }

    // Check the results.
    int new_num_data = data_host.size();
    auto pos_host = Cabana::slice<1>( data_host );

    for ( int i = 0; i < new_num_data; ++i )
    {
        // Make sure particles haven't moved if within allowable halo mesh and
        // migrate is not being forced.
        if ( !force_comm && test_halo_width < halo_width )
        {
            EXPECT_DOUBLE_EQ( pos_host( i, Cajita::Dim::I ),
                              pos_initial( i, Cajita::Dim::I ) );
            EXPECT_DOUBLE_EQ( pos_host( i, Cajita::Dim::J ),
                              pos_initial( i, Cajita::Dim::J ) );
            EXPECT_DOUBLE_EQ( pos_host( i, Cajita::Dim::K ),
                              pos_initial( i, Cajita::Dim::K ) );
        }
        else
        {
            // Make sure everything was wrapped into the global domain.
            EXPECT_LE( pos_host( i, Cajita::Dim::I ), test_data.global_hi_x );
            EXPECT_LE( pos_host( i, Cajita::Dim::J ), test_data.global_hi_y );
            EXPECT_LE( pos_host( i, Cajita::Dim::K ), test_data.global_hi_z );
            EXPECT_GE( pos_host( i, Cajita::Dim::I ), test_data.global_lo_x );
            EXPECT_GE( pos_host( i, Cajita::Dim::J ), test_data.global_lo_y );
            EXPECT_GE( pos_host( i, Cajita::Dim::K ), test_data.global_lo_z );

            // Make sure everything was wrapped into the local domain.
            EXPECT_LE( pos_host( i, Cajita::Dim::I ), test_data.hi_x );
            EXPECT_LE( pos_host( i, Cajita::Dim::J ), test_data.hi_y );
            EXPECT_LE( pos_host( i, Cajita::Dim::K ), test_data.hi_z );
            EXPECT_GE( pos_host( i, Cajita::Dim::I ), test_data.lo_x );
            EXPECT_GE( pos_host( i, Cajita::Dim::J ), test_data.lo_y );
            EXPECT_GE( pos_host( i, Cajita::Dim::K ), test_data.lo_z );
        }
    }

    // Make sure the particles are still unique.
    int final_total = 0;
    int initial_total = 0;
    MPI_Allreduce( &new_num_data, &final_total, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    MPI_Allreduce( &num_data, &initial_total, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    EXPECT_EQ( final_total, initial_total );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
/*
TEST( TEST_CATEGORY, periodic_test_migrate_aosoa )
{
    // Call with varied halo data and halo width, without forced communication
    for ( int i = 1; i < 4; i++ )
        for ( int j = 0; j < 4; j++ )
            testMigrate( i, j, false, 0 );

    // Call with forced communication
    testMigrate( 1, 1, true, 0 );
}

TEST( TEST_CATEGORY, periodic_test_migrate_aosoa_separate )
{
    for ( int i = 1; i < 4; i++ )
        for ( int j = 1; j < 4; j++ )
            testMigrate( i, j, false, 1 );

    testMigrate( 1, 1, true, 1 );
}

TEST( TEST_CATEGORY, periodic_test_migrate_slice )
{
    for ( int i = 1; i < 4; i++ )
        for ( int j = 1; j < 4; j++ )
            testMigrate( i, j, false, 2 );

    testMigrate( 1, 1, true, 2 );
}
*/
//---------------------------------------------------------------------------//
void redistributeTest( const Cajita::ManualPartitioner& partitioner,
                       const std::array<bool, 3>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 22, 19, 21 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    // Create local block with a halo of 2.
    const int halo_size = 2;
    auto block = Cajita::createLocalGrid( global_grid, halo_size );
    auto local_mesh = Cajita::createLocalMesh<Kokkos::HostSpace>( *block );

    // Allocate a maximum number of particles assuming we have a halo on every
    // boundary.
    auto ghosted_cell_space =
        block->indexSpace( Cajita::Ghost(), Cajita::Cell(), Cajita::Local() );
    int num_particle = ghosted_cell_space.size();
    using MemberTypes = Cabana::MemberTypes<double[3], int>;
    using ParticleContainer = Cabana::AoSoA<MemberTypes, Kokkos::HostSpace>;
    ParticleContainer particles( "particles", num_particle );
    auto coords = Cabana::slice<0>( particles, "coords" );
    auto linear_ids = Cabana::slice<1>( particles, "linear_ids" );

    // Put particles in the center of every cell including halo cells if we
    // have them. Their ids should be equivalent to that of the rank they are
    // going to.
    int pid = 0;
    for ( int nk = -1; nk < 2; ++nk )
        for ( int nj = -1; nj < 2; ++nj )
            for ( int ni = -1; ni < 2; ++ni )
            {
                auto neighbor_rank = block->neighborRank( ni, nj, nk );
                if ( neighbor_rank >= 0 )
                {
                    auto shared_space = block->sharedIndexSpace(
                        Cajita::Ghost(), Cajita::Cell(), ni, nj, nk );
                    for ( int k = shared_space.min( Dim::K );
                          k < shared_space.max( Dim::K ); ++k )
                        for ( int j = shared_space.min( Dim::J );
                              j < shared_space.max( Dim::J ); ++j )
                            for ( int i = shared_space.min( Dim::I );
                                  i < shared_space.max( Dim::I ); ++i )
                            {
                                // Set the coordinates at the cell center.
                                coords( pid, Dim::I ) =
                                    local_mesh.lowCorner( Cajita::Ghost(),
                                                          Dim::I ) +
                                    ( i + 0.5 ) * cell_size;
                                coords( pid, Dim::J ) =
                                    local_mesh.lowCorner( Cajita::Ghost(),
                                                          Dim::J ) +
                                    ( j + 0.5 ) * cell_size;
                                coords( pid, Dim::K ) =
                                    local_mesh.lowCorner( Cajita::Ghost(),
                                                          Dim::K ) +
                                    ( k + 0.5 ) * cell_size;

                                // Set the linear ids as the linear rank of
                                // the neighbor.
                                linear_ids( pid ) = neighbor_rank;

                                // Increment the particle count.
                                ++pid;
                            }
                }
            }
    num_particle = pid;

    // Copy to the device space.
    particles.resize( num_particle );

    auto particles_mirror =
        Cabana::create_mirror_view_and_copy( TEST_DEVICE(), particles );

    // Redistribute the particles.
    auto coords_mirror = Cabana::slice<0>( particles_mirror, "coords" );
    Cajita::particleGridMigrate( *block, coords_mirror, particles_mirror, 0,
                                 true );

    // Copy back to check.
    particles = Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                     particles_mirror );
    coords = Cabana::slice<0>( particles, "coords" );
    linear_ids = Cabana::slice<1>( particles, "linear_ids" );

    // Check that we got as many particles as we should have.
    EXPECT_EQ( coords.size(), num_particle );
    EXPECT_EQ( linear_ids.size(), num_particle );

    // Check that all of the particle ids are equal to this rank id.
    for ( int p = 0; p < num_particle; ++p )
        EXPECT_EQ( linear_ids( p ), global_grid->blockId() );

    // Check that all of the particles are now in the local domain.
    double low_c[3] = { local_mesh.lowCorner( Cajita::Own(), Dim::I ),
                        local_mesh.lowCorner( Cajita::Own(), Dim::J ),
                        local_mesh.lowCorner( Cajita::Own(), Dim::K ) };
    double high_c[3] = { local_mesh.highCorner( Cajita::Own(), Dim::I ),
                         local_mesh.highCorner( Cajita::Own(), Dim::J ),
                         local_mesh.highCorner( Cajita::Own(), Dim::K ) };
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_TRUE( coords( p, d ) >= low_c[d] );
            EXPECT_TRUE( coords( p, d ) <= high_c[d] );
        }
}

//---------------------------------------------------------------------------//
// The objective of this test is to check how the redistribution works when we
// have no particles to redistribute. In this case we put no particles in the
// halo so no communication should occur. This ensures the graph communication
// works when some neighbors get no data.
void localOnlyTest( const Cajita::ManualPartitioner& partitioner,
                    const std::array<bool, 3>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 22, 19, 21 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    // Get the local block with a halo of 2.
    const int halo_size = 2;
    auto block = Cajita::createLocalGrid( global_grid, halo_size );
    auto local_mesh = Cajita::createLocalMesh<Kokkos::HostSpace>( *block );

    // Allocate particles
    auto owned_cell_space =
        block->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    int num_particle = owned_cell_space.size();
    using MemberTypes = Cabana::MemberTypes<double[3], int>;
    using ParticleContainer = Cabana::AoSoA<MemberTypes, Kokkos::HostSpace>;
    ParticleContainer particles( "particles", num_particle );
    auto coords = Cabana::slice<0>( particles, "coords" );
    auto linear_ids = Cabana::slice<1>( particles, "linear_ids" );

    // Put particles in the center of every local cell.
    int pid = 0;
    for ( int k = 0; k < owned_cell_space.extent( Dim::K ); ++k )
        for ( int j = 0; j < owned_cell_space.extent( Dim::J ); ++j )
            for ( int i = 0; i < owned_cell_space.extent( Dim::I ); ++i )
            {
                // Set the coordinates at the cell center.
                coords( pid, Dim::I ) =
                    local_mesh.lowCorner( Cajita::Own(), Dim::I ) +
                    ( i + 0.5 ) * cell_size;
                coords( pid, Dim::J ) =
                    local_mesh.lowCorner( Cajita::Own(), Dim::J ) +
                    ( j + 0.5 ) * cell_size;
                coords( pid, Dim::K ) =
                    local_mesh.lowCorner( Cajita::Own(), Dim::K ) +
                    ( k + 0.5 ) * cell_size;

                // Set the linear rank
                linear_ids( pid ) = global_grid->blockId();

                // Increment the particle count.
                ++pid;
            }

    // Copy to the device space.
    auto particles_mirror =
        Cabana::create_mirror_view_and_copy( TEST_DEVICE(), particles );

    // Redistribute the particles.
    auto coords_mirror = Cabana::slice<0>( particles_mirror, "coords" );
    Cajita::particleGridMigrate( *block, coords_mirror, particles_mirror, 0,
                                 true );

    // Copy back to check.
    particles = Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                     particles_mirror );
    coords = Cabana::slice<0>( particles, "coords" );
    linear_ids = Cabana::slice<1>( particles, "linear_ids" );

    // Check that we got as many particles as we should have.
    EXPECT_EQ( coords.size(), num_particle );
    EXPECT_EQ( linear_ids.size(), num_particle );

    // Check that all of the particle ids are equal to this rank id.
    for ( int p = 0; p < num_particle; ++p )
        EXPECT_EQ( linear_ids( p ), global_grid->blockId() );

    // Check that all of the particles are now in the local domain.
    double low_c[3] = { local_mesh.lowCorner( Cajita::Own(), Dim::I ),
                        local_mesh.lowCorner( Cajita::Own(), Dim::J ),
                        local_mesh.lowCorner( Cajita::Own(), Dim::K ) };
    double high_c[3] = { local_mesh.highCorner( Cajita::Own(), Dim::I ),
                         local_mesh.highCorner( Cajita::Own(), Dim::J ),
                         local_mesh.highCorner( Cajita::Own(), Dim::K ) };
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_TRUE( coords( p, d ) >= low_c[d] );
            EXPECT_TRUE( coords( p, d ) <= high_c[d] );
        }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, not_periodic_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    // Boundaries are not periodic.
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    redistributeTest( partitioner, is_dim_periodic );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( partitioner, is_dim_periodic );
    }
    if ( ranks_per_dim[0] != ranks_per_dim[2] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[2] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( partitioner, is_dim_periodic );
    }
    if ( ranks_per_dim[1] != ranks_per_dim[2] )
    {
        std::swap( ranks_per_dim[1], ranks_per_dim[2] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( partitioner, is_dim_periodic );
    }
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, periodic_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    // Every boundary is periodic
    std::array<bool, 3> is_dim_periodic = { true, true, true };

    redistributeTest( partitioner, is_dim_periodic );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( partitioner, is_dim_periodic );
    }
    if ( ranks_per_dim[0] != ranks_per_dim[2] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[2] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( partitioner, is_dim_periodic );
    }
    if ( ranks_per_dim[1] != ranks_per_dim[2] )
    {
        std::swap( ranks_per_dim[1], ranks_per_dim[2] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( partitioner, is_dim_periodic );
    }
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, local_only_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    // Every boundary is periodic
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    localOnlyTest( partitioner, is_dim_periodic );
}
//---------------------------------------------------------------------------//

} // end namespace Test
