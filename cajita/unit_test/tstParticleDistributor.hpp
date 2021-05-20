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
        Cajita::gridMigrate( local_grid, data_src,
                             std::integral_constant<std::size_t, 1>(),
                             test_halo_width, force_comm );

        data_host.resize( data_src.size() );
        Cabana::deep_copy( data_host, data_src );
    }
    // Do the migration with a separate destination AoSoA.
    else if ( test_type == 1 )
    {
        Cabana::AoSoA<DataTypes, TEST_MEMSPACE> data_dst( "data_dst",
                                                          num_data );
        Cajita::gridMigrate( local_grid, data_src,
                             std::integral_constant<std::size_t, 1>(), data_dst,
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
                Cajita::createGridDistributor( local_grid, pos_src );
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

//---------------------------------------------------------------------------//

} // end namespace Test
