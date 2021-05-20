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
#include <Cabana_Distributor.hpp>

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>
#include <Cajita_ManualPartitioner.hpp>
#include <Cajita_ParticleGridDistributor.hpp>
#include <Cajita_Types.hpp>

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
template <class GridType>
void redistributeTest( const GridType global_grid, const double cell_size,
                       const int data_halo_size, const int test_halo_size,
                       const bool force_comm, const int test_type )
{
    // Create local block with varying halo size.
    auto block = Cajita::createLocalGrid( global_grid, data_halo_size );
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
    particles.resize( num_particle );

    ParticleContainer particles_initial( "initial_particles", num_particle );
    Cabana::deep_copy( particles_initial, particles );
    auto coords_initial = Cabana::slice<0>( particles_initial );

    // Copy to the device space.
    auto particles_mirror =
        Cabana::create_mirror_view_and_copy( TEST_DEVICE(), particles );
    auto coords_mirror = Cabana::slice<0>( particles_mirror, "coords" );

    // Redistribute the particle AoSoA in place.
    if ( test_type == 0 )
    {
        Cajita::particleGridMigrate( *block, coords_mirror, particles_mirror,
                                     test_halo_size, force_comm );

        // Copy back to check.
        particles = Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                         particles_mirror );
    }
    // Do the migration with a separate destination AoSoA.
    else if ( test_type == 1 )
    {
        auto particles_dst =
            Cabana::create_mirror_view( TEST_MEMSPACE(), particles_mirror );
        Cajita::particleGridMigrate( *block, coords_mirror, particles_mirror,
                                     particles_dst, test_halo_size,
                                     force_comm );
        // Copy back to check.
        particles = Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                         particles_dst );
    }

    coords = Cabana::slice<0>( particles, "coords" );
    linear_ids = Cabana::slice<1>( particles, "linear_ids" );

    // Check that we got as many particles as we should have.
    EXPECT_EQ( coords.size(), num_particle );
    EXPECT_EQ( linear_ids.size(), num_particle );

    std::array<double, 3> local_low = {
        local_mesh.lowCorner( Cajita::Own(), Dim::I ),
        local_mesh.lowCorner( Cajita::Own(), Dim::J ),
        local_mesh.lowCorner( Cajita::Own(), Dim::K ) };
    std::array<double, 3> local_high = {
        local_mesh.highCorner( Cajita::Own(), Dim::I ),
        local_mesh.highCorner( Cajita::Own(), Dim::J ),
        local_mesh.highCorner( Cajita::Own(), Dim::K ) };

    for ( int p = 0; p < num_particle; ++p )
    {
        // Particles should be redistributed if forcing communication or if
        // anything is outside the minimum halo width (currently every case
        // except test_halo_size=0)
        if ( force_comm || test_halo_size > 0 )
        {
            // Check that all of the particle ids are equal to this rank id.
            EXPECT_EQ( linear_ids( p ), global_grid->blockId() );

            // Check that all of the particles are now in the local domain.
            for ( int d = 0; d < 3; ++d )
            {
                EXPECT_GE( coords( p, d ), local_low[d] );
                EXPECT_LE( coords( p, d ), local_high[d] );
            }
        }
        else
        {
            // If only partially moving (test_halo_size < data_halo_size)
            // particles are allowed outside the local domain (and with
            // different rank id), but must be inside the test_halo_size.
            for ( int d = 0; d < 3; ++d )
                EXPECT_DOUBLE_EQ( coords( p, d ), coords_initial( p, d ) );
        }
    }
} // namespace Test

//---------------------------------------------------------------------------//
// The objective of this test is to check how the redistribution works when we
// have no particles to redistribute. In this case we put no particles in the
// halo so no communication should occur. This ensures the graph communication
// works when some neighbors get no data.
template <class GridType>
void localOnlyTest( const GridType global_grid, const double cell_size )
{
    // Get the local block with a halo of 2.
    const int data_halo_size = 2;
    auto block = Cajita::createLocalGrid( global_grid, data_halo_size );
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
    double local_low[3] = { local_mesh.lowCorner( Cajita::Own(), Dim::I ),
                            local_mesh.lowCorner( Cajita::Own(), Dim::J ),
                            local_mesh.lowCorner( Cajita::Own(), Dim::K ) };
    double local_high[3] = { local_mesh.highCorner( Cajita::Own(), Dim::I ),
                             local_mesh.highCorner( Cajita::Own(), Dim::J ),
                             local_mesh.highCorner( Cajita::Own(), Dim::K ) };
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_GE( coords( p, d ), local_low[d] );
            EXPECT_LE( coords( p, d ), local_high[d] );
        }
}

auto createGrid( const Cajita::ManualPartitioner& partitioner,
                 const std::array<bool, 3>& is_periodic,
                 const double cell_size )
{
    // Create the global grid.
    std::array<int, 3> global_num_cell = { 18, 15, 9 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_periodic, partitioner );
    return global_grid;
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
    std::array<bool, 3> is_periodic = { false, false, false };

    // Create global grid.
    double cell_size = 0.23;
    auto global_grid = createGrid( partitioner, is_periodic, cell_size );

    // Test in-place and new AoSoA
    for ( int t = 0; t < 2; t++ )
        redistributeTest( global_grid, cell_size, 2, 2, false, t );

    // Test with forced communication.
    redistributeTest( global_grid, cell_size, 2, 2, true, 0 );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( global_grid, cell_size, 2, 2, true, 0 );
    }
    if ( ranks_per_dim[0] != ranks_per_dim[2] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[2] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( global_grid, cell_size, 2, 2, true, 0 );
    }
    if ( ranks_per_dim[1] != ranks_per_dim[2] )
    {
        std::swap( ranks_per_dim[1], ranks_per_dim[2] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( global_grid, cell_size, 2, 2, true, 0 );
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
    std::array<bool, 3> is_periodic = { true, true, true };

    // Create global grid.
    double cell_size = 0.23;
    auto global_grid = createGrid( partitioner, is_periodic, cell_size );

    // Test in-place
    // Test multiple system halo sizes
    for ( int i = 0; i < 3; i++ )
        // Test multiple minimum_halo_width
        for ( int j = 0; j < 3; j++ )
            redistributeTest( global_grid, cell_size, 1, j, false, 0 );

    // Retest with separate destination AoSoA.
    redistributeTest( global_grid, cell_size, 2, 2, true, 1 );

    // Test with forced communication.
    redistributeTest( global_grid, cell_size, 2, 2, true, 0 );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( global_grid, cell_size, 2, 2, true, 0 );
    }
    if ( ranks_per_dim[0] != ranks_per_dim[2] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[2] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( global_grid, cell_size, 2, 2, true, 0 );
    }
    if ( ranks_per_dim[1] != ranks_per_dim[2] )
    {
        std::swap( ranks_per_dim[1], ranks_per_dim[2] );
        partitioner = Cajita::ManualPartitioner( ranks_per_dim );
        redistributeTest( global_grid, cell_size, 2, 2, true, 0 );
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
    std::array<bool, 3> is_periodic = { true, true, true };

    // Create global grid.
    double cell_size = 0.23;
    auto global_grid = createGrid( partitioner, is_periodic, cell_size );

    localOnlyTest( global_grid, cell_size );
}
//---------------------------------------------------------------------------//

} // end namespace Test
