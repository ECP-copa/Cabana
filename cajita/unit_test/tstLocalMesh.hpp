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

#include <Kokkos_Core.hpp>

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>
#include <Cajita_ManualPartitioner.hpp>
#include <Cajita_Types.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cmath>
#include <vector>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
template <class LocalMeshType, class LocalGridType>
void uniformLocalMeshTest( const LocalMeshType &local_mesh,
                           const LocalGridType &local_grid,
                           const std::array<double, 3> &low_corner,
                           const double cell_size, const int halo_width )
{
    // Get the global grid.
    const auto &global_grid = local_grid.globalGrid();

    // Check the low and high corners.
    Kokkos::View<double[3], TEST_DEVICE> own_lc( "own_lc" );
    Kokkos::View<double[3], TEST_DEVICE> own_hc( "own_hc" );
    Kokkos::View<double[3], TEST_DEVICE> own_ext( "own_extent" );
    Kokkos::View<double[3], TEST_DEVICE> ghost_lc( "ghost_lc" );
    Kokkos::View<double[3], TEST_DEVICE> ghost_hc( "ghost_hc" );
    Kokkos::View<double[3], TEST_DEVICE> ghost_ext( "ghost_extent" );

    Kokkos::parallel_for(
        "get_corners", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 3 ),
        KOKKOS_LAMBDA( const int d ) {
            own_lc( d ) = local_mesh.lowCorner( Own(), d );
            own_hc( d ) = local_mesh.highCorner( Own(), d );
            own_ext( d ) = local_mesh.extent( Own(), d );
            ghost_lc( d ) = local_mesh.lowCorner( Ghost(), d );
            ghost_hc( d ) = local_mesh.highCorner( Ghost(), d );
            ghost_ext( d ) = local_mesh.extent( Ghost(), d );
        } );

    auto own_lc_m =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), own_lc );
    auto own_hc_m =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), own_hc );
    auto own_ext_m =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), own_ext );
    auto ghost_lc_m =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ghost_lc );
    auto ghost_hc_m =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ghost_hc );
    auto ghost_ext_m =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ghost_ext );

    for ( int d = 0; d < 3; ++d )
        EXPECT_FLOAT_EQ( own_lc_m( d ),
                         low_corner[d] +
                             cell_size * global_grid.globalOffset( d ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_FLOAT_EQ( own_hc_m( d ),
                         low_corner[d] +
                             cell_size * ( global_grid.globalOffset( d ) +
                                           global_grid.ownedNumCell( d ) ) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_FLOAT_EQ( own_ext_m( d ),
                         cell_size * global_grid.ownedNumCell( d ) );

    for ( int d = 0; d < 3; ++d )
    {
        int ghost_lc_offset =
            ( global_grid.isPeriodic( d ) || global_grid.dimBlockId( d ) > 0 )
                ? halo_width
                : 0;
        EXPECT_FLOAT_EQ( ghost_lc_m( d ),
                         low_corner[d] +
                             cell_size * ( global_grid.globalOffset( d ) -
                                           ghost_lc_offset ) );

        int ghost_hc_offset =
            ( global_grid.isPeriodic( d ) ||
              global_grid.dimBlockId( d ) < global_grid.dimNumBlock( d ) - 1 )
                ? local_grid.haloCellWidth()
                : 0;
        EXPECT_FLOAT_EQ( ghost_hc_m( d ),
                         low_corner[d] +
                             cell_size * ( global_grid.globalOffset( d ) +
                                           ghost_hc_offset +
                                           global_grid.ownedNumCell( d ) ) );

        EXPECT_FLOAT_EQ( ghost_ext_m( d ),
                         cell_size * ( global_grid.ownedNumCell( d ) +
                                       ghost_hc_offset + ghost_lc_offset ) );
    }

    // Check the cell locations and measures.
    auto cell_space = local_grid.indexSpace( Ghost(), Cell(), Local() );
    {
        auto measure = createView<double, TEST_DEVICE>( "measure", cell_space );
        auto loc_x = createView<double, TEST_DEVICE>( "loc_x", cell_space );
        auto loc_y = createView<double, TEST_DEVICE>( "loc_y", cell_space );
        auto loc_z = createView<double, TEST_DEVICE>( "loc_z", cell_space );
        Kokkos::parallel_for(
            "get_cell_coord",
            createExecutionPolicy( cell_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double loc[3];
                int idx[3] = { i, j, k };
                local_mesh.coordinates( Cell(), idx, loc );
                loc_x( i, j, k ) = loc[Dim::I];
                loc_y( i, j, k ) = loc[Dim::J];
                loc_z( i, j, k ) = loc[Dim::K];
                measure( i, j, k ) = local_mesh.measure( Cell(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        auto loc_z_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_z );
        for ( int i = 0; i < cell_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < cell_space.extent( Dim::J ); ++j )
                for ( int k = 0; k < cell_space.extent( Dim::K ); ++k )
                {
                    EXPECT_FLOAT_EQ( loc_x_m( i, j, k ),
                                     ghost_lc_m( Dim::I ) +
                                         cell_size * ( i + 0.5 ) );
                    EXPECT_FLOAT_EQ( loc_y_m( i, j, k ),
                                     ghost_lc_m( Dim::J ) +
                                         cell_size * ( j + 0.5 ) );
                    EXPECT_FLOAT_EQ( loc_z_m( i, j, k ),
                                     ghost_lc_m( Dim::K ) +
                                         cell_size * ( k + 0.5 ) );
                    EXPECT_FLOAT_EQ( measure_m( i, j, k ),
                                     cell_size * cell_size * cell_size );
                }
    }

    // Check the node locations and measures.
    auto node_space = local_grid.indexSpace( Ghost(), Node(), Local() );
    {
        auto measure = createView<double, TEST_DEVICE>( "measure", node_space );
        auto loc_x = createView<double, TEST_DEVICE>( "loc_x", node_space );
        auto loc_y = createView<double, TEST_DEVICE>( "loc_y", node_space );
        auto loc_z = createView<double, TEST_DEVICE>( "loc_z", node_space );
        Kokkos::parallel_for(
            "get_node_coord",
            createExecutionPolicy( node_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double loc[3];
                int idx[3] = { i, j, k };
                local_mesh.coordinates( Node(), idx, loc );
                loc_x( i, j, k ) = loc[Dim::I];
                loc_y( i, j, k ) = loc[Dim::J];
                loc_z( i, j, k ) = loc[Dim::K];
                measure( i, j, k ) = local_mesh.measure( Node(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        auto loc_z_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_z );
        for ( int i = 0; i < node_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < node_space.extent( Dim::J ); ++j )
                for ( int k = 0; k < node_space.extent( Dim::K ); ++k )
                {
                    EXPECT_FLOAT_EQ( loc_x_m( i, j, k ),
                                     ghost_lc_m( Dim::I ) + cell_size * i );
                    EXPECT_FLOAT_EQ( loc_y_m( i, j, k ),
                                     ghost_lc_m( Dim::J ) + cell_size * j );
                    EXPECT_FLOAT_EQ( loc_z_m( i, j, k ),
                                     ghost_lc_m( Dim::K ) + cell_size * k );
                    EXPECT_FLOAT_EQ( measure_m( i, j, k ), 0.0 );
                }
    }

    // Check the I-face locations and measures
    auto face_i_space =
        local_grid.indexSpace( Ghost(), Face<Dim::I>(), Local() );
    {
        auto measure =
            createView<double, TEST_DEVICE>( "measure", face_i_space );
        auto loc_x = createView<double, TEST_DEVICE>( "loc_x", face_i_space );
        auto loc_y = createView<double, TEST_DEVICE>( "loc_y", face_i_space );
        auto loc_z = createView<double, TEST_DEVICE>( "loc_z", face_i_space );
        Kokkos::parallel_for(
            "get_face_i_coord",
            createExecutionPolicy( face_i_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double loc[3];
                int idx[3] = { i, j, k };
                local_mesh.coordinates( Face<Dim::I>(), idx, loc );
                loc_x( i, j, k ) = loc[Dim::I];
                loc_y( i, j, k ) = loc[Dim::J];
                loc_z( i, j, k ) = loc[Dim::K];
                measure( i, j, k ) = local_mesh.measure( Face<Dim::I>(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        auto loc_z_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_z );
        for ( int i = 0; i < face_i_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < face_i_space.extent( Dim::J ); ++j )
                for ( int k = 0; k < face_i_space.extent( Dim::K ); ++k )
                {
                    EXPECT_FLOAT_EQ( loc_x_m( i, j, k ),
                                     ghost_lc_m( Dim::I ) + cell_size * i );
                    EXPECT_FLOAT_EQ( loc_y_m( i, j, k ),
                                     ghost_lc_m( Dim::J ) +
                                         cell_size * ( j + 0.5 ) );
                    EXPECT_FLOAT_EQ( loc_z_m( i, j, k ),
                                     ghost_lc_m( Dim::K ) +
                                         cell_size * ( k + 0.5 ) );
                    EXPECT_FLOAT_EQ( measure_m( i, j, k ),
                                     cell_size * cell_size );
                }
    }

    // Check the J-face locations and measures.
    auto face_j_space =
        local_grid.indexSpace( Ghost(), Face<Dim::J>(), Local() );
    {
        auto measure =
            createView<double, TEST_DEVICE>( "measure", face_j_space );
        auto loc_x = createView<double, TEST_DEVICE>( "loc_x", face_j_space );
        auto loc_y = createView<double, TEST_DEVICE>( "loc_y", face_j_space );
        auto loc_z = createView<double, TEST_DEVICE>( "loc_z", face_j_space );
        Kokkos::parallel_for(
            "get_face_j_coord",
            createExecutionPolicy( face_j_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double loc[3];
                int idx[3] = { i, j, k };
                local_mesh.coordinates( Face<Dim::J>(), idx, loc );
                loc_x( i, j, k ) = loc[Dim::I];
                loc_y( i, j, k ) = loc[Dim::J];
                loc_z( i, j, k ) = loc[Dim::K];
                measure( i, j, k ) = local_mesh.measure( Face<Dim::J>(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        auto loc_z_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_z );
        for ( int i = 0; i < face_j_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < face_j_space.extent( Dim::J ); ++j )
                for ( int k = 0; k < face_j_space.extent( Dim::K ); ++k )
                {
                    EXPECT_FLOAT_EQ( loc_x_m( i, j, k ),
                                     ghost_lc_m( Dim::I ) +
                                         cell_size * ( i + 0.5 ) );
                    EXPECT_FLOAT_EQ( loc_y_m( i, j, k ),
                                     ghost_lc_m( Dim::J ) + cell_size * j );
                    EXPECT_FLOAT_EQ( loc_z_m( i, j, k ),
                                     ghost_lc_m( Dim::K ) +
                                         cell_size * ( k + 0.5 ) );
                    EXPECT_FLOAT_EQ( measure_m( i, j, k ),
                                     cell_size * cell_size );
                }
    }

    // Check the K-face locations and measures.
    auto face_k_space =
        local_grid.indexSpace( Ghost(), Face<Dim::K>(), Local() );
    {
        auto measure =
            createView<double, TEST_DEVICE>( "measure", face_k_space );
        auto loc_x = createView<double, TEST_DEVICE>( "loc_x", face_k_space );
        auto loc_y = createView<double, TEST_DEVICE>( "loc_y", face_k_space );
        auto loc_z = createView<double, TEST_DEVICE>( "loc_z", face_k_space );
        Kokkos::parallel_for(
            "get_face_k_coord",
            createExecutionPolicy( face_k_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double loc[3];
                int idx[3] = { i, j, k };
                local_mesh.coordinates( Face<Dim::K>(), idx, loc );
                loc_x( i, j, k ) = loc[Dim::I];
                loc_y( i, j, k ) = loc[Dim::J];
                loc_z( i, j, k ) = loc[Dim::K];
                measure( i, j, k ) = local_mesh.measure( Face<Dim::K>(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        auto loc_z_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_z );
        for ( int i = 0; i < face_k_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < face_k_space.extent( Dim::J ); ++j )
                for ( int k = 0; k < face_k_space.extent( Dim::K ); ++k )
                {
                    EXPECT_FLOAT_EQ( loc_x_m( i, j, k ),
                                     ghost_lc_m( Dim::I ) +
                                         cell_size * ( i + 0.5 ) );
                    EXPECT_FLOAT_EQ( loc_y_m( i, j, k ),
                                     ghost_lc_m( Dim::J ) +
                                         cell_size * ( j + 0.5 ) );
                    EXPECT_FLOAT_EQ( loc_z_m( i, j, k ),
                                     ghost_lc_m( Dim::K ) + cell_size * k );
                    EXPECT_FLOAT_EQ( measure_m( i, j, k ),
                                     cell_size * cell_size );
                }
    }

    // Check the I-edge locations and measures.
    auto edge_i_space =
        local_grid.indexSpace( Ghost(), Edge<Dim::I>(), Local() );
    {
        auto measure =
            createView<double, TEST_DEVICE>( "measure", edge_i_space );
        auto loc_x = createView<double, TEST_DEVICE>( "loc_x", edge_i_space );
        auto loc_y = createView<double, TEST_DEVICE>( "loc_y", edge_i_space );
        auto loc_z = createView<double, TEST_DEVICE>( "loc_z", edge_i_space );
        Kokkos::parallel_for(
            "get_edge_i_coord",
            createExecutionPolicy( edge_i_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double loc[3];
                int idx[3] = { i, j, k };
                local_mesh.coordinates( Edge<Dim::I>(), idx, loc );
                loc_x( i, j, k ) = loc[Dim::I];
                loc_y( i, j, k ) = loc[Dim::J];
                loc_z( i, j, k ) = loc[Dim::K];
                measure( i, j, k ) = local_mesh.measure( Edge<Dim::I>(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        auto loc_z_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_z );
        for ( int i = 0; i < edge_i_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < edge_i_space.extent( Dim::J ); ++j )
                for ( int k = 0; k < edge_i_space.extent( Dim::K ); ++k )
                {
                    EXPECT_FLOAT_EQ( loc_x_m( i, j, k ),
                                     ghost_lc_m( Dim::I ) +
                                         cell_size * ( i + 0.5 ) );
                    EXPECT_FLOAT_EQ( loc_y_m( i, j, k ),
                                     ghost_lc_m( Dim::J ) + cell_size * j );
                    EXPECT_FLOAT_EQ( loc_z_m( i, j, k ),
                                     ghost_lc_m( Dim::K ) + cell_size * k );
                    EXPECT_FLOAT_EQ( measure_m( i, j, k ), cell_size );
                }
    }

    // Check the J-edge locations and measures.
    auto edge_j_space =
        local_grid.indexSpace( Ghost(), Edge<Dim::J>(), Local() );
    {
        auto measure =
            createView<double, TEST_DEVICE>( "measure", edge_j_space );
        auto loc_x = createView<double, TEST_DEVICE>( "loc_x", edge_j_space );
        auto loc_y = createView<double, TEST_DEVICE>( "loc_y", edge_j_space );
        auto loc_z = createView<double, TEST_DEVICE>( "loc_z", edge_j_space );
        Kokkos::parallel_for(
            "get_edge_j_coord",
            createExecutionPolicy( edge_j_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double loc[3];
                int idx[3] = { i, j, k };
                local_mesh.coordinates( Edge<Dim::J>(), idx, loc );
                loc_x( i, j, k ) = loc[Dim::I];
                loc_y( i, j, k ) = loc[Dim::J];
                loc_z( i, j, k ) = loc[Dim::K];
                measure( i, j, k ) = local_mesh.measure( Edge<Dim::J>(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        auto loc_z_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_z );
        for ( int i = 0; i < edge_j_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < edge_j_space.extent( Dim::J ); ++j )
                for ( int k = 0; k < edge_j_space.extent( Dim::K ); ++k )
                {
                    EXPECT_FLOAT_EQ( loc_x_m( i, j, k ),
                                     ghost_lc_m( Dim::I ) + cell_size * i );
                    EXPECT_FLOAT_EQ( loc_y_m( i, j, k ),
                                     ghost_lc_m( Dim::J ) +
                                         cell_size * ( j + 0.5 ) );
                    EXPECT_FLOAT_EQ( loc_z_m( i, j, k ),
                                     ghost_lc_m( Dim::K ) + cell_size * k );
                    EXPECT_FLOAT_EQ( measure_m( i, j, k ), cell_size );
                }
    }

    // Check the K-edge locations and measures.
    auto edge_k_space =
        local_grid.indexSpace( Ghost(), Edge<Dim::K>(), Local() );
    {
        auto measure =
            createView<double, TEST_DEVICE>( "measure", edge_k_space );
        auto loc_x = createView<double, TEST_DEVICE>( "loc_x", edge_k_space );
        auto loc_y = createView<double, TEST_DEVICE>( "loc_y", edge_k_space );
        auto loc_z = createView<double, TEST_DEVICE>( "loc_z", edge_k_space );
        Kokkos::parallel_for(
            "get_edge_k_coord",
            createExecutionPolicy( edge_k_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double loc[3];
                int idx[3] = { i, j, k };
                local_mesh.coordinates( Edge<Dim::K>(), idx, loc );
                loc_x( i, j, k ) = loc[Dim::I];
                loc_y( i, j, k ) = loc[Dim::J];
                loc_z( i, j, k ) = loc[Dim::K];
                measure( i, j, k ) = local_mesh.measure( Edge<Dim::K>(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        auto loc_z_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_z );
        for ( int i = 0; i < edge_k_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < edge_k_space.extent( Dim::J ); ++j )
                for ( int k = 0; k < edge_k_space.extent( Dim::K ); ++k )
                {
                    EXPECT_FLOAT_EQ( loc_x_m( i, j, k ),
                                     ghost_lc_m( Dim::I ) + cell_size * i );
                    EXPECT_FLOAT_EQ( loc_y_m( i, j, k ),
                                     ghost_lc_m( Dim::J ) + cell_size * j );
                    EXPECT_FLOAT_EQ( loc_z_m( i, j, k ),
                                     ghost_lc_m( Dim::K ) +
                                         cell_size * ( k + 0.5 ) );
                    EXPECT_FLOAT_EQ( measure_m( i, j, k ), cell_size );
                }
    }
}

//---------------------------------------------------------------------------//
void uniformTest( const std::array<int, 3> &ranks_per_dim,
                  const std::array<bool, 3> &is_dim_periodic )
{
    // Create the global mesh.
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high_corner = { -0.3, 9.5, 2.3 };
    double cell_size = 0.05;
    auto global_mesh =
        createUniformGlobalMesh( low_corner, high_corner, cell_size );

    // Create the global grid.
    ManualPartitioner partitioner( ranks_per_dim );
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid
    int halo_width = 1;
    auto local_grid = createLocalGrid( global_grid, halo_width );

    // Create the local mesh.
    auto local_mesh = createLocalMesh<TEST_DEVICE>( *local_grid );

    // Test the local mesh.
    uniformLocalMeshTest( local_mesh, *local_grid, low_corner, cell_size,
                          halo_width );
}

//---------------------------------------------------------------------------//
void nonUniformTest( const std::array<int, 3> &ranks_per_dim,
                     const std::array<bool, 3> &is_dim_periodic )
{
    // Create the global mesh.
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    double cell_size = 0.05;
    std::array<int, 3> num_cell = { 18, 188, 24 };
    std::array<std::vector<double>, 3> edges;
    for ( int d = 0; d < 3; ++d )
        for ( int n = 0; n < num_cell[d] + 1; ++n )
            edges[d].push_back( low_corner[d] + n * cell_size );
    auto global_mesh = createNonUniformGlobalMesh( edges[Dim::I], edges[Dim::J],
                                                   edges[Dim::K] );

    // Create the global grid.
    ManualPartitioner partitioner( ranks_per_dim );
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid.
    int halo_width = 1;
    auto local_grid = createLocalGrid( global_grid, halo_width );

    // Create the local mesh.
    auto local_mesh = createLocalMesh<TEST_DEVICE>( *local_grid );

    // Test the local mesh.
    uniformLocalMeshTest( local_mesh, *local_grid, low_corner, cell_size,
                          halo_width );
}

//---------------------------------------------------------------------------//
void irregularTest( const std::array<int, 3> &ranks_per_dim )
{
    // Create the global mesh using functions to build the edges. Use a cyclic
    // pattern for the cell sizes so we can easily compute cell sizes from
    // periodic wrap-around.
    std::array<double, 3> low_corner = { 3.1, 4.1, -2.8 };
    int ncell = 20;
    double ref_cell_size = 8.0 * std::atan( 1.0 ) / ncell;
    std::array<int, 3> num_cell = { ncell, ncell, ncell };

    auto i_func = [=]( const int i ) {
        return 0.5 * std::cos( i * ref_cell_size ) + low_corner[Dim::I];
    };
    auto j_func = [=]( const int j ) {
        return 2.0 * std::cos( j * ref_cell_size ) + low_corner[Dim::J];
    };
    auto k_func = [=]( const int k ) {
        return 1.5 * std::cos( k * ref_cell_size ) + low_corner[Dim::K];
    };

    std::array<std::vector<double>, 3> edges;
    for ( int n = 0; n < num_cell[Dim::I] + 1; ++n )
        edges[Dim::I].push_back( i_func( n ) );
    for ( int n = 0; n < num_cell[Dim::J] + 1; ++n )
        edges[Dim::J].push_back( j_func( n ) );
    for ( int n = 0; n < num_cell[Dim::K] + 1; ++n )
        edges[Dim::K].push_back( k_func( n ) );

    auto global_mesh = createNonUniformGlobalMesh( edges[Dim::I], edges[Dim::J],
                                                   edges[Dim::K] );

    // Create the global grid.
    std::array<bool, 3> periodic = { true, true, true };
    ManualPartitioner partitioner( ranks_per_dim );
    auto global_grid =
        createGlobalGrid( MPI_COMM_WORLD, global_mesh, periodic, partitioner );

    // Create a local grid.
    int halo_width = 1;
    auto local_grid = createLocalGrid( global_grid, halo_width );

    // Create the local mesh.
    auto local_mesh = createLocalMesh<TEST_DEVICE>( *local_grid );

    // Get index spaces
    auto ghost_cell_local_space =
        local_grid->indexSpace( Ghost(), Cell(), Local() );
    auto own_cell_global_space =
        local_grid->indexSpace( Own(), Cell(), Global() );
    auto own_cell_local_space =
        local_grid->indexSpace( Own(), Cell(), Local() );

    // Check the low and high corners.
    Kokkos::View<double[3], TEST_DEVICE> own_lc( "own_lc" );
    Kokkos::View<double[3], TEST_DEVICE> own_hc( "own_hc" );
    Kokkos::View<double[3], TEST_DEVICE> ghost_lc( "ghost_lc" );
    Kokkos::View<double[3], TEST_DEVICE> ghost_hc( "ghost_hc" );

    Kokkos::parallel_for(
        "get_corners", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 3 ),
        KOKKOS_LAMBDA( const int d ) {
            own_lc( d ) = local_mesh.lowCorner( Own(), d );
            own_hc( d ) = local_mesh.highCorner( Own(), d );
            ghost_lc( d ) = local_mesh.lowCorner( Ghost(), d );
            ghost_hc( d ) = local_mesh.highCorner( Ghost(), d );
        } );

    auto own_lc_m =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), own_lc );
    auto own_hc_m =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), own_hc );
    auto ghost_lc_m =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ghost_lc );
    auto ghost_hc_m =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), ghost_hc );

    EXPECT_FLOAT_EQ( own_lc_m( Dim::I ),
                     i_func( own_cell_global_space.min( Dim::I ) ) );
    EXPECT_FLOAT_EQ( own_lc_m( Dim::J ),
                     j_func( own_cell_global_space.min( Dim::J ) ) );
    EXPECT_FLOAT_EQ( own_lc_m( Dim::K ),
                     k_func( own_cell_global_space.min( Dim::K ) ) );

    EXPECT_FLOAT_EQ( own_hc_m( Dim::I ),
                     i_func( own_cell_global_space.max( Dim::I ) ) );
    EXPECT_FLOAT_EQ( own_hc_m( Dim::J ),
                     j_func( own_cell_global_space.max( Dim::J ) ) );
    EXPECT_FLOAT_EQ( own_hc_m( Dim::K ),
                     k_func( own_cell_global_space.max( Dim::K ) ) );

    EXPECT_FLOAT_EQ( ghost_lc_m( Dim::I ),
                     i_func( own_cell_global_space.min( Dim::I ) -
                             own_cell_local_space.min( Dim::I ) ) );
    EXPECT_FLOAT_EQ( ghost_lc_m( Dim::J ),
                     j_func( own_cell_global_space.min( Dim::J ) -
                             own_cell_local_space.min( Dim::J ) ) );
    EXPECT_FLOAT_EQ( ghost_lc_m( Dim::K ),
                     k_func( own_cell_global_space.min( Dim::K ) -
                             own_cell_local_space.min( Dim::K ) ) );

    EXPECT_FLOAT_EQ( ghost_hc_m( Dim::I ),
                     i_func( own_cell_global_space.max( Dim::I ) +
                             own_cell_local_space.min( Dim::I ) ) );
    EXPECT_FLOAT_EQ( ghost_hc_m( Dim::J ),
                     j_func( own_cell_global_space.max( Dim::J ) +
                             own_cell_local_space.min( Dim::J ) ) );
    EXPECT_FLOAT_EQ( ghost_hc_m( Dim::K ),
                     k_func( own_cell_global_space.max( Dim::K ) +
                             own_cell_local_space.min( Dim::K ) ) );

    // Check the cell locations and measures.
    auto cell_measure = createView<double, TEST_DEVICE>(
        "cell_measures", ghost_cell_local_space );
    auto cell_location_x = createView<double, TEST_DEVICE>(
        "cell_locations_x", ghost_cell_local_space );
    auto cell_location_y = createView<double, TEST_DEVICE>(
        "cell_locations_y", ghost_cell_local_space );
    auto cell_location_z = createView<double, TEST_DEVICE>(
        "cell_locations_z", ghost_cell_local_space );
    Kokkos::parallel_for(
        "get_cell_data",
        createExecutionPolicy( own_cell_local_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            int idx[3] = { i, j, k };
            double loc[3];
            local_mesh.coordinates( Cell(), idx, loc );
            cell_location_x( i, j, k ) = loc[Dim::I];
            cell_location_y( i, j, k ) = loc[Dim::J];
            cell_location_z( i, j, k ) = loc[Dim::K];
            cell_measure( i, j, k ) = local_mesh.measure( Cell(), idx );
        } );
    auto cell_measure_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), cell_measure );
    auto cell_location_x_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), cell_location_x );
    auto cell_location_y_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), cell_location_y );
    auto cell_location_z_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), cell_location_z );
    for ( int i = own_cell_global_space.min( Dim::I );
          i < own_cell_global_space.max( Dim::I ); ++i )
        for ( int j = own_cell_global_space.min( Dim::J );
              j < own_cell_global_space.max( Dim::J ); ++j )
            for ( int k = own_cell_global_space.min( Dim::K );
                  k < own_cell_global_space.max( Dim::K ); ++k )
            {
                double measure = ( i_func( i + 1 ) - i_func( i ) ) *
                                 ( j_func( j + 1 ) - j_func( j ) ) *
                                 ( k_func( k + 1 ) - k_func( k ) );
                double compute_m =
                    cell_measure_h( i - own_cell_global_space.min( Dim::I ) +
                                        own_cell_local_space.min( Dim::I ),
                                    j - own_cell_global_space.min( Dim::J ) +
                                        own_cell_local_space.min( Dim::J ),
                                    k - own_cell_global_space.min( Dim::K ) +
                                        own_cell_local_space.min( Dim::K ) );
                EXPECT_FLOAT_EQ( measure, compute_m );

                double x_loc = ( i_func( i + 1 ) + i_func( i ) ) / 2.0;
                double compute_x =
                    cell_location_x_h( i - own_cell_global_space.min( Dim::I ) +
                                           own_cell_local_space.min( Dim::I ),
                                       j - own_cell_global_space.min( Dim::J ) +
                                           own_cell_local_space.min( Dim::J ),
                                       k - own_cell_global_space.min( Dim::K ) +
                                           own_cell_local_space.min( Dim::K ) );
                EXPECT_FLOAT_EQ( x_loc, compute_x );

                double y_loc = ( j_func( j + 1 ) + j_func( j ) ) / 2.0;
                double compute_y =
                    cell_location_y_h( i - own_cell_global_space.min( Dim::I ) +
                                           own_cell_local_space.min( Dim::I ),
                                       j - own_cell_global_space.min( Dim::J ) +
                                           own_cell_local_space.min( Dim::J ),
                                       k - own_cell_global_space.min( Dim::K ) +
                                           own_cell_local_space.min( Dim::K ) );
                EXPECT_FLOAT_EQ( y_loc, compute_y );

                double z_loc = ( k_func( k + 1 ) + k_func( k ) ) / 2.0;
                double compute_z =
                    cell_location_z_h( i - own_cell_global_space.min( Dim::I ) +
                                           own_cell_local_space.min( Dim::I ),
                                       j - own_cell_global_space.min( Dim::J ) +
                                           own_cell_local_space.min( Dim::J ),
                                       k - own_cell_global_space.min( Dim::K ) +
                                           own_cell_local_space.min( Dim::K ) );
                EXPECT_FLOAT_EQ( z_loc, compute_z );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( mesh, periodic_uniform_test )
{
    std::array<bool, 3> is_dim_periodic = { true, true, true };

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    uniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    uniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    uniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    uniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    uniformTest( ranks_per_dim, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( mesh, periodic_non_uniform_test )
{
    std::array<bool, 3> is_dim_periodic = { true, true, true };

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    nonUniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    nonUniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    nonUniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    nonUniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    nonUniformTest( ranks_per_dim, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( mesh, non_periodic_uniform_test )
{
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    uniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    uniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    uniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    uniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    uniformTest( ranks_per_dim, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( mesh, non_periodic_non_uniform_test )
{
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    nonUniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    nonUniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    nonUniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    nonUniformTest( ranks_per_dim, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    nonUniformTest( ranks_per_dim, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( mesh, irregular_non_uniform_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    irregularTest( ranks_per_dim );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    irregularTest( ranks_per_dim );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    irregularTest( ranks_per_dim );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    irregularTest( ranks_per_dim );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    irregularTest( ranks_per_dim );
}

//---------------------------------------------------------------------------//

} // end namespace Test
