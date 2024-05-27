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

#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Grid_LocalMesh.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_Types.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using namespace Cabana::Grid;

namespace Test
{

//---------------------------------------------------------------------------//
template <class LocalMeshType, class LocalGridType>
void uniformLocalMeshTest2d( const LocalMeshType& local_mesh,
                             const LocalGridType& local_grid,
                             const std::array<double, 2>& low_corner,
                             const double cell_size, const int halo_width )
{
    // Get the global grid.
    const auto& global_grid = local_grid.globalGrid();

    // Check the low and high corners.
    Kokkos::View<double[2], TEST_MEMSPACE> own_lc( "own_lc" );
    Kokkos::View<double[2], TEST_MEMSPACE> own_hc( "own_hc" );
    Kokkos::View<double[2], TEST_MEMSPACE> own_ext( "own_extent" );
    Kokkos::View<double[2], TEST_MEMSPACE> ghost_lc( "ghost_lc" );
    Kokkos::View<double[2], TEST_MEMSPACE> ghost_hc( "ghost_hc" );
    Kokkos::View<double[2], TEST_MEMSPACE> ghost_ext( "ghost_extent" );

    Kokkos::parallel_for(
        "get_corners", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 2 ),
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

    for ( int d = 0; d < 2; ++d )
        EXPECT_FLOAT_EQ( own_lc_m( d ),
                         low_corner[d] +
                             cell_size * global_grid.globalOffset( d ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_FLOAT_EQ( own_hc_m( d ),
                         low_corner[d] +
                             cell_size * ( global_grid.globalOffset( d ) +
                                           global_grid.ownedNumCell( d ) ) );

    for ( int d = 0; d < 2; ++d )
        EXPECT_FLOAT_EQ( own_ext_m( d ),
                         cell_size * global_grid.ownedNumCell( d ) );

    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_FLOAT_EQ(
            ghost_lc_m( d ),
            low_corner[d] +
                cell_size * ( global_grid.globalOffset( d ) - halo_width ) );

        EXPECT_FLOAT_EQ( ghost_hc_m( d ),
                         low_corner[d] +
                             cell_size * ( global_grid.globalOffset( d ) +
                                           global_grid.ownedNumCell( d ) +
                                           halo_width ) );

        EXPECT_FLOAT_EQ(
            ghost_ext_m( d ),
            cell_size * ( global_grid.ownedNumCell( d ) + 2 * halo_width ) );
    }

    // Check the cell locations and measures.
    auto cell_space = local_grid.indexSpace( Ghost(), Cell(), Local() );
    {
        auto measure =
            createView<double, TEST_MEMSPACE>( "measure", cell_space );
        auto loc_x = createView<double, TEST_MEMSPACE>( "loc_x", cell_space );
        auto loc_y = createView<double, TEST_MEMSPACE>( "loc_y", cell_space );
        Kokkos::parallel_for(
            "get_cell_coord",
            createExecutionPolicy( cell_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                double loc[2];
                int idx[2] = { i, j };
                local_mesh.coordinates( Cell(), idx, loc );
                loc_x( i, j ) = loc[Dim::I];
                loc_y( i, j ) = loc[Dim::J];
                measure( i, j ) = local_mesh.measure( Cell(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        for ( int i = 0; i < cell_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < cell_space.extent( Dim::J ); ++j )
            {
                EXPECT_FLOAT_EQ( loc_x_m( i, j ), ghost_lc_m( Dim::I ) +
                                                      cell_size * ( i + 0.5 ) );
                EXPECT_FLOAT_EQ( loc_y_m( i, j ), ghost_lc_m( Dim::J ) +
                                                      cell_size * ( j + 0.5 ) );
                EXPECT_FLOAT_EQ( measure_m( i, j ), cell_size * cell_size );
            }
    }

    // Check the node locations and measures.
    auto node_space = local_grid.indexSpace( Ghost(), Node(), Local() );
    {
        auto measure =
            createView<double, TEST_MEMSPACE>( "measure", node_space );
        auto loc_x = createView<double, TEST_MEMSPACE>( "loc_x", node_space );
        auto loc_y = createView<double, TEST_MEMSPACE>( "loc_y", node_space );
        Kokkos::parallel_for(
            "get_node_coord",
            createExecutionPolicy( node_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                double loc[2];
                int idx[2] = { i, j };
                local_mesh.coordinates( Node(), idx, loc );
                loc_x( i, j ) = loc[Dim::I];
                loc_y( i, j ) = loc[Dim::J];
                measure( i, j ) = local_mesh.measure( Node(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        for ( int i = 0; i < node_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < node_space.extent( Dim::J ); ++j )
            {
                EXPECT_FLOAT_EQ( loc_x_m( i, j ),
                                 ghost_lc_m( Dim::I ) + cell_size * i );
                EXPECT_FLOAT_EQ( loc_y_m( i, j ),
                                 ghost_lc_m( Dim::J ) + cell_size * j );
                EXPECT_FLOAT_EQ( measure_m( i, j ), 0.0 );
            }
    }

    // Check the I-face locations and measures
    auto face_i_space =
        local_grid.indexSpace( Ghost(), Face<Dim::I>(), Local() );
    {
        auto measure =
            createView<double, TEST_MEMSPACE>( "measure", face_i_space );
        auto loc_x = createView<double, TEST_MEMSPACE>( "loc_x", face_i_space );
        auto loc_y = createView<double, TEST_MEMSPACE>( "loc_y", face_i_space );
        Kokkos::parallel_for(
            "get_face_i_coord",
            createExecutionPolicy( face_i_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                double loc[2];
                int idx[2] = { i, j };
                local_mesh.coordinates( Face<Dim::I>(), idx, loc );
                loc_x( i, j ) = loc[Dim::I];
                loc_y( i, j ) = loc[Dim::J];
                measure( i, j ) = local_mesh.measure( Face<Dim::I>(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        for ( int i = 0; i < face_i_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < face_i_space.extent( Dim::J ); ++j )
            {
                EXPECT_FLOAT_EQ( loc_x_m( i, j ),
                                 ghost_lc_m( Dim::I ) + cell_size * i );
                EXPECT_FLOAT_EQ( loc_y_m( i, j ), ghost_lc_m( Dim::J ) +
                                                      cell_size * ( j + 0.5 ) );
                EXPECT_FLOAT_EQ( measure_m( i, j ), cell_size );
            }
    }

    // Check the J-face locations and measures.
    auto face_j_space =
        local_grid.indexSpace( Ghost(), Face<Dim::J>(), Local() );
    {
        auto measure =
            createView<double, TEST_MEMSPACE>( "measure", face_j_space );
        auto loc_x = createView<double, TEST_MEMSPACE>( "loc_x", face_j_space );
        auto loc_y = createView<double, TEST_MEMSPACE>( "loc_y", face_j_space );
        Kokkos::parallel_for(
            "get_face_j_coord",
            createExecutionPolicy( face_j_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                double loc[2];
                int idx[2] = { i, j };
                local_mesh.coordinates( Face<Dim::J>(), idx, loc );
                loc_x( i, j ) = loc[Dim::I];
                loc_y( i, j ) = loc[Dim::J];
                measure( i, j ) = local_mesh.measure( Face<Dim::J>(), idx );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        for ( int i = 0; i < face_j_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < face_j_space.extent( Dim::J ); ++j )
            {
                EXPECT_FLOAT_EQ( loc_x_m( i, j ), ghost_lc_m( Dim::I ) +
                                                      cell_size * ( i + 0.5 ) );
                EXPECT_FLOAT_EQ( loc_y_m( i, j ),
                                 ghost_lc_m( Dim::J ) + cell_size * j );
                EXPECT_FLOAT_EQ( measure_m( i, j ), cell_size );
            }
    }

    // Extra check for using local mesh functions with shared index space.
    auto shared_cell_space = local_grid.sharedIndexSpace( Own(), Cell(), 0, 1 );
    std::cout << shared_cell_space.extent( 0 ) << " "
              << shared_cell_space.extent( 1 ) << std::endl;
    {
        auto measure =
            createView<double, TEST_MEMSPACE>( "measure", cell_space );
        auto loc_x = createView<double, TEST_MEMSPACE>( "loc_x", cell_space );
        auto loc_y = createView<double, TEST_MEMSPACE>( "loc_y", cell_space );
        std::cout << loc_x.extent( 0 ) << " " << loc_x.extent( 1 ) << std::endl;
        Kokkos::parallel_for(
            "get_cell_coord",
            createExecutionPolicy( shared_cell_space, TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                Kokkos::Array<double, 2> loc;
                // Passing this index to the local mesh is the relevant check.
                auto idx = shared_cell_space.min();
                local_mesh.coordinates( Cell(), idx.data(), loc.data() );
                loc_x( i, j ) = loc[Dim::I];
                loc_y( i, j ) = loc[Dim::J];
                measure( i, j ) = local_mesh.measure( Cell(), idx.data() );
            } );
        auto measure_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), measure );
        auto loc_x_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_x );
        auto loc_y_m =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), loc_y );
        for ( int i = 0; i < shared_cell_space.extent( Dim::I ); ++i )
            for ( int j = 0; j < shared_cell_space.extent( Dim::J ); ++j )
            {
                int i_shared = shared_cell_space.min( Dim::I ) + i;
                int j_shared = shared_cell_space.min( Dim::J ) + j;

                // This is specific to the neighbor chosen above at (0,1).
                EXPECT_FLOAT_EQ( loc_x_m( i_shared, j_shared ),
                                 own_lc_m( Dim::I ) + cell_size * 0.5 );
                EXPECT_FLOAT_EQ( loc_y_m( i_shared, j_shared ),
                                 own_hc_m( Dim::J ) - cell_size * 0.5 );
                EXPECT_FLOAT_EQ( measure_m( i_shared, j_shared ),
                                 cell_size * cell_size );
            }
    }
}

//---------------------------------------------------------------------------//
void uniformTest2d( const std::array<int, 2>& ranks_per_dim,
                    const std::array<bool, 2>& is_dim_periodic )
{
    // Create the global mesh.
    std::array<double, 2> low_corner = { -1.2, 0.1 };
    std::array<double, 2> high_corner = { -0.3, 9.5 };
    double cell_size = 0.05;
    auto global_mesh =
        createUniformGlobalMesh( low_corner, high_corner, cell_size );

    // Create the global grid.
    ManualBlockPartitioner<2> partitioner( ranks_per_dim );
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid
    int halo_width = 1;
    auto local_grid = createLocalGrid( global_grid, halo_width );

    // Create the local mesh.
    auto local_mesh = createLocalMesh<TEST_MEMSPACE>( *local_grid );

    // Test the local mesh.
    uniformLocalMeshTest2d( local_mesh, *local_grid, low_corner, cell_size,
                            halo_width );
}

//---------------------------------------------------------------------------//
void nonUniformTest2d( const std::array<int, 2>& ranks_per_dim,
                       const std::array<bool, 2>& is_dim_periodic )
{
    // Create the global mesh.
    std::array<double, 2> low_corner = { -1.2, 0.1 };
    double cell_size = 0.05;
    std::array<int, 2> num_cell = { 18, 188 };
    std::array<std::vector<double>, 2> edges;
    for ( int d = 0; d < 2; ++d )
        for ( int n = 0; n < num_cell[d] + 1; ++n )
            edges[d].push_back( low_corner[d] + n * cell_size );
    auto global_mesh =
        createNonUniformGlobalMesh( edges[Dim::I], edges[Dim::J] );

    // Create the global grid.
    ManualBlockPartitioner<2> partitioner( ranks_per_dim );
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid.
    int halo_width = 1;
    auto local_grid = createLocalGrid( global_grid, halo_width );

    // Create the local mesh.
    auto local_mesh = createLocalMesh<TEST_MEMSPACE>( *local_grid );

    // Test the local mesh.
    uniformLocalMeshTest2d( local_mesh, *local_grid, low_corner, cell_size,
                            halo_width );
}

//---------------------------------------------------------------------------//
void irregularTest2d( const std::array<int, 2>& ranks_per_dim )
{
    // Create the global mesh using functions to build the edges. Use a cyclic
    // pattern for the cell sizes so we can easily compute cell sizes from
    // periodic wrap-around.
    std::array<double, 2> low_corner = { 3.1, 4.1 };
    int ncell = 20;
    double ref_cell_size = 8.0 * std::atan( 1.0 ) / ncell;
    std::array<int, 2> num_cell = { ncell, ncell };

    auto i_func = [=]( const int i )
    { return 0.5 * std::cos( i * ref_cell_size ) + low_corner[Dim::I]; };
    auto j_func = [=]( const int j )
    { return 2.0 * std::cos( j * ref_cell_size ) + low_corner[Dim::J]; };

    std::array<std::vector<double>, 2> edges;
    for ( int n = 0; n < num_cell[Dim::I] + 1; ++n )
        edges[Dim::I].push_back( i_func( n ) );
    for ( int n = 0; n < num_cell[Dim::J] + 1; ++n )
        edges[Dim::J].push_back( j_func( n ) );

    auto global_mesh =
        createNonUniformGlobalMesh( edges[Dim::I], edges[Dim::J] );

    // Create the global grid.
    std::array<bool, 2> periodic = { true, true };
    ManualBlockPartitioner<2> partitioner( ranks_per_dim );
    auto global_grid =
        createGlobalGrid( MPI_COMM_WORLD, global_mesh, periodic, partitioner );

    // Create a local grid.
    int halo_width = 1;
    auto local_grid = createLocalGrid( global_grid, halo_width );

    // Create the local mesh.
    auto local_mesh = createLocalMesh<TEST_MEMSPACE>( *local_grid );

    // Get index spaces
    auto ghost_cell_local_space =
        local_grid->indexSpace( Ghost(), Cell(), Local() );
    auto own_cell_global_space =
        local_grid->indexSpace( Own(), Cell(), Global() );
    auto own_cell_local_space =
        local_grid->indexSpace( Own(), Cell(), Local() );

    // Check the low and high corners.
    Kokkos::View<double[2], TEST_MEMSPACE> own_lc( "own_lc" );
    Kokkos::View<double[2], TEST_MEMSPACE> own_hc( "own_hc" );
    Kokkos::View<double[2], TEST_MEMSPACE> ghost_lc( "ghost_lc" );
    Kokkos::View<double[2], TEST_MEMSPACE> ghost_hc( "ghost_hc" );

    Kokkos::parallel_for(
        "get_corners", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 2 ),
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

    EXPECT_FLOAT_EQ( own_hc_m( Dim::I ),
                     i_func( own_cell_global_space.max( Dim::I ) ) );
    EXPECT_FLOAT_EQ( own_hc_m( Dim::J ),
                     j_func( own_cell_global_space.max( Dim::J ) ) );

    EXPECT_FLOAT_EQ( ghost_lc_m( Dim::I ),
                     i_func( own_cell_global_space.min( Dim::I ) -
                             own_cell_local_space.min( Dim::I ) ) );
    EXPECT_FLOAT_EQ( ghost_lc_m( Dim::J ),
                     j_func( own_cell_global_space.min( Dim::J ) -
                             own_cell_local_space.min( Dim::J ) ) );

    EXPECT_FLOAT_EQ( ghost_hc_m( Dim::I ),
                     i_func( own_cell_global_space.max( Dim::I ) +
                             own_cell_local_space.min( Dim::I ) ) );
    EXPECT_FLOAT_EQ( ghost_hc_m( Dim::J ),
                     j_func( own_cell_global_space.max( Dim::J ) +
                             own_cell_local_space.min( Dim::J ) ) );

    // Check the cell locations and measures.
    auto cell_measure = createView<double, TEST_MEMSPACE>(
        "cell_measures", ghost_cell_local_space );
    auto cell_location_x = createView<double, TEST_MEMSPACE>(
        "cell_locations_x", ghost_cell_local_space );
    auto cell_location_y = createView<double, TEST_MEMSPACE>(
        "cell_locations_y", ghost_cell_local_space );
    Kokkos::parallel_for(
        "get_cell_data",
        createExecutionPolicy( own_cell_local_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            int idx[2] = { i, j };
            double loc[2];
            local_mesh.coordinates( Cell(), idx, loc );
            cell_location_x( i, j ) = loc[Dim::I];
            cell_location_y( i, j ) = loc[Dim::J];
            cell_measure( i, j ) = local_mesh.measure( Cell(), idx );
        } );
    auto cell_measure_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), cell_measure );
    auto cell_location_x_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), cell_location_x );
    auto cell_location_y_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), cell_location_y );
    for ( int i = own_cell_global_space.min( Dim::I );
          i < own_cell_global_space.max( Dim::I ); ++i )
        for ( int j = own_cell_global_space.min( Dim::J );
              j < own_cell_global_space.max( Dim::J ); ++j )
        {
            double measure = ( i_func( i + 1 ) - i_func( i ) ) *
                             ( j_func( j + 1 ) - j_func( j ) );
            double compute_m =
                cell_measure_h( i - own_cell_global_space.min( Dim::I ) +
                                    own_cell_local_space.min( Dim::I ),
                                j - own_cell_global_space.min( Dim::J ) +
                                    own_cell_local_space.min( Dim::J ) );
            EXPECT_FLOAT_EQ( measure, compute_m );

            double x_loc = ( i_func( i + 1 ) + i_func( i ) ) / 2.0;
            double compute_x =
                cell_location_x_h( i - own_cell_global_space.min( Dim::I ) +
                                       own_cell_local_space.min( Dim::I ),
                                   j - own_cell_global_space.min( Dim::J ) +
                                       own_cell_local_space.min( Dim::J ) );
            EXPECT_FLOAT_EQ( x_loc, compute_x );

            double y_loc = ( j_func( j + 1 ) + j_func( j ) ) / 2.0;
            double compute_y =
                cell_location_y_h( i - own_cell_global_space.min( Dim::I ) +
                                       own_cell_local_space.min( Dim::I ),
                                   j - own_cell_global_space.min( Dim::J ) +
                                       own_cell_local_space.min( Dim::J ) );
            EXPECT_FLOAT_EQ( y_loc, compute_y );
        }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( mesh, periodic_uniform_test )
{
    std::array<bool, 2> is_dim_periodic = { true, true };

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> ranks_per_dim = { 0, 0 };
    MPI_Dims_create( comm_size, 2, ranks_per_dim.data() );

    uniformTest2d( ranks_per_dim, is_dim_periodic );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        uniformTest2d( ranks_per_dim, is_dim_periodic );
    }
}

//---------------------------------------------------------------------------//
TEST( mesh, periodic_non_uniform_test )
{
    std::array<bool, 2> is_dim_periodic = { true, true };

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> ranks_per_dim = { 0, 0 };
    MPI_Dims_create( comm_size, 2, ranks_per_dim.data() );

    nonUniformTest2d( ranks_per_dim, is_dim_periodic );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        nonUniformTest2d( ranks_per_dim, is_dim_periodic );
    }
}

//---------------------------------------------------------------------------//
TEST( mesh, non_periodic_uniform_test )
{
    std::array<bool, 2> is_dim_periodic = { false, false };

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> ranks_per_dim = { 0, 0 };
    MPI_Dims_create( comm_size, 2, ranks_per_dim.data() );

    uniformTest2d( ranks_per_dim, is_dim_periodic );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        uniformTest2d( ranks_per_dim, is_dim_periodic );
    }
}

//---------------------------------------------------------------------------//
TEST( mesh, non_periodic_non_uniform_test )
{
    std::array<bool, 2> is_dim_periodic = { false, false };

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> ranks_per_dim = { 0, 0 };
    MPI_Dims_create( comm_size, 2, ranks_per_dim.data() );

    nonUniformTest2d( ranks_per_dim, is_dim_periodic );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        nonUniformTest2d( ranks_per_dim, is_dim_periodic );
    }
}

//---------------------------------------------------------------------------//
TEST( mesh, irregular_non_uniform_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> ranks_per_dim = { 0, 0 };
    MPI_Dims_create( comm_size, 2, ranks_per_dim.data() );

    irregularTest2d( ranks_per_dim );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        irregularTest2d( ranks_per_dim );
    }
}

//---------------------------------------------------------------------------//

} // end namespace Test
