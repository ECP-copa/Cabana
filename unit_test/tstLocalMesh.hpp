/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Kokkos_Core.hpp>

#include <Cajita_Types.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_ManualPartitioner.hpp>
#include <Cajita_Block.hpp>
#include <Cajita_LocalMesh.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <vector>
#include <cmath>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
template<class LocalMeshType, class BlockType>
void uniformLocalMeshTest( const LocalMeshType& local_mesh,
                           const BlockType& block,
                           const std::array<double,3>& low_corner,
                           const double cell_size,
                           const int halo_width )
{
    // Get the global grid.
    const auto& global_grid = block.globalGrid();

    // Check the low and high corners.
    Kokkos::View<double[3],TEST_DEVICE> own_lc( "own_lc" );
    Kokkos::View<double[3],TEST_DEVICE> own_hc( "own_hc" );
    Kokkos::View<double[3],TEST_DEVICE> ghost_lc( "ghost_lc" );
    Kokkos::View<double[3],TEST_DEVICE> ghost_hc( "ghost_hc" );

    Kokkos::parallel_for(
        "get_corners",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,3),
        KOKKOS_LAMBDA( const int d ){
            own_lc(d) = local_mesh.lowCorner( Own(), d );
            own_hc(d) = local_mesh.highCorner( Own(), d );
            ghost_lc(d) = local_mesh.lowCorner( Ghost(), d );
            ghost_hc(d) = local_mesh.highCorner( Ghost(), d );
        });

    auto own_lc_m = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), own_lc );
    auto own_hc_m = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), own_hc );
    auto ghost_lc_m = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), ghost_lc );
    auto ghost_hc_m = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), ghost_hc );

    for ( int d = 0; d < 3; ++d )
        EXPECT_FLOAT_EQ(
            own_lc_m(d), low_corner[d] + cell_size * global_grid.globalOffset(d) );

    for ( int d = 0; d < 3; ++d )
        EXPECT_FLOAT_EQ(
            own_hc_m(d), low_corner[d] + cell_size * (
                global_grid.globalOffset(d) + global_grid.ownedNumCell(d) ) );

    for ( int d = 0; d < 3; ++d )
    {
        int ghost_lc_offset =
            ( global_grid.isPeriodic( d ) || global_grid.dimBlockId( d ) > 0 )
            ? halo_width : 0;
        EXPECT_FLOAT_EQ(
            ghost_lc_m(d), low_corner[d] + cell_size *
            ( global_grid.globalOffset(d) - ghost_lc_offset ) );
    }

    for ( int d = 0; d < 3; ++d )
    {
        int ghost_hc_offset =
            ( global_grid.isPeriodic( d ) ||
              global_grid.dimBlockId( d ) < global_grid.dimNumBlock( d ) - 1 )
            ? block.haloCellWidth() : 0;
        EXPECT_FLOAT_EQ(
            ghost_hc_m(d), low_corner[d] + cell_size *
            ( global_grid.globalOffset(d) + ghost_hc_offset +
              global_grid.ownedNumCell(d) ) );
    }

    // Check the cell locations.
    auto cell_space = block.indexSpace( Ghost(), Cell(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        Kokkos::View<double*,TEST_DEVICE> coord( "coord", cell_space.extent(d) );
        Kokkos::parallel_for(
            "get_cell_coord",
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,cell_space.extent(d)),
            KOKKOS_LAMBDA( const int c ){
                coord(c) = local_mesh.coordinate( Cell(), c, d );
            });
        auto coord_m = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), coord );
        for ( int c = 0; c < cell_space.extent(d); ++c )
            EXPECT_FLOAT_EQ( coord_m(c), ghost_lc_m(d) + cell_size * (c+0.5) );
    }

    // Check the node locations.
    auto node_space = block.indexSpace( Ghost(), Node(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        Kokkos::View<double*,TEST_DEVICE> coord( "coord", node_space.extent(d) );
        Kokkos::parallel_for(
            "get_node_coord",
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,node_space.extent(d)),
            KOKKOS_LAMBDA( const int c ){
                coord(c) = local_mesh.coordinate( Node(), c, d );
            });
        auto coord_m = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), coord );
        for ( int c = 0; c < node_space.extent(d); ++c )
            EXPECT_FLOAT_EQ( coord_m(c), ghost_lc_m(d) + cell_size * c );
    }

    // Check the I-face locations.
    auto face_i_space = block.indexSpace( Ghost(), Face<Dim::I>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        Kokkos::View<double*,TEST_DEVICE> coord( "coord", face_i_space.extent(d) );
        Kokkos::parallel_for(
            "get_face_i_coord",
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,face_i_space.extent(d)),
            KOKKOS_LAMBDA( const int c ){
                coord(c) = local_mesh.coordinate( Face<Dim::I>(), c, d );
            });
        auto coord_m = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), coord );
        double offset = ( d == Dim::I ) ? 0.0 : 0.5;
        for ( int c = 0; c < face_i_space.extent(d); ++c )
            EXPECT_FLOAT_EQ( coord_m(c), ghost_lc_m(d) + cell_size * (c + offset) );
    }

    // Check the J-face locations.
    auto face_j_space = block.indexSpace( Ghost(), Face<Dim::J>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        Kokkos::View<double*,TEST_DEVICE> coord( "coord", face_j_space.extent(d) );
        Kokkos::parallel_for(
            "get_face_j_coord",
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,face_j_space.extent(d)),
            KOKKOS_LAMBDA( const int c ){
                coord(c) = local_mesh.coordinate( Face<Dim::J>(), c, d );
            });
        auto coord_m = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), coord );
        double offset = ( d == Dim::J ) ? 0.0 : 0.5;
        for ( int c = 0; c < face_j_space.extent(d); ++c )
            EXPECT_FLOAT_EQ( coord_m(c), ghost_lc_m(d) + cell_size * (c + offset) );
    }

    // Check the K-face locations.
    auto face_k_space = block.indexSpace( Ghost(), Face<Dim::K>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        Kokkos::View<double*,TEST_DEVICE> coord( "coord", face_k_space.extent(d) );
        Kokkos::parallel_for(
            "get_face_k_coord",
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,face_k_space.extent(d)),
            KOKKOS_LAMBDA( const int c ){
                coord(c) = local_mesh.coordinate( Face<Dim::K>(), c, d );
            });
        auto coord_m = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), coord );
        double offset = ( d == Dim::K ) ? 0.0 : 0.5;
        for ( int c = 0; c < face_k_space.extent(d); ++c )
            EXPECT_FLOAT_EQ( coord_m(c), ghost_lc_m(d) + cell_size * (c + offset) );
    }

    // Check the I-edge locations.
    auto edge_i_space = block.indexSpace( Ghost(), Edge<Dim::I>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        Kokkos::View<double*,TEST_DEVICE> coord( "coord", edge_i_space.extent(d) );
        Kokkos::parallel_for(
            "get_edge_i_coord",
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,edge_i_space.extent(d)),
            KOKKOS_LAMBDA( const int c ){
                coord(c) = local_mesh.coordinate( Edge<Dim::I>(), c, d );
            });
        auto coord_m = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), coord );
        double offset = ( d == Dim::I ) ? 0.5 : 0.0;
        for ( int c = 0; c < edge_i_space.extent(d); ++c )
            EXPECT_FLOAT_EQ( coord_m(c), ghost_lc_m(d) + cell_size * (c + offset) );
    }

    // Check the J-edge locations.
    auto edge_j_space = block.indexSpace( Ghost(), Edge<Dim::J>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        Kokkos::View<double*,TEST_DEVICE> coord( "coord", edge_j_space.extent(d) );
        Kokkos::parallel_for(
            "get_edge_j_coord",
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,edge_j_space.extent(d)),
            KOKKOS_LAMBDA( const int c ){
                coord(c) = local_mesh.coordinate( Edge<Dim::J>(), c, d );
            });
        auto coord_m = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), coord );
        double offset = ( d == Dim::J ) ? 0.5 : 0.0;
        for ( int c = 0; c < edge_j_space.extent(d); ++c )
            EXPECT_FLOAT_EQ( coord_m(c), ghost_lc_m(d) + cell_size * (c + offset) );
    }

    // Check the K-edge locations.
    auto edge_k_space = block.indexSpace( Ghost(), Edge<Dim::K>(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        Kokkos::View<double*,TEST_DEVICE> coord( "coord", edge_k_space.extent(d) );
        Kokkos::parallel_for(
            "get_edge_k_coord",
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,edge_k_space.extent(d)),
            KOKKOS_LAMBDA( const int c ){
                coord(c) = local_mesh.coordinate( Edge<Dim::K>(), c, d );
            });
        auto coord_m = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), coord );
        double offset = ( d == Dim::K ) ? 0.5 : 0.0;
        for ( int c = 0; c < edge_k_space.extent(d); ++c )
            EXPECT_FLOAT_EQ( coord_m(c), ghost_lc_m(d) + cell_size * (c + offset) );
    }
}

//---------------------------------------------------------------------------//
void uniformTest( const std::array<int,3>& ranks_per_dim,
                  const std::array<bool,3>& is_dim_periodic )
{
    // Create the global mesh.
    std::array<double,3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double,3> high_corner = { -0.3, 9.5, 2.3 };
    double cell_size = 0.05;
    auto global_mesh = createUniformGlobalMesh(
        low_corner, high_corner, cell_size );

    // Create the global grid.
    ManualPartitioner partitioner( ranks_per_dim );
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         global_mesh,
                                         is_dim_periodic,
                                         partitioner );

    // Create a  grid block.
    int halo_width = 1;
    auto block = createBlock( global_grid, halo_width );

    // Create the local mesh.
    auto local_mesh = createLocalMesh<TEST_DEVICE>( *block );

    // Test the local mesh.
    uniformLocalMeshTest( local_mesh, *block, low_corner,
                          cell_size, halo_width );
}

//---------------------------------------------------------------------------//
void nonUniformTest( const std::array<int,3>& ranks_per_dim,
                     const std::array<bool,3>& is_dim_periodic )
{
    // Create the global mesh.
    std::array<double,3> low_corner = { -1.2, 0.1, 1.1 };
    double cell_size = 0.05;
    std::array<int,3> num_cell = { 18, 188, 24 };
    std::array<std::vector<double>,3> edges;
    for ( int d = 0; d < 3; ++d )
        for ( int n = 0; n < num_cell[d] + 1; ++n )
            edges[d].push_back( low_corner[d] + n * cell_size );
    auto global_mesh = createNonUniformGlobalMesh(
        edges[Dim::I], edges[Dim::J], edges[Dim::K] );

    // Create the global grid.
    ManualPartitioner partitioner( ranks_per_dim );
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         global_mesh,
                                         is_dim_periodic,
                                         partitioner );

    // Create a  grid block.
    int halo_width = 1;
    auto block = createBlock( global_grid, halo_width );

    // Create the local mesh.
    auto local_mesh = createLocalMesh<TEST_DEVICE>( *block );

    // Test the local mesh.
    uniformLocalMeshTest( local_mesh, *block, low_corner,
                          cell_size, halo_width );
}

//---------------------------------------------------------------------------//
void irregularTest( const std::array<int,3>& ranks_per_dim )
{
    // Create the global mesh using functions to build the edges. Use a cyclic
    // pattern for the cell sizes so we can easily compute cell sizes from
    // periodic wrap-around.
    std::array<double,3> low_corner = { 3.1, 4.1, -2.8 };
    int ncell = 20;
    double ref_cell_size = 8.0 * std::atan(1.0) / ncell;
    std::array<int,3> num_cell = { ncell, ncell, ncell };

    auto i_func = [=]( const int i )
                  { return 0.5*std::cos(i*ref_cell_size)+low_corner[Dim::I]; };
    auto j_func = [=]( const int j )
                  { return 2.0*std::cos(j*ref_cell_size)+low_corner[Dim::J]; };
    auto k_func = [=]( const int k )
                  { return 1.5*std::cos(k*ref_cell_size)+low_corner[Dim::K]; };

    std::array<std::vector<double>,3> edges;
    for ( int n = 0; n < num_cell[Dim::I] + 1; ++n )
        edges[Dim::I].push_back( i_func(n) );
    for ( int n = 0; n < num_cell[Dim::J] + 1; ++n )
        edges[Dim::J].push_back( j_func(n) );
    for ( int n = 0; n < num_cell[Dim::K] + 1; ++n )
        edges[Dim::K].push_back( k_func(n) );

    auto global_mesh = createNonUniformGlobalMesh(
        edges[Dim::I], edges[Dim::J], edges[Dim::K] );

    // Create the global grid.
    std::array<bool,3> periodic = {true,true,true};
    ManualPartitioner partitioner( ranks_per_dim );
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         global_mesh,
                                         periodic,
                                         partitioner );

    // Create a  grid block.
    int halo_width = 1;
    auto block = createBlock( global_grid, halo_width );

    // Create the local mesh.
    auto local_mesh = createLocalMesh<TEST_DEVICE>( *block );

    // Get index spaces
    auto own_cell_local_space = block->indexSpace( Own(), Cell(), Local() );
    auto ghost_cell_local_space = block->indexSpace( Ghost(), Cell(), Local() );
    auto own_cell_global_space = block->indexSpace( Own(), Cell(), Global() );
    auto ghost_cell_global_space = block->indexSpace( Ghost(), Cell(), Global() );

    // Check the low and high corners.
    Kokkos::View<double[3],TEST_DEVICE> own_lc( "own_lc" );
    Kokkos::View<double[3],TEST_DEVICE> own_hc( "own_hc" );
    Kokkos::View<double[3],TEST_DEVICE> ghost_lc( "ghost_lc" );
    Kokkos::View<double[3],TEST_DEVICE> ghost_hc( "ghost_hc" );

    Kokkos::parallel_for(
        "get_corners",
        Kokkos::RangePolicy<TEST_EXECSPACE>(0,3),
        KOKKOS_LAMBDA( const int d ){
            own_lc(d) = local_mesh.lowCorner( Own(), d );
            own_hc(d) = local_mesh.highCorner( Own(), d );
            ghost_lc(d) = local_mesh.lowCorner( Ghost(), d );
            ghost_hc(d) = local_mesh.highCorner( Ghost(), d );
        });

    auto own_lc_m = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), own_lc );
    auto own_hc_m = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), own_hc );
    auto ghost_lc_m = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), ghost_lc );
    auto ghost_hc_m = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), ghost_hc );

    EXPECT_FLOAT_EQ( own_lc_m(Dim::I), i_func(own_cell_global_space.min(Dim::I)) );
    EXPECT_FLOAT_EQ( own_lc_m(Dim::J), j_func(own_cell_global_space.min(Dim::J)) );
    EXPECT_FLOAT_EQ( own_lc_m(Dim::K), k_func(own_cell_global_space.min(Dim::K)) );

    EXPECT_FLOAT_EQ( own_hc_m(Dim::I), i_func(own_cell_global_space.max(Dim::I)) );
    EXPECT_FLOAT_EQ( own_hc_m(Dim::J), j_func(own_cell_global_space.max(Dim::J)) );
    EXPECT_FLOAT_EQ( own_hc_m(Dim::K), k_func(own_cell_global_space.max(Dim::K)) );

    EXPECT_FLOAT_EQ( ghost_lc_m(Dim::I),
                     i_func(ghost_cell_global_space.min(Dim::I)) );
    EXPECT_FLOAT_EQ( ghost_lc_m(Dim::J),
                     j_func(ghost_cell_global_space.min(Dim::J)) );
    EXPECT_FLOAT_EQ( ghost_lc_m(Dim::K),
                     k_func(ghost_cell_global_space.min(Dim::K)) );

    EXPECT_FLOAT_EQ( ghost_hc_m(Dim::I),
                     i_func(ghost_cell_global_space.max(Dim::I)) );
    EXPECT_FLOAT_EQ( ghost_hc_m(Dim::J),
                     j_func(ghost_cell_global_space.max(Dim::J)) );
    EXPECT_FLOAT_EQ( ghost_hc_m(Dim::K),
                     k_func(ghost_cell_global_space.max(Dim::K)) );

    // Check the cell measures.
    auto cell_measure = createView<double,TEST_DEVICE>(
        "cell_measures", ghost_cell_local_space );
    Kokkos::parallel_for(
        "get_cell_measure",
        createExecutionPolicy( own_cell_local_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            cell_measure(i,j,k) =
                local_mesh.measure( Cell(), i, j, k );
        });
    auto cell_measure_h = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), cell_measure );
    for ( int i = own_cell_global_space.min(Dim::I);
          i < own_cell_global_space.max(Dim::I); ++i )
        for ( int j = own_cell_global_space.min(Dim::J);
              j < own_cell_global_space.max(Dim::J); ++j )
            for ( int k = own_cell_global_space.min(Dim::K);
                  k < own_cell_global_space.max(Dim::K); ++k )
            {
                double measure = ( i_func(i+1) - i_func(i) ) *
                                 ( j_func(j+1) - j_func(j) ) *
                                 ( k_func(k+1) - k_func(k) );
                double value = cell_measure_h(
                    i - ghost_cell_global_space.min(Dim::I),
                    j - ghost_cell_global_space.min(Dim::J),
                    k - ghost_cell_global_space.min(Dim::K) );
                EXPECT_FLOAT_EQ( measure, value );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( mesh, periodic_uniform_test )
{
    std::array<bool,3> is_dim_periodic = {true,true,true};

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int,3> ranks_per_dim = {0,0,0};
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
    std::array<bool,3> is_dim_periodic = {true,true,true};

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int,3> ranks_per_dim = {0,0,0};
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
    std::array<bool,3> is_dim_periodic = {false,false,false};

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int,3> ranks_per_dim = {0,0,0};
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
    std::array<bool,3> is_dim_periodic = {false,false,false};

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int,3> ranks_per_dim = {0,0,0};
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
    std::array<int,3> ranks_per_dim = {0,0,0};
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
