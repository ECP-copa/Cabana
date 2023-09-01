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

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <numeric>
#include <vector>

using Cabana::Grid::Cell;
using Cabana::Grid::Dim;
using Cabana::Grid::Edge;
using Cabana::Grid::Face;
using Cabana::Grid::Ghost;
using Cabana::Grid::Global;
using Cabana::Grid::Local;
using Cabana::Grid::Node;
using Cabana::Grid::Own;

namespace Test
{
//---------------------------------------------------------------------------//
void layoutTest()
{
    // Let MPI compute the partitioning for this test.
    Cabana::Grid::DimBlockPartitioner<2> partitioner;

    // Create the global .
    double cell_size = 0.23;
    std::array<int, 2> global_num_cell = { 37, 15 };
    std::array<bool, 2> is_dim_periodic = { true, true };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create an array layout on the nodes.
    int halo_width = 2;
    int dofs_per_node = 4;
    auto node_layout = Cabana::Grid::createArrayLayout( global_grid, halo_width,
                                                        dofs_per_node, Node() );

    // Check the owned index_space.
    auto array_node_owned_space = node_layout->indexSpace( Own(), Local() );
    auto grid_node_owned_space =
        node_layout->localGrid()->indexSpace( Own(), Node(), Local() );
    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_EQ( array_node_owned_space.min( d ),
                   grid_node_owned_space.min( d ) );
        EXPECT_EQ( array_node_owned_space.max( d ),
                   grid_node_owned_space.max( d ) );
    }
    EXPECT_EQ( array_node_owned_space.min( 2 ), 0 );
    EXPECT_EQ( array_node_owned_space.max( 2 ), dofs_per_node );

    // Check the ghosted index_space.
    auto array_node_ghosted_space = node_layout->indexSpace( Ghost(), Local() );
    auto grid_node_ghosted_space =
        node_layout->localGrid()->indexSpace( Ghost(), Node(), Local() );
    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_EQ( array_node_ghosted_space.min( d ),
                   grid_node_ghosted_space.min( d ) );
        EXPECT_EQ( array_node_ghosted_space.max( d ),
                   grid_node_ghosted_space.max( d ) );
    }
    EXPECT_EQ( array_node_ghosted_space.min( 2 ), 0 );
    EXPECT_EQ( array_node_ghosted_space.max( 2 ), dofs_per_node );

    // Check the shared owned index_space.
    auto array_node_shared_owned_space =
        node_layout->sharedIndexSpace( Own(), -1, 0 );
    auto grid_node_shared_owned_space =
        node_layout->localGrid()->sharedIndexSpace( Own(), Node(), -1, 0 );
    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_EQ( array_node_shared_owned_space.min( d ),
                   grid_node_shared_owned_space.min( d ) );
        EXPECT_EQ( array_node_shared_owned_space.max( d ),
                   grid_node_shared_owned_space.max( d ) );
    }
    EXPECT_EQ( array_node_shared_owned_space.min( 2 ), 0 );
    EXPECT_EQ( array_node_shared_owned_space.max( 2 ), dofs_per_node );

    // Check the shared ghosted index_space.
    auto array_node_shared_ghosted_space =
        node_layout->sharedIndexSpace( Ghost(), 1, -1 );
    auto grid_node_shared_ghosted_space =
        node_layout->localGrid()->sharedIndexSpace( Ghost(), Node(), 1, -1 );
    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_EQ( array_node_shared_ghosted_space.min( d ),
                   grid_node_shared_ghosted_space.min( d ) );
        EXPECT_EQ( array_node_shared_ghosted_space.max( d ),
                   grid_node_shared_ghosted_space.max( d ) );
    }
    EXPECT_EQ( array_node_shared_ghosted_space.min( 2 ), 0 );
    EXPECT_EQ( array_node_shared_ghosted_space.max( 2 ), dofs_per_node );

    // Create an array layout on the cells.
    int dofs_per_cell = 4;
    auto cell_layout = Cabana::Grid::createArrayLayout( global_grid, halo_width,
                                                        dofs_per_cell, Cell() );

    // Check the owned index_space.
    auto array_cell_owned_space = cell_layout->indexSpace( Own(), Local() );
    auto grid_cell_owned_space =
        cell_layout->localGrid()->indexSpace( Own(), Cell(), Local() );
    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_EQ( array_cell_owned_space.min( d ),
                   grid_cell_owned_space.min( d ) );
        EXPECT_EQ( array_cell_owned_space.max( d ),
                   grid_cell_owned_space.max( d ) );
    }
    EXPECT_EQ( array_cell_owned_space.min( 2 ), 0 );
    EXPECT_EQ( array_cell_owned_space.max( 2 ), dofs_per_cell );

    // Check the ghosted index_space.
    auto array_cell_ghosted_space = cell_layout->indexSpace( Ghost(), Local() );
    auto grid_cell_ghosted_space =
        cell_layout->localGrid()->indexSpace( Ghost(), Cell(), Local() );
    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_EQ( array_cell_ghosted_space.min( d ),
                   grid_cell_ghosted_space.min( d ) );
        EXPECT_EQ( array_cell_ghosted_space.max( d ),
                   grid_cell_ghosted_space.max( d ) );
    }
    EXPECT_EQ( array_cell_ghosted_space.min( 2 ), 0 );
    EXPECT_EQ( array_cell_ghosted_space.max( 2 ), dofs_per_cell );

    // Check the shared owned index_space.
    auto array_cell_shared_owned_space =
        cell_layout->sharedIndexSpace( Own(), 0, 1 );
    auto grid_cell_shared_owned_space =
        cell_layout->localGrid()->sharedIndexSpace( Own(), Cell(), 0, 1 );
    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_EQ( array_cell_shared_owned_space.min( d ),
                   grid_cell_shared_owned_space.min( d ) );
        EXPECT_EQ( array_cell_shared_owned_space.max( d ),
                   grid_cell_shared_owned_space.max( d ) );
    }
    EXPECT_EQ( array_cell_shared_owned_space.min( 2 ), 0 );
    EXPECT_EQ( array_cell_shared_owned_space.max( 2 ), dofs_per_cell );

    // Check the shared ghosted index_space.
    auto array_cell_shared_ghosted_space =
        cell_layout->sharedIndexSpace( Ghost(), 1, 1 );
    auto grid_cell_shared_ghosted_space =
        cell_layout->localGrid()->sharedIndexSpace( Ghost(), Cell(), 1, 1 );
    for ( int d = 0; d < 2; ++d )
    {
        EXPECT_EQ( array_cell_shared_ghosted_space.min( d ),
                   grid_cell_shared_ghosted_space.min( d ) );
        EXPECT_EQ( array_cell_shared_ghosted_space.max( d ),
                   grid_cell_shared_ghosted_space.max( d ) );
    }
    EXPECT_EQ( array_cell_shared_ghosted_space.min( 2 ), 0 );
    EXPECT_EQ( array_cell_shared_ghosted_space.max( 2 ), dofs_per_cell );
}

//---------------------------------------------------------------------------//
void arrayTest()
{
    // Let MPI compute the partitioning for this test.
    Cabana::Grid::DimBlockPartitioner<2> partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 2> global_num_cell = { 37, 15 };
    std::array<bool, 2> is_dim_periodic = { true, true };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create an array layout on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout = Cabana::Grid::createArrayLayout( global_grid, halo_width,
                                                        dofs_per_cell, Cell() );

    // Create an array.
    std::string label( "test_array" );
    auto array =
        Cabana::Grid::createArray<double, TEST_MEMSPACE>( label, cell_layout );

    // Check the array.
    EXPECT_EQ( label, array->label() );
    auto space = cell_layout->indexSpace( Ghost(), Local() );
    auto view = array->view();
    EXPECT_EQ( label, view.label() );
    EXPECT_EQ( view.size(), space.size() );
    for ( int i = 0; i < 3; ++i )
        EXPECT_EQ( view.extent( i ), space.extent( i ) );
}

//---------------------------------------------------------------------------//
void arrayOpTest()
{
    // Let MPI compute the partitioning for this test.
    Cabana::Grid::DimBlockPartitioner<2> partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 2> global_num_cell = { 37, 15 };
    std::array<bool, 2> is_dim_periodic = { true, true };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create an array layout on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout = Cabana::Grid::createArrayLayout( global_grid, halo_width,
                                                        dofs_per_cell, Cell() );

    // Create an array.
    std::string label( "test_array" );
    auto array =
        Cabana::Grid::createArray<double, TEST_MEMSPACE>( label, cell_layout );

    // Assign a value to the entire the array.
    Cabana::Grid::ArrayOp::assign( *array, 2.0, Ghost() );
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          array->view() );
    auto ghosted_space = array->layout()->indexSpace( Ghost(), Local() );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long l = 0; l < ghosted_space.extent( 2 ); ++l )
                EXPECT_DOUBLE_EQ( host_view( i, j, l ), 2.0 );

    // Scale the entire array with a single value.
    Cabana::Grid::ArrayOp::scale( *array, 0.5, Ghost() );
    Kokkos::deep_copy( host_view, array->view() );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long l = 0; l < ghosted_space.extent( 2 ); ++l )
                EXPECT_DOUBLE_EQ( host_view( i, j, l ), 1.0 );

    // Scale each array component by a different value.
    std::vector<double> scales = { 2.3, 1.5, 8.9, -12.1 };
    Cabana::Grid::ArrayOp::scale( *array, scales, Ghost() );
    Kokkos::deep_copy( host_view, array->view() );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long l = 0; l < ghosted_space.extent( 2 ); ++l )
                EXPECT_DOUBLE_EQ( host_view( i, j, l ), scales[l] );

    // Create another array and update.
    auto array_2 =
        Cabana::Grid::createArray<double, TEST_MEMSPACE>( label, cell_layout );
    Cabana::Grid::ArrayOp::assign( *array_2, 0.5, Ghost() );
    Cabana::Grid::ArrayOp::update( *array, 3.0, *array_2, 2.0, Ghost() );
    Kokkos::deep_copy( host_view, array->view() );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long l = 0; l < ghosted_space.extent( 2 ); ++l )
                EXPECT_DOUBLE_EQ( host_view( i, j, l ), 3.0 * scales[l] + 1.0 );

    // Check the subarray.
    auto subarray = createSubarray( *array, 2, 4 );
    auto sub_ghosted_space = subarray->layout()->indexSpace( Ghost(), Local() );
    EXPECT_EQ( sub_ghosted_space.rank(), 3 );
    EXPECT_EQ( sub_ghosted_space.extent( Dim::I ),
               ghosted_space.extent( Dim::I ) );
    EXPECT_EQ( sub_ghosted_space.extent( Dim::J ),
               ghosted_space.extent( Dim::J ) );
    auto host_subview = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), subarray->view() );
    for ( long i = 0; i < sub_ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < sub_ghosted_space.extent( Dim::J ); ++j )
            for ( long l = 0; l < sub_ghosted_space.extent( 2 ); ++l )
                EXPECT_DOUBLE_EQ( host_subview( i, j, l ),
                                  3.0 * scales[l + 2] + 1.0 );

#ifndef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
    // Compute the dot product of the two arrays.
    std::vector<double> dots( dofs_per_cell );
    Cabana::Grid::ArrayOp::dot( *array, *array_2, dots );
    int total_num_cell = global_grid->globalNumEntity( Cell(), Dim::I ) *
                         global_grid->globalNumEntity( Cell(), Dim::J );
    for ( int n = 0; n < dofs_per_cell; ++n )
        EXPECT_FLOAT_EQ( dots[n],
                         ( 3.0 * scales[n] + 1.0 ) * 0.5 * total_num_cell );

    // Compute the two-norm of the array components
    std::vector<double> norm_2( dofs_per_cell );
    Cabana::Grid::ArrayOp::norm2( *array, norm_2 );
    for ( int n = 0; n < dofs_per_cell; ++n )
        EXPECT_FLOAT_EQ( norm_2[n],
                         std::sqrt( std::pow( 3.0 * scales[n] + 1.0, 2.0 ) *
                                    total_num_cell ) );

    // Compute the one-norm of the array components
    std::vector<double> norm_1( dofs_per_cell );
    Cabana::Grid::ArrayOp::norm1( *array, norm_1 );
    for ( int n = 0; n < dofs_per_cell; ++n )
        EXPECT_FLOAT_EQ( norm_1[n],
                         fabs( 3.0 * scales[n] + 1.0 ) * total_num_cell );

    // Compute the infinity-norm of the array components
    std::vector<double> large_vals = { -1939304932.2, 20399994.532,
                                       9098201010.114, -89877402343.99 };
    for ( int n = 0; n < dofs_per_cell; ++n )
        host_view( 4, 4, n ) = large_vals[n];
    Kokkos::deep_copy( array->view(), host_view );
    std::vector<double> norm_inf( dofs_per_cell );
    Cabana::Grid::ArrayOp::normInf( *array, norm_inf );
    for ( int n = 0; n < dofs_per_cell; ++n )
        EXPECT_FLOAT_EQ( norm_inf[n], fabs( large_vals[n] ) );
#endif

    // Check the copy.
    Cabana::Grid::ArrayOp::copy( *array, *array_2, Own() );
    Kokkos::deep_copy( host_view, array->view() );
    auto owned_space = array->layout()->indexSpace( Own(), Local() );
    for ( long i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( long j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
            for ( long l = 0; l < owned_space.extent( 2 ); ++l )
                EXPECT_DOUBLE_EQ( host_view( i, j, l ), 0.5 );

    // Now make a clone and copy.
    auto array_3 = Cabana::Grid::ArrayOp::clone( *array );
    Cabana::Grid::ArrayOp::copy( *array_3, *array, Own() );
    Kokkos::deep_copy( host_view, array_3->view() );
    for ( long i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( long j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
            for ( long l = 0; l < owned_space.extent( 2 ); ++l )
                EXPECT_DOUBLE_EQ( host_view( i, j, l ), 0.5 );

    // Test the fused clone copy.
    auto array_4 = Cabana::Grid::ArrayOp::cloneCopy( *array, Own() );
    Kokkos::deep_copy( host_view, array_4->view() );
    for ( long i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( long j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
            for ( long l = 0; l < owned_space.extent( 2 ); ++l )
                EXPECT_DOUBLE_EQ( host_view( i, j, l ), 0.5 );

    // Do a 3 vector update.
    Cabana::Grid::ArrayOp::assign( *array, 1.0, Ghost() );
    Cabana::Grid::ArrayOp::scale( *array, scales, Ghost() );
    Cabana::Grid::ArrayOp::assign( *array_2, 0.5, Ghost() );
    Cabana::Grid::ArrayOp::scale( *array_2, scales, Ghost() );
    Cabana::Grid::ArrayOp::assign( *array_3, 1.5, Ghost() );
    Cabana::Grid::ArrayOp::scale( *array_3, scales, Ghost() );
    Cabana::Grid::ArrayOp::update( *array, 3.0, *array_2, 2.0, *array_3, 4.0,
                                   Ghost() );
    Kokkos::deep_copy( host_view, array->view() );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long l = 0; l < ghosted_space.extent( 2 ); ++l )
                EXPECT_DOUBLE_EQ( host_view( i, j, l ),
                                  ( 3.0 + 1.0 + 6.0 ) * scales[l] );
}

//---------------------------------------------------------------------------//
template <class DecompositionType, class EntityType>
void arrayBoundaryTest()
{
    // Let MPI compute the partitioning for this test.
    Cabana::Grid::DimBlockPartitioner<2> partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 2> global_num_cell = { 37, 15 };
    std::array<bool, 2> is_dim_periodic = { false, false };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create an array layout on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto node_layout = Cabana::Grid::createArrayLayout(
        global_grid, halo_width, dofs_per_cell, EntityType() );

    // Create an array.
    std::string label( "test_array" );
    auto array =
        Cabana::Grid::createArray<double, TEST_MEMSPACE>( label, node_layout );

    // Assign a value to the entire the array.
    // This test is simply to ensure the boundary index space is valid for the
    // array.
    Cabana::Grid::ArrayOp::assign( *array, 2.0, Ghost() );
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          array->view() );
    for ( int x = -1; x <= 1; x++ )
    {
        for ( int y = -1; y <= 1; y++ )
        {
            std::array<int, 2> dir = { x, y };
            auto boundary_space =
                array->layout()->localGrid()->boundaryIndexSpace(
                    Ghost(), EntityType(), dir );
            for ( long i = boundary_space.min( Dim::I );
                  i < boundary_space.max( Dim::I ); ++i )
                for ( long j = boundary_space.min( Dim::J );
                      j < boundary_space.max( Dim::J ); ++j )
                    for ( long l = 0; l < dofs_per_cell; ++l )
                        EXPECT_DOUBLE_EQ( host_view( i, j, l ), 2.0 );
        }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( array, array_test )
{
    layoutTest();
    arrayTest();
    arrayOpTest();
}
TEST( array, array_boundary_test )
{
    arrayBoundaryTest<Own, Cell>();
    arrayBoundaryTest<Own, Node>();
    arrayBoundaryTest<Own, Face<Dim::I>>();
    arrayBoundaryTest<Own, Face<Dim::J>>();

    arrayBoundaryTest<Ghost, Cell>();
    arrayBoundaryTest<Ghost, Node>();
    arrayBoundaryTest<Ghost, Face<Dim::I>>();
    arrayBoundaryTest<Ghost, Face<Dim::J>>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
