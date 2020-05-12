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

#include <Cajita_Array.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <numeric>
#include <vector>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void layoutTest()
{
    // Let MPI compute the partitioning for this test.
    UniformDimPartitioner partitioner;

    // Create the global .
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 101, 85, 99 };
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create an array layout on the nodes.
    int halo_width = 2;
    int dofs_per_node = 4;
    auto node_layout =
        createArrayLayout( global_grid, halo_width, dofs_per_node, Node() );

    // Check the owned index_space.
    auto array_node_owned_space = node_layout->indexSpace( Own(), Local() );
    auto grid_node_owned_space =
        node_layout->localGrid()->indexSpace( Own(), Node(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_node_owned_space.min( d ),
                   grid_node_owned_space.min( d ) );
        EXPECT_EQ( array_node_owned_space.max( d ),
                   grid_node_owned_space.max( d ) );
    }
    EXPECT_EQ( array_node_owned_space.min( 3 ), 0 );
    EXPECT_EQ( array_node_owned_space.max( 3 ), dofs_per_node );

    // Check the ghosted index_space.
    auto array_node_ghosted_space = node_layout->indexSpace( Ghost(), Local() );
    auto grid_node_ghosted_space =
        node_layout->localGrid()->indexSpace( Ghost(), Node(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_node_ghosted_space.min( d ),
                   grid_node_ghosted_space.min( d ) );
        EXPECT_EQ( array_node_ghosted_space.max( d ),
                   grid_node_ghosted_space.max( d ) );
    }
    EXPECT_EQ( array_node_ghosted_space.min( 3 ), 0 );
    EXPECT_EQ( array_node_ghosted_space.max( 3 ), dofs_per_node );

    // Check the shared owned index_space.
    auto array_node_shared_owned_space =
        node_layout->sharedIndexSpace( Own(), -1, 0, 1 );
    auto grid_node_shared_owned_space =
        node_layout->localGrid()->sharedIndexSpace( Own(), Node(), -1, 0, 1 );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_node_shared_owned_space.min( d ),
                   grid_node_shared_owned_space.min( d ) );
        EXPECT_EQ( array_node_shared_owned_space.max( d ),
                   grid_node_shared_owned_space.max( d ) );
    }
    EXPECT_EQ( array_node_shared_owned_space.min( 3 ), 0 );
    EXPECT_EQ( array_node_shared_owned_space.max( 3 ), dofs_per_node );

    // Check the shared ghosted index_space.
    auto array_node_shared_ghosted_space =
        node_layout->sharedIndexSpace( Ghost(), 1, -1, 0 );
    auto grid_node_shared_ghosted_space =
        node_layout->localGrid()->sharedIndexSpace( Ghost(), Node(), 1, -1, 0 );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_node_shared_ghosted_space.min( d ),
                   grid_node_shared_ghosted_space.min( d ) );
        EXPECT_EQ( array_node_shared_ghosted_space.max( d ),
                   grid_node_shared_ghosted_space.max( d ) );
    }
    EXPECT_EQ( array_node_shared_ghosted_space.min( 3 ), 0 );
    EXPECT_EQ( array_node_shared_ghosted_space.max( 3 ), dofs_per_node );

    // Create an array layout on the cells.
    int dofs_per_cell = 4;
    auto cell_layout =
        createArrayLayout( global_grid, halo_width, dofs_per_cell, Cell() );

    // Check the owned index_space.
    auto array_cell_owned_space = cell_layout->indexSpace( Own(), Local() );
    auto grid_cell_owned_space =
        cell_layout->localGrid()->indexSpace( Own(), Cell(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_cell_owned_space.min( d ),
                   grid_cell_owned_space.min( d ) );
        EXPECT_EQ( array_cell_owned_space.max( d ),
                   grid_cell_owned_space.max( d ) );
    }
    EXPECT_EQ( array_cell_owned_space.min( 3 ), 0 );
    EXPECT_EQ( array_cell_owned_space.max( 3 ), dofs_per_cell );

    // Check the ghosted index_space.
    auto array_cell_ghosted_space = cell_layout->indexSpace( Ghost(), Local() );
    auto grid_cell_ghosted_space =
        cell_layout->localGrid()->indexSpace( Ghost(), Cell(), Local() );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_cell_ghosted_space.min( d ),
                   grid_cell_ghosted_space.min( d ) );
        EXPECT_EQ( array_cell_ghosted_space.max( d ),
                   grid_cell_ghosted_space.max( d ) );
    }
    EXPECT_EQ( array_cell_ghosted_space.min( 3 ), 0 );
    EXPECT_EQ( array_cell_ghosted_space.max( 3 ), dofs_per_cell );

    // Check the shared owned index_space.
    auto array_cell_shared_owned_space =
        cell_layout->sharedIndexSpace( Own(), 0, 1, -1 );
    auto grid_cell_shared_owned_space =
        cell_layout->localGrid()->sharedIndexSpace( Own(), Cell(), 0, 1, -1 );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_cell_shared_owned_space.min( d ),
                   grid_cell_shared_owned_space.min( d ) );
        EXPECT_EQ( array_cell_shared_owned_space.max( d ),
                   grid_cell_shared_owned_space.max( d ) );
    }
    EXPECT_EQ( array_cell_shared_owned_space.min( 3 ), 0 );
    EXPECT_EQ( array_cell_shared_owned_space.max( 3 ), dofs_per_cell );

    // Check the shared ghosted index_space.
    auto array_cell_shared_ghosted_space =
        cell_layout->sharedIndexSpace( Ghost(), 1, 1, 1 );
    auto grid_cell_shared_ghosted_space =
        cell_layout->localGrid()->sharedIndexSpace( Ghost(), Cell(), 1, 1, 1 );
    for ( int d = 0; d < 3; ++d )
    {
        EXPECT_EQ( array_cell_shared_ghosted_space.min( d ),
                   grid_cell_shared_ghosted_space.min( d ) );
        EXPECT_EQ( array_cell_shared_ghosted_space.max( d ),
                   grid_cell_shared_ghosted_space.max( d ) );
    }
    EXPECT_EQ( array_cell_shared_ghosted_space.min( 3 ), 0 );
    EXPECT_EQ( array_cell_shared_ghosted_space.max( 3 ), dofs_per_cell );
}

//---------------------------------------------------------------------------//
void arrayTest()
{
    // Let MPI compute the partitioning for this test.
    UniformDimPartitioner partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 101, 85, 99 };
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create an array layout on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout =
        createArrayLayout( global_grid, halo_width, dofs_per_cell, Cell() );

    // Create an array.
    std::string label( "test_array" );
    auto array = createArray<double, TEST_DEVICE>( label, cell_layout );

    // Check the array.
    EXPECT_EQ( label, array->label() );
    auto space = cell_layout->indexSpace( Ghost(), Local() );
    auto view = array->view();
    EXPECT_EQ( label, view.label() );
    EXPECT_EQ( view.size(), space.size() );
    for ( int i = 0; i < 4; ++i )
        EXPECT_EQ( view.extent( i ), space.extent( i ) );
}

//---------------------------------------------------------------------------//
void arrayOpTest()
{
    // Let MPI compute the partitioning for this test.
    UniformDimPartitioner partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 101, 85, 99 };
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create an array layout on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout =
        createArrayLayout( global_grid, halo_width, dofs_per_cell, Cell() );

    // Create an array.
    std::string label( "test_array" );
    auto array = createArray<double, TEST_DEVICE>( label, cell_layout );

    // Assign a value to the entire the array.
    ArrayOp::assign( *array, 2.0, Ghost() );
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          array->view() );
    auto ghosted_space = array->layout()->indexSpace( Ghost(), Local() );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long k = 0; k < ghosted_space.extent( Dim::K ); ++k )
                for ( long l = 0; l < ghosted_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_view( i, j, k, l ), 2.0 );

    // Scale the entire array with a single value.
    ArrayOp::scale( *array, 0.5, Ghost() );
    Kokkos::deep_copy( host_view, array->view() );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long k = 0; k < ghosted_space.extent( Dim::K ); ++k )
                for ( long l = 0; l < ghosted_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_view( i, j, k, l ), 1.0 );

    // Scale each array component by a different value.
    std::vector<double> scales = { 2.3, 1.5, 8.9, -12.1 };
    ArrayOp::scale( *array, scales, Ghost() );
    Kokkos::deep_copy( host_view, array->view() );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long k = 0; k < ghosted_space.extent( Dim::K ); ++k )
                for ( long l = 0; l < ghosted_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_view( i, j, k, l ), scales[l] );

    // Create another array and update.
    auto array_2 = createArray<double, TEST_DEVICE>( label, cell_layout );
    ArrayOp::assign( *array_2, 0.5, Ghost() );
    ArrayOp::update( *array, 3.0, *array_2, 2.0, Ghost() );
    Kokkos::deep_copy( host_view, array->view() );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long k = 0; k < ghosted_space.extent( Dim::K ); ++k )
                for ( long l = 0; l < ghosted_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_view( i, j, k, l ), 3.0 * scales[l] + 1.0 );

    // Check the subarray.
    auto subarray = createSubarray( *array, 2, 4 );
    auto sub_ghosted_space = subarray->layout()->indexSpace( Ghost(), Local() );
    EXPECT_EQ( sub_ghosted_space.rank(), 4 );
    EXPECT_EQ( sub_ghosted_space.extent( Dim::I ),
               ghosted_space.extent( Dim::I ) );
    EXPECT_EQ( sub_ghosted_space.extent( Dim::J ),
               ghosted_space.extent( Dim::J ) );
    EXPECT_EQ( sub_ghosted_space.extent( Dim::K ),
               ghosted_space.extent( Dim::K ) );
    EXPECT_EQ( sub_ghosted_space.extent( 3 ), 2 );
    auto host_subview = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), subarray->view() );
    for ( long i = 0; i < sub_ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < sub_ghosted_space.extent( Dim::J ); ++j )
            for ( long k = 0; k < sub_ghosted_space.extent( Dim::K ); ++k )
                for ( long l = 0; l < sub_ghosted_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_subview( i, j, k, l ),
                               3.0 * scales[l + 2] + 1.0 );

    // Compute the dot product of the two arrays.
    std::vector<double> dots( dofs_per_cell );
    ArrayOp::dot( *array, *array_2, dots );
    int total_num_node = global_grid->globalNumEntity( Node(), Dim::I ) *
                         global_grid->globalNumEntity( Node(), Dim::J ) *
                         global_grid->globalNumEntity( Node(), Dim::K );
    for ( int n = 0; n < dofs_per_cell; ++n )
        EXPECT_FLOAT_EQ( dots[n],
                         ( 3.0 * scales[n] + 1.0 ) * 0.5 * total_num_node );

    // Compute the two-norm of the array components
    std::vector<double> norm_2( dofs_per_cell );
    ArrayOp::norm2( *array, norm_2 );
    for ( int n = 0; n < dofs_per_cell; ++n )
        EXPECT_FLOAT_EQ( norm_2[n],
                         std::sqrt( std::pow( 3.0 * scales[n] + 1.0, 2.0 ) *
                                    total_num_node ) );

    // Compute the one-norm of the array components
    std::vector<double> norm_1( dofs_per_cell );
    ArrayOp::norm1( *array, norm_1 );
    for ( int n = 0; n < dofs_per_cell; ++n )
        EXPECT_FLOAT_EQ( norm_1[n],
                         fabs( 3.0 * scales[n] + 1.0 ) * total_num_node );

    // Compute the infinity-norm of the array components
    std::vector<double> large_vals = { -1939304932.2, 20399994.532,
                                       9098201010.114, -89877402343.99 };
    for ( int n = 0; n < dofs_per_cell; ++n )
        host_view( 4, 4, 4, n ) = large_vals[n];
    Kokkos::deep_copy( array->view(), host_view );
    std::vector<double> norm_inf( dofs_per_cell );
    ArrayOp::normInf( *array, norm_inf );
    for ( int n = 0; n < dofs_per_cell; ++n )
        EXPECT_FLOAT_EQ( norm_inf[n], fabs( large_vals[n] ) );

    // Check the copy.
    ArrayOp::copy( *array, *array_2, Own() );
    Kokkos::deep_copy( host_view, array->view() );
    auto owned_space = array->layout()->indexSpace( Own(), Local() );
    for ( long i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( long j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
            for ( long k = owned_space.min( Dim::K );
                  k < owned_space.max( Dim::K ); ++k )
                for ( long l = 0; l < owned_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_view( i, j, k, l ), 0.5 );

    // Now make a clone and copy.
    auto array_3 = ArrayOp::clone( *array );
    ArrayOp::copy( *array_3, *array, Own() );
    Kokkos::deep_copy( host_view, array_3->view() );
    for ( long i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( long j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
            for ( long k = owned_space.min( Dim::K );
                  k < owned_space.max( Dim::K ); ++k )
                for ( long l = 0; l < owned_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_view( i, j, k, l ), 0.5 );

    // Test the fused clone copy.
    auto array_4 = ArrayOp::cloneCopy( *array, Own() );
    Kokkos::deep_copy( host_view, array_4->view() );
    for ( long i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( long j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
            for ( long k = owned_space.min( Dim::K );
                  k < owned_space.max( Dim::K ); ++k )
                for ( long l = 0; l < owned_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_view( i, j, k, l ), 0.5 );
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

//---------------------------------------------------------------------------//

} // end namespace Test
