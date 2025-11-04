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
#include <Cabana_Grid_BovWriter.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_HaloBase.hpp>
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cmath>
#include <fstream>
#include <memory>

using namespace Cabana::Grid;

namespace Test
{
//---------------------------------------------------------------------------//
void writeTest3d()
{
    // Create the global mesh.
    DimBlockPartitioner<3> partitioner;
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 22, 19, 21 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Device-accessible mesh data.
    Kokkos::Array<int, 3> num_cell_dev = {
        global_num_cell[0], global_num_cell[1], global_num_cell[2] };

    // Get the global ijk offsets.
    auto off_i = global_grid->globalOffset( Dim::I );
    auto off_j = global_grid->globalOffset( Dim::J );
    auto off_k = global_grid->globalOffset( Dim::K );

    // Field data values.
    double pi2 = 8.0 * atan( 1.0 );
    {
        // Create a scalar cell field and fill it with data.
        auto cell_layout = createArrayLayout( global_grid, 0, 1, Cell() );
        auto cell_field =
            createArray<double, TEST_MEMSPACE>( "cell_field_3d", cell_layout );
        auto cell_data = cell_field->view();

        Kokkos::parallel_for(
            "fill_cell_field",
            createExecutionPolicy(
                cell_layout->localGrid()->indexSpace( Own(), Cell(), Local() ),
                TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double xarg = double( off_i + i ) / num_cell_dev[0];
                double yarg = double( off_j + j ) / num_cell_dev[1];
                double zarg = double( off_k + k ) / num_cell_dev[2];
                cell_data( i, j, k, 0 ) =
                    1.0 + fabs( Kokkos::cos( pi2 * xarg ) *
                                Kokkos::cos( pi2 * yarg ) *
                                Kokkos::cos( pi2 * zarg ) );
            } );

        // Create a vector node field and fill it with data.
        auto node_layout = createArrayLayout( global_grid, 0, 3, Node() );
        auto node_field =
            createArray<double, TEST_MEMSPACE>( "node_field_3d", node_layout );
        auto node_data = node_field->view();
        Kokkos::parallel_for(
            "fill_node_field",
            createExecutionPolicy(
                node_layout->localGrid()->indexSpace( Own(), Node(), Local() ),
                TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double xarg = double( off_i + i ) / num_cell_dev[0];
                double yarg = double( off_j + j ) / num_cell_dev[1];
                double zarg = double( off_k + k ) / num_cell_dev[2];
                node_data( i, j, k, Dim::I ) =
                    1.0 + fabs( Kokkos::cos( pi2 * xarg ) );
                node_data( i, j, k, Dim::J ) =
                    1.0 + fabs( Kokkos::cos( pi2 * yarg ) );
                node_data( i, j, k, Dim::K ) =
                    1.0 + fabs( Kokkos::cos( pi2 * zarg ) );
            } );

        // Gather the node data.
        auto node_halo = createHalo( NodeHaloPattern<3>(), 0, *node_field );
        node_halo->gather( TEST_EXECSPACE(), *node_field );

        // Write the fields to a file.
        Experimental::BovWriter::writeTimeStep( "grid_cell_field_3d", 302, 3.43,
                                                *cell_field );
        Experimental::BovWriter::writeTimeStep( "grid_node_field_3d", 1972,
                                                12.457, *node_field );
    }
    // Read the data back in on rank 0 and make sure it is OK.
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( 0 == rank )
    {
        // Open the cell file.
        std::fstream cell_data_file;
        cell_data_file.open( "grid_cell_field_3d_000302.dat",
                             std::fstream::in | std::fstream::binary );

        // The cell file data is ordered KJI
        double cell_value;
        int cell_id = 0;
        for ( int k = 0; k < global_grid->globalNumEntity( Cell(), Dim::K );
              ++k )
            for ( int j = 0; j < global_grid->globalNumEntity( Cell(), Dim::J );
                  ++j )
                for ( int i = 0;
                      i < global_grid->globalNumEntity( Cell(), Dim::I ); ++i )
                {
                    double xarg = double( i ) / global_num_cell[0];
                    double yarg = double( j ) / global_num_cell[1];
                    double zarg = double( k ) / global_num_cell[2];

                    cell_data_file.seekg( cell_id * sizeof( double ) );
                    cell_data_file.read( (char*)&cell_value, sizeof( double ) );

                    EXPECT_FLOAT_EQ( cell_value,
                                     1.0 + fabs( Kokkos::cos( pi2 * xarg ) *
                                                 Kokkos::cos( pi2 * yarg ) *
                                                 Kokkos::cos( pi2 * zarg ) ) );
                    ++cell_id;
                }

        // Close the cell file.
        cell_data_file.close();

        // Open the node file.
        std::fstream node_data_file;
        node_data_file.open( "grid_node_field_3d_001972.dat",
                             std::fstream::in | std::fstream::binary );

        // The node file data is ordered KJI
        double node_value;
        int node_id = 0;
        for ( int k = 0; k < global_grid->globalNumEntity( Cell(), Dim::K ) + 1;
              ++k )
            for ( int j = 0;
                  j < global_grid->globalNumEntity( Cell(), Dim::J ) + 1; ++j )
                for ( int i = 0;
                      i < global_grid->globalNumEntity( Cell(), Dim::I ) + 1;
                      ++i )
                {
                    double xarg = double( i ) / global_num_cell[0];
                    double yarg = double( j ) / global_num_cell[1];
                    double zarg = double( k ) / global_num_cell[2];

                    node_data_file.seekg( node_id * sizeof( double ) );
                    node_data_file.read( (char*)&node_value, sizeof( double ) );
                    EXPECT_FLOAT_EQ( node_value,
                                     1.0 + fabs( Kokkos::cos( pi2 * xarg ) ) );
                    ++node_id;

                    node_data_file.seekg( node_id * sizeof( double ) );
                    node_data_file.read( (char*)&node_value, sizeof( double ) );
                    EXPECT_FLOAT_EQ( node_value,
                                     1.0 + fabs( Kokkos::cos( pi2 * yarg ) ) );
                    ++node_id;

                    node_data_file.seekg( node_id * sizeof( double ) );
                    node_data_file.read( (char*)&node_value, sizeof( double ) );
                    EXPECT_FLOAT_EQ( node_value,
                                     1.0 + fabs( Kokkos::cos( pi2 * zarg ) ) );
                    ++node_id;
                }

        // Close the node file.
        node_data_file.close();
    }
}

//---------------------------------------------------------------------------//
void writeTest2d()
{
    // Create the global mesh.
    DimBlockPartitioner<2> partitioner;
    double cell_size = 0.23;
    std::array<int, 2> global_num_cell = { 22, 19 };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    std::array<bool, 2> is_dim_periodic = { true, true };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Device-accessible mesh data.
    Kokkos::Array<int, 2> num_cell_dev = { global_num_cell[0],
                                           global_num_cell[1] };

    // Get the global ijk offsets.
    auto off_i = global_grid->globalOffset( Dim::I );
    auto off_j = global_grid->globalOffset( Dim::J );

    // Field data values.
    double pi2 = 8.0 * atan( 1.0 );

    {
        // Create a scalar cell field and fill it with data.
        auto cell_layout = createArrayLayout( global_grid, 0, 1, Cell() );
        auto cell_field =
            createArray<double, TEST_MEMSPACE>( "cell_field_2d", cell_layout );
        auto cell_data = cell_field->view();

        Kokkos::parallel_for(
            "fill_cell_field",
            createExecutionPolicy(
                cell_layout->localGrid()->indexSpace( Own(), Cell(), Local() ),
                TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                double xarg = double( off_i + i ) / num_cell_dev[0];
                double yarg = double( off_j + j ) / num_cell_dev[1];
                cell_data( i, j, 0 ) = 1.0 + fabs( Kokkos::cos( pi2 * xarg ) *
                                                   Kokkos::cos( pi2 * yarg ) );
            } );

        // Create a vector node field and fill it with data.
        auto node_layout = createArrayLayout( global_grid, 0, 2, Node() );
        auto node_field =
            createArray<double, TEST_MEMSPACE>( "node_field_2d", node_layout );
        auto node_data = node_field->view();
        Kokkos::parallel_for(
            "fill_node_field",
            createExecutionPolicy(
                node_layout->localGrid()->indexSpace( Own(), Node(), Local() ),
                TEST_EXECSPACE() ),
            KOKKOS_LAMBDA( const int i, const int j ) {
                double xarg = double( off_i + i ) / num_cell_dev[0];
                double yarg = double( off_j + j ) / num_cell_dev[1];
                node_data( i, j, Dim::I ) =
                    1.0 + fabs( Kokkos::cos( pi2 * xarg ) );
                node_data( i, j, Dim::J ) =
                    1.0 + fabs( Kokkos::cos( pi2 * yarg ) );
            } );

        // Gather the node data.
        auto node_halo = createHalo( NodeHaloPattern<2>(), 0, *node_field );
        node_halo->gather( TEST_EXECSPACE(), *node_field );

        // Write the fields to a file.
        Experimental::BovWriter::writeTimeStep( "grid_cell_field_2d", 302, 3.43,
                                                *cell_field );
        Experimental::BovWriter::writeTimeStep( "grid_node_field_2d", 1972,
                                                12.457, *node_field );
    }
    // Read the data back in on rank 0 and make sure it is OK.
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( 0 == rank )
    {
        // Open the cell file.
        std::fstream cell_data_file;
        cell_data_file.open( "grid_cell_field_2d_000302.dat",
                             std::fstream::in | std::fstream::binary );

        // The cell file data is ordered KJI
        double cell_value;
        int cell_id = 0;
        for ( int j = 0; j < global_grid->globalNumEntity( Cell(), Dim::J );
              ++j )
            for ( int i = 0; i < global_grid->globalNumEntity( Cell(), Dim::I );
                  ++i )
            {
                double xarg = double( i ) / global_num_cell[0];
                double yarg = double( j ) / global_num_cell[1];

                cell_data_file.seekg( cell_id * sizeof( double ) );
                cell_data_file.read( (char*)&cell_value, sizeof( double ) );

                EXPECT_FLOAT_EQ( cell_value,
                                 1.0 + fabs( Kokkos::cos( pi2 * xarg ) *
                                             Kokkos::cos( pi2 * yarg ) ) );
                ++cell_id;
            }

        // Close the cell file.
        cell_data_file.close();

        // Open the node file.
        std::fstream node_data_file;
        node_data_file.open( "grid_node_field_2d_001972.dat",
                             std::fstream::in | std::fstream::binary );

        // The node file data is ordered KJI
        double node_value;
        int node_id = 0;
        for ( int j = 0; j < global_grid->globalNumEntity( Cell(), Dim::J ) + 1;
              ++j )
            for ( int i = 0;
                  i < global_grid->globalNumEntity( Cell(), Dim::I ) + 1; ++i )
            {
                double xarg = double( i ) / global_num_cell[0];
                double yarg = double( j ) / global_num_cell[1];

                node_data_file.seekg( node_id * sizeof( double ) );
                node_data_file.read( (char*)&node_value, sizeof( double ) );
                EXPECT_FLOAT_EQ( node_value,
                                 1.0 + fabs( Kokkos::cos( pi2 * xarg ) ) );
                ++node_id;

                node_data_file.seekg( node_id * sizeof( double ) );
                node_data_file.read( (char*)&node_value, sizeof( double ) );
                EXPECT_FLOAT_EQ( node_value,
                                 1.0 + fabs( Kokkos::cos( pi2 * yarg ) ) );
                ++node_id;
            }

        // Close the node file.
        node_data_file.close();
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( Bov, Write3d ) { writeTest3d(); }

TEST( Bov, Write2d ) { writeTest2d(); }

//---------------------------------------------------------------------------//

} // end namespace Test
