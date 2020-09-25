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
#include <Cajita_Experimental::BovWriter.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_Halo.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <fstream>
#include <memory>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void writeTest()
{
    // Create the global mesh.
    UniformDimPartitioner partitioner;
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

    // Get the global ijk offsets.
    auto off_i = global_grid->globalOffset( Dim::I );
    auto off_j = global_grid->globalOffset( Dim::J );
    auto off_k = global_grid->globalOffset( Dim::K );

    // Create a scalar cell field and fill it with data.
    auto cell_layout = createArrayLayout( global_grid, 0, 1, Cell() );
    auto cell_field =
        createArray<double, TEST_DEVICE>( "cell_field", cell_layout );
    auto cell_data = cell_field->view();
    Kokkos::parallel_for(
        "fill_cell_field",
        createExecutionPolicy(
            cell_layout->localGrid()->indexSpace( Own(), Cell(), Local() ),
            TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            cell_data( i, j, k, 0 ) = i + off_i + j + off_j + k + off_k;
        } );

    // Create a vector node field and fill it with data.
    auto node_layout = createArrayLayout( global_grid, 0, 3, Node() );
    auto node_field =
        createArray<int, TEST_DEVICE>( "node_field", node_layout );
    auto node_data = node_field->view();
    Kokkos::parallel_for(
        "fill_node_field",
        createExecutionPolicy(
            node_layout->localGrid()->indexSpace( Own(), Node(), Local() ),
            TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            node_data( i, j, k, Dim::I ) = i + off_i;
            node_data( i, j, k, Dim::J ) = j + off_j;
            node_data( i, j, k, Dim::K ) = k + off_k;
        } );

    // Gather the node data.
    auto node_halo = createHalo( *node_field, FullHaloPattern() );
    node_halo->gather( *node_field );

    // Write the fields to a file.
    Experimental::BovWriter::writeTimeStep( 302, 3.43, *cell_field );
    Experimental::BovWriter::writeTimeStep( 1972, 12.457, *node_field );

    // Read the data back in on rank 0 and make sure it is OK.
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( 0 == rank )
    {
        // Open the cell file.
        std::fstream cell_data_file;
        cell_data_file.open( "grid_cell_field_000302.dat",
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
                    cell_data_file.seekg( cell_id * sizeof( double ) );
                    cell_data_file.read( (char *)&cell_value,
                                         sizeof( double ) );
                    EXPECT_EQ( cell_value, i + j + k );
                    ++cell_id;
                }

        // Close the cell file.
        cell_data_file.close();

        // Open the node file.
        std::fstream node_data_file;
        node_data_file.open( "grid_node_field_001972.dat",
                             std::fstream::in | std::fstream::binary );

        // The node file data is ordered KJI
        int node_value;
        int node_id = 0;
        for ( int k = 0; k < global_grid->globalNumEntity( Cell(), Dim::K ) + 1;
              ++k )
            for ( int j = 0;
                  j < global_grid->globalNumEntity( Cell(), Dim::J ) + 1; ++j )
                for ( int i = 0;
                      i < global_grid->globalNumEntity( Cell(), Dim::I ) + 1;
                      ++i )
                {
                    auto ival =
                        ( i == global_grid->globalNumEntity( Cell(), Dim::I ) )
                            ? 0
                            : i;
                    node_data_file.seekg( node_id * sizeof( int ) );
                    node_data_file.read( (char *)&node_value, sizeof( int ) );
                    EXPECT_EQ( node_value, ival );
                    ++node_id;

                    auto jval =
                        ( j == global_grid->globalNumEntity( Cell(), Dim::J ) )
                            ? 0
                            : j;
                    node_data_file.seekg( node_id * sizeof( int ) );
                    node_data_file.read( (char *)&node_value, sizeof( int ) );
                    EXPECT_EQ( node_value, jval );
                    ++node_id;

                    auto kval =
                        ( k == global_grid->globalNumEntity( Cell(), Dim::K ) )
                            ? 0
                            : k;
                    node_data_file.seekg( node_id * sizeof( int ) );
                    node_data_file.read( (char *)&node_value, sizeof( int ) );
                    EXPECT_EQ( node_value, kval );
                    ++node_id;
                }

        // Close the node file.
        node_data_file.close();
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, write_test ) { writeTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
