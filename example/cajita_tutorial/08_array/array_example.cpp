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

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

#include <mpi.h>

//---------------------------------------------------------------------------//
// Array example.
//---------------------------------------------------------------------------//
void arrayExample()
{
    /*
      The array and corresponding array layout allocates and stores the physical
      quantities on the grid - prior to creating an Array, only the size, shape,
      and indexing of the grid is defined.
    */

    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    if ( comm_rank == 0 )
    {
        std::cout << "Cajita Array Example" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    using exec_space = Kokkos::DefaultHostExecutionSpace;
    using device_type = exec_space::device_type;

    /*
      The array layout is an intermediate class between the local grid and array
      - it does not store the mesh field data. Instead it holds the layout of
      the field arrays.

      As with the previous examples, we define everything necessary to create
      the local grid.
    */

    // Let MPI compute the partitioning for this test.
    Cajita::DimBlockPartitioner<3> partitioner;

    // Create the global mesh.
    double cell_size = 0.50;
    std::array<int, 3> global_num_cell = { 37, 15, 20 };
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    /*
      An array layout includes the local grid size information together with the
      dimensionality of the field data, the degrees of freedom on the mesh.
    */
    int halo_width = 2;
    int dofs_per_node = 4;
    auto node_layout = Cajita::createArrayLayout(
        global_grid, halo_width, dofs_per_node, Cajita::Node() );

    /*
      Similar to the local grid, the array layout holds index spaces for
      parallel iteration over mesh fields. These again include options for mesh
      entities and owned/owned+ghosted.
    */
    auto array_node_owned_space =
        node_layout->indexSpace( Cajita::Own(), Cajita::Local() );
    std::cout << "Array layout (Own, Local) \nMin: ";
    for ( int d = 0; d < dofs_per_node; ++d )
        std::cout << array_node_owned_space.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < dofs_per_node; ++d )
        std::cout << array_node_owned_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    auto array_node_ghosted_space =
        node_layout->indexSpace( Cajita::Ghost(), Cajita::Local() );
    std::cout << "Array layout (Ghost, Local) \nMin: ";
    for ( int d = 0; d < dofs_per_node; ++d )
        std::cout << array_node_ghosted_space.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < dofs_per_node; ++d )
        std::cout << array_node_ghosted_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    auto array_node_shared_owned_space =
        node_layout->sharedIndexSpace( Cajita::Own(), -1, 0, 1 );
    std::cout << "Array layout (Shared, Own, -1,0,1) \nMin: ";
    for ( int d = 0; d < dofs_per_node; ++d )
        std::cout << array_node_shared_owned_space.min( d ) << " ";
    std::cout << "\nMax: ";
    for ( int d = 0; d < dofs_per_node; ++d )
        std::cout << array_node_shared_owned_space.max( d ) << " ";
    std::cout << "\n" << std::endl;

    /*
      Next, we create an array from the array layout. This directly stores the
      Kokkos View with field data that can be extracted for parallel
      computation.
    */
    std::string label( "example_array" );
    auto array = Cajita::createArray<double, device_type>( label, node_layout );

    auto view = array->view();
    std::cout << "Array total size: " << view.size() << std::endl;
    std::cout << "Array extents: ";
    for ( int i = 0; i < 4; ++i )
        std::cout << view.extent( i ) << " ";
    std::cout << "\n" << std::endl;

    /*
      Many array-based operations are available for convenience:
       - assign a value to every field on the mesh
       - scale the entire array
       - update
       - dot product between two arrays
       - two-norm
       - one-norm
       - infinity-norm

      For each operation we print the first value in the array - each value is
      updated.
    */
    Cajita::ArrayOp::assign( *array, 2.0, Cajita::Ghost() );

    // Scale the entire array with a single value.
    Cajita::ArrayOp::scale( *array, 0.5, Cajita::Ghost() );

    // Compute the dot product of the two arrays.
    std::vector<double> dots( dofs_per_node );
    auto array_2 =
        Cajita::createArray<double, device_type>( label, node_layout );
    Cajita::ArrayOp::assign( *array_2, 0.5, Cajita::Ghost() );
    Cajita::ArrayOp::update( *array, 3.0, *array_2, 2.0, Cajita::Ghost() );
    Cajita::ArrayOp::dot( *array, *array_2, dots );
    std::cout << "Array dot product: ";
    std::cout << dots[0] << std::endl;

    // Compute the two-norm of the array components
    std::cout << "Array two-norm: ";
    std::vector<double> norm_2( dofs_per_node );
    Cajita::ArrayOp::norm2( *array, norm_2 );
    std::cout << norm_2[0] << std::endl;

    // Compute the one-norm of the array components
    std::cout << "Array one-norm: ";
    std::vector<double> norm_1( dofs_per_node );
    Cajita::ArrayOp::norm1( *array, norm_1 );
    std::cout << norm_1[0] << std::endl;

    // Compute the infinity-norm of the array components
    std::cout << "Array infinity-norm: ";
    view = array->view();
    std::vector<double> large_vals = { -1939304932.2, 20399994.532,
                                       9098201010.114, -89877402343.99 };
    for ( int n = 0; n < dofs_per_node; ++n )
        view( 4, 4, 4, n ) = large_vals[n];
    std::vector<double> norm_inf( dofs_per_node );
    Cajita::ArrayOp::normInf( *array, norm_inf );
    std::cout << norm_inf[0] << std::endl;

    // Check the copy.
    Cajita::ArrayOp::copy( *array, *array_2, Cajita::Own() );

    // Now make a clone and copy.
    auto array_3 = Cajita::ArrayOp::clone( *array );
    Cajita::ArrayOp::copy( *array_3, *array, Cajita::Own() );

    // Test the fused clone copy.
    auto array_4 = Cajita::ArrayOp::cloneCopy( *array, Cajita::Own() );

    // Do a 3 vector update.
    std::vector<double> scales = { 2.3, 1.5, 8.9, -12.1 };
    Cajita::ArrayOp::assign( *array, 1.0, Cajita::Ghost() );
    Cajita::ArrayOp::scale( *array, scales, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *array_2, 0.5, Cajita::Ghost() );
    Cajita::ArrayOp::scale( *array_2, scales, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *array_3, 1.5, Cajita::Ghost() );
    Cajita::ArrayOp::scale( *array_3, scales, Cajita::Ghost() );
    Cajita::ArrayOp::update( *array, 3.0, *array_2, 2.0, *array_3, 4.0,
                             Cajita::Ghost() );

    /*
      It is also possible to create sub-arrays of existing arrays.
    */
    auto subarray = Cajita::createSubarray( *array, 2, 4 );
    auto sub_ghosted_space =
        subarray->layout()->indexSpace( Cajita::Ghost(), Cajita::Local() );
    std::cout << "\nSub-array index space size: ";
    std::cout << sub_ghosted_space.size() << std::endl;
    std::cout << "Sub-array index space extents: ";
    for ( int n = 0; n < dofs_per_node; ++n )
        std::cout << sub_ghosted_space.extent( n ) << " ";
    std::cout << "\n" << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        arrayExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
