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

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

//---------------------------------------------------------------------------//
// Grid parallel example.
//---------------------------------------------------------------------------//
void gridParallelExample()
{
    /*
      Just as the core library extends the Kokkos::parallel_for and reduce
      concepts for the AoSoA and neighbor lists, Cabana::Grid provides
      capabilities for parallel iteration over structured grids.

      Most use cases will likely use the interface which use the local grids
      directly, but there are additional options for using index spaces,
      including multiple index spaces.

      All data structures (global mesh/grid, local grid/mesh, and array data)
      are created first.
    */
    std::cout << "Cabana::Grid Grid Parallel Example\n" << std::endl;

    using exec_space = Kokkos::DefaultHostExecutionSpace;
    using device_type = exec_space::device_type;

    // Let MPI compute the partitioning for this example.
    Cabana::Grid::DimBlockPartitioner<3> partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 39, 42, 55 };
    std::array<bool, 3> is_dim_periodic = { false, true, true };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create an array layout on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout = Cabana::Grid::createArrayLayout(
        global_grid, halo_width, dofs_per_cell, Cabana::Grid::Cell() );
    auto local_grid = cell_layout->localGrid();

    // Create an array.
    std::string label( "example_array" );
    auto array =
        Cabana::Grid::createArray<double, device_type>( label, cell_layout );
    //-----------------------------------------------------------------------//

    /*
      The grid parallel functions first take a kernel label, then an execution
      space rather than a direct Kokkos range policy. Internally, a
      multidimensional range policy (Kokkos::MDRangePolicy) is created with the
      execution space given. Next, the Own or Ghost tag is passed to define
      whether the kernel should include ghosted entities or not (remember that
      specifying Ghost includes both owned and ghosted entities). The mesh
      entity is passed next and finally, the parallel kernel.

      Here we use a lambda function for simplicity, but a functor with or
      without tag arguments are also options. Note that the kernel signature
      uses the three indices of the 3D grid. We set every value on the cells in
      the grid to 1.
    */
    auto array_view = array->view();
    Cabana::Grid::grid_parallel_for(
        "local_grid_for", exec_space(), *local_grid, Cabana::Grid::Ghost(),
        Cabana::Grid::Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            for ( int l = 0; l < 4; ++l )
                array_view( i, j, k, l ) = 1.0;
        } );

    /*
      We can similarly do grid reductions with the same arguments as
      parallel_for; the only change is the reduction variable in the kernel
      signature and the final return value.
    */
    double sum = 0.0;
    Cabana::Grid::grid_parallel_reduce(
        "local_grid_reduce", exec_space(), *local_grid, Cabana::Grid::Ghost(),
        Cabana::Grid::Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ) {
            for ( int l = 0; l < 4; ++l )
                result += array_view( i, j, k, l );
        },
        sum );

    /*
      Here, the total should match the total grid size (including ghosts) times
      the four field values for each grid point.
    */
    auto ghost_is = local_grid->indexSpace(
        Cabana::Grid::Ghost(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
    std::cout << "Local grid sum: " << sum << " (should be "
              << 4 * ghost_is.size() << ")\n"
              << std::endl;
    //-----------------------------------------------------------------------//

    /*
      Grid parallel operations are also possible by directly using an index
      space instead of using the index space from within the local grid. Here we
      subtract 2 from every entity.
    */
    auto own_is = local_grid->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
    Cabana::Grid::grid_parallel_for(
        "index_space_for", exec_space(), own_is,
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            for ( int l = 0; l < 4; ++l )
                array_view( i, j, k, l ) -= 2.0;
        } );

    /*
      Just as before with the local grid we can do a grid reduction with the
      index space as well.
    */
    double sum_is = 0.0;
    Cabana::Grid::grid_parallel_reduce(
        "index_space_reduce", exec_space(), own_is,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ) {
            for ( int l = 0; l < 4; ++l )
                result += array_view( i, j, k, l );
        },
        sum_is );
    /*
      Now the total should be the negative of the total owned grid size, again
      times the four field values for each grid point.
    */
    std::cout << "Index space grid sum: " << sum_is << " (should be "
              << -4 * own_is.size() << ")\n"
              << std::endl;
    //-----------------------------------------------------------------------//

    /*
      An additional option in using the index space interface is using the index
      space returned by the array layout. Note that the functor signature
      changes here since the field dimension is included in the index space
      (rather than looping in serial over the last dimension as in the previous
      cases).

      For this kernel we divide every entity by the total number of cells.
    */
    auto own_array_is =
        cell_layout->indexSpace( Cabana::Grid::Own(), Cabana::Grid::Local() );
    Cabana::Grid::grid_parallel_for(
        "index_space_for", exec_space(), own_array_is,
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ) {
            array_view( i, j, k, l ) /= sum_is;
        } );

    /*
      Do another grid reduction to check the result.
    */
    double sum_layout_is = 0.0;
    Cabana::Grid::grid_parallel_reduce(
        "index_space_reduce", exec_space(), own_array_is,
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l,
                       double& result ) { result += array_view( i, j, k, l ); },
        sum_layout_is );
    /*
      Now the total should be 1.
    */
    std::cout << "Array layout index space grid sum: " << sum_layout_is
              << " (should be 1)\n"
              << std::endl;
    //-----------------------------------------------------------------------//

    /*
      One potentially useful extension of directly using index spaces is that
      they can be fused together into a single parallel kernel. Here we iterate
      over both the local and boundary index spaces at the same time, combined
      in a Kokkos::Array and differentiated with the index in the array in the
      kernel.
    */
    auto boundary_is = local_grid->boundaryIndexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), 1, 0, 0 );
    Cabana::Grid::grid_parallel_for(
        "multi_space_for", exec_space{},
        Kokkos::Array<Cabana::Grid::IndexSpace<3>, 2>{ own_is, boundary_is },
        // The first index in the functor signature is which index space is
        // being used (from the array in the previous argument).
        KOKKOS_LAMBDA( const int s, const int i, const int j, const int k ) {
            for ( int l = 0; l < 4; ++l )
            {
                // Now we can differentiate updates in kernel based on which
                // index space we're using. We set the value for the owned space
                // and increment (afterwards) only on the boundary.
                if ( 0 == s )
                    array_view( i, j, k, l ) = 0.5;
                else if ( 1 == s )
                    array_view( i, j, k, l ) += 1.0;
            }
        } );
    /*
      Here we reduce only over each index space separately to check the value.
    */
    double sum_bound = 0.0;
    Cabana::Grid::grid_parallel_reduce(
        "boundary_space_reduce", exec_space(), own_is,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ) {
            for ( int l = 0; l < 4; ++l )
                result += array_view( i, j, k, l );
        },
        sum_bound );
    /*
      This total should be the size of the owned space, times the value, times
      the four field values for each grid point plus the boundary space size.
    */
    std::cout << "Multiple index space grid sum: " << sum_bound
              << " (should be "
              << 4 * 0.5 * own_is.size() + 4 * boundary_is.size() << ")\n"
              << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // MPI only needed to create the grid/mesh. Not intended to be run with
    // multiple ranks.
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        gridParallelExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
