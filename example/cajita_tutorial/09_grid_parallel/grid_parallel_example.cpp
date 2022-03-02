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

//---------------------------------------------------------------------------//
// Grid parallel example.
//---------------------------------------------------------------------------//
void gridParallelExample()
{
    /*
      Just as the core library extends the Kokkos::parallel_for and reduce
      concepts for the AoSoA and neighbor lists, Cajita provides capabilities
      for parallel iteration over structured grids.

      Most use cases will likely use the interface which use the local grids
      directly, but there are additional options for using index spaces,
      including multiple index spaces.

      All data structures (global mesh/grid, local grid/mesh, and array data) is
      created first.
    */
    std::cout << "Cajita Grid Parallel Example\n" << std::endl;

    using exec_space = Kokkos::DefaultHostExecutionSpace;
    using device_type = exec_space::device_type;

    // Let MPI compute the partitioning for this example.
    Cajita::UniformDimPartitioner partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 39, 42, 55 };
    std::array<bool, 3> is_dim_periodic = { false, true, true };
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

    // Create an array layout on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout = Cajita::createArrayLayout(
        global_grid, halo_width, dofs_per_cell, Cajita::Cell() );
    auto local_grid = cell_layout->localGrid();

    // Create an array.
    std::string label( "example_array" );
    auto array = Cajita::createArray<double, device_type>( label, cell_layout );
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
      uses the three indicies of the 3D grid. We set every value on the cells in
      the grid to 1.
    */
    auto array_view = array->view();
    Cajita::grid_parallel_for(
        "local_grid_for", exec_space(), *local_grid, Cajita::Ghost(),
        Cajita::Cell(), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            for ( int l = 0; l < 4; ++l )
                array_view( i, j, k, l ) = 1.0;
        } );

    /*
      We can similarly do grid reductions with the same arguments as
      parallel_for; the only change is the reduction variable in the kernel
      signature and the final return value.
    */
    double sum = 0.0;
    Cajita::grid_parallel_reduce(
        "local_grid_reduce", exec_space(), *local_grid, Cajita::Ghost(),
        Cajita::Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ) {
            for ( int l = 0; l < 4; ++l )
                result += array_view( i, j, k, l );
        },
        sum );

    /*
      Here, the total should match the total grid size (including ghosts) times
      the four field values for each grid point.
    */
    auto ghost_is = local_grid->indexSpace( Cajita::Ghost(), Cajita::Cell(),
                                            Cajita::Local() );
    std::cout << "Total grid sum: " << sum << " (should be "
              << 4 * ghost_is.size() << ")" << std::endl;

    /*
      Grid parallel operations are also possible by directly using an index
      space instead of using the index space from within the local grid. Here we
      subtract 2 from every entity.
    */
    auto own_is = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(),
                                          Cajita::Local() );
    Cajita::grid_parallel_for(
        "index_space_for", exec_space(), own_is,
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            for ( int l = 0; l < 4; ++l )
                array_view( i, j, k, l ) -= 2.0;
        } );

    /*
      Just as before we can do a grid reduction as well.
    */
    double sum_is = 0.0;
    Cajita::grid_parallel_reduce(
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
    std::cout << "Total grid sum: " << sum_is << " (should be "
              << -4 * own_is.size() << ")" << std::endl;

    /*
      One potentially useful extension of directly using index spaces is that
      they can be fused together into a single parallel kernel. Here we iterate
      over both the local and boundary index spaces at the same time, combined
      in a Kokkos::Array and differentiated with the index in the array in the
      kernel.
    */
    auto boundary_is = local_grid->boundaryIndexSpace(
        Cajita::Own(), Cajita::Cell(), 1, 0, 0 );
    Cajita::grid_parallel_for(
        "multi_space_for", exec_space{},
        Kokkos::Array<Cajita::IndexSpace<3>, 2>{ boundary_is, own_is },
        KOKKOS_LAMBDA( const int s, const int i, const int j, const int k ) {
            for ( int l = 0; l < 4; ++l )
            {
                if ( 0 == s )
                    array_view( i, j, k, l ) = 5.0;
                else if ( 1 == s )
                    array_view( i, j, k, l ) = 3.0;
            }
        } );
    /*
      Here we reduce only over each index space separately to check the value.
    */
    double sum_bound = 0.0;
    Cajita::grid_parallel_reduce(
        "boundary_space_reduce", exec_space(), boundary_is,
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ) {
            for ( int l = 0; l < 4; ++l )
                result += array_view( i, j, k, l );
        },
        sum_bound );
    /*
      This total should be the size of the boundary, times the value, times the
      four field values for each grid point.
    */
    std::cout << "Low-X-boundary grid sum: " << sum_bound << " (should be "
              << 4 * 5 * boundary_is.size() << ")" << std::endl;
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
