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

#include <mpi.h>

//---------------------------------------------------------------------------//
// Local Mesh example.
//---------------------------------------------------------------------------//
void localMeshExample()
{
    /*
      Just as the local grid describes the subdomain of the global grid, the
      local mesh describes a subdomain of the global mesh. Again, the mesh
      describes the physical geometry and size, while the grid defines only the
      indexing. In contrast to the global data structures, the local mesh is
      created using the local grid. This is because the local indexing is
      required to correctly extract the correct subset of information for the
      local mesh, including ghost information.
    */

    using execution_space = Kokkos::DefaultHostExecutionSpace;
    using memory_space = execution_space::memory_space;

    /*
      As with the previous examples, we first create the global mesh and grid,
      as well as the local grid.
    */
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high_corner = { -0.3, 9.5, 2.3 };
    double cell_size = 0.05;
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, cell_size );

    // Here we partition only in x to simplify the example below.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> input_ranks_per_dim = { comm_size, 1, 1 };
    Cabana::Grid::ManualBlockPartitioner<3> partitioner( input_ranks_per_dim );

    // Create the global grid.
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Get the current rank for printing output.
    int comm_rank = global_grid->blockId();
    if ( comm_rank == 0 )
    {
        std::cout << "Cabana::Grid Local Mesh Example" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    // Create a local grid
    int halo_width = 1;
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );

    /*
      Create the local mesh from the local grid. The device type template
      parameter defines both the Kokkos memory and execution spaces that will be
      able to access the resulting local mesh data.
    */
    auto local_mesh =
        Cabana::Grid::createLocalMesh<memory_space>( *local_grid );

    /*
      Just like the global mesh, the local mesh holds information about the
      physical exents and dimensions of the cells. The local mesh is distinct in
      that it is intended to be regularly used at the application level and is
      designed to be used on the host or in device kernels.
    */
    std::cout << "Low corner local: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.lowCorner( Cabana::Grid::Own(), d ) << " ";
    std::cout << "\nHigh corner local: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.highCorner( Cabana::Grid::Own(), d ) << " ";
    std::cout << "\nExtent local: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.extent( Cabana::Grid::Own(), d ) << " ";
    std::cout << "\nLow corner ghost: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.lowCorner( Cabana::Grid::Ghost(), d ) << " ";
    std::cout << "\nHigh corner ghost: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.highCorner( Cabana::Grid::Ghost(), d ) << " ";
    std::cout << "\nExtent ghost: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.extent( Cabana::Grid::Ghost(), d ) << " ";

    /*
      Note that this information is taken directly from the global grid and mesh
      information.
    */
    std::cout << "\n\nLow corner from global offset ";
    for ( int d = 0; d < 3; ++d )
        std::cout << cell_size * global_grid->globalOffset( d ) << " ";

    /*
      The local mesh is most often useful to get information about individual
      cells within a kernel. Note this is done on the host here, but would most
      often be in a Kokkos parallel kernel. The local mesh is designed such that
      it can be captured by value in Kokkos parallel kernels.

      This information includes the coordinates and the measure (size). The
      measure depends on the entity type used: Nodes return zero, Edges return
      length, Faces return area, and Cells return volume.
    */
    double loc[3];
    int idx[3] = { 27, 17, 15 };
    local_mesh.coordinates( Cabana::Grid::Cell(), idx, loc );
    std::cout << "\nRandom cell coordinates: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << loc[d] << " ";
    std::cout << "\nRandom cell measure: "
              << local_mesh.measure( Cabana::Grid::Cell(), idx ) << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        localMeshExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
