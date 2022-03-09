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
// Local Mesh example.
//---------------------------------------------------------------------------//
void localMeshExample()
{
    /*
      Just as the local grid describes the subdomain of the global grid, the
      local mesh describes a subdomain of the global mesh. In contrast to the
      global data structures, the local mesh is created using the local grid.
      This is because the local indexing in required to correctly extract the
      necessary information from the local mesh.
    */

    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    if ( comm_rank == 0 )
    {
        std::cout << "Cajita Local Mesh Example" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    using exec_space = Kokkos::DefaultHostExecutionSpace;
    using device_type = exec_space::device_type;

    /*
      As with the previous examples, we first create the global mesh and grid,
      as well as the local grid.
    */
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high_corner = { -0.3, 9.5, 2.3 };
    double cell_size = 0.05;
    auto global_mesh =
        Cajita::createUniformGlobalMesh( low_corner, high_corner, cell_size );

    // Here we partition only in x to simplify the example below.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> input_ranks_per_dim = { comm_size, 1, 1 };
    Cajita::ManualBlockPartitioner<3> partitioner( input_ranks_per_dim );

    // Create the global grid.
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    // Create a local grid
    int halo_width = 1;
    auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );

    // Create the local mesh from the local grid.
    auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

    /*
      Just like the global mesh, the local mesh holds information about the
      physical exents and dimensions of the cells. The local mesh is distinct in
      that it is intended to be regularly used at the application level and is
      designed to be used on the host or in device kernels.
    */
    std::cout << "Low corner local: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.lowCorner( Cajita::Own(), d ) << " ";
    std::cout << "\nHigh corner local: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.highCorner( Cajita::Own(), d ) << " ";
    std::cout << "\nExtent local: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.extent( Cajita::Own(), d ) << " ";
    std::cout << "\nLow corner ghost: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.lowCorner( Cajita::Ghost(), d ) << " ";
    std::cout << "\nHigh corner ghost: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.highCorner( Cajita::Ghost(), d ) << " ";
    std::cout << "\nExtent ghost: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << local_mesh.extent( Cajita::Ghost(), d ) << " ";

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
      often be in a Kokkos parallel kernel.

      This information includes the coordinates and the measure (size). The
      measure depends on the entity type used: Nodes return zero, Edges return
      length, Faces return area, and Cells return volume.
    */
    double loc[3];
    int idx[3] = { 27, 17, 15 };
    local_mesh.coordinates( Cajita::Cell(), idx, loc );
    std::cout << "\nRandom cell coordinates: ";
    for ( int d = 0; d < 3; ++d )
        std::cout << loc[d] << " ";
    std::cout << "\nRandom cell measure: "
              << local_mesh.measure( Cajita::Cell(), idx ) << std::endl;
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
