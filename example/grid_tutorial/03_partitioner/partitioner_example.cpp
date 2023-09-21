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

#include <iostream>

//---------------------------------------------------------------------------//
// Partitioner example.
//---------------------------------------------------------------------------//
void partitionerExample()
{
    /*
      The Cabana::Grid partitioner splits the global mesh across the available
      MPI ranks in a spatial decomposition scheme. Both manual and near-uniform
      block partitioning are available.
    */
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    if ( comm_rank == 0 )
    {
        std::cout << "Cabana::Grid Partitioner Example" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    /*
      The simpler Cabana::Grid partitioner, DimBlockPartitioner sets a uniform
      decomposition internally using MPI. This partitioning is best only if the
      global mesh is a uniform cube (square) and the particles within it are
      evenly distributed. This partitioning may not produce the best results for
      an elongated domain, for example (due to the communication surface area to
      volume ratio). The DimBlockPartitioner is templated on spatial dimension.
    */
    Cabana::Grid::DimBlockPartitioner<3> dim_block_partitioner;

    /*
      Extract the MPI ranks per spatial dimension. The second argument, global
      cells per dimension, is unused in this case, but useful to make choices
      about partition shapes within more complex strategies.
    */
    std::array<int, 3> ranks_per_dim_block =
        dim_block_partitioner.ranksPerDimension( MPI_COMM_WORLD, { 0, 0, 0 } );

    // Print the created decomposition.
    if ( comm_rank == 0 )
    {
        std::cout << "Ranks per dimension (automatic): ";
        for ( int d = 0; d < 3; ++d )
            std::cout << ranks_per_dim_block[d] << " ";
        std::cout << std::endl;
    }

    /*
      Instead, the manual variant, ManualBlockPartitioner, accepts a user
      decomposition (the number of MPI ranks per dimension). This is useful if
      an efficient partitioning is known a priori.

      Here, we use the total number of MPI ranks and partition only in X.
    */
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> input_ranks_per_dim = { comm_size, 1 };

    // Create the manual partitioner in 2D.
    Cabana::Grid::ManualBlockPartitioner<2> manual_partitioner(
        input_ranks_per_dim );

    /*
      Extract the MPI ranks per spatial dimension. Again, the second argument,
      global cells per dimension, is unused in this case.
    */
    std::array<int, 2> ranks_per_dim_manual =
        manual_partitioner.ranksPerDimension( MPI_COMM_WORLD, { 0, 0 } );

    // Print the created decomposition.
    if ( comm_rank == 0 )
    {
        std::cout << "Ranks per dimension (manual): ";
        for ( int d = 0; d < 2; ++d )
            std::cout << ranks_per_dim_manual[d] << " ";
        std::cout << std::endl;
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        partitionerExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
