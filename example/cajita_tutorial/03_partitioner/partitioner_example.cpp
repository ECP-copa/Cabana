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

#include <iostream>

//---------------------------------------------------------------------------//
// Types example.
//---------------------------------------------------------------------------//
void partitionerExample()
{
    /*
      The Cajita partitioner splits the global mesh across the available MPI
      ranks in a spatial decomposition scheme. Both manual and near-uniform
      block partitioning are available.
    */
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    if ( comm_rank == 0 )
    {
        std::cout << "Cajita Partitioner Example" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    /*
      The simpler Cajita partitioner, DimBlockPartitioner finds the
      decomposition closest to uniform internally using MPI. It is templated on
      spatial dimension.
    */
    Cajita::DimBlockPartitioner<3> auto_partitioner;

    /*
      Extract the MPI ranks per spatial dimension. The second argument, global
      cells per dimension, is unused in this case.
    */
    std::array<int, 3> ranks_per_dim_3 =
        auto_partitioner.ranksPerDimension( MPI_COMM_WORLD, { 0, 0, 0 } );

    // Print the created decomposition.
    if ( comm_rank == 0 )
    {
        std::cout << "Ranks per dimension (automatic): ";
        for ( int d = 0; d < 3; ++d )
            std::cout << ranks_per_dim_3[d] << " ";
        std::cout << std::endl;
    }

    /*
      Instead, the manual variant, ManualBlockPartitioner, accepts a user
      decomposition (the number of MPI ranks per dimension).

      Here, MPI is still used to set the initial ranks per dimension to ensure
      it matches the available number of ranks.
    */
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> input_ranks_per_dim = { 0, 0 };
    MPI_Dims_create( comm_size, 2, input_ranks_per_dim.data() );

    // Swap the MPI created decomposition because you can.
    std::swap( input_ranks_per_dim[0], input_ranks_per_dim[1] );

    // Create the manual partitioner.
    Cajita::ManualBlockPartitioner<2> manual_partitioner( input_ranks_per_dim );

    /*
      Extract the MPI ranks per spatial dimension. The second argument, global
      cells per dimension, is unused in this case.
    */
    std::array<int, 2> ranks_per_dim_2 =
        manual_partitioner.ranksPerDimension( MPI_COMM_WORLD, { 0, 0 } );

    // Print the created decomposition.
    if ( comm_rank == 0 )
    {
        std::cout << "Ranks per dimension (manual): ";
        for ( int d = 0; d < 2; ++d )
            std::cout << ranks_per_dim_2[d] << " ";
        std::cout << std::endl;
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    Kokkos::ScopeGuard scope_guard( argc, argv );

    partitionerExample();

    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
