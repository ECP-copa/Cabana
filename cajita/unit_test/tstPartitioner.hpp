/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita_DynamicPartitioner.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_ManualPartitioner.hpp>
#include <Cajita_Partitioner.hpp>
#include <Kokkos_Core.hpp>

#include <ctime>
#include <gtest/gtest.h>
#include <set>

#include <mpi.h>

using namespace Cajita;

namespace Test
{
template <typename PartitionerType>
void owned_cell_info_test_3d( PartitionerType& partitioner )
{

    // Create Ground Truth Settings
    std::array<int, 3> local_num_cell = { 104, 55, 97 };
    std::array<int, 3> global_num_cell = { 0, 0, 0 };

    auto ranks_per_dim =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, global_num_cell );
    // Extract the periodicity of the boundary as integers.
    std::array<int, 3> periodic_dims = { (int)true, (int)true, (int)true };

    // Generate a communicator with a Cartesian topology.
    MPI_Comm cart_comm;
    int reorder_cart_ranks = 1;
    MPI_Cart_create( MPI_COMM_WORLD, 3, ranks_per_dim.data(),
                     periodic_dims.data(), reorder_cart_ranks, &cart_comm );

    // Get the Cartesian topology index of this rank.
    int linear_rank;
    std::array<int, 3> cart_rank;
    MPI_Comm_rank( cart_comm, &linear_rank );
    MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

    // Set the cells per dimension and the remainde groud truth.
    std::array<int, 3> owned_num_cell_gt;
    std::array<int, 3> global_cell_offset_gt;
    // Creating GT value from local_num_cell:
    //   for each dimension, the N-th rank will have local_num_cell[d]-1 cells
    //   the others will have local_num_cell[d]
    for ( int d = 0; d < 3; ++d )
    {
        // the N-th rank will have local_num_cell[d]-1
        if ( cart_rank[d] == ranks_per_dim[d] - 1 )
            owned_num_cell_gt[d] = local_num_cell[d] - 1;
        else // the others will have local_num_cell[d]
            owned_num_cell_gt[d] = local_num_cell[d];

        global_cell_offset_gt[d] = local_num_cell[d] * cart_rank[d];
        global_num_cell[d] = local_num_cell[d] * ranks_per_dim[d] - 1;
    }

    // Get the cells per dimension and the remainder with partitioner
    std::array<int, 3> owned_num_cell;
    std::array<int, 3> global_cell_offset;

    // The total global cells are avaragely assgined to all ranks
    // The remainder is averagely spreaded in the first several ranks
    partitioner.ownedCellInfo( cart_comm, global_num_cell, owned_num_cell,
                               global_cell_offset );

    for ( std::size_t d = 0; d < 3; ++d )
    {
        EXPECT_EQ( owned_num_cell[d], owned_num_cell_gt[d] );
        EXPECT_EQ( global_cell_offset[d], global_cell_offset_gt[d] );
    }
}

template <typename PartitionerType>
void owned_cell_info_test_2d( PartitionerType& partitioner )
{

    // Create Ground Truth Settings
    std::array<int, 2> local_num_cell = { 69, 203 };
    std::array<int, 2> global_num_cell = { 0, 0 };

    auto ranks_per_dim =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, global_num_cell );
    // Extract the periodicity of the boundary as integers.
    std::array<int, 2> periodic_dims = { (int)true, (int)true };

    // Generate a communicator with a Cartesian topology.
    MPI_Comm cart_comm;
    int reorder_cart_ranks = 1;
    MPI_Cart_create( MPI_COMM_WORLD, 2, ranks_per_dim.data(),
                     periodic_dims.data(), reorder_cart_ranks, &cart_comm );

    // Get the Cartesian topology index of this rank.
    int linear_rank;
    std::array<int, 2> cart_rank;
    MPI_Comm_rank( cart_comm, &linear_rank );
    MPI_Cart_coords( cart_comm, linear_rank, 2, cart_rank.data() );

    // Set the cells per dimension and the remainde groud truth.
    std::array<int, 2> owned_num_cell_gt;
    std::array<int, 2> global_cell_offset_gt;
    // Creating GT value from local_num_cell:
    //   for each dimension, the N-th rank will have local_num_cell[d]-1 cells
    //   the others will have local_num_cell[d]
    for ( int d = 0; d < 2; ++d )
    {
        if ( cart_rank[d] == ranks_per_dim[d] - 1 )
            owned_num_cell_gt[d] = local_num_cell[d] - 1;
        else
            owned_num_cell_gt[d] = local_num_cell[d];

        global_cell_offset_gt[d] = local_num_cell[d] * cart_rank[d];
        global_num_cell[d] = local_num_cell[d] * ranks_per_dim[d] - 1;
    }

    // Get the cells per dimension and the remainder with partitioner
    std::array<int, 2> owned_num_cell;
    std::array<int, 2> global_cell_offset;

    // The total global cells are avaragely assgined to all ranks
    // The remainder is averagely spreaded in the first several ranks
    partitioner.ownedCellInfo( cart_comm, global_num_cell, owned_num_cell,
                               global_cell_offset );

    for ( std::size_t d = 0; d < 2; ++d )
    {
        EXPECT_EQ( owned_num_cell[d], owned_num_cell_gt[d] );
        EXPECT_EQ( global_cell_offset[d], global_cell_offset_gt[d] );
    }
}

void testBlockPartitioner3d()
{
    DimBlockPartitioner<3> partitioner;
    owned_cell_info_test_3d( partitioner );
}

void testManualPartitioner3d()
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    {
        ManualBlockPartitioner<3> partitioner( ranks_per_dim );
        owned_cell_info_test_3d( partitioner );
    }

    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        ManualBlockPartitioner<3> partitioner( ranks_per_dim );
        owned_cell_info_test_3d( partitioner );
    }
    if ( ranks_per_dim[0] != ranks_per_dim[2] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[2] );
        ManualBlockPartitioner<3> partitioner( ranks_per_dim );
        owned_cell_info_test_3d( partitioner );
    }
    if ( ranks_per_dim[1] != ranks_per_dim[2] )
    {
        std::swap( ranks_per_dim[1], ranks_per_dim[2] );
        ManualBlockPartitioner<3> partitioner( ranks_per_dim );
        owned_cell_info_test_3d( partitioner );
    }
}

void testBlockPartitioner2d()
{
    DimBlockPartitioner<2> partitioner;
    owned_cell_info_test_2d( partitioner );
}

void testManualPartitioner2d()
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> ranks_per_dim = { 0, 0 };
    MPI_Dims_create( comm_size, 2, ranks_per_dim.data() );

    {
        ManualBlockPartitioner<2> partitioner( ranks_per_dim );
        owned_cell_info_test_2d( partitioner );
    }

    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        ManualBlockPartitioner<2> partitioner( ranks_per_dim );
        owned_cell_info_test_2d( partitioner );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, partitioner_owned_cell_info_test_3d )
{
    testBlockPartitioner3d();
    testManualPartitioner3d();
}

TEST( TEST_CATEGORY, partitioner_owned_cell_info_test_2d )
{
    testBlockPartitioner2d();
    testManualPartitioner2d();
}
} // namespace Test
