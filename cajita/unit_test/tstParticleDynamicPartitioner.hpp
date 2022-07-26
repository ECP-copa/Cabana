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

#include <Cajita_ParticleDynamicPartitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Kokkos_Core.hpp>

#include <ctime>
#include <gtest/gtest.h>
#include <set>

#include <mpi.h>

using namespace Cajita;

namespace Test
{

auto generate_random_particles(
    const std::array<std::vector<int>, 3>& gt_partition,
    const Kokkos::Array<int, 3>& cart_rank, int occupy_par_num_per_rank,
    const std::array<double, 3> global_low_corner, double dx,
    int cell_num_per_tile_dim ) -> Kokkos::View<double* [3], TEST_MEMSPACE>
{
    std::set<std::array<double, 3>> par_set;

    double start[3], size[3];
    for ( int d = 0; d < 3; ++d )
    {
        start[d] =
            ( gt_partition[d][cart_rank[d]] * cell_num_per_tile_dim + 0.5 ) *
                dx +
            global_low_corner[d];

        size[d] =
            ( ( gt_partition[d][cart_rank[d] + 1] * cell_num_per_tile_dim ) -
              ( gt_partition[d][cart_rank[d]] * cell_num_per_tile_dim ) ) *
            dx;
    }
    // insert the corner tiles to the set, to ensure the uniqueness of the
    // ground truth partition
    par_set.insert(
        { start[0] + 0.01 * dx, start[1] + 0.01 * dx, start[2] + 0.01 * dx } );
    par_set.insert( {
        start[0] + size[0] - dx - 0.01 * dx,
        start[1] + size[1] - dx - 0.01 * dx,
        start[2] + size[2] - dx - 0.01 * dx,
    } );

    // insert random tiles to the set
    while ( static_cast<int>( par_set.size() ) < occupy_par_num_per_rank )
    {
        double rand_offset[3];
        for ( int d = 0; d < 3; ++d )
            rand_offset[d] = (double)std::rand() / RAND_MAX;
        par_set.insert( { start[0] + rand_offset[0] * ( size[0] - dx ),
                          start[1] + rand_offset[1] * ( size[1] - dx ),
                          start[2] + rand_offset[2] * ( size[2] - dx ) } );
    }

    // particle_set => particle view (host)
    typedef typename TEST_EXECSPACE::array_layout layout;
    Kokkos::View<double* [3], layout, Kokkos::HostSpace> par_view_host(
        "particle_view_host", par_set.size() );
    int i = 0;
    for ( auto it = par_set.begin(); it != par_set.end(); ++it )
    {
        for ( int d = 0; d < 3; ++d )
            par_view_host( i, d ) = ( *it )[d];
        i++;
    }

    // create tiles view on device
    Kokkos::View<double* [3], TEST_MEMSPACE> par_view =
        Kokkos::create_mirror_view_and_copy( TEST_MEMSPACE(), par_view_host );
    return par_view;
}

/*!
  \brief In this test, the ground truth partition is first randomly chosen, then
  a given number of tiles are regiestered on each rank (the most bottom-left and
  top-right tiles are always registered to ensure the uniqueness of the ground
  truth partition )
  \param occupy_num_per_rank the tile number that will be registered on each MPI
  rank
*/
void random_distribution_automatic_rank( int occupy_num_per_rank )
{
    // define the domain size
    constexpr int size_tile_per_dim = 32;
    constexpr int cell_per_tile_dim = 4;
    constexpr int size_per_dim = size_tile_per_dim * cell_per_tile_dim;
    srand( time( 0 ) );

    // some settings for partitioner
    int max_optimize_iteration = 10;

    std::array<int, 3> global_cells_per_dim = { size_per_dim, size_per_dim,
                                                size_per_dim };

    // partitioner
    DynamicPartitioner<TEST_DEVICE, cell_per_tile_dim> partitioner(
        MPI_COMM_WORLD, global_cells_per_dim, max_optimize_iteration );

    // check the value of some pre-computed constants
    auto cbptd = partitioner.cell_bits_per_tile_dim;
    EXPECT_EQ( cbptd, 2 );

    auto cnptd = partitioner.cell_num_per_tile_dim;
    EXPECT_EQ( cnptd, 4 );

    // ranks per dim test
    auto ranks_per_dim =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, global_cells_per_dim );

    EXPECT_EQ( ranks_per_dim[0] >= 1, true );
    EXPECT_EQ( ranks_per_dim[1] >= 1, true );
    EXPECT_EQ( ranks_per_dim[2] >= 1, true );

    // compute the rank ID
    Kokkos::Array<int, 3> cart_rank;
    std::array<int, 3> periodic_dims = { 0, 0, 0 };
    int reordered_cart_ranks = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create( MPI_COMM_WORLD, 3, ranks_per_dim.data(),
                     periodic_dims.data(), reordered_cart_ranks, &cart_comm );
    int linear_rank;
    MPI_Comm_rank( cart_comm, &linear_rank );
    MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

    // generate random ground truth partition on the root rank
    std::array<std::set<int>, 3> gt_partition_set;
    std::array<std::vector<int>, 3> gt_partition;
    int world_rank, world_size;
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    for ( int d = 0; d < 3; ++d )
    {
        gt_partition[d].resize( ranks_per_dim[d] + 1 );
    }

    if ( world_rank == 0 )
    {
        for ( int d = 0; d < 3; ++d )
        {
            gt_partition_set[d].insert( 0 );
            while ( static_cast<int>( gt_partition_set[d].size() ) <
                    ranks_per_dim[d] )
            {
                int rand_num = std::rand() % size_tile_per_dim;
                gt_partition_set[d].insert( rand_num );
            }
            gt_partition_set[d].insert( size_tile_per_dim );
            int i = 0;
            for ( auto it = gt_partition_set[d].begin();
                  it != gt_partition_set[d].end(); ++it )
            {
                gt_partition[d][i++] = *it;
            }
        }
    }

    // broadcast the ground truth partition to all ranks
    for ( int d = 0; d < 3; ++d )
    {
        MPI_Barrier( MPI_COMM_WORLD );
        MPI_Bcast( gt_partition[d].data(), gt_partition[d].size(), MPI_INT, 0,
                   MPI_COMM_WORLD );
        MPI_Barrier( MPI_COMM_WORLD );
    }

    // init partitions (average partition)
    std::array<std::vector<int>, 3> rec_partitions;
    for ( int d = 0; d < 3; ++d )
    {
        int ele = size_tile_per_dim / ranks_per_dim[d];
        int part = 0;
        for ( int i = 0; i < ranks_per_dim[d]; ++i )
        {
            rec_partitions[d].push_back( part );
            part += ele;
        }
        rec_partitions[d].push_back( size_tile_per_dim );
    }

    partitioner.initializeRecPartition( rec_partitions[0], rec_partitions[1],
                                        rec_partitions[2] );

    // basic settings for domain size and position
    double cell_size = 0.1;
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };

    // randomly generate a fixed number of particles on each MPI rank
    auto particle_view = generate_random_particles(
        gt_partition, cart_rank, occupy_num_per_rank, global_low_corner,
        cell_size, cell_per_tile_dim );
    // compute workload from a particle view and do partition optimization
    auto pws = createParticleDynamicPartitionerWorkloadMeasurer<
        partitioner.cell_num_per_tile_dim, partitioner.num_space_dim,
        TEST_DEVICE>( particle_view, occupy_num_per_rank, global_low_corner,
                      cell_size, MPI_COMM_WORLD );
    partitioner.setLocalWorkload( &pws );
    partitioner.optimizePartition( MPI_COMM_WORLD );

    // check results (should be the same as the gt_partition)
    auto part = partitioner.getCurrentPartition();
    for ( int d = 0; d < 3; ++d )
    {
        for ( int id = 0; id < ranks_per_dim[d] + 1; id++ )
            EXPECT_EQ( part[d][id], gt_partition[d][id] );
    }

    auto imbalance_factor = partitioner.computeImbalanceFactor( cart_comm );
    EXPECT_FLOAT_EQ( imbalance_factor, 1.0f );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( sparse_dim_partitioner, sparse_dim_partitioner_random_par_test )
{
    random_distribution_automatic_rank( 50 );
}
//---------------------------------------------------------------------------//
} // end namespace Test
