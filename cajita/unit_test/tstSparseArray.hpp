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

#include <Cajita_SparseArray.hpp>
#include <Cajita_SparseDimPartitioner.hpp>
#include <Cajita_SparseLocalGrid.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

using namespace Cajita;
using namespace Cajita::Experimental;

namespace Test
{

// generate a random partition, to mimic a random simulation status
std::array<std::vector<int>, 3>
generate_random_partition( std::array<int, 3> ranks_per_dim,
                           int size_tile_per_dim )
{
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

    return gt_partition;
}

// return random generated particles and occupied tile numbers (last param)
template <typename scalar_type>
auto generate_random_particles(
    const std::array<std::vector<int>, 3>& gt_partition,
    const Kokkos::Array<int, 3>& cart_rank, const int size_tile_per_dim,
    const int particle_number, std::array<int, 3>& occupy_tile_num )
    -> Kokkos::View<scalar_type* [3], TEST_MEMSPACE>
{
    // register valid tiles in each MPI rank
    // compute the sub-domain size (divided by the ground truth partition)
    // const int area_size = size_tile_per_dim * size_tile_per_dim;
    // occupy_tile_num_per_rank = occupy_tile_num_per_rank >= area_size
    //                                ? area_size
    //                                : occupy_tile_num_per_rank;
    std::set<std::array<scalar_type, 3>> tiles_set;

    int start[3], size[3];
    for ( int d = 0; d < 3; ++d )
    {
        start[d] = gt_partition[d][cart_rank[d]];
        size[d] = gt_partition[d][cart_rank[d] + 1] - start[d];
    }

    // insert the corner tiles to the set, to ensure the uniqueness of the
    // ground truth partition
    tiles_set.insert( { start[0], start[1], start[2] } );
    tiles_set.insert( { start[0] + size[0] - 1, start[1] + size[1] - 1,
                        start[2] + size[2] - 1 } );

    // insert random tiles to the set
    while ( static_cast<int>( tiles_set.size() ) < occupy_tile_num_per_rank )
    {
        int rand_offset[3];
        for ( int d = 0; d < 3; ++d )
            rand_offset[d] = std::rand() % size[d];
        tiles_set.insert( { start[0] + rand_offset[0],
                            start[1] + rand_offset[1],
                            start[2] + rand_offset[2] } );
    }

    // tiles_set => tiles_view (host)
    typedef typename TEST_EXECSPACE::array_layout layout;
    Kokkos::View<int* [3], layout, Kokkos::HostSpace> tiles_view_host(
        "tiles_view_host", tiles_set.size() );
    int i = 0;
    for ( auto it = tiles_set.begin(); it != tiles_set.end(); ++it )
    {
        for ( int d = 0; d < 3; ++d )
            tiles_view_host( i, d ) = ( *it )[d];
        i++;
    }

    // create tiles view on device
    Kokkos::View<int* [3], TEST_MEMSPACE> tiles_view =
        Kokkos::create_mirror_view_and_copy( TEST_MEMSPACE(), tiles_view_host );
    return tiles_view;
}

void sparse_array_test()
{
    // basic senario information
    constexpr int dim_n = 3;
    constexpr int size_tile_per_dim = 32;
    constexpr int cell_per_tile_dim = 4;
    constexpr int size_per_dim = size_tile_per_dim * cell_per_tile_dim;

    int pre_alloc_size = size_per_dim * size_per_dim;

    using scalar_type = float;
    // Create global mesh
    scalar_type cell_size = 0.1f;
    std::array<int, dim_n> global_num_cell(
        { size_per_dim, size_per_dim, size_per_dim } );
    // global low corners: random numbuers
    std::array<scalar_type, dim_n> global_low_corner = { 1.2f, 3.3f, -2.8f };
    std::array<scalar_type, dim_n> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    std::array<bool, dim_n> is_dim_periodic = { false, false, false };

    // sparse partitioner
    scalar_type max_workload_coeff = 1.5;
    int workload_num = size_per_dim * size_per_dim * size_per_dim;
    int num_step_rebalance = 200;
    int max_optimize_iteration = 10;
    SparseDimPartitioner<TEST_DEVICE, cell_per_tile_dim> partitioner(
        MPI_COMM_WORLD, max_workload_coeff, workload_num, num_step_rebalance,
        global_num_cell, max_optimize_iteration );

    // rank-related information
    Kokkos::Array<int, dim_n> cart_rank;
    std::array<int, dim_n> periodic_dims = { 0, 0, 0 };
    int reordered_cart_ranks = 1;
    MPI_Comm cart_comm;
    int linear_rank;

    // MPI rank topo and rank ID
    auto ranks_per_dim =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, global_num_cell );
    MPI_Cart_create( MPI_COMM_WORLD, dim_n, ranks_per_dim.data(),
                     periodic_dims.data(), reordered_cart_ranks, &cart_comm );
    MPI_Comm_rank( cart_comm, &linear_rank );
    MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

    // scene initialization
    auto gt_partitions =
        generate_random_partition( ranks_per_dim, size_tile_per_dim );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( sparse_array, 3d_sparse_array ) { sparse_array_test(); }

//---------------------------------------------------------------------------//
} // namespace Test