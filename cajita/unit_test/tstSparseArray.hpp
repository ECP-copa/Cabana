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

// convert std::set to device-side view
template <typename T>
auto set2view( const std::set<std::array<T, 3>>& in_set )
    -> Kokkos::View<T* [3], TEST_MEMSPACE>
{
    // set => view (host)
    typedef typename TEST_EXECSPACE::array_layout layout;
    Kokkos::View<T* [3], layout, Kokkos::HostSpace> host_view( "view_host",
                                                               in_set.size() );
    for ( int i = 0, auto it = tiles_set.begin(); it != tiles_set.end();
          ++it, ++i )
    {
        for ( int d = 0; d < 3; ++d )
            host_view( i, d ) = ( *it )[d];
    }

    // create tiles view on device
    Kokkos::View<int* [3], TEST_MEMSPACE> dev_view =
        Kokkos::create_mirror_view_and_copy( TEST_MEMSPACE(), host_view );
    return dev_view;
}

// return random generated particles and occupied tile numbers (last two params)
template <typename T>
void generate_random_particles( const int particle_number,
                                const std::array<int, 3>& part_start,
                                const std::array<int, 3>& part_end,
                                const int cell_per_tile_dim,
                                const std::array<T, 3> global_low_corner,
                                const T cell_size,
                                std::set<std::array<int, 3>>& tile_set,
                                std::set<std::array<scalar, 3>>& par_pos_set )
{
    // range of particle positions
    T start[3], size[3];
    for ( int d = 0; d < 3; ++d )
    {
        // because each particle will activate three around tiles, we apply
        // 1.01 cell_size offset compared to the real partition to ensure
        // all the activated tiles sit inside the valid partition range
        start[d] = global_low_corner[d] +
                   cell_size * ( 1.01f + cell_per_tile_dim * (T)part_start[d] );
        size[d] =
            cell_size *
            ( cell_per_tile_dim * (T)( part_end[d] - part_start[d] ) - 2.02f );
    }

    // insert random particles to the set
    while ( static_cast<int>( par_pos_set.size() ) < occupy_tile_num_per_rank )
    {
        T rand_offset[3];
        for ( int d = 0; d < 3; ++d )
            rand_offset[d] = std::rand() / RAND_MAX * size[d];
        std::array<T, 3> new_pos = { start[0] + rand_offset[0],
                                     start[1] + rand_offset[1],
                                     start[2] + rand_offset[2] };
        auto old_size = par_pos_set.size();
        par_pos_set.insert( new_pos );
        if ( old_size == par_pos_set.size() )
            continue;

        std::array<int, 3> grid_base;
        for ( int d = 0; d < 3; ++d )
        {
            grid_base[d] = int( std::lround( new_pos[d] / cell_size ) ) - 1;
        }

        for ( int i = 0; i <= 2; i++ )
            for ( int j = 0; j <= 2; j++ )
                for ( int k = 0; k <= 2; k++ )
                {
                    tile_set.insert( {
                        ( grid_base[0] + i ) / cell_per_tile_dim,
                        ( grid_base[1] + j ) / cell_per_tile_dim,
                        ( grid_base[2] + k ) / cell_per_tile_dim,
                    } );
                }
    }
}
template <typename EntityType>
void sparse_array_test( int par_num, EntityType e )
{
    // basic senario information
    constexpr int size_tile_per_dim = 32;
    constexpr int cell_per_tile_dim = 4;
    constexpr int cell_per_tile =
        cell_per_tile_dim * cell_per_tile_dim * cell_per_tile_dim;
    constexpr int size_per_dim = size_tile_per_dim * cell_per_tile_dim;

    int pre_alloc_size = size_per_dim * size_per_dim;

    using T = float;
    // Create global mesh
    T cell_size = 0.1f;
    std::array<int, 3> global_num_cell(
        { size_per_dim, size_per_dim, size_per_dim } );
    // global low corners: random numbuers
    std::array<T, 3> global_low_corner = { 1.2f, 3.3f, -2.8f };
    std::array<T, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // sparse partitioner
    T max_workload_coeff = 1.5;
    int workload_num = size_per_dim * size_per_dim * size_per_dim;
    int num_step_rebalance = 200;
    int max_optimize_iteration = 10;
    SparseDimPartitioner<TEST_DEVICE, cell_per_tile_dim> partitioner(
        MPI_COMM_WORLD, max_workload_coeff, workload_num, num_step_rebalance,
        global_num_cell, max_optimize_iteration );

    // rank-related information
    Kokkos::Array<int, 3> cart_rank;
    std::array<int, 3> periodic_dims = { 0, 0, 0 };
    int reordered_cart_ranks = 1;
    MPI_Comm cart_comm;
    int linear_rank;

    // MPI rank topo and rank ID
    auto ranks_per_dim =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, global_num_cell );
    MPI_Cart_create( MPI_COMM_WORLD, 3, ranks_per_dim.data(),
                     periodic_dims.data(), reordered_cart_ranks, &cart_comm );
    MPI_Comm_rank( cart_comm, &linear_rank );
    MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

    // scene initialization
    auto gt_partitions =
        generate_random_partition( ranks_per_dim, size_tile_per_dim );
    partitioner.initializeRecPartition( gt_partitions[0], gt_partitions[1],
                                        gt_partitions[2] );

    std::set<std::array<int, 3>> tile_set;
    std::set<std::array<T, 3>> par_pos_set;
    generate_random_particles( par_num,
                               { gt_partitions[0][cart_rank[0]],
                                 gt_partitions[1][cart_rank[1]],
                                 gt_partitions[2][cart_rank[2]] },
                               { gt_partitions[0][cart_rank[0] + 1],
                                 gt_partitions[1][cart_rank[1] + 1],
                                 gt_partitions[2][cart_rank[2] + 1] },
                               cell_per_tile_dim, global_low_corner, cell_size,
                               tile_set, par_pos_set );
    auto tile_view = set2view( tile_set );
    auto par_view = set2view( par_pos_set );

    // DEBUG [TODO]
    printf( "rank %d (%d, %d, %d), par_num = %d, par_pos_set size = %llu, "
            "tile_set_size = %llu\n",
            linear_rank, cart_rank[0], cart_rank[1], cart_rank[2], par_num,
            par_pos_set.size(), tile_set.size() );
    // END [TODO]

    // mesh/grid related initilization
    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    int halo_width = 2;
    auto local_grid =
        createSparseLocalGrid( global_grid, halo_width, cell_per_tile_dim );

    auto sparse_map =
        createSparseMap<TEST_EXECSPACE>( global_mesh, pre_alloc_size );

    // def test sparse array
    using DataTypes = Cabana::MemberTypes<int, float, double[3]>;
    auto test_layout =
        createSparseArrayLayout<DataTypes>( local_grid, sparse_map, e );
    auto test_array = createSparseArray<TEST_DEVICE>(
        std::string( "test_sparse_grid" ), test_layout );

    // insert particles
    test_array.registerSparseMap( par_view );
    test_array.reserve( 1.2 );

    // size-realted tests
    EXPECT_EQ( test_array.size(), sparse_map.sizeCell() );
    EXPECT_EQ( test_array.capacity() >= test_layout.reservedCellSize( 1.2 ),
               true );
    EXPECT_EQ( test_array.empty(), false );
    EXPECT_EQ( test_array.numSoA(), sparse_map.sizeTile() );
    for ( std::size_t i = 0; i < test_array.numSoA(); ++i )
        EXPECT_EQ( test_array.arraySize( i ), cell_per_dim )

    // some test [TODO]

    // test end

    test_array.clear();
    EXPECT_EQ( test_array.size(), 0 );
    EXPECT_EQ( test_layout.sparseMap().sizeTile(), 0 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( sparse_array, 3d_sparse_array ) { sparse_array_test(); }

//---------------------------------------------------------------------------//
} // namespace Test