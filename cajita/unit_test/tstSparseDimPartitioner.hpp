#include <Cajita_SparseDimPartitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

using namespace Cajita;

namespace Test
{
void uniform_distribution_automatic_rank()
{
    constexpr int size_tile_per_dim = 16;
    constexpr int cell_per_tile_dim = 4;
    constexpr int size_per_dim = size_tile_per_dim * cell_per_tile_dim;
    constexpr int total_size = size_per_dim * size_per_dim * size_per_dim;

    float max_workload_coeff = 1.5;
    int particle_num = total_size;
    int num_step_rebalance = 100;

    std::array<int, 3> global_cells_per_dim = {
        size_tile_per_dim * cell_per_tile_dim,
        size_tile_per_dim * cell_per_tile_dim,
        size_tile_per_dim * cell_per_tile_dim };

    SparseDimPartitioner<TEST_MEMSPACE, TEST_EXECSPACE, cell_per_tile_dim>
        partitioner( MPI_COMM_WORLD, max_workload_coeff, particle_num,
                     num_step_rebalance, global_cells_per_dim );

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

    // init partitions
    std::array<std::vector<int>, 3> rec_partitions;
    for ( int d = 0; d < 3; ++d )
    {
        int ele = size_tile_per_dim / ranks_per_dim[d];
        int part = 0;
        for ( int i = 0; i < ranks_per_dim[0]; ++i )
        {
            rec_partitions[d].push_back( part );
            part += ele;
        }
        rec_partitions[d].push_back( size_tile_per_dim );
    }
    partitioner.initialize_rec_partition( rec_partitions[0], rec_partitions[1],
                                          rec_partitions[2] );

    // test getCurrentPartition
    {
        auto part = partitioner.get_current_partition();
        for ( int d = 0; d < 3; ++d )
            for ( int id = 0; id < ranks_per_dim[d] + 1; id++ )
                EXPECT_EQ( part[d][id], rec_partitions[d][id] );
    }

    // test ownedCellsPerDimension
    std::array<int, 3> cart_rank;
    std::array<int, 3> periodic_dims = { 0, 0, 0 };
    int reordered_cart_ranks = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create( MPI_COMM_WORLD, 3, ranks_per_dim.data(),
                     periodic_dims.data(), reordered_cart_ranks, &cart_comm );
    int linear_rank;
    MPI_Comm_rank( cart_comm, &linear_rank );
    // make a new communicater with MPI_Cart_create
    MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

    auto owned_cells_per_dim =
        partitioner.ownedCellsPerDimension( cart_comm, global_cells_per_dim );
    auto owned_tiles_per_dim = partitioner.ownedTilesPerDimension( cart_comm );
    for ( int d = 0; d < 3; ++d )
    {
        auto gt_tile = rec_partitions[d][cart_rank[d] + 1] -
                       rec_partitions[d][cart_rank[d]];
        EXPECT_EQ( owned_tiles_per_dim[d], gt_tile );
        EXPECT_EQ( owned_cells_per_dim[d], gt_tile * cell_per_tile_dim *
                                               cell_per_tile_dim *
                                               cell_per_tile_dim );
    }

    // init sparseMap
    double cell_size = 0.1;
    int pre_alloc_size = size_per_dim * size_per_dim;
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_cells_per_dim[0],
        global_low_corner[1] + cell_size * global_cells_per_dim[1],
        global_low_corner[2] + cell_size * global_cells_per_dim[2] };
    auto global_mesh = createSparseGlobalMesh(
        global_low_corner, global_high_corner, global_cells_per_dim );
    auto sis = createSparseMap<TEST_EXECSPACE>( global_mesh, pre_alloc_size );
    Kokkos::parallel_for(
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, size_per_dim ),
        KOKKOS_LAMBDA( int i ) {
            for ( int j = 0; j < size_per_dim; j++ )
                for ( int k = 0; k < size_per_dim; k++ )
                {
                    sis.insertCell( i, j, k );
                }
        } );

    // compute workload and do optimization
    partitioner.computeLocalWorkLoad( sis );
    partitioner.computeFullPrefixSum( MPI_COMM_WORLD );
    partitioner.optimizePartition();

    // check results
    owned_cells_per_dim =
        partitioner.ownedCellsPerDimension( cart_comm, global_cells_per_dim );
    for ( int d = 0; d < 3; ++d )
    {
        auto gt_tile = rec_partitions[d][cart_rank[d] + 1] -
                       rec_partitions[d][cart_rank[d]];

        EXPECT_EQ( owned_cells_per_dim[d], gt_tile * cell_per_tile_dim *
                                               cell_per_tile_dim *
                                               cell_per_tile_dim );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( sparse_dim_partitioner, sparse_dim_partitioner_test )
{
    uniform_distribution_automatic_rank();
}

//---------------------------------------------------------------------------//
} // end namespace Test