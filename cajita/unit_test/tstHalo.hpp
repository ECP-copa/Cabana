/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita_Array.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_Halo.hpp>
#include <Cajita_ManualPartitioner.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cmath>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
// Halo padding in each dimension for different entity types.
int haloPad( Cell, int ) { return 0; }

int haloPad( Node, int ) { return 1; }

template <int D>
int haloPad( Face<D>, int d )
{
    return ( d == D ) ? 1 : 0;
}

template <int D>
int haloPad( Edge<D>, int d )
{
    return ( d == D ) ? 0 : 1;
}

//---------------------------------------------------------------------------//
// Check initial array gather. We should get 1 everywhere in the array now
// where there was ghost overlap. Otherwise there will still be 0.
template <class Array>
void checkGather( const int halo_width, const Array &array )
{
    auto owned_space = array.layout()->indexSpace( Own(), Local() );
    auto ghosted_space = array.layout()->indexSpace( Ghost(), Local() );
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          array.view() );
    auto pad_i = haloPad( typename Array::entity_type(), Dim::I );
    auto pad_j = haloPad( typename Array::entity_type(), Dim::J );
    auto pad_k = haloPad( typename Array::entity_type(), Dim::K );
    for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
        for ( unsigned j = 0; j < ghosted_space.extent( 1 ); ++j )
            for ( unsigned k = 0; k < ghosted_space.extent( 2 ); ++k )
                for ( unsigned l = 0; l < ghosted_space.extent( 3 ); ++l )
                    if ( i < owned_space.min( Dim::I ) - halo_width ||
                         i >= owned_space.max( Dim::I ) + halo_width + pad_i ||
                         j < owned_space.min( Dim::J ) - halo_width ||
                         j >= owned_space.max( Dim::J ) + halo_width + pad_j ||
                         k < owned_space.min( Dim::K ) - halo_width ||
                         k >= owned_space.max( Dim::K ) + halo_width + pad_k )
                        EXPECT_EQ( host_view( i, j, k, l ), 0.0 );
                    else
                        EXPECT_EQ( host_view( i, j, k, l ), 1.0 );
}

//---------------------------------------------------------------------------//
// Check array scatter. The value of the cell should be a function of how many
// neighbors it has. Corner neighbors get 8, edge neighbors get 4, face
// neighbors get 2, and no neighbors remain at 1.
template <class Array>
void checkScatter( const std::array<bool, 3> &is_dim_periodic,
                   const int halo_width, const Array &array )
{
    // Get data.
    auto owned_space = array.layout()->indexSpace( Own(), Local() );
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          array.view() );
    const auto &global_grid = array.layout()->localGrid()->globalGrid();

    // This function checks if an index is in the halo of a low neighbor in
    // the given dimension
    auto in_dim_min_halo = [&]( const int i, const int dim ) {
        if ( is_dim_periodic[dim] || global_grid.dimBlockId( dim ) > 0 )
            return i < ( owned_space.min( dim ) + halo_width +
                         haloPad( typename Array::entity_type(), dim ) );
        else
            return false;
    };

    // This function checks if an index is in the halo of a high neighbor in
    // the given dimension
    auto in_dim_max_halo = [&]( const int i, const int dim ) {
        if ( is_dim_periodic[dim] || global_grid.dimBlockId( dim ) <
                                         global_grid.dimNumBlock( dim ) - 1 )
            return i >= ( owned_space.max( dim ) - halo_width );
        else
            return false;
    };

    // Check results. Use the halo functions to figure out how many neighbor
    // a given cell was ghosted to.
    for ( unsigned i = owned_space.min( 0 ); i < owned_space.max( 0 ); ++i )
        for ( unsigned j = owned_space.min( 1 ); j < owned_space.max( 1 ); ++j )
            for ( unsigned k = owned_space.min( 2 ); k < owned_space.max( 2 );
                  ++k )
            {
                int num_n = 0;
                if ( in_dim_min_halo( i, Dim::I ) ||
                     in_dim_max_halo( i, Dim::I ) )
                    ++num_n;
                if ( in_dim_min_halo( j, Dim::J ) ||
                     in_dim_max_halo( j, Dim::J ) )
                    ++num_n;
                if ( in_dim_min_halo( k, Dim::K ) ||
                     in_dim_max_halo( k, Dim::K ) )
                    ++num_n;
                double scatter_val = std::pow( 2.0, num_n );
                for ( unsigned l = 0; l < owned_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_view( i, j, k, l ), scatter_val );
            }
}

//---------------------------------------------------------------------------//
void gatherScatterTest( const ManualPartitioner &partitioner,
                        const std::array<bool, 3> &is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 32, 23, 41 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Array parameters.
    unsigned array_halo_width = 3;

    // Loop over halo sizes up to the size of the array halo width.
    for ( unsigned halo_width = 1; halo_width <= array_halo_width;
          ++halo_width )
    {
        // Create a cell array.
        auto layout =
            createArrayLayout( global_grid, array_halo_width, 4, Cell() );
        auto array = createArray<double, TEST_DEVICE>( "array", layout );

        // Assign the owned cells a value of 1 and the rest 0.
        ArrayOp::assign( *array, 0.0, Ghost() );
        ArrayOp::assign( *array, 1.0, Own() );

        // Create a halo.
        auto halo = createHalo( *array, FullHaloPattern(), halo_width );

        // Gather into the ghosts.
        halo->gather( TEST_EXECSPACE(), *array );

        // Check the gather.
        checkGather( halo_width, *array );

        // Scatter from the ghosts back to owned.
        halo->scatter( TEST_EXECSPACE(), ScatterReduce::Sum(), *array );

        // Check the scatter.
        checkScatter( is_dim_periodic, halo_width, *array );
    }

    // Repeat the process but this time with multiple arrays in a Halo
    for ( unsigned halo_width = 1; halo_width <= array_halo_width;
          ++halo_width )
    {
        // Create arrays of different layouts and dof counts.
        auto cell_layout =
            createArrayLayout( global_grid, array_halo_width, 4, Cell() );
        auto cell_array =
            createArray<double, TEST_DEVICE>( "cell_array", cell_layout );

        auto node_layout =
            createArrayLayout( global_grid, array_halo_width, 3, Node() );
        auto node_array =
            createArray<float, TEST_DEVICE>( "node_array", node_layout );

        auto face_i_layout = createArrayLayout( global_grid, array_halo_width,
                                                4, Face<Dim::I>() );
        auto face_i_array =
            createArray<double, TEST_DEVICE>( "face_i_array", face_i_layout );

        auto face_j_layout = createArrayLayout( global_grid, array_halo_width,
                                                1, Face<Dim::J>() );
        auto face_j_array =
            createArray<double, TEST_DEVICE>( "face_j_array", face_j_layout );

        auto face_k_layout = createArrayLayout( global_grid, array_halo_width,
                                                2, Face<Dim::K>() );
        auto face_k_array =
            createArray<float, TEST_DEVICE>( "face_k_array", face_k_layout );

        auto edge_i_layout = createArrayLayout( global_grid, array_halo_width,
                                                3, Edge<Dim::I>() );
        auto edge_i_array =
            createArray<float, TEST_DEVICE>( "edge_i_array", edge_i_layout );

        auto edge_j_layout = createArrayLayout( global_grid, array_halo_width,
                                                2, Edge<Dim::J>() );
        auto edge_j_array =
            createArray<float, TEST_DEVICE>( "edge_j_array", edge_j_layout );

        auto edge_k_layout = createArrayLayout( global_grid, array_halo_width,
                                                1, Edge<Dim::K>() );
        auto edge_k_array =
            createArray<double, TEST_DEVICE>( "edge_k_array", edge_k_layout );

        // Assign the owned cells a value of 1 and the rest 0.
        ArrayOp::assign( *cell_array, 0.0, Ghost() );
        ArrayOp::assign( *cell_array, 1.0, Own() );

        ArrayOp::assign( *node_array, 0.0, Ghost() );
        ArrayOp::assign( *node_array, 1.0, Own() );

        ArrayOp::assign( *face_i_array, 0.0, Ghost() );
        ArrayOp::assign( *face_i_array, 1.0, Own() );

        ArrayOp::assign( *face_j_array, 0.0, Ghost() );
        ArrayOp::assign( *face_j_array, 1.0, Own() );

        ArrayOp::assign( *face_k_array, 0.0, Ghost() );
        ArrayOp::assign( *face_k_array, 1.0, Own() );

        ArrayOp::assign( *edge_i_array, 0.0, Ghost() );
        ArrayOp::assign( *edge_i_array, 1.0, Own() );

        ArrayOp::assign( *edge_j_array, 0.0, Ghost() );
        ArrayOp::assign( *edge_j_array, 1.0, Own() );

        ArrayOp::assign( *edge_k_array, 0.0, Ghost() );
        ArrayOp::assign( *edge_k_array, 1.0, Own() );

        // Create a multihalo.
        auto halo =
            createHalo( FullHaloPattern(), halo_width, *cell_array, *node_array,
                        *face_i_array, *face_j_array, *face_k_array,
                        *edge_i_array, *edge_j_array, *edge_k_array );

        // Gather into the ghosts.
        halo->gather( TEST_EXECSPACE(), *cell_array, *node_array, *face_i_array,
                      *face_j_array, *face_k_array, *edge_i_array,
                      *edge_j_array, *edge_k_array );

        // Check the gather.
        checkGather( halo_width, *cell_array );
        checkGather( halo_width, *node_array );
        checkGather( halo_width, *face_i_array );
        checkGather( halo_width, *face_j_array );
        checkGather( halo_width, *face_k_array );
        checkGather( halo_width, *edge_i_array );
        checkGather( halo_width, *edge_j_array );
        checkGather( halo_width, *edge_k_array );

        // Scatter from the ghosts back to owned.
        halo->scatter( TEST_EXECSPACE(), ScatterReduce::Sum(), *cell_array,
                       *node_array, *face_i_array, *face_j_array, *face_k_array,
                       *edge_i_array, *edge_j_array, *edge_k_array );

        // Check the scatter.
        checkScatter( is_dim_periodic, halo_width, *cell_array );
        checkScatter( is_dim_periodic, halo_width, *node_array );
        checkScatter( is_dim_periodic, halo_width, *face_i_array );
        checkScatter( is_dim_periodic, halo_width, *face_j_array );
        checkScatter( is_dim_periodic, halo_width, *face_k_array );
        checkScatter( is_dim_periodic, halo_width, *edge_i_array );
        checkScatter( is_dim_periodic, halo_width, *edge_j_array );
        checkScatter( is_dim_periodic, halo_width, *edge_k_array );
    }
}

//---------------------------------------------------------------------------//
template <class ReduceFunc>
struct TestHaloReduce;

template <>
struct TestHaloReduce<ScatterReduce::Min>
{
    template <class ViewType>
    static void check( ViewType view, int neighbor_rank, int comm_rank,
                       const int i, const int j, const int k, const int l )
    {
        if ( neighbor_rank < comm_rank )
            EXPECT_EQ( view( i, j, k, l ), neighbor_rank );
        else
            EXPECT_EQ( view( i, j, k, l ), comm_rank );
    }
};

template <>
struct TestHaloReduce<ScatterReduce::Max>
{
    template <class ViewType>
    static void check( ViewType view, int neighbor_rank, int comm_rank,
                       const int i, const int j, const int k, const int l )
    {
        if ( neighbor_rank > comm_rank )
            EXPECT_EQ( view( i, j, k, l ), neighbor_rank );
        else
            EXPECT_EQ( view( i, j, k, l ), comm_rank );
    }
};

template <>
struct TestHaloReduce<ScatterReduce::Replace>
{
    template <class ViewType>
    static void check( ViewType view, int neighbor_rank, int, const int i,
                       const int j, const int k, const int l )
    {
        EXPECT_EQ( view( i, j, k, l ), neighbor_rank );
    }
};

template <class ReduceFunc>
void scatterReduceTest( const ReduceFunc &reduce )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 32, 23, 41 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    auto global_grid =
        createGlobalGrid( MPI_COMM_WORLD, global_mesh, is_dim_periodic,
                          Cajita::UniformDimPartitioner() );

    // Create an array on the cells.
    unsigned array_halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout = createArrayLayout( global_grid, array_halo_width,
                                          dofs_per_cell, Cell() );
    auto array = createArray<double, TEST_DEVICE>( "array", cell_layout );

    // Assign the rank to the array.
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    ArrayOp::assign( *array, comm_rank, Ghost() );

    // Create a halo pattern - just write to your 8 corner neighbors so we can
    // eliminate overlap between neighbors and not need to resolve the
    // collision.
    HaloPattern pattern;
    std::vector<std::array<int, 3>> neighbors = {
        { -1, -1, -1 }, { 1, -1, -1 }, { -1, 1, -1 }, { 1, 1, -1 },
        { -1, -1, 1 },  { 1, -1, 1 },  { -1, 1, 1 },  { 1, 1, 1 } };
    pattern.setNeighbors( neighbors );

    // Create a halo.
    auto halo = createHalo( *array, pattern );

    // Scatter.
    halo->scatter( TEST_EXECSPACE(), reduce, *array );

    // Check the reduction.
    auto host_array = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           array->view() );
    for ( const auto &n : neighbors )
    {
        auto neighbor_rank =
            cell_layout->localGrid()->neighborRank( n[0], n[1], n[2] );
        auto shared_space = cell_layout->localGrid()->sharedIndexSpace(
            Cajita::Own(), Cajita::Cell(), n[0], n[1], n[2] );
        for ( int i = shared_space.min( Dim::I );
              i < shared_space.max( Dim::I ); ++i )
            for ( int j = shared_space.min( Dim::J );
                  j < shared_space.max( Dim::J ); ++j )
                for ( int k = shared_space.min( Dim::K );
                      k < shared_space.max( Dim::K ); ++k )
                    for ( int l = 0; l < 4; ++l )
                    {
                        TestHaloReduce<ReduceFunc>::check(
                            host_array, neighbor_rank, comm_rank, i, j, k, l );
                    }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, not_periodic_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    ManualPartitioner partitioner( ranks_per_dim );

    // Boundaries are not periodic.
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, periodic_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    ManualPartitioner partitioner( ranks_per_dim );

    // Every boundary is periodic
    std::array<bool, 3> is_dim_periodic = { true, true, true };

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[2] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[1], ranks_per_dim[2] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
    std::swap( ranks_per_dim[0], ranks_per_dim[1] );
    partitioner = ManualPartitioner( ranks_per_dim );
    gatherScatterTest( partitioner, is_dim_periodic );
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, scatter_reduce_max_test )
{
    scatterReduceTest( ScatterReduce::Max() );
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, scatter_reduce_min_test )
{
    scatterReduceTest( ScatterReduce::Min() );
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, scatter_reduce_replace_test )
{
    scatterReduceTest( ScatterReduce::Replace() );
}

//---------------------------------------------------------------------------//

} // end namespace Test
