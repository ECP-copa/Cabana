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

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_HaloBase.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cmath>

using namespace Cabana::Grid;

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

//---------------------------------------------------------------------------//
// Check initial array gather. We should get 1 everywhere in the array now
// where there was ghost overlap. Otherwise there will still be 0.
template <class Array>
void checkGather( const std::array<bool, 2>& is_dim_periodic,
                  const int halo_width, const Array& array )
{
    auto owned_space = array.layout()->indexSpace( Own(), Local() );
    auto ghosted_space = array.layout()->indexSpace( Ghost(), Local() );
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          array.view() );
    auto pad_i = haloPad( typename Array::entity_type(), Dim::I );
    auto pad_j = haloPad( typename Array::entity_type(), Dim::J );
    const auto& global_grid = array.layout()->localGrid()->globalGrid();

    // This function checks if an index is in the low boundary halo in the
    // given dimension
    auto in_boundary_min_halo = [&]( const int i, const int dim )
    {
        if ( is_dim_periodic[dim] || !global_grid.onLowBoundary( dim ) )
            return false;
        else
            return ( i < owned_space.min( dim ) );
    };

    // This function checks if an index is in the high boundary halo of in the
    // given dimension
    auto in_boundary_max_halo = [&]( const int i, const int dim )
    {
        if ( is_dim_periodic[dim] || !global_grid.onHighBoundary( dim ) )
            return false;
        else
            return ( i >= owned_space.max( dim ) );
    };

    // Check the gather.
    for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
        for ( unsigned j = 0; j < ghosted_space.extent( 1 ); ++j )
            for ( unsigned l = 0; l < ghosted_space.extent( 2 ); ++l )
                if ( in_boundary_min_halo( i, Dim::I ) ||
                     in_boundary_min_halo( j, Dim::J ) ||
                     in_boundary_max_halo( i, Dim::I ) ||
                     in_boundary_max_halo( j, Dim::J ) )
                {
                    EXPECT_DOUBLE_EQ( host_view( i, j, l ), 0.0 );
                }
                else if ( i < owned_space.min( Dim::I ) - halo_width ||
                          i >= owned_space.max( Dim::I ) + halo_width + pad_i ||
                          j < owned_space.min( Dim::J ) - halo_width ||
                          j >= owned_space.max( Dim::J ) + halo_width + pad_j )
                {
                    EXPECT_DOUBLE_EQ( host_view( i, j, l ), 0.0 );
                }
                else
                {
                    EXPECT_DOUBLE_EQ( host_view( i, j, l ), 1.0 );
                }
}

//---------------------------------------------------------------------------//
// Check array scatter. The value of the cell should be a function of how many
// neighbors it has. Corner neighbors get 8, edge neighbors get 4, face
// neighbors get 2, and no neighbors remain at 1.
template <class Array>
void checkScatter( const std::array<bool, 2>& is_dim_periodic,
                   const int halo_width, const Array& array )
{
    // Get data.
    auto owned_space = array.layout()->indexSpace( Own(), Local() );
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          array.view() );
    const auto& global_grid = array.layout()->localGrid()->globalGrid();

    // This function checks if an index is in the halo of a low neighbor in
    // the given dimension
    auto in_dim_min_halo = [&]( const int i, const int dim )
    {
        if ( is_dim_periodic[dim] || global_grid.dimBlockId( dim ) > 0 )
            return i < ( owned_space.min( dim ) + halo_width +
                         haloPad( typename Array::entity_type(), dim ) );
        else
            return false;
    };

    // This function checks if an index is in the halo of a high neighbor in
    // the given dimension
    auto in_dim_max_halo = [&]( const int i, const int dim )
    {
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
        {
            int num_n = 0;
            if ( in_dim_min_halo( i, Dim::I ) || in_dim_max_halo( i, Dim::I ) )
                ++num_n;
            if ( in_dim_min_halo( j, Dim::J ) || in_dim_max_halo( j, Dim::J ) )
                ++num_n;
            double scatter_val = std::pow( 2.0, num_n );
            for ( unsigned l = 0; l < owned_space.extent( 2 ); ++l )
                EXPECT_EQ( host_view( i, j, l ), scatter_val );
        }
}

//---------------------------------------------------------------------------//
template <class TestCommSpace>
void gatherScatterTest( const ManualBlockPartitioner<2>& partitioner,
                        const std::array<bool, 2>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.43;
    std::array<int, 2> global_num_cell = { 19, 27 };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
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
        auto array = createArray<double, TEST_MEMSPACE>( "array", layout );

        // Assign the owned cells a value of 1 and the rest 0.
        ArrayOp::assign( *array, 0.0, Ghost() );
        ArrayOp::assign( *array, 1.0, Own() );

        // Create a halo.
        auto halo = createHalo<TestCommSpace>( NodeHaloPattern<2>(), halo_width,
                                               *array );

        // Gather into the ghosts.
        halo->gather( TEST_EXECSPACE(), *array );

        // Check the gather.
        checkGather( is_dim_periodic, halo_width, *array );

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
            createArray<double, TEST_MEMSPACE>( "cell_array", cell_layout );

        auto node_layout =
            createArrayLayout( global_grid, array_halo_width, 3, Node() );
        auto node_array =
            createArray<float, TEST_MEMSPACE>( "node_array", node_layout );

        auto face_i_layout = createArrayLayout( global_grid, array_halo_width,
                                                4, Face<Dim::I>() );
        auto face_i_array =
            createArray<double, TEST_MEMSPACE>( "face_i_array", face_i_layout );

        auto face_j_layout = createArrayLayout( global_grid, array_halo_width,
                                                1, Face<Dim::J>() );
        auto face_j_array =
            createArray<double, TEST_MEMSPACE>( "face_j_array", face_j_layout );

        // Assign the owned cells a value of 1 and the rest 0.
        ArrayOp::assign( *cell_array, 0.0, Ghost() );
        ArrayOp::assign( *cell_array, 1.0, Own() );

        ArrayOp::assign( *node_array, 0.0, Ghost() );
        ArrayOp::assign( *node_array, 1.0, Own() );

        ArrayOp::assign( *face_i_array, 0.0, Ghost() );
        ArrayOp::assign( *face_i_array, 1.0, Own() );

        ArrayOp::assign( *face_j_array, 0.0, Ghost() );
        ArrayOp::assign( *face_j_array, 1.0, Own() );

        // Create a multihalo.
        auto halo = createHalo<TestCommSpace>( NodeHaloPattern<2>(), halo_width,
                                               *cell_array, *node_array,
                                               *face_i_array, *face_j_array );

        // Gather into the ghosts.
        halo->gather( TEST_EXECSPACE(), *cell_array, *node_array, *face_i_array,
                      *face_j_array );

        // Check the gather.
        checkGather( is_dim_periodic, halo_width, *cell_array );
        checkGather( is_dim_periodic, halo_width, *node_array );
        checkGather( is_dim_periodic, halo_width, *face_i_array );
        checkGather( is_dim_periodic, halo_width, *face_j_array );

        // Scatter from the ghosts back to owned.
        halo->scatter( TEST_EXECSPACE(), ScatterReduce::Sum(), *cell_array,
                       *node_array, *face_i_array, *face_j_array );

        // Check the scatter.
        checkScatter( is_dim_periodic, halo_width, *cell_array );
        checkScatter( is_dim_periodic, halo_width, *node_array );
        checkScatter( is_dim_periodic, halo_width, *face_i_array );
        checkScatter( is_dim_periodic, halo_width, *face_j_array );
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
                       const int i, const int j, const int l )
    {
        if ( neighbor_rank < comm_rank )
            EXPECT_EQ( view( i, j, l ), neighbor_rank );
        else
            EXPECT_EQ( view( i, j, l ), comm_rank );
    }
};

template <>
struct TestHaloReduce<ScatterReduce::Max>
{
    template <class ViewType>
    static void check( ViewType view, int neighbor_rank, int comm_rank,
                       const int i, const int j, const int l )
    {
        if ( neighbor_rank > comm_rank )
            EXPECT_EQ( view( i, j, l ), neighbor_rank );
        else
            EXPECT_EQ( view( i, j, l ), comm_rank );
    }
};

template <>
struct TestHaloReduce<ScatterReduce::Replace>
{
    template <class ViewType>
    static void check( ViewType view, int neighbor_rank, int, const int i,
                       const int j, const int l )
    {
        EXPECT_EQ( view( i, j, l ), neighbor_rank );
    }
};

template <class TestCommSpace, class ReduceFunc>
void scatterReduceTest( const ReduceFunc& reduce )
{
    // Create the global grid.
    double cell_size = 0.43;
    std::array<int, 2> global_num_cell = { 19, 27 };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    std::array<bool, 2> is_dim_periodic = { true, true };
    auto global_grid =
        createGlobalGrid( MPI_COMM_WORLD, global_mesh, is_dim_periodic,
                          DimBlockPartitioner<2>() );

    // Create an array on the cells.
    unsigned array_halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout = createArrayLayout( global_grid, array_halo_width,
                                          dofs_per_cell, Cell() );
    auto array = createArray<double, TEST_MEMSPACE>( "array", cell_layout );

    // Assign the rank to the array.
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    ArrayOp::assign( *array, comm_rank, Ghost() );

    // Create a halo pattern - just write to your 4 corner neighbors so we can
    // eliminate overlap between neighbors and not need to resolve the
    // collision.
    HaloPattern<2> pattern;
    std::vector<std::array<int, 2>> neighbors = {
        { -1, -1 }, { 1, -1 }, { -1, 1 }, { 1, 1 } };
    pattern.setNeighbors( neighbors );

    // Create a halo.
    auto halo = createHalo<TestCommSpace>( pattern, array_halo_width, *array );

    // Scatter.
    halo->scatter( TEST_EXECSPACE(), reduce, *array );

    // Check the reduction.
    auto host_array = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                           array->view() );
    for ( const auto& n : neighbors )
    {
        auto neighbor_rank =
            cell_layout->localGrid()->neighborRank( n[0], n[1] );
        auto shared_space = cell_layout->localGrid()->sharedIndexSpace(
            Own(), Cell(), n[0], n[1] );
        for ( int i = shared_space.min( Dim::I );
              i < shared_space.max( Dim::I ); ++i )
            for ( int j = shared_space.min( Dim::J );
                  j < shared_space.max( Dim::J ); ++j )
                for ( int l = 0; l < 4; ++l )
                {
                    TestHaloReduce<ReduceFunc>::check(
                        host_array, neighbor_rank, comm_rank, i, j, l );
                }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
template <typename TestCommSpace>
class Halo2dTypedTest : public ::testing::Test
{
};

// Add additional backends to test when implemented.
using CommSpaceTypes = ::testing::Types<Cabana::Mpi>;

// Need a trailing comma
// to avoid an error when compiling with clang++
TYPED_TEST_SUITE( Halo2dTypedTest, CommSpaceTypes, );

TYPED_TEST( Halo2dTypedTest, NonPeriodic2d )
{
    // Extract communication backend type
    using commspace_type = TypeParam;

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> ranks_per_dim = { 0, 0 };
    MPI_Dims_create( comm_size, 2, ranks_per_dim.data() );
    ManualBlockPartitioner<2> partitioner( ranks_per_dim );

    // Boundaries are not periodic.
    std::array<bool, 2> is_dim_periodic = { false, false };

    gatherScatterTest<commspace_type>( partitioner, is_dim_periodic );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        partitioner = ManualBlockPartitioner<2>( ranks_per_dim );
        gatherScatterTest<commspace_type>( partitioner, is_dim_periodic );
    }
}

//---------------------------------------------------------------------------//
TYPED_TEST( Halo2dTypedTest, Periodic2d )
{
    // Extract communication backend type
    using commspace_type = TypeParam;

    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 2> ranks_per_dim = { 0, 0 };
    MPI_Dims_create( comm_size, 2, ranks_per_dim.data() );
    ManualBlockPartitioner<2> partitioner( ranks_per_dim );

    // Every boundary is periodic
    std::array<bool, 2> is_dim_periodic = { true, true };

    gatherScatterTest<commspace_type>( partitioner, is_dim_periodic );

    // Test with different block configurations to make sure all the
    // dimensions get partitioned even at small numbers of ranks.
    // gatherScatterTest( partitioner, is_dim_periodic );
    if ( ranks_per_dim[0] != ranks_per_dim[1] )
    {
        std::swap( ranks_per_dim[0], ranks_per_dim[1] );
        partitioner = ManualBlockPartitioner<2>( ranks_per_dim );
        gatherScatterTest<commspace_type>( partitioner, is_dim_periodic );
    }
}

//---------------------------------------------------------------------------//
TYPED_TEST( Halo2dTypedTest, ScatterReduceMax2d )
{
    // Extract communication backend type
    using commspace_type = TypeParam;

    scatterReduceTest<commspace_type>( ScatterReduce::Max() );
}

//---------------------------------------------------------------------------//
TYPED_TEST( Halo2dTypedTest, ScatterReduceMin2d )
{
    // Extract communication backend type
    using commspace_type = TypeParam;

    scatterReduceTest<commspace_type>( ScatterReduce::Min() );
}

//---------------------------------------------------------------------------//
TYPED_TEST( Halo2dTypedTest, ScatterReduceReplace2d )
{
    // Extract communication backend type
    using commspace_type = TypeParam;

    scatterReduceTest<commspace_type>( ScatterReduce::Replace() );
}

//---------------------------------------------------------------------------//

} // end namespace Test
