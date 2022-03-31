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
#include <Cajita_Hilbert.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_ManualPartitioner.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

namespace Test
{
//---------------------------------------------------------------------------//
void LayoutHilbert3DSubviewTest()
{
    // typedef
    typedef
        typename Kokkos::View<double****, Kokkos::LayoutHilbert3D, TEST_DEVICE>
            view_type;

    // typedef
    typedef typename Kokkos::View<double****, TEST_DEVICE> buff_type;

    // Set dimensions
    int dim1 = 45;
    int dim2 = 51;
    int dim3 = 1;
    int dim4 = 2;

    // View Index Space
    auto view_space =
        Cajita::IndexSpace<4>( { 0, 0, 0, 0 }, { dim1, dim2, dim3, dim4 } );

    // Create Hilbert View
    Kokkos::View<double****, Kokkos::LayoutHilbert3D, TEST_DEVICE>
        hilbert_array( "Hilbert", dim1, dim2, dim3, dim4 );

    // Duplicate Hilbert View
    Kokkos::View<double****, Kokkos::LayoutHilbert3D, TEST_DEVICE>
        hilbert_array2( "Hilbert", dim1, dim2, dim3, dim4 );

    // Test shallow copy and dimension methods
    hilbert_array2 = hilbert_array;

    // Create Regular View
    Kokkos::View<double****, TEST_DEVICE> regular_array( "Regular", dim1, dim2,
                                                         dim3, dim4 );

    // Loop over both views and assign values ( in typical increase LayoutRight
    // order )
    Kokkos::parallel_for(
        "Initialize",
        Cajita::createExecutionPolicy( view_space,
                                       view_type::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ) {
            hilbert_array( i, j, k, l ) =
                i + dim1 * ( j + dim2 * ( k + (dim3)*l ) );
            regular_array( i, j, k, l ) =
                i + dim1 * ( j + dim2 * ( k + (dim3)*l ) );
        } );
    view_type::execution_space().fence();

    // Check that the Hilbert View has been assigned consistently with the
    // Regular Array
    for ( int i = 0; i < dim1; i++ )
        for ( int j = 0; j < dim2; j++ )
            for ( int k = 0; k < dim3; k++ )
                for ( int l = 0; l < dim4; l++ )
                    EXPECT_EQ( hilbert_array( i, j, k, l ),
                               regular_array( i, j, k, l ) );

    // Create subview index space - mimicking a halo subview of width 2
    Cajita::IndexSpace<4> space;
    space = Cajita::IndexSpace<4>( { 0, 0, 0, 0 }, { 2, dim2, dim3, dim4 } );

    // Create Hilbert subview from Hilbert View
    auto hilbert_sub =
        Kokkos::subview( hilbert_array, space.range( 0 ), space.range( 1 ),
                         space.range( 2 ), space.range( 3 ) );

    // Create Regular subview from Regular View
    auto regular_sub =
        Kokkos::subview( regular_array, space.range( 0 ), space.range( 1 ),
                         space.range( 2 ), space.range( 3 ) );

    // Set replacement value
    int replace_val = 7012;

    // Create Small Regular View the same dimensions as the subview
    Kokkos::View<double****, TEST_DEVICE> regular_small(
        "RegularSmall", space.extent( 0 ), space.extent( 1 ), space.extent( 2 ),
        space.extent( 3 ) );

    // Loop over all indices in Small Regular View and set each value to
    // replacement value
    Kokkos::parallel_for(
        "SmallInitialize",
        Cajita::createExecutionPolicy( space, view_type::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ) {
            regular_small( i, j, k, l ) = replace_val;
        } );
    view_type::execution_space().fence();

    // Deep copy Small Regular View over to the Hilbert Subview
    Kokkos::deep_copy( hilbert_sub, regular_small );

    // Check that the replacement value got copied over correctly from the Small
    // Regular View, to the Hilbert Subview and hence the original Hilbert view
    for ( int i = 0; i < dim1; i++ )
        for ( int j = 0; j < dim2; j++ )
            for ( int k = 0; k < dim3; k++ )
                for ( int l = 0; l < dim4; l++ )
                    if ( i >= space.min( 0 ) && i < space.max( 0 ) &&
                         j >= space.min( 1 ) && j < space.max( 1 ) &&
                         k >= space.min( 2 ) && k < space.max( 2 ) &&
                         l >= space.min( 3 ) && l < space.max( 3 ) )
                        EXPECT_EQ( hilbert_array( i, j, k, l ), replace_val );
                    else
                        EXPECT_EQ( hilbert_array( i, j, k, l ),
                                   i + dim1 * ( j + dim2 * ( k + (dim3)*l ) ) );
}

//---------------------------------------------------------------------------//
void LayoutHilbert3DArrayOpTest()
{
    // typedef
    typedef typename Kokkos::View<double****, TEST_DEVICE> buff_type;

    // Let MPI compute the partitioning for this test.
    Cajita::UniformDimPartitioner partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 101, 85, 99 };
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    // Create an array layout on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout = Cajita::createArrayLayout(
        global_grid, halo_width, dofs_per_cell, Cajita::Cell() );

    // Create an array.
    std::string label( "test_array" );
    auto array =
        Cajita::createArray<double, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
            label, cell_layout );

    // Assign a value to the entire the array.
    Cajita::ArrayOp::assign( *array, 2.0, Cajita::Ghost() );

    // Create copy on host to check
    buff_type dev_view( "dev_view", array->view().extent( 0 ),
                        array->view().extent( 1 ), array->view().extent( 2 ),
                        array->view().extent( 3 ) );
    auto host_view = Kokkos::create_mirror( dev_view );

    Kokkos::deep_copy( dev_view, array->view() );
    Kokkos::deep_copy( host_view, dev_view );

    auto ghosted_space =
        array->layout()->indexSpace( Cajita::Ghost(), Cajita::Local() );
    for ( long i = 0; i < ghosted_space.extent( Cajita::Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Cajita::Dim::J ); ++j )
            for ( long k = 0; k < ghosted_space.extent( Cajita::Dim::K ); ++k )
                for ( long l = 0; l < ghosted_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_view( i, j, k, l ), 2.0 );
}

//---------------------------------------------------------------------------//
// Halo padding in each dimension for different entity types.
int haloPad( Cajita::Cell, int ) { return 0; }

int haloPad( Cajita::Node, int ) { return 1; }

template <int D>
int haloPad( Cajita::Face<D>, int d )
{
    return ( d == D ) ? 1 : 0;
}

template <int D>
int haloPad( Cajita::Edge<D>, int d )
{
    return ( d == D ) ? 0 : 1;
}

//---------------------------------------------------------------------------//
// Check initial array gather. We should get 1 everywhere in the array now
// where there was ghost overlap. Otherwise there will still be 0.
template <class Array>
void checkGather( const std::array<bool, 3>& is_dim_periodic,
                  const int halo_width, const Array& array )
{
    auto owned_space =
        array.layout()->indexSpace( Cajita::Own(), Cajita::Local() );
    auto ghosted_space =
        array.layout()->indexSpace( Cajita::Ghost(), Cajita::Local() );
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          array.view() );
    auto pad_i = haloPad( typename Array::entity_type(), Cajita::Dim::I );
    auto pad_j = haloPad( typename Array::entity_type(), Cajita::Dim::J );
    auto pad_k = haloPad( typename Array::entity_type(), Cajita::Dim::K );
    const auto& global_grid = array.layout()->localGrid()->globalGrid();

    // This function checks if an index is in the low boundary halo in the
    // given dimension
    auto in_boundary_min_halo = [&]( const int i, const int dim ) {
        if ( is_dim_periodic[dim] || !global_grid.onLowBoundary( dim ) )
            return false;
        else
            return ( i < owned_space.min( dim ) );
    };

    // This function checks if an index is in the high boundary halo of in the
    // given dimension
    auto in_boundary_max_halo = [&]( const int i, const int dim ) {
        if ( is_dim_periodic[dim] || !global_grid.onHighBoundary( dim ) )
            return false;
        else
            return ( i >= owned_space.max( dim ) );
    };

    // Check the gather.
    for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
        for ( unsigned j = 0; j < ghosted_space.extent( 1 ); ++j )
            for ( unsigned k = 0; k < ghosted_space.extent( 2 ); ++k )
                for ( unsigned l = 0; l < ghosted_space.extent( 3 ); ++l )
                    if ( in_boundary_min_halo( i, Cajita::Dim::I ) ||
                         in_boundary_min_halo( j, Cajita::Dim::J ) ||
                         in_boundary_min_halo( k, Cajita::Dim::K ) ||
                         in_boundary_max_halo( i, Cajita::Dim::I ) ||
                         in_boundary_max_halo( j, Cajita::Dim::J ) ||
                         in_boundary_max_halo( k, Cajita::Dim::K ) )
                    {
                        EXPECT_DOUBLE_EQ( host_view( i, j, k, l ), 0.0 );
                    }
                    else if ( i < owned_space.min( Cajita::Dim::I ) -
                                      halo_width ||
                              i >= owned_space.max( Cajita::Dim::I ) +
                                       halo_width + pad_i ||
                              j < owned_space.min( Cajita::Dim::J ) -
                                      halo_width ||
                              j >= owned_space.max( Cajita::Dim::J ) +
                                       halo_width + pad_j ||
                              k < owned_space.min( Cajita::Dim::K ) -
                                      halo_width ||
                              k >= owned_space.max( Cajita::Dim::K ) +
                                       halo_width + pad_k )
                    {
                        EXPECT_DOUBLE_EQ( host_view( i, j, k, l ), 0.0 );
                    }
                    else
                    {
                        EXPECT_DOUBLE_EQ( host_view( i, j, k, l ), 1.0 );
                    }
}

//---------------------------------------------------------------------------//
// Check array scatter. The value of the cell should be a function of how many
// neighbors it has. Corner neighbors get 8, edge neighbors get 4, face
// neighbors get 2, and no neighbors remain at 1.
template <class Array>
void checkScatter( const std::array<bool, 3>& is_dim_periodic,
                   const int halo_width, const Array& array )
{
    // Get data.
    auto owned_space =
        array.layout()->indexSpace( Cajita::Own(), Cajita::Local() );
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          array.view() );
    const auto& global_grid = array.layout()->localGrid()->globalGrid();

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
                if ( in_dim_min_halo( i, Cajita::Dim::I ) ||
                     in_dim_max_halo( i, Cajita::Dim::I ) )
                    ++num_n;
                if ( in_dim_min_halo( j, Cajita::Dim::J ) ||
                     in_dim_max_halo( j, Cajita::Dim::J ) )
                    ++num_n;
                if ( in_dim_min_halo( k, Cajita::Dim::K ) ||
                     in_dim_max_halo( k, Cajita::Dim::K ) )
                    ++num_n;
                double scatter_val = std::pow( 2.0, num_n );
                for ( unsigned l = 0; l < owned_space.extent( 3 ); ++l )
                    EXPECT_EQ( host_view( i, j, k, l ), scatter_val );
            }
}

//---------------------------------------------------------------------------//
void gatherScatterTest( const Cajita::ManualBlockPartitioner<3>& partitioner,
                        const std::array<bool, 3>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 17, 20, 21 };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );

    // Create the global grid.
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    // Array parameters.
    unsigned array_halo_width = 3;

    // Loop over halo sizes up to the size of the array halo width.
    for ( unsigned halo_width = 1; halo_width <= array_halo_width;
          ++halo_width )
    {
        // Create a cell array.
        auto layout = Cajita::createArrayLayout( global_grid, array_halo_width,
                                                 4, Cajita::Cell() );
        auto array =
            Cajita::createArray<double, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
                "array", layout );

        // Assign the owned cells a value of 1 and the rest 0.
        Cajita::ArrayOp::assign( *array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *array, 1.0, Cajita::Own() );

        // Create a halo.
        auto halo =
            Cajita::createHalo( *array, Cajita::FullHaloPattern(), halo_width );

        // Gather into the ghosts.
        halo->gather( TEST_EXECSPACE(), *array );

        // Check the gather.
        checkGather( is_dim_periodic, halo_width, *array );

        // Scatter from the ghosts back to owned.
        halo->scatter( TEST_EXECSPACE(), Cajita::ScatterReduce::Sum(), *array );

        // Check the scatter.
        checkScatter( is_dim_periodic, halo_width, *array );
    }

    // Repeat the process but this time with multiple arrays in a Halo
    for ( unsigned halo_width = 1; halo_width <= array_halo_width;
          ++halo_width )
    {
        // Create arrays of different layouts and dof counts.
        auto cell_layout = Cajita::createArrayLayout(
            global_grid, array_halo_width, 4, Cajita::Cell() );
        auto cell_array =
            Cajita::createArray<double, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
                "cell_array", cell_layout );

        auto node_layout = Cajita::createArrayLayout(
            global_grid, array_halo_width, 3, Cajita::Node() );
        auto node_array =
            Cajita::createArray<float, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
                "node_array", node_layout );

        auto face_i_layout = Cajita::createArrayLayout(
            global_grid, array_halo_width, 4, Cajita::Face<Cajita::Dim::I>() );
        auto face_i_array =
            Cajita::createArray<double, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
                "face_i_array", face_i_layout );

        auto face_j_layout = Cajita::createArrayLayout(
            global_grid, array_halo_width, 1, Cajita::Face<Cajita::Dim::J>() );
        auto face_j_array =
            Cajita::createArray<double, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
                "face_j_array", face_j_layout );

        auto face_k_layout = Cajita::createArrayLayout(
            global_grid, array_halo_width, 2, Cajita::Face<Cajita::Dim::K>() );
        auto face_k_array =
            Cajita::createArray<float, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
                "face_k_array", face_k_layout );

        auto edge_i_layout = Cajita::createArrayLayout(
            global_grid, array_halo_width, 3, Cajita::Edge<Cajita::Dim::I>() );
        auto edge_i_array =
            Cajita::createArray<float, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
                "edge_i_array", edge_i_layout );

        auto edge_j_layout = Cajita::createArrayLayout(
            global_grid, array_halo_width, 2, Cajita::Edge<Cajita::Dim::J>() );
        auto edge_j_array =
            Cajita::createArray<float, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
                "edge_j_array", edge_j_layout );

        auto edge_k_layout = Cajita::createArrayLayout(
            global_grid, array_halo_width, 1, Cajita::Edge<Cajita::Dim::K>() );
        auto edge_k_array =
            Cajita::createArray<double, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
                "edge_k_array", edge_k_layout );

        // Assign the owned cells a value of 1 and the rest 0.
        Cajita::ArrayOp::assign( *cell_array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *cell_array, 1.0, Cajita::Own() );

        Cajita::ArrayOp::assign( *node_array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *node_array, 1.0, Cajita::Own() );

        Cajita::ArrayOp::assign( *face_i_array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *face_i_array, 1.0, Cajita::Own() );

        Cajita::ArrayOp::assign( *face_j_array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *face_j_array, 1.0, Cajita::Own() );

        Cajita::ArrayOp::assign( *face_k_array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *face_k_array, 1.0, Cajita::Own() );

        Cajita::ArrayOp::assign( *edge_i_array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *edge_i_array, 1.0, Cajita::Own() );

        Cajita::ArrayOp::assign( *edge_j_array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *edge_j_array, 1.0, Cajita::Own() );

        Cajita::ArrayOp::assign( *edge_k_array, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *edge_k_array, 1.0, Cajita::Own() );

        // Create a multihalo.
        auto halo = Cajita::createHalo(
            Cajita::FullHaloPattern(), halo_width, *cell_array, *node_array,
            *face_i_array, *face_j_array, *face_k_array, *edge_i_array,
            *edge_j_array, *edge_k_array );

        // Gather into the ghosts.
        halo->gather( TEST_EXECSPACE(), *cell_array, *node_array, *face_i_array,
                      *face_j_array, *face_k_array, *edge_i_array,
                      *edge_j_array, *edge_k_array );

        // Check the gather.
        checkGather( is_dim_periodic, halo_width, *cell_array );
        checkGather( is_dim_periodic, halo_width, *node_array );
        checkGather( is_dim_periodic, halo_width, *face_i_array );
        checkGather( is_dim_periodic, halo_width, *face_j_array );
        checkGather( is_dim_periodic, halo_width, *face_k_array );
        checkGather( is_dim_periodic, halo_width, *edge_i_array );
        checkGather( is_dim_periodic, halo_width, *edge_j_array );
        checkGather( is_dim_periodic, halo_width, *edge_k_array );

        // Scatter from the ghosts back to owned.
        halo->scatter( TEST_EXECSPACE(), Cajita::ScatterReduce::Sum(),
                       *cell_array, *node_array, *face_i_array, *face_j_array,
                       *face_k_array, *edge_i_array, *edge_j_array,
                       *edge_k_array );

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
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( layout_hilbert, layout_hilbert_subview_test )
{
    // Test Subview Functionality
    LayoutHilbert3DSubviewTest();
}

TEST( layout_hilbert, layout_hilbert_arrayop_test )
{
    // ArrayOp Test
    LayoutHilbert3DArrayOpTest();
}

TEST( TEST_CATEGORY, not_periodic_test )
{
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    Cajita::ManualBlockPartitioner<3> partitioner( ranks_per_dim );

    // Boundaries are not periodic.
    std::array<bool, 3> is_dim_periodic = { false, false, false };

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
    Cajita::ManualBlockPartitioner<3> partitioner( ranks_per_dim );

    // Every boundary is periodic
    std::array<bool, 3> is_dim_periodic = { true, true, true };

    gatherScatterTest( partitioner, is_dim_periodic );
}

//---------------------------------------------------------------------------//

} // namespace Test