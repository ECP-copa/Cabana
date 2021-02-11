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

    // Create copies on host to check
    buff_type dev_view( "dev_view", hilbert_array.extent( 0 ),
                        hilbert_array.extent( 1 ), hilbert_array.extent( 2 ),
                        hilbert_array.extent( 3 ) );
    auto host_view_hilbert = Kokkos::create_mirror( dev_view );

    Kokkos::deep_copy( dev_view, hilbert_array );
    Kokkos::deep_copy( host_view_hilbert, dev_view );

    auto host_view_regular = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), regular_array );

    // Check that the Hilbert View has been assigned consistently with the
    // Regular Array
    for ( int i = 0; i < dim1; i++ )
        for ( int j = 0; j < dim2; j++ )
            for ( int k = 0; k < dim3; k++ )
                for ( int l = 0; l < dim4; l++ )
                    EXPECT_EQ( host_view_hilbert( i, j, k, l ),
                               host_view_regular( i, j, k, l ) );

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

    // Create copy on host to check
    buff_type dev_view_new(
        "dev_view_new", hilbert_array.extent( 0 ), hilbert_array.extent( 1 ),
        hilbert_array.extent( 2 ), hilbert_array.extent( 3 ) );
    auto host_view_hilbert_new = Kokkos::create_mirror( dev_view_new );

    Kokkos::deep_copy( dev_view_new, hilbert_array );
    Kokkos::deep_copy( host_view_hilbert_new, dev_view_new );

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
                        EXPECT_EQ( host_view_hilbert_new( i, j, k, l ),
                                   replace_val );
                    else
                        EXPECT_EQ( host_view_hilbert_new( i, j, k, l ),
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
void LayoutHilbert3DGatherTest( const Cajita::ManualPartitioner& partitioner,
                                const std::array<bool, 3>& is_dim_periodic )
{
    // typedef
    typedef
        typename Kokkos::View<double****, Kokkos::LayoutHilbert3D, TEST_DEVICE>
            view_type;

    // typedef
    typedef typename Kokkos::View<double****, TEST_DEVICE> buff_type;

    // Get rank
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // Define Cell Size and Number of Halo Cells in Each Direction
    double cell_size = 0.25;
    int halo_width = 2;

    // Set grid information
    std::array<int, 3> global_num_cell = { 104, 104, 1 };
    std::array<double, 3> global_low_corner = { 0.0, 0.0, 0.0 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };

    // Create local grid
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );
    auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );

    // Create vector layout with 2 dofs
    auto cell_vector_layout =
        Cajita::createArrayLayout( local_grid, 2, Cajita::Cell() );

    // Create array with LayoutHilbert3D
    auto array =
        Cajita::createArray<double, Kokkos::LayoutHilbert3D, TEST_DEVICE>(
            "array", cell_vector_layout );

    // Create halo
    auto halo = createHalo( Cajita::FullHaloPattern(), halo_width, *array );

    // Get owned and ghosted index spaces
    auto owned_space =
        cell_vector_layout->indexSpace( Cajita::Own(), Cajita::Local() );
    auto ghosted_space =
        cell_vector_layout->indexSpace( Cajita::Ghost(), Cajita::Local() );

    // Get underlying view for assignment
    auto array_view = array->view();

    // Generate Kokkos Views to store neighbor data for later use
    Kokkos::View<unsigned int**, TEST_DEVICE> owned_shared_spaces(
        "Owned_Shared_Spaces", 27, 6 );
    Kokkos::View<unsigned int**, TEST_DEVICE> ghosted_shared_spaces(
        "Ghosted_Shared_Spaces", 27, 6 );

    // Create copies on host to populate
    auto host_owned_shared_spaces = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), owned_shared_spaces );
    auto host_ghosted_shared_spaces = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), ghosted_shared_spaces );

    int num_neighbors = 0;
    // Loop over all possible neighbors
    for ( int i = -1; i < 2; ++i )
    {
        for ( int j = -1; j < 2; ++j )
        {
            for ( int k = -1; k < 2; ++k )
            {
                if ( !( i == 0 && j == 0 && k == 0 ) )
                {
                    int neighbor = local_grid->neighborRank( i, j, k );
                    // Only if neighbor exists
                    if ( neighbor != -1 )
                    {
                        // Ghost cells we are receiving ( in our ghost space,
                        // but in our neighbors owned space )
                        auto shared_recv_cells = local_grid->sharedIndexSpace(
                            Cajita::Ghost(), Cajita::Cell(), i, j, k );

                        // Cells we are sending ( in our owned space, but in our
                        // neighbors ghost space )
                        auto shared_send_cells = local_grid->sharedIndexSpace(
                            Cajita::Own(), Cajita::Cell(), i, j, k );

                        // Add index spaces to host mirror views
                        host_owned_shared_spaces( num_neighbors, 0 ) =
                            shared_send_cells.min( 0 );
                        host_owned_shared_spaces( num_neighbors, 1 ) =
                            shared_send_cells.min( 1 );
                        host_owned_shared_spaces( num_neighbors, 2 ) =
                            shared_send_cells.min( 2 );
                        host_owned_shared_spaces( num_neighbors, 3 ) =
                            shared_send_cells.max( 0 );
                        host_owned_shared_spaces( num_neighbors, 4 ) =
                            shared_send_cells.max( 1 );
                        host_owned_shared_spaces( num_neighbors, 5 ) =
                            shared_send_cells.max( 2 );

                        host_ghosted_shared_spaces( num_neighbors, 0 ) =
                            shared_recv_cells.min( 0 );
                        host_ghosted_shared_spaces( num_neighbors, 1 ) =
                            shared_recv_cells.min( 1 );
                        host_ghosted_shared_spaces( num_neighbors, 2 ) =
                            shared_recv_cells.min( 2 );
                        host_ghosted_shared_spaces( num_neighbors, 3 ) =
                            shared_recv_cells.max( 0 );
                        host_ghosted_shared_spaces( num_neighbors, 4 ) =
                            shared_recv_cells.max( 1 );
                        host_ghosted_shared_spaces( num_neighbors, 5 ) =
                            shared_recv_cells.max( 2 );

                        // Increase neighbor count
                        num_neighbors++;
                    }
                }
            }
        }
    }

    // Deep copy from host to device
    Kokkos::deep_copy( owned_shared_spaces, host_owned_shared_spaces );
    Kokkos::deep_copy( ghosted_shared_spaces, host_ghosted_shared_spaces );

    // Value to set to Cells we are sending ( in our owned space, but in our
    // neighbors ghost space )
    double shared_halo_value = 2.0;

    // Set view values such that
    // My ghost cells = 0.0
    // Cells we are sending ( in our owned space, but in our neighbors ghost
    // space ) = shared_halo_value Our remaining owned cells = 1.0 Loop over
    // entire index space of local view ( owned + ghost cells )
    Kokkos::parallel_for(
        "HilbertInitialize",
        Cajita::createExecutionPolicy( ghosted_space,
                                       view_type::execution_space() ),
        KOKKOS_LAMBDA( const unsigned i, const unsigned j, const unsigned k,
                       const unsigned l ) {
            // My ghost cells = 0.0
            if ( i < owned_space.min( Cajita::Dim::I ) ||
                 i >= owned_space.max( Cajita::Dim::I ) ||
                 j < owned_space.min( Cajita::Dim::J ) ||
                 j >= owned_space.max( Cajita::Dim::J ) ||
                 k < owned_space.min( Cajita::Dim::K ) ||
                 k >= owned_space.max( Cajita::Dim::K ) )
                array_view( i, j, k, l ) = 0.0;
            else
            {
                // Loop over all neighbors
                for ( int n = 0; n < num_neighbors; n++ )
                {
                    // Get shared index space with current neighbor
                    auto shared_min0 = owned_shared_spaces( n, 0 );
                    auto shared_min1 = owned_shared_spaces( n, 1 );
                    auto shared_min2 = owned_shared_spaces( n, 2 );
                    auto shared_max0 = owned_shared_spaces( n, 3 );
                    auto shared_max1 = owned_shared_spaces( n, 4 );
                    auto shared_max2 = owned_shared_spaces( n, 5 );

                    // Cells we are sending ( in our owned space, but in
                    // our neighbors ghost space ) = shared_halo_value
                    if ( i >= shared_min0 && i < shared_max0 &&
                         j >= shared_min1 && j < shared_max1 &&
                         k >= shared_min2 && k < shared_max2 )
                        array_view( i, j, k, l ) = shared_halo_value;
                }
                // Our remaining owned cells = 1.0
                if ( array_view( i, j, k, l ) != shared_halo_value )
                    array_view( i, j, k, l ) = 1.0;
            }
        } );
    view_type::execution_space().fence();

    // Gather
    halo->gather( TEST_EXECSPACE(), *array );

    // Create copy on host to check
    buff_type dev_view( "dev_view", array->view().extent( 0 ),
                        array->view().extent( 1 ), array->view().extent( 2 ),
                        array->view().extent( 3 ) );
    auto host_view = Kokkos::create_mirror( dev_view );

    Kokkos::deep_copy( dev_view, array->view() );
    Kokkos::deep_copy( host_view, dev_view );

    // Test if Halo succeeded as expected
    // Result should be:
    // My ghost cells = shared_halo_value ( We check )
    // Cells we are sending ( in our owned space, but in our neighbors ghost
    // space ) = shared_halo_value ( We check ) Our remaining owned cells = 1.0
    // ( We don't check ) Loop over entire index space of local view ( owned +
    // ghost cells )
    for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
    {
        for ( unsigned j = 0; j < ghosted_space.extent( 1 ); ++j )
        {
            for ( unsigned k = 0; k < ghosted_space.extent( 2 ); ++k )
            {
                for ( unsigned l = 0; l < ghosted_space.extent( 3 ); ++l )
                {
                    // My ghost cells = shared_halo_value
                    if ( i < owned_space.min( Cajita::Dim::I ) ||
                         i >= owned_space.max( Cajita::Dim::I ) ||
                         j < owned_space.min( Cajita::Dim::J ) ||
                         j >= owned_space.max( Cajita::Dim::J ) ||
                         k < owned_space.min( Cajita::Dim::K ) ||
                         k >= owned_space.max( Cajita::Dim::K ) )
                    {
                        EXPECT_EQ( host_view( i, j, k, l ), shared_halo_value );
                    }
                    else
                    {
                        // Loop over all neighbors
                        for ( int n = 0; n < num_neighbors; n++ )
                        {
                            // Get shared index space with current neighbor
                            auto shared_min0 = host_owned_shared_spaces( n, 0 );
                            auto shared_min1 = host_owned_shared_spaces( n, 1 );
                            auto shared_min2 = host_owned_shared_spaces( n, 2 );
                            auto shared_max0 = host_owned_shared_spaces( n, 3 );
                            auto shared_max1 = host_owned_shared_spaces( n, 4 );
                            auto shared_max2 = host_owned_shared_spaces( n, 5 );

                            // Cells we are sending ( in our owned space, but in
                            // our neighbors ghost space ) = shared_halo_value
                            if ( i >= shared_min0 && i < shared_max0 &&
                                 j >= shared_min1 && j < shared_max1 &&
                                 k >= shared_min2 && k < shared_max2 )
                            {
                                EXPECT_EQ( host_view( i, j, k, l ),
                                           shared_halo_value );
                            }
                        }
                    }
                }
            }
        }
    }
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
void checkGather( const int halo_width, const Array& array )
{
    // typedef
    typedef typename Kokkos::View<double****, TEST_DEVICE> buff_type;

    auto owned_space =
        array.layout()->indexSpace( Cajita::Own(), Cajita::Local() );
    auto ghosted_space =
        array.layout()->indexSpace( Cajita::Ghost(), Cajita::Local() );

    // Create copy on host to check
    buff_type dev_view( "dev_view", array.view().extent( 0 ),
                        array.view().extent( 1 ), array.view().extent( 2 ),
                        array.view().extent( 3 ) );
    auto host_view = Kokkos::create_mirror( dev_view );

    Kokkos::deep_copy( dev_view, array.view() );
    Kokkos::deep_copy( host_view, dev_view );

    auto pad_i = haloPad( typename Array::entity_type(), Cajita::Dim::I );
    auto pad_j = haloPad( typename Array::entity_type(), Cajita::Dim::J );
    auto pad_k = haloPad( typename Array::entity_type(), Cajita::Dim::K );
    for ( unsigned i = 0; i < ghosted_space.extent( 0 ); ++i )
        for ( unsigned j = 0; j < ghosted_space.extent( 1 ); ++j )
            for ( unsigned k = 0; k < ghosted_space.extent( 2 ); ++k )
                for ( unsigned l = 0; l < ghosted_space.extent( 3 ); ++l )
                    if ( i < owned_space.min( Cajita::Dim::I ) - halo_width ||
                         i >= owned_space.max( Cajita::Dim::I ) + halo_width +
                                  pad_i ||
                         j < owned_space.min( Cajita::Dim::J ) - halo_width ||
                         j >= owned_space.max( Cajita::Dim::J ) + halo_width +
                                  pad_j ||
                         k < owned_space.min( Cajita::Dim::K ) - halo_width ||
                         k >= owned_space.max( Cajita::Dim::K ) + halo_width +
                                  pad_k )
                        EXPECT_EQ( host_view( i, j, k, l ), 0.0 );
                    else
                        EXPECT_EQ( host_view( i, j, k, l ), 1.0 );
}

//---------------------------------------------------------------------------//
// Check array scatter. The value of the cell should be a function of how many
// neighbors it has. Corner neighbors get 8, edge neighbors get 4, face
// neighbors get 2, and no neighbors remain at 1.
template <class Array>
void checkScatter( const std::array<bool, 3>& is_dim_periodic,
                   const int halo_width, const Array& array )
{
    // typedef
    typedef typename Kokkos::View<double****, TEST_DEVICE> buff_type;

    // Get data.
    auto owned_space =
        array.layout()->indexSpace( Cajita::Own(), Cajita::Local() );

    // Create copy on host to check
    buff_type dev_view( "dev_view", array.view().extent( 0 ),
                        array.view().extent( 1 ), array.view().extent( 2 ),
                        array.view().extent( 3 ) );
    auto host_view = Kokkos::create_mirror( dev_view );

    Kokkos::deep_copy( dev_view, array.view() );
    Kokkos::deep_copy( host_view, dev_view );

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
void LayoutHilbert3DScatterTest( const Cajita::ManualPartitioner& partitioner,
                                 const std::array<bool, 3>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 32, 23, 41 };
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
        checkGather( halo_width, *array );

        // Scatter from the ghosts back to owned.
        halo->scatter( TEST_EXECSPACE(), Cajita::ScatterReduce::Sum(), *array );

        // Check the scatter.
        checkScatter( is_dim_periodic, halo_width, *array );
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

TEST( layout_hilbert, layout_hilbert_gather_test )
{
    // Test Halo Gather Routine
    // Let MPI compute the partitioning for this test.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    // Perform a 2-D partitioning for now
    int x_ranks = comm_size;
    while ( x_ranks % 2 == 0 && x_ranks > 2 )
    {
        x_ranks /= 2;
    }
    int y_ranks = comm_size / x_ranks;
    std::array<int, 3> ranks_per_dim = { x_ranks, y_ranks, 1 };

    // Create 2-D partitioner
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    // Test the non-periodic case
    std::array<bool, 3> dim_not_periodic = { false, false, false };

    // Gather Test
    LayoutHilbert3DGatherTest( partitioner, dim_not_periodic );

    // Scatter Test
    LayoutHilbert3DScatterTest( partitioner, dim_not_periodic );

    // Test the periodic case
    // std::array<bool, 3> dim_periodic = {true, true, true};

    // Gather Test
    // LayoutHilbert3DGatherTest( partitioner, dim_periodic );

    // Scatter Test
    // LayoutHilbert3DScatterTest( partitioner, dim_periodic );
}

//---------------------------------------------------------------------------//

} // namespace Test