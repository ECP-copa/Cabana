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

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

namespace Test
{
//---------------------------------------------------------------------------//

void LayoutHilbert2DSubviewTest()
{
    // typedef
    typedef
        typename Kokkos::View<double ****, Kokkos::LayoutHilbert2D, TEST_DEVICE>
            view_type;

    // Set dimensions
    int dim1 = 27;
    int dim2 = 51;
    int dim3 = 1;
    int dim4 = 2;

    // View Index Space
    auto view_space =
        Cajita::IndexSpace<4>( {0, 0, 0, 0}, {dim1, dim2, dim3, dim4} );

    // Create Hilbert View
    Kokkos::View<double ****, Kokkos::LayoutHilbert2D, TEST_DEVICE>
        HilbertArray( "Hilbert", dim1, dim2, dim3, dim4 );

    // Create Regular View
    Kokkos::View<double ****, TEST_DEVICE> RegularArray( "Regular", dim1, dim2,
                                                         dim3, dim4 );

    // Loop over both views and assign values ( in typical increase LayoutRight
    // order )
    Kokkos::parallel_for(
        "Initialize",
        Cajita::createExecutionPolicy( view_space,
                                       view_type::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ) {
            HilbertArray( i, j, k, l ) =
                i + dim1 * ( j + dim2 * ( k + (dim3)*l ) );
            RegularArray( i, j, k, l ) =
                i + dim1 * ( j + dim2 * ( k + (dim3)*l ) );
        } );
    view_type::execution_space().fence();

    // Create copies on host to check
    auto host_view_hilbert = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), HilbertArray );
    auto host_view_regular = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), RegularArray );

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
    space = Cajita::IndexSpace<4>( {0, 0, 0, 0}, {2, dim2, dim3, dim4} );

    // Create Hilbert subview from Hilbert View
    auto HilbertSub =
        Kokkos::subview( HilbertArray, space.range( 0 ), space.range( 1 ),
                         space.range( 2 ), space.range( 3 ) );

    // Create Regular subview from Regular View
    auto RegularSub =
        Kokkos::subview( RegularArray, space.range( 0 ), space.range( 1 ),
                         space.range( 2 ), space.range( 3 ) );

    // Set replacement value
    int replaceVal = 7012;

    // Create Small Regular View the same dimensions as the subview
    Kokkos::View<double ****, TEST_DEVICE> RegularSmall(
        "RegularSmall", space.extent( 0 ), space.extent( 1 ), space.extent( 2 ),
        space.extent( 3 ) );

    // Loop over all indices in Small Regular View and set each value to
    // replacement value
    Kokkos::parallel_for(
        "SmallInitialize",
        Cajita::createExecutionPolicy( space, view_type::execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int l ) {
            RegularSmall( i, j, k, l ) = replaceVal;
        } );
    view_type::execution_space().fence();

    // Deep copy Small Regular View over to the Hilbert Subview
    Kokkos::deep_copy( HilbertSub, RegularSmall );

    // Create copy on host to check
    auto host_view_hilbert_new = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), HilbertArray );

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
                                   replaceVal );
                    else
                        EXPECT_EQ( host_view_hilbert_new( i, j, k, l ),
                                   i + dim1 * ( j + dim2 * ( k + (dim3)*l ) ) );
}

//---------------------------------------------------------------------------//
void LayoutHilbert2DGatherTest( const Cajita::ManualPartitioner &partitioner,
                                const std::array<bool, 3> &is_dim_periodic )
{
    // typedef
    typedef
        typename Kokkos::View<double ****, Kokkos::LayoutHilbert2D, TEST_DEVICE>
            view_type;

    // Get rank
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // Define Cell Size and Number of Halo Cells in Each Direction
    double cell_size = 0.25;
    int halo_width = 2;

    // Set grid information
    std::array<int, 3> global_num_cell = {104, 104, 1};
    std::array<double, 3> global_low_corner = {0.0, 0.0, 0.0};
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2]};

    // Create local grid
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );
    auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );

    // Create vector layout with 2 dofs
    auto cell_vector_layout =
        Cajita::createArrayLayout( local_grid, 2, Cajita::Cell() );

    // Create array with LayoutHilbert2D
    auto array =
        Cajita::createArray<double, Kokkos::LayoutHilbert2D, TEST_DEVICE>(
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
    Kokkos::View<unsigned int **, TEST_DEVICE> owned_shared_spaces(
        "Owned_Shared_Spaces", 27, 6 );
    Kokkos::View<unsigned int **, TEST_DEVICE> ghosted_shared_spaces(
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

    // TODO: ArrayOp does not appear to work correctly
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
    auto host_view = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                          array->view() );

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
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( layout_hilbert, layout_hilbert2d_test )
{
    // Test Subview Functionality
    LayoutHilbert2DSubviewTest();

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
    std::array<int, 3> ranks_per_dim = {x_ranks, y_ranks, 1};

    // Create 2-D partitioner
    Cajita::ManualPartitioner partitioner( ranks_per_dim );

    // Test the non-periodic case
    std::array<bool, 3> is_dim_periodic = {false, false, false};

    LayoutHilbert2DGatherTest( partitioner, is_dim_periodic );
}

//---------------------------------------------------------------------------//

} // namespace Test