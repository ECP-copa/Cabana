/****************************************************************************
 * Copyright (c) 2019-2020 by the Cajita authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita_Halo.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_Array.hpp>
#include <Cajita_ManualPartitioner.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <cmath>
#include <array>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void gatherScatterTest( const ManualPartitioner& partitioner,
                        const std::array<bool,3>& is_dim_periodic )
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int,3> global_num_cell = { 32, 23, 41 };
    std::array<double,3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double,3> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh( global_low_corner,
                                                global_high_corner,
                                                global_num_cell );

    // Create the global grid.
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         global_mesh,
                                         is_dim_periodic,
                                         partitioner );

    // Create an array on the cells.
    unsigned array_halo_width = 3;
    int dofs_per_cell = 4;
    auto cell_layout =
        createArrayLayout( global_grid, array_halo_width, dofs_per_cell, Cell() );

    // Loop over halo sizes up to the size of the array halo width.
    for ( unsigned halo_width = 1; halo_width <= array_halo_width; ++halo_width )
    {
        // Assign the owned cells a value of 1 and the rest 0.
        auto array = createArray<double,TEST_DEVICE>( "array", cell_layout );
        ArrayOp::assign( *array, 0.0, Ghost() );
        ArrayOp::assign( *array, 1.0, Own() );

        // Create a halo.
        auto halo = createHalo( *array, FullHaloPattern(), halo_width );

        // Gather into the ghosts.
        halo->gather( *array );

        // Check the gather. We should get 1 everywhere in the array now where
        // there was ghost overlap. Otherwise there will still be 0.
        auto owned_space = cell_layout->indexSpace( Own(), Local() );
        auto ghosted_space = cell_layout->indexSpace( Ghost(), Local() );
        auto host_view = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), array->view() );
        for ( unsigned i = 0; i < ghosted_space.extent(0); ++i )
            for ( unsigned j = 0; j < ghosted_space.extent(1); ++j )
                for ( unsigned k = 0; k < ghosted_space.extent(2); ++k )
                    for ( unsigned l = 0; l < ghosted_space.extent(3); ++l )
                        if ( i < owned_space.min(Dim::I) - halo_width ||
                             i >= owned_space.max(Dim::I) + halo_width ||
                             j < owned_space.min(Dim::J) - halo_width ||
                             j >= owned_space.max(Dim::J) + halo_width ||
                             k < owned_space.min(Dim::K) - halo_width ||
                             k >= owned_space.max(Dim::K) + halo_width )
                            EXPECT_EQ( host_view(i,j,k,l), 0.0 );
                        else
                            EXPECT_EQ( host_view(i,j,k,l), 1.0 );

        // Scatter from the ghosts back to owned.
        halo->scatter( *array );

        // Check the scatter. The value of the cell should be a function of how
        // many neighbors it has. Corner neighbors get 8, edge neighbors get 4,
        // face neighbors get 2, and no neighbors remain at 1.

        // This function checks if an index is in the halo of a low neighbor in
        // the given dimension
        auto in_dim_min_halo =
            [=]( const int i, const int dim ){
                if ( is_dim_periodic[dim] || global_grid->dimBlockId(dim) > 0 )
                    return i < (owned_space.min(dim) + halo_width);
                else
                    return false;
            };

        // This function checks if an index is in the halo of a high neighbor in
        // the given dimension
        auto in_dim_max_halo =
            [=]( const int i, const int dim ){
                if ( is_dim_periodic[dim] ||
                     global_grid->dimBlockId(dim) <
                     global_grid->dimNumBlock(dim) - 1 )
                    return i >= (owned_space.max(dim) - halo_width);
                else
                    return false;
            };

        // Check results. Use the halo functions to figure out how many neighbor a
        // given cell was ghosted to.
        Kokkos::deep_copy( host_view, array->view() );
        for ( unsigned i = owned_space.min(0); i < owned_space.max(0); ++i )
            for ( unsigned j = owned_space.min(1); j < owned_space.max(1); ++j )
                for ( unsigned k = owned_space.min(2); k < owned_space.max(2); ++k )
                {
                    int num_n = 0;
                    if ( in_dim_min_halo(i,Dim::I) || in_dim_max_halo(i,Dim::I) )
                        ++num_n;
                    if ( in_dim_min_halo(j,Dim::J) || in_dim_max_halo(j,Dim::J) )
                        ++num_n;
                    if ( in_dim_min_halo(k,Dim::K) || in_dim_max_halo(k,Dim::K) )
                        ++num_n;
                    double scatter_val = std::pow( 2.0, num_n );
                    for ( unsigned l = 0; l < owned_space.extent(3); ++l )
                        EXPECT_EQ( host_view(i,j,k,l), scatter_val );
                }
    }
}

//---------------------------------------------------------------------------//
void scatterMinTest()
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int,3> global_num_cell = { 32, 23, 41 };
    std::array<double,3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double,3> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh( global_low_corner,
                                                global_high_corner,
                                                global_num_cell );

    // Create the global grid.
    std::array<bool,3> is_dim_periodic = {true,true,true};
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         global_mesh,
                                         is_dim_periodic,
                                         Cajita::UniformDimPartitioner() );

    // Create an array on the cells.
    unsigned array_halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout =
        createArrayLayout( global_grid, array_halo_width, dofs_per_cell, Cell() );
    auto array = createArray<double,TEST_DEVICE>( "array", cell_layout );

    // Assign the rank to the array.
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    ArrayOp::assign( *array, comm_rank, Ghost() );

    // Create a halo pattern - just write to your 8 corner neighbors so we can
    // eliminate overlap between neighbors and not need to resolve the
    // collision.
    HaloPattern pattern;
    std::vector<std::array<int,3>> neighbors =
        { {-1,-1,-1}, {1,-1,-1}, {-1,1,-1}, {1,1,-1},
          {-1,-1,1}, {1,-1,1}, {-1,1,1}, {1,1,1} };
    pattern.setNeighbors( neighbors );

    // Create a halo.
    auto halo = createHalo( *array, pattern );

    // Scatter.
    halo->scatter( *array, ScatterReduce::Min() );

    // Check the reduction.
    auto host_array = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), array->view() );
    for ( const auto& n : neighbors )
    {
        auto neighbor_rank =
            cell_layout->localGrid()->neighborRank(n[0],n[1],n[2]);
        auto shared_space =
            cell_layout->localGrid()->sharedIndexSpace(
                Cajita::Own(), Cajita::Cell(), n[0], n[1], n[2] );
        for ( int i = shared_space.min(Dim::I);
              i < shared_space.max(Dim::I);
              ++i )
            for ( int j = shared_space.min(Dim::J);
                  j < shared_space.max(Dim::J);
                  ++j )
                for ( int k = shared_space.min(Dim::K);
                      k < shared_space.max(Dim::K);
                      ++k )
                    for ( int l = 0; l < 4; ++l )
                    {
                        if ( neighbor_rank < comm_rank )
                            EXPECT_EQ( host_array(i,j,k,l), neighbor_rank );
                        else
                            EXPECT_EQ( host_array(i,j,k,l), comm_rank );
                    }
    }
}

//---------------------------------------------------------------------------//
void scatterMaxTest()
{
    // Create the global grid.
    double cell_size = 0.23;
    std::array<int,3> global_num_cell = { 32, 23, 41 };
    std::array<double,3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double,3> global_high_corner =
        { global_low_corner[0] + cell_size * global_num_cell[0],
          global_low_corner[1] + cell_size * global_num_cell[1],
          global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh( global_low_corner,
                                                global_high_corner,
                                                global_num_cell );

    // Create the global grid.
    std::array<bool,3> is_dim_periodic = {true,true,true};
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         global_mesh,
                                         is_dim_periodic,
                                         Cajita::UniformDimPartitioner() );

    // Create an array on the cells.
    unsigned array_halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout =
        createArrayLayout( global_grid, array_halo_width, dofs_per_cell, Cell() );
    auto array = createArray<double,TEST_DEVICE>( "array", cell_layout );

    // Assign the rank to the array.
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    ArrayOp::assign( *array, comm_rank, Ghost() );

    // Create a halo pattern - just write to your 8 corner neighbors so we can
    // eliminate overlap between neighbors and not need to resolve the
    // collision.
    HaloPattern pattern;
    std::vector<std::array<int,3>> neighbors =
        { {-1,-1,-1}, {1,-1,-1}, {-1,1,-1}, {1,1,-1},
          {-1,-1,1}, {1,-1,1}, {-1,1,1}, {1,1,1} };
    pattern.setNeighbors( neighbors );

    // Create a halo.
    auto halo = createHalo( *array, pattern );

    // Scatter.
    halo->scatter( *array, ScatterReduce::Max() );

    // Check the reduction.
    auto host_array = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), array->view() );
    for ( const auto& n : neighbors )
    {
        auto neighbor_rank =
            cell_layout->localGrid()->neighborRank(n[0],n[1],n[2]);
        auto shared_space =
            cell_layout->localGrid()->sharedIndexSpace(
                Cajita::Own(), Cajita::Cell(), n[0], n[1], n[2] );
        for ( int i = shared_space.min(Dim::I);
              i < shared_space.max(Dim::I);
              ++i )
            for ( int j = shared_space.min(Dim::J);
                  j < shared_space.max(Dim::J);
                  ++j )
                for ( int k = shared_space.min(Dim::K);
                      k < shared_space.max(Dim::K);
                      ++k )
                    for ( int l = 0; l < 4; ++l )
                    {
                        if ( neighbor_rank > comm_rank )
                            EXPECT_EQ( host_array(i,j,k,l), neighbor_rank );
                        else
                            EXPECT_EQ( host_array(i,j,k,l), comm_rank );
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
    std::array<int,3> ranks_per_dim = {0,0,0};
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    ManualPartitioner partitioner( ranks_per_dim );

    // Boundaries are not periodic.
    std::array<bool,3> is_dim_periodic = {false,false,false};

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
    std::array<int,3> ranks_per_dim = {0,0,0};
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );
    ManualPartitioner partitioner( ranks_per_dim );

    // Every boundary is periodic
    std::array<bool,3> is_dim_periodic = {true,true,true};

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
TEST( TEST_CATEGORY, scatter_reduce_test )
{
    scatterMinTest();
    scatterMaxTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
