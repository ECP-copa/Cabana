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
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Grid_Parallel.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Cabana::Grid;

namespace Test
{

//---------------------------------------------------------------------------//
// Tag functor.
struct ForTag
{
};
struct ReduceTag
{
};

struct TestFunctor1
{
    Kokkos::View<double*, TEST_MEMSPACE> v;

    KOKKOS_INLINE_FUNCTION
    void operator()( const ForTag&, const int i ) const { v( i ) = 2.0; }

    KOKKOS_INLINE_FUNCTION
    void operator()( const ReduceTag&, const int i, double& result ) const
    {
        result += v( i );
    }
};

struct TestFunctor2
{
    Kokkos::View<double**, TEST_MEMSPACE> v;

    KOKKOS_INLINE_FUNCTION
    void operator()( const ForTag&, const int i, const int j ) const
    {
        v( i, j ) = 2.0;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const ReduceTag&, const int i, const int j,
                     double& result ) const
    {
        result += v( i, j );
    }
};

struct TestFunctorArray
{
    Kokkos::View<double****, TEST_MEMSPACE> v;

    KOKKOS_INLINE_FUNCTION
    void operator()( const ForTag&, const int i, const int j,
                     const int k ) const
    {
        for ( int l = 0; l < 4; ++l )
            v( i, j, k, l ) = 2.0;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const ReduceTag&, const int i, const int j, const int k,
                     double& result ) const
    {
        for ( int l = 0; l < 4; ++l )
            result += v( i, j, k, l );
    }
};

//---------------------------------------------------------------------------//
void parallelIndexSpaceTest()
{
    // Rank-1 index space without tag.
    int min_i = 4;
    int max_i = 8;
    int size_i = 12;
    IndexSpace<1> is1( { min_i }, { max_i } );
    Kokkos::View<double*, TEST_MEMSPACE> v1( "v1", size_i );
    grid_parallel_for(
        "fill_rank_1", TEST_EXECSPACE(), is1,
        KOKKOS_LAMBDA( const int i ) { v1( i ) = 1.0; } );
    auto v1_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), v1 );
    for ( int i = 0; i < size_i; ++i )
    {
        if ( is1.min( 0 ) <= i && i < is1.max( 0 ) )
            EXPECT_DOUBLE_EQ( v1_mirror( i ), 1.0 );
        else
            EXPECT_DOUBLE_EQ( v1_mirror( i ), 0.0 );
    }

    // check reduction.
    double sum1 = 0.0;
    grid_parallel_reduce(
        "reduce_rank_1", TEST_EXECSPACE(), is1,
        KOKKOS_LAMBDA( const int i, double& result ) { result += v1( i ); },
        sum1 );
    EXPECT_EQ( sum1, is1.size() );

    // Rank-1 index space with tag.
    TestFunctor1 func1;
    func1.v = v1;
    grid_parallel_for( "fill_rank_1", TEST_EXECSPACE(), is1, ForTag(), func1 );
    Kokkos::deep_copy( v1_mirror, v1 );
    for ( int i = 0; i < size_i; ++i )
    {
        if ( is1.min( 0 ) <= i && i < is1.max( 0 ) )
            EXPECT_DOUBLE_EQ( v1_mirror( i ), 2.0 );
        else
            EXPECT_DOUBLE_EQ( v1_mirror( i ), 0.0 );
    }

    // check reduction.
    double sum1_tag = 0.0;
    grid_parallel_reduce( "reduce_rank_1_tag", TEST_EXECSPACE(), is1,
                          ReduceTag(), func1, sum1_tag );
    EXPECT_EQ( sum1_tag, 2.0 * is1.size() );

    // Rank-2 index space without tag.
    int min_j = 3;
    int max_j = 9;
    int size_j = 18;
    IndexSpace<2> is2( { min_i, min_j }, { max_i, max_j } );
    Kokkos::View<double**, TEST_MEMSPACE> v2( "v2", size_i, size_j );
    grid_parallel_for(
        "fill_rank_2", TEST_EXECSPACE(), is2,
        KOKKOS_LAMBDA( const int i, const int j ) { v2( i, j ) = 1.0; } );
    auto v2_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), v2 );
    for ( int i = 0; i < size_i; ++i )
        for ( int j = 0; j < size_j; ++j )
        {
            if ( is2.min( 0 ) <= i && i < is2.max( 0 ) && is2.min( 1 ) <= j &&
                 j < is2.max( 1 ) )
                EXPECT_DOUBLE_EQ( v2_mirror( i, j ), 1.0 );
            else
                EXPECT_DOUBLE_EQ( v2_mirror( i, j ), 0.0 );
        }

    // check reduction.
    double sum2 = 0.0;
    grid_parallel_reduce(
        "reduce_rank_2", TEST_EXECSPACE(), is2,
        KOKKOS_LAMBDA( const int i, const int j, double& result ) {
            result += v2( i, j );
        },
        sum2 );
    EXPECT_EQ( sum2, is2.size() );

    // Rank-2 index space with tag.
    TestFunctor2 func2;
    func2.v = v2;
    grid_parallel_for( "fill_rank_2", TEST_EXECSPACE(), is2, ForTag(), func2 );
    Kokkos::deep_copy( v2_mirror, v2 );
    for ( int i = 0; i < size_i; ++i )
        for ( int j = 0; j < size_j; ++j )
        {
            if ( is2.min( 0 ) <= i && i < is2.max( 0 ) && is2.min( 1 ) <= j &&
                 j < is2.max( 1 ) )
                EXPECT_DOUBLE_EQ( v2_mirror( i, j ), 2.0 );
            else
                EXPECT_DOUBLE_EQ( v2_mirror( i, j ), 0.0 );
        }

    // check reduction.
    double sum2_tag = 0.0;
    grid_parallel_reduce( "reduce_rank_2_tag", TEST_EXECSPACE(), is2,
                          ReduceTag(), func2, sum2_tag );
    EXPECT_EQ( sum2_tag, 2.0 * is2.size() );
}

//---------------------------------------------------------------------------//
void parallelLocalGridTest()
{
    // Let MPI compute the partitioning for this test.
    DimBlockPartitioner<3> partitioner;

    // Create the global mesh.
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 39, 42, 55 };
    std::array<bool, 3> is_dim_periodic = { true, true, true };
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

    // Create an array layout on the cells.
    int halo_width = 2;
    int dofs_per_cell = 4;
    auto cell_layout =
        createArrayLayout( global_grid, halo_width, dofs_per_cell, Cell() );
    auto local_grid = cell_layout->localGrid();

    // Create an array.
    std::string label( "test_array" );
    auto array = createArray<double, TEST_MEMSPACE>( label, cell_layout );

    // Assign a value to the entire the array.
    auto array_view = array->view();
    grid_parallel_for(
        "fill_array", TEST_EXECSPACE(), *local_grid, Ghost(), Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            for ( int l = 0; l < 4; ++l )
                array_view( i, j, k, l ) = 1.0;
        } );
    auto host_view =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), array_view );
    auto ghosted_space = array->layout()->indexSpace( Ghost(), Local() );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long k = 0; k < ghosted_space.extent( Dim::K ); ++k )
                for ( long l = 0; l < ghosted_space.extent( 3 ); ++l )
                    EXPECT_DOUBLE_EQ( host_view( i, j, k, l ), 1.0 );

    // check reduction.
    double sum = 0.0;
    grid_parallel_reduce(
        "reduce_array", TEST_EXECSPACE(), *local_grid, Ghost(), Cell(),
        KOKKOS_LAMBDA( const int i, const int j, const int k, double& result ) {
            for ( int l = 0; l < 4; ++l )
                result += array_view( i, j, k, l );
        },
        sum );
    EXPECT_EQ( sum, ghosted_space.size() );

    // Assign a value again using a tag.
    TestFunctorArray func;
    func.v = array_view;
    grid_parallel_for( "fill_array", TEST_EXECSPACE(), *local_grid, Ghost(),
                       Cell(), ForTag(), func );
    Kokkos::deep_copy( host_view, array_view );
    for ( long i = 0; i < ghosted_space.extent( Dim::I ); ++i )
        for ( long j = 0; j < ghosted_space.extent( Dim::J ); ++j )
            for ( long k = 0; k < ghosted_space.extent( Dim::K ); ++k )
                for ( long l = 0; l < ghosted_space.extent( 3 ); ++l )
                    EXPECT_DOUBLE_EQ( host_view( i, j, k, l ), 2.0 );

    // check reduction.
    double sum_tag = 0.0;
    grid_parallel_reduce( "reduce_array", TEST_EXECSPACE(), *local_grid,
                          Ghost(), Cell(), ReduceTag(), func, sum_tag );
    EXPECT_EQ( sum_tag, 2.0 * ghosted_space.size() );
}

//---------------------------------------------------------------------------//
void parallelMultiSpaceTest()
{
    // 2D
    IndexSpace<2> is2_0( { 10, 10 } );
    IndexSpace<2> is2_1( { 1, 1 }, { 4, 4 } );
    IndexSpace<2> is2_2( { 9, 9 }, { 10, 10 } );

    auto data_2d = createView<double, TEST_MEMSPACE>( "data_2d", is2_0 );
    Kokkos::deep_copy( data_2d, 0.0 );

    grid_parallel_for(
        "multi_space_2d", TEST_EXECSPACE{},
        Kokkos::Array<IndexSpace<2>, 2>{ is2_1, is2_2 },
        KOKKOS_LAMBDA( const int s, const int i, const int j ) {
            if ( 0 == s )
                data_2d( i, j ) = 1.0;
            else if ( 1 == s )
                data_2d( i, j ) = 2.0;
        } );

    auto host_data_2d =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, data_2d );

    for ( int i = 0; i < is2_0.extent( Dim::I ); ++i )
        for ( int j = 0; j < is2_0.extent( Dim::J ); ++j )
        {
            long idx[2] = { i, j };
            if ( is2_1.inRange( idx ) )
            {
                EXPECT_DOUBLE_EQ( 1.0, host_data_2d( i, j ) );
            }
            else if ( is2_2.inRange( idx ) )
            {
                EXPECT_DOUBLE_EQ( 2.0, host_data_2d( i, j ) );
            }
            else
            {
                EXPECT_DOUBLE_EQ( 0.0, host_data_2d( i, j ) );
            }
        }

    // 3D
    IndexSpace<3> is3_0( { 10, 10, 10 } );
    IndexSpace<3> is3_1( { 1, 1, 1 }, { 4, 4, 4 } );
    IndexSpace<3> is3_2( { 9, 9, 9 }, { 10, 10, 10 } );

    auto data_3d = createView<double, TEST_MEMSPACE>( "data_3d", is3_0 );
    Kokkos::deep_copy( data_3d, 0.0 );

    grid_parallel_for(
        "multi_space_3d", TEST_EXECSPACE{},
        Kokkos::Array<IndexSpace<3>, 2>{ is3_1, is3_2 },
        KOKKOS_LAMBDA( const int s, const int i, const int j, const int k ) {
            if ( 0 == s )
                data_3d( i, j, k ) = 1.0;
            else if ( 1 == s )
                data_3d( i, j, k ) = 2.0;
        } );

    auto host_data_3d =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, data_3d );

    for ( int i = 0; i < is3_0.extent( Dim::I ); ++i )
        for ( int j = 0; j < is3_0.extent( Dim::J ); ++j )
            for ( int k = 0; k < is3_0.extent( Dim::K ); ++k )
            {
                long idx[3] = { i, j, k };
                if ( is3_1.inRange( idx ) )
                {
                    EXPECT_DOUBLE_EQ( 1.0, host_data_3d( i, j, k ) );
                }
                else if ( is3_2.inRange( idx ) )
                {
                    EXPECT_DOUBLE_EQ( 2.0, host_data_3d( i, j, k ) );
                }
                else
                {
                    EXPECT_DOUBLE_EQ( 0.0, host_data_3d( i, j, k ) );
                }
            }

    // 4D
    IndexSpace<4> is4_0( { 10, 10, 10, 10 } );
    IndexSpace<4> is4_1( { 1, 1, 1, 1 }, { 4, 4, 4, 4 } );
    IndexSpace<4> is4_2( { 9, 9, 9, 9 }, { 10, 10, 10, 10 } );

    auto data_4d = createView<double, TEST_MEMSPACE>( "data_4d", is4_0 );
    Kokkos::deep_copy( data_4d, 0.0 );

    grid_parallel_for(
        "multi_space_4d", TEST_EXECSPACE{},
        Kokkos::Array<IndexSpace<4>, 2>{ is4_1, is4_2 },
        KOKKOS_LAMBDA( const int s, const int i, const int j, const int k,
                       const int l ) {
            if ( 0 == s )
                data_4d( i, j, k, l ) = 1.0;
            else if ( 1 == s )
                data_4d( i, j, k, l ) = 2.0;
        } );

    auto host_data_4d =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, data_4d );

    for ( int i = 0; i < is4_0.extent( Dim::I ); ++i )
        for ( int j = 0; j < is4_0.extent( Dim::J ); ++j )
            for ( int k = 0; k < is4_0.extent( Dim::K ); ++k )
                for ( int l = 0; l < is4_0.extent( 3 ); ++l )
                {
                    long idx[4] = { i, j, k, l };
                    if ( is4_1.inRange( idx ) )
                    {
                        EXPECT_DOUBLE_EQ( 1.0, host_data_4d( i, j, k, l ) );
                    }
                    else if ( is4_2.inRange( idx ) )
                    {
                        EXPECT_DOUBLE_EQ( 2.0, host_data_4d( i, j, k, l ) );
                    }
                    else
                    {
                        EXPECT_DOUBLE_EQ( 0.0, host_data_4d( i, j, k, l ) );
                    }
                }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, parallel_index_space_test ) { parallelIndexSpaceTest(); }

TEST( TEST_CATEGORY, parallel_local_grid_test ) { parallelLocalGridTest(); }

TEST( TEST_CATEGORY, parallel_multispace_test ) { parallelMultiSpaceTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
