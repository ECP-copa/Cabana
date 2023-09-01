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

#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_LoadBalancer.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_Types.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <cmath>

using namespace Cabana::Grid;

namespace Test
{

//---------------------------------------------------------------------------//
double lbWork( std::array<double, 6>& verts )
{
    // work density function: x*y*z
    const double x12 = verts[0] * verts[0];
    const double y12 = verts[1] * verts[1];
    const double z12 = verts[2] * verts[2];
    const double x22 = verts[3] * verts[3];
    const double y22 = verts[4] * verts[4];
    const double z22 = verts[5] * verts[5];
    return ( x12 - x22 ) * ( y12 - y22 ) * ( z12 - z22 ) / -8.;
}

double lbWork( std::array<double, 4>& verts )
{
    // work density function: x*y
    const double x12 = verts[0] * verts[0];
    const double y12 = verts[1] * verts[1];
    const double x22 = verts[2] * verts[2];
    const double y22 = verts[3] * verts[3];
    return ( x12 - x22 ) * ( y12 - y22 ) / 4.;
}

void lbTest3d()
{
    std::array<double, 8> expected_imbalance = { 0,
                                                 0.5,
                                                 0.66666666666666663,
                                                 0.79999999999999993,
                                                 0.80000000000000016,
                                                 0.87499999999999989,
                                                 0.8571428571428571,
                                                 0.92857142857142871 };
    const double imbalance_limit = 0.04;
    const std::size_t test_steps = 2000; // Number of times to rebalance
    const DimBlockPartitioner<3> partitioner;
    std::array<int, 3> empty_array = { 0 };
    std::array<int, 3> ranks =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, empty_array );

    // Global mesh
    double cell_size = 0.23;
    std::array<int, 3> global_num_cell = { 47 * ranks[0], 38 * ranks[1],
                                           53 * ranks[2] };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // Global grid
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    auto lb = Experimental::createLoadBalancer( MPI_COMM_WORLD, global_grid );
    std::array<double, 6> vertices = lb->getVertices();
    std::array<double, 6> internal_vertices = lb->getInternalVertices();
    double work = lbWork( vertices );
    global_grid =
        lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
    double imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %.17g\n", imbalance );
    if ( size < static_cast<int>( expected_imbalance.size() ) )
    {
        EXPECT_DOUBLE_EQ( imbalance, expected_imbalance[size - 1] );
    }
    for ( std::size_t step = 1; step < test_steps; ++step )
    {
        work = lbWork( vertices );
        global_grid =
            lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
        vertices = lb->getVertices();
        internal_vertices = lb->getInternalVertices();
        imbalance = lb->getImbalance();
        // if ( rank == 0 )
        //    printf( "LB Imbalance: %g\n", imbalance );
        if ( imbalance < imbalance_limit )
            break;
    }
    double final_imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %g\n", final_imbalance );
    EXPECT_LT( final_imbalance, imbalance_limit );
}

void lbTest3dMinSizeScalar()
{
    std::array<double, 8> expected_imbalance = { 0,
                                                 0.5,
                                                 0.66666666666666663,
                                                 0.79999999999999993,
                                                 0.80000000000000016,
                                                 0.87499999999999989,
                                                 0.8571428571428571,
                                                 0.92857142857142871 };
    const double imbalance_limit = 0.04;
    const std::size_t test_steps = 2000; // Number of times to rebalance
    DimBlockPartitioner<3> partitioner;
    std::array<int, 3> empty_array = { 0 };
    std::array<int, 3> ranks =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, empty_array );

    // Global mesh
    double cell_size = 0.23;
    const double min_domain_size = 3 * cell_size;
    std::array<int, 3> global_num_cell = { 47 * ranks[0], 38 * ranks[1],
                                           53 * ranks[2] };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // Global grid
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    auto lb = Experimental::createLoadBalancer( MPI_COMM_WORLD, global_grid,
                                                min_domain_size );
    std::array<double, 6> vertices = lb->getVertices();
    std::array<double, 6> internal_vertices = lb->getInternalVertices();
    double work = lbWork( vertices );
    global_grid =
        lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
    double imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %.17g\n", imbalance );
    if ( size < static_cast<int>( expected_imbalance.size() ) )
    {
        EXPECT_DOUBLE_EQ( imbalance, expected_imbalance[size - 1] );
    }
    for ( std::size_t step = 0; step < test_steps; ++step )
    {
        work = lbWork( vertices );
        global_grid =
            lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
        vertices = lb->getVertices();
        EXPECT_LE( min_domain_size, vertices[3] - vertices[0] );
        EXPECT_LE( min_domain_size, vertices[4] - vertices[1] );
        EXPECT_LE( min_domain_size, vertices[5] - vertices[2] );
        internal_vertices = lb->getInternalVertices();
        EXPECT_LE( min_domain_size,
                   internal_vertices[3] - internal_vertices[0] );
        EXPECT_LE( min_domain_size,
                   internal_vertices[4] - internal_vertices[1] );
        EXPECT_LE( min_domain_size,
                   internal_vertices[5] - internal_vertices[2] );
        imbalance = lb->getImbalance();
        // if ( rank == 0 )
        //    printf( "LB Imbalance: %g\n", imbalance );
        if ( imbalance < imbalance_limit )
            break;
    }
    double final_imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %g\n", final_imbalance );
    EXPECT_LT( final_imbalance,
               imbalance_limit ); // Will fail for large amount of ranks
}

void lbTest3dMinSizeArray()
{
    std::array<double, 8> expected_imbalance = { 0,
                                                 0.5,
                                                 0.66666666666666663,
                                                 0.79999999999999993,
                                                 0.80000000000000016,
                                                 0.87499999999999989,
                                                 0.8571428571428571,
                                                 0.92857142857142871 };
    const double imbalance_limit = 0.04;
    const std::size_t test_steps = 2000; // Number of times to rebalance
    DimBlockPartitioner<3> partitioner;
    std::array<int, 3> empty_array = { 0 };
    std::array<int, 3> ranks =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, empty_array );

    // Global mesh
    double cell_size = 0.23;
    const std::array<double, 3> min_domain_size = {
        3 * cell_size, 5 * cell_size, 10 * cell_size };
    std::array<int, 3> global_num_cell = { 47 * ranks[0], 38 * ranks[1],
                                           53 * ranks[2] };
    std::array<double, 3> global_low_corner = { 1.2, 3.3, -2.8 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // Global grid
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    auto lb = Experimental::createLoadBalancer( MPI_COMM_WORLD, global_grid,
                                                min_domain_size );
    std::array<double, 6> vertices = lb->getVertices();
    std::array<double, 6> internal_vertices = lb->getInternalVertices();
    double work = lbWork( vertices );
    global_grid =
        lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
    double imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %.17g\n", imbalance );
    if ( size < static_cast<int>( expected_imbalance.size() ) )
    {
        EXPECT_DOUBLE_EQ( imbalance, expected_imbalance[size - 1] );
    }
    for ( std::size_t step = 0; step < test_steps; ++step )
    {
        double work = lbWork( vertices );
        global_grid =
            lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
        vertices = lb->getVertices();
        EXPECT_LE( min_domain_size[0], vertices[3] - vertices[0] );
        EXPECT_LE( min_domain_size[1], vertices[4] - vertices[1] );
        EXPECT_LE( min_domain_size[2], vertices[5] - vertices[2] );
        internal_vertices = lb->getInternalVertices();
        EXPECT_LE( min_domain_size[0],
                   internal_vertices[3] - internal_vertices[0] );
        EXPECT_LE( min_domain_size[1],
                   internal_vertices[4] - internal_vertices[1] );
        EXPECT_LE( min_domain_size[2],
                   internal_vertices[5] - internal_vertices[2] );
        double imbalance = lb->getImbalance();
        // if ( rank == 0 )
        //    printf( "LB Imbalance: %g\n", imbalance );
        if ( imbalance < imbalance_limit )
            break;
    }
    double final_imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %g\n", final_imbalance );
    EXPECT_LT( final_imbalance,
               imbalance_limit ); // Will fail for large amount of ranks
}

void lbTest2d()
{
    std::array<double, 8> expected_imbalance = { 0,
                                                 0.49999999999999989,
                                                 0.66666666666666663,
                                                 0.80000000000000004,
                                                 0.80000000000000004,
                                                 0.87499999999999989,
                                                 0.85714285714285721,
                                                 0.90909090909090906 };
    const double imbalance_limit = 0.04;
    const std::size_t test_steps = 2000; // Number of times to rebalance
    const DimBlockPartitioner<2> partitioner;
    std::array<int, 2> empty_array = { 0 };
    std::array<int, 2> ranks =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, empty_array );

    // Global mesh
    double cell_size = 0.23;
    std::array<int, 2> global_num_cell = { 47 * ranks[0], 38 * ranks[1] };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    std::array<bool, 2> is_dim_periodic = { false, false };

    // Global grid
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    auto lb = Experimental::createLoadBalancer( MPI_COMM_WORLD, global_grid );
    std::array<double, 4> vertices = lb->getVertices();
    std::array<double, 4> internal_vertices = lb->getInternalVertices();
    double work = lbWork( vertices );
    global_grid =
        lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
    double imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %.17g\n", imbalance );
    if ( size < static_cast<int>( expected_imbalance.size() ) )
    {
        EXPECT_DOUBLE_EQ( imbalance, expected_imbalance[size - 1] );
    }
    for ( std::size_t step = 0; step < test_steps; ++step )
    {
        double work = lbWork( vertices );
        global_grid =
            lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
        vertices = lb->getVertices();
        internal_vertices = lb->getInternalVertices();
        double imbalance = lb->getImbalance();
        // if ( rank == 0 )
        //    printf( "LB Imbalance: %g\n", imbalance );
        if ( imbalance < imbalance_limit )
            break;
    }
    double final_imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %g\n", final_imbalance );
    EXPECT_LT( final_imbalance,
               imbalance_limit ); // Will fail for large amount of ranks
}

void lbTest2dMinSizeScalar()
{
    std::array<double, 8> expected_imbalance = { 0,
                                                 0.49999999999999989,
                                                 0.66666666666666663,
                                                 0.80000000000000004,
                                                 0.80000000000000004,
                                                 0.87499999999999989,
                                                 0.85714285714285721,
                                                 0.90909090909090906 };
    const double imbalance_limit = 0.04;
    const std::size_t test_steps = 2000; // Number of times to rebalance
    DimBlockPartitioner<2> partitioner;
    std::array<int, 2> empty_array = { 0 };
    std::array<int, 2> ranks =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, empty_array );

    // Global mesh
    double cell_size = 0.23;
    const double min_domain_size = 3 * cell_size;
    std::array<int, 2> global_num_cell = { 47 * ranks[0], 38 * ranks[1] };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    std::array<bool, 2> is_dim_periodic = { false, false };

    // Global grid
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    auto lb = Experimental::createLoadBalancer( MPI_COMM_WORLD, global_grid,
                                                min_domain_size );
    std::array<double, 4> vertices = lb->getVertices();
    std::array<double, 4> internal_vertices = lb->getInternalVertices();
    double work = lbWork( vertices );
    global_grid =
        lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
    double imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %.17g\n", imbalance );
    if ( size < static_cast<int>( expected_imbalance.size() ) )
    {
        EXPECT_DOUBLE_EQ( imbalance, expected_imbalance[size - 1] );
    }
    for ( std::size_t step = 0; step < test_steps; ++step )
    {
        double work = lbWork( vertices );
        global_grid =
            lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
        vertices = lb->getVertices();
        EXPECT_LE( min_domain_size, vertices[2] - vertices[0] );
        EXPECT_LE( min_domain_size, vertices[3] - vertices[1] );
        internal_vertices = lb->getInternalVertices();
        EXPECT_LE( min_domain_size,
                   internal_vertices[2] - internal_vertices[0] );
        EXPECT_LE( min_domain_size,
                   internal_vertices[3] - internal_vertices[1] );
        double imbalance = lb->getImbalance();
        // if ( rank == 0 )
        //    printf( "LB Imbalance: %g\n", imbalance );
        if ( imbalance < imbalance_limit )
            break;
    }
    double final_imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %g\n", final_imbalance );
    EXPECT_LT( final_imbalance,
               imbalance_limit ); // Will fail for large amount of ranks
}

void lbTest2dMinSizeArray()
{
    std::array<double, 8> expected_imbalance = { 0,
                                                 0.49999999999999989,
                                                 0.66666666666666663,
                                                 0.80000000000000004,
                                                 0.80000000000000004,
                                                 0.87499999999999989,
                                                 0.85714285714285721,
                                                 0.90909090909090906 };
    const double imbalance_limit = 0.04;
    const std::size_t test_steps = 2000; // Number of times to rebalance
    DimBlockPartitioner<2> partitioner;
    std::array<int, 2> empty_array = { 0 };
    std::array<int, 2> ranks =
        partitioner.ranksPerDimension( MPI_COMM_WORLD, empty_array );

    // Global mesh
    double cell_size = 0.23;
    const std::array<double, 2> min_domain_size = { 3 * cell_size,
                                                    5 * cell_size };
    std::array<int, 2> global_num_cell = { 47 * ranks[0], 38 * ranks[1] };
    std::array<double, 2> global_low_corner = { 1.2, 3.3 };
    std::array<double, 2> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1] };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, global_num_cell );
    std::array<bool, 2> is_dim_periodic = { false, false };

    // Global grid
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    auto lb = Experimental::createLoadBalancer( MPI_COMM_WORLD, global_grid,
                                                min_domain_size );
    std::array<double, 4> vertices = lb->getVertices();
    std::array<double, 4> internal_vertices = lb->getInternalVertices();
    double work = lbWork( vertices );
    global_grid =
        lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
    double imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %.17g\n", imbalance );
    if ( size < static_cast<int>( expected_imbalance.size() ) )
    {
        EXPECT_DOUBLE_EQ( imbalance, expected_imbalance[size - 1] );
    }
    for ( std::size_t step = 0; step < test_steps; ++step )
    {
        double work = lbWork( vertices );
        global_grid =
            lb->createBalancedGlobalGrid( global_mesh, partitioner, work );
        vertices = lb->getVertices();
        EXPECT_LE( min_domain_size[0], vertices[2] - vertices[0] );
        EXPECT_LE( min_domain_size[1], vertices[3] - vertices[1] );
        internal_vertices = lb->getInternalVertices();
        EXPECT_LE( min_domain_size[0],
                   internal_vertices[2] - internal_vertices[0] );
        EXPECT_LE( min_domain_size[1],
                   internal_vertices[3] - internal_vertices[1] );
        double imbalance = lb->getImbalance();
        // if ( rank == 0 )
        //    printf( "LB Imbalance: %g\n", imbalance );
        if ( imbalance < imbalance_limit )
            break;
    }
    double final_imbalance = lb->getImbalance();
    if ( rank == 0 )
        printf( "LB Imbalance: %g\n", final_imbalance );
    EXPECT_LT( final_imbalance,
               imbalance_limit ); // Will fail for large amount of ranks
}
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( load_balancer, 3d_lb_test ) { lbTest3d(); }
TEST( load_balancer, 3d_lb_test_min_domain_size_scalar )
{
    lbTest3dMinSizeScalar();
}
TEST( load_balancer, 3d_lb_test_min_domain_size_array )
{
    lbTest3dMinSizeArray();
}
TEST( load_balancer, 2d_lb_test ) { lbTest2d(); }
TEST( load_balancer, 2d_lb_test_min_domain_size_scalar )
{
    lbTest2dMinSizeScalar();
}
TEST( load_balancer, 2d_lb_test_min_domain_size_array )
{
    lbTest2dMinSizeArray();
}
//---------------------------------------------------------------------------//
} // end namespace Test
