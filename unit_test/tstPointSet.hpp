/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Kokkos_Core.hpp>

#include <Cajita_Types.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_UniformDimPartitioner.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>
#include <Cajita_Splines.hpp>
#include <Cajita_PointSet.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <array>
#include <vector>
#include <cmath>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void pointSetTest()
{
    // Create the global mesh.
    std::array<double,3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double,3> high_corner = { -0.3, 9.5, 2.3 };
    double cell_size = 0.05;
    auto global_mesh = createUniformGlobalMesh(
        low_corner, high_corner, cell_size );

    // Create the global grid.
    UniformDimPartitioner partitioner;
    std::array<bool,3> is_dim_periodic = {true,true,true};
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         global_mesh,
                                         is_dim_periodic,
                                         partitioner );

    // Create a local grid.
    int halo_width = 1;
    auto local_grid = createLocalGrid( global_grid, halo_width );
    auto local_mesh = createLocalMesh<TEST_DEVICE>( *local_grid );

    // Create a point in the center of every cell.
    auto cell_space =
        local_grid->indexSpace( Own(), Cell(), Local() );
    int num_point = cell_space.size();
    Kokkos::View<double*[3],TEST_DEVICE> points(
        Kokkos::ViewAllocateWithoutInitializing("points"), num_point );
    Kokkos::parallel_for(
        "fill_points",
        createExecutionPolicy(cell_space,TEST_EXECSPACE()),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            int pi = i - halo_width;
            int pj = j - halo_width;
            int pk = k - halo_width;
            int pid = pi + cell_space.extent(Dim::I) * (
                pj + cell_space.extent(Dim::J) * pk );
            double x[3];
            int idx[3] = {i,j,k};
            local_mesh.coordinates( Cell(), idx, x );
            points(pid,Dim::I) = x[Dim::I];
            points(pid,Dim::J) = x[Dim::J];
            points(pid,Dim::K) = x[Dim::K];
        });

    // Create a point set with linear spline interpolation to the nodes.
    auto point_set = createPointSet(
        points, num_point, num_point, *local_grid, Node(), Spline<1>() );

    // Check the point set data.
    EXPECT_EQ( point_set.num_point, num_point );
    EXPECT_EQ( point_set.dx, cell_size );
    EXPECT_EQ( point_set.rdx, 1.0 / cell_size );
    double xn_low[3];
    int idx_low[3] = {0,0,0};
    local_mesh.coordinates( Node(), idx_low, xn_low );
    for ( int d = 0; d < 3; ++d )
        EXPECT_EQ( point_set.low_corner[d], xn_low[d] );

    // Check logical coordinates
    auto logical_coords_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), point_set.logical_coords );
    for ( int i = cell_space.min(Dim::I); i < cell_space.max(Dim::I); ++i )
        for ( int j = cell_space.min(Dim::J); j < cell_space.max(Dim::J); ++j )
            for ( int k = cell_space.min(Dim::K); k < cell_space.max(Dim::K); ++k )
            {
                int pi = i - halo_width;
                int pj = j - halo_width;
                int pk = k - halo_width;
                int pid = pi + cell_space.extent(Dim::I) * (
                    pj + cell_space.extent(Dim::J) * pk );
                EXPECT_FLOAT_EQ( logical_coords_host(pid,Dim::I), i + 0.5 );
                EXPECT_FLOAT_EQ( logical_coords_host(pid,Dim::J), j + 0.5 );
                EXPECT_FLOAT_EQ( logical_coords_host(pid,Dim::K), k + 0.5 );
            }

    // Check stencil.
    auto stencil_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), point_set.stencil );
    for ( int i = cell_space.min(Dim::I); i < cell_space.max(Dim::I); ++i )
        for ( int j = cell_space.min(Dim::J); j < cell_space.max(Dim::J); ++j )
            for ( int k = cell_space.min(Dim::K); k < cell_space.max(Dim::K); ++k )
            {
                int pi = i - halo_width;
                int pj = j - halo_width;
                int pk = k - halo_width;
                int pid = pi + cell_space.extent(Dim::I) * (
                    pj + cell_space.extent(Dim::J) * pk );
                EXPECT_EQ( stencil_host(pid,0,Dim::I), i );
                EXPECT_EQ( stencil_host(pid,1,Dim::I), i + 1 );
                EXPECT_EQ( stencil_host(pid,0,Dim::J), j );
                EXPECT_EQ( stencil_host(pid,1,Dim::J), j + 1 );
                EXPECT_EQ( stencil_host(pid,0,Dim::K), k );
                EXPECT_EQ( stencil_host(pid,1,Dim::K), k + 1 );
            }

    // Check values.
    auto values_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), point_set.value );
    for ( int i = cell_space.min(Dim::I); i < cell_space.max(Dim::I); ++i )
        for ( int j = cell_space.min(Dim::J); j < cell_space.max(Dim::J); ++j )
            for ( int k = cell_space.min(Dim::K); k < cell_space.max(Dim::K); ++k )
            {
                int pi = i - halo_width;
                int pj = j - halo_width;
                int pk = k - halo_width;
                int pid = pi + cell_space.extent(Dim::I) * (
                    pj + cell_space.extent(Dim::J) * pk );
                EXPECT_FLOAT_EQ( values_host(pid,0,Dim::I), 0.5 );
                EXPECT_FLOAT_EQ( values_host(pid,1,Dim::I), 0.5 );
                EXPECT_FLOAT_EQ( values_host(pid,0,Dim::J), 0.5 );
                EXPECT_FLOAT_EQ( values_host(pid,1,Dim::J), 0.5 );
                EXPECT_FLOAT_EQ( values_host(pid,0,Dim::K), 0.5 );
                EXPECT_FLOAT_EQ( values_host(pid,1,Dim::K), 0.5 );
            }

    // Check gradients.
    auto gradients_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), point_set.gradient );
    for ( int i = cell_space.min(Dim::I); i < cell_space.max(Dim::I); ++i )
        for ( int j = cell_space.min(Dim::J); j < cell_space.max(Dim::J); ++j )
            for ( int k = cell_space.min(Dim::K); k < cell_space.max(Dim::K); ++k )
            {
                int pi = i - halo_width;
                int pj = j - halo_width;
                int pk = k - halo_width;
                int pid = pi + cell_space.extent(Dim::I) * (
                    pj + cell_space.extent(Dim::J) * pk );

                EXPECT_FLOAT_EQ( gradients_host(pid,0,0,0,Dim::I), -0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,0,0,0,Dim::J), -0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,0,0,0,Dim::K), -0.25/cell_size );

                EXPECT_FLOAT_EQ( gradients_host(pid,1,0,0,Dim::I), 0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,1,0,0,Dim::J), -0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,1,0,0,Dim::K), -0.25/cell_size );

                EXPECT_FLOAT_EQ( gradients_host(pid,1,1,0,Dim::I), 0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,1,1,0,Dim::J), 0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,1,1,0,Dim::K), -0.25/cell_size );

                EXPECT_FLOAT_EQ( gradients_host(pid,0,1,0,Dim::I), -0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,0,1,0,Dim::J), 0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,0,1,0,Dim::K), -0.25/cell_size );

                EXPECT_FLOAT_EQ( gradients_host(pid,0,0,1,Dim::I), -0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,0,0,1,Dim::J), -0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,0,0,1,Dim::K), 0.25/cell_size );

                EXPECT_FLOAT_EQ( gradients_host(pid,1,0,1,Dim::I), 0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,1,0,1,Dim::J), -0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,1,0,1,Dim::K), 0.25/cell_size );

                EXPECT_FLOAT_EQ( gradients_host(pid,1,1,1,Dim::I), 0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,1,1,1,Dim::J), 0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,1,1,1,Dim::K), 0.25/cell_size );

                EXPECT_FLOAT_EQ( gradients_host(pid,0,1,1,Dim::I), -0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,0,1,1,Dim::J), 0.25/cell_size );
                EXPECT_FLOAT_EQ( gradients_host(pid,0,1,1,Dim::K), 0.25/cell_size );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( point_set, update_test )
{
    pointSetTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
