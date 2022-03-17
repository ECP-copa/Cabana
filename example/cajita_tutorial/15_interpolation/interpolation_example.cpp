/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <iostream>

//---------------------------------------------------------------------------//
// Interpolation example.
//---------------------------------------------------------------------------//
void interpolationExample()
{
    /*
      Cajita provides various particle-to-grid (p2g) and grid-to-particle (g2p)
      interpolation methods, based on the rank of the interpolated field.
    */
    std::cout << "Cajita Interpolation Example\n" << std::endl;

    /*
      First, we need some setup to demonstrate the use of Cajita interpolation.
      This includes the creation of a simple uniform mesh and various fields on
      particles and the mesh.
    */
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

    // Create the global mesh.
    std::array<double, 2> low_corner = { -1.2, 0.1 };
    std::array<double, 2> high_corner = { -0.2, 9.5 };
    double cell_size = 0.1;
    auto global_mesh =
        Cajita::createUniformGlobalMesh( low_corner, high_corner, cell_size );

    // Create the global grid.
    Cajita::DimBlockPartitioner<2> partitioner;
    std::array<bool, 2> is_dim_periodic = { true, true };
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    // Create a  grid local_grid.
    int halo_width = 1;
    auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );
    auto local_mesh = Cajita::createLocalMesh<ExecutionSpace>( *local_grid );

    // Create a point in the center of every cell.
    auto cell_space = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(),
                                              Cajita::Local() );
    int num_point = cell_space.size();
    Kokkos::View<double* [2], ExecutionSpace> points(
        Kokkos::ViewAllocateWithoutInitializing( "points" ), num_point );
    Kokkos::parallel_for(
        "fill_points", createExecutionPolicy( cell_space, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            int pi = i - halo_width;
            int pj = j - halo_width;
            int pid = pi + cell_space.extent( Cajita::Dim::I ) * pj;
            int idx[2] = { i, j };
            double x[2];
            local_mesh.coordinates( Cajita::Cell(), idx, x );
            points( pid, Cajita::Dim::I ) = x[Cajita::Dim::I];
            points( pid, Cajita::Dim::J ) = x[Cajita::Dim::J];
        } );

    // Next, we use Cajita functionality to create grid data fields.
    // Create a scalar field on the grid. See the Array tutorial example for
    // information on these functions.
    auto scalar_layout =
        Cajita::createArrayLayout( local_grid, 1, Cajita::Node() );
    auto scalar_grid_field = Cajita::createArray<double, ExecutionSpace>(
        "scalar_grid_field", scalar_layout );

    // Create a halo for scatter operations. This concept is discussed in more
    // detail in the Halo tutorial example.
    auto scalar_halo =
        Cajita::createHalo( *scalar_grid_field, Cajita::NodeHaloPattern<2>() );
    auto scalar_grid_host =
        Kokkos::create_mirror_view( scalar_grid_field->view() );

    // Create a vector field on the grid.
    auto vector_layout =
        Cajita::createArrayLayout( local_grid, 2, Cajita::Node() );
    auto vector_grid_field = Cajita::createArray<double, ExecutionSpace>(
        "vector_grid_field", vector_layout );
    auto vector_halo =
        Cajita::createHalo( *vector_grid_field, Cajita::NodeHaloPattern<2>() );
    auto vector_grid_host =
        Kokkos::create_mirror_view( vector_grid_field->view() );

    // Simple Kokkos::Views may be used to represent particle data.
    // Create a scalar point field.
    Kokkos::View<double*, ExecutionSpace> scalar_point_field(
        Kokkos::ViewAllocateWithoutInitializing( "scalar_point_field" ),
        num_point );
    auto scalar_point_host = Kokkos::create_mirror_view( scalar_point_field );

    // Create a vector point field.
    Kokkos::View<double* [2], ExecutionSpace> vector_point_field(
        Kokkos::ViewAllocateWithoutInitializing( "vector_point_field" ),
        num_point );
    auto vector_point_host = Kokkos::create_mirror_view( vector_point_field );

    // Create a tensor point field.
    Kokkos::View<double* [2][2], ExecutionSpace> tensor_point_field(
        Kokkos::ViewAllocateWithoutInitializing( "tensor_point_field" ),
        num_point );
    auto tensor_point_host = Kokkos::create_mirror_view( tensor_point_field );

    /***************************************************************************
     * P2G
     **************************************************************************/
    /*
     * The Cajita::P2G namespace contains several methods for interpolating data
     * from particles to the grid. These interpolations are inherently scatter
     * operations for particle-based threading (a single particle maps to
     * several grid nodes), which requires an underlying Kokkos::ScatterView for
     * the data being interpolated. Of note, these methods perform
     * interpolations for a single particle datum. They may ...
     *
     * Cajita also provides a convenience interface for defining field-based P2G
     * or G2P operators, by wrapping the single-particle interpolation methods
     * with loops over all particles inside: Cajita::p2g().
     */

    Kokkos::deep_copy( scalar_point_field, 3.5 );
    Kokkos::deep_copy( vector_point_field, 3.5 );
    Kokkos::deep_copy( tensor_point_field, 3.5 );

    // Interpolate a scalar point value to the grid.
    Cajita::ArrayOp::assign( *scalar_grid_field, 0.0, Cajita::Ghost() );
    auto scalar_p2g = Cajita::createScalarValueP2G( scalar_point_field, -0.5 );
    Cajita::p2g( scalar_p2g, points, num_point, Cajita::Spline<1>(),
                 *scalar_halo, *scalar_grid_field );

    // Interpolate a vector point value to the grid.
    Cajita::ArrayOp::assign( *vector_grid_field, 0.0, Cajita::Ghost() );
    auto vector_p2g = Cajita::createVectorValueP2G( vector_point_field, -0.5 );
    Cajita::p2g( vector_p2g, points, num_point, Cajita::Spline<1>(),
                 *vector_halo, *vector_grid_field );

    // Interpolate a scalar point gradient value to the grid.
    Cajita::ArrayOp::assign( *vector_grid_field, 0.0, Cajita::Ghost() );
    auto scalar_grad_p2g =
        Cajita::createScalarGradientP2G( scalar_point_field, -0.5 );
    Cajita::p2g( scalar_grad_p2g, points, num_point, Cajita::Spline<1>(),
                 *vector_halo, *vector_grid_field );

    // Interpolate a tensor point divergence value to the grid.
    Cajita::ArrayOp::assign( *vector_grid_field, 0.0, Cajita::Ghost() );
    auto tensor_div_p2g =
        Cajita::createTensorDivergenceP2G( tensor_point_field, -0.5 );
    Cajita::p2g( tensor_div_p2g, points, num_point, Cajita::Spline<1>(),
                 *vector_halo, *vector_grid_field );

    /***************************************************************************
     * G2P
     **************************************************************************/
    /*
     * In addition to P2G, The Cajita::G2P namespace contains several methods
     * for interpolating data from the grid to particles. These interpolations
     * are inherently gather operations for particle-based threading (multiple
     * grid values are gathered to a single point).
     *
     * Here we again focus on the Cajita::g2p() interface to interpolate from
     * all grid points to particles.
     */

    // Interpolate a scalar grid value to the points.
    Kokkos::deep_copy( scalar_point_field, 0.0 );
    auto scalar_value_g2p =
        Cajita::createScalarValueG2P( scalar_point_field, -0.5 );
    Cajita::g2p( *scalar_grid_field, *scalar_halo, points, num_point,
                 Cajita::Spline<1>(), scalar_value_g2p );
    Kokkos::deep_copy( scalar_point_host, scalar_point_field );

    // Interpolate a vector grid value to the points.
    Kokkos::deep_copy( vector_point_field, 0.0 );
    auto vector_value_g2p =
        Cajita::createVectorValueG2P( vector_point_field, -0.5 );
    Cajita::g2p( *vector_grid_field, *vector_halo, points, num_point,
                 Cajita::Spline<1>(), vector_value_g2p );
    Kokkos::deep_copy( vector_point_host, vector_point_field );

    // Interpolate a scalar grid gradient to the points.
    Kokkos::deep_copy( vector_point_field, 0.0 );
    auto scalar_gradient_g2p =
        Cajita::createScalarGradientG2P( vector_point_field, -0.5 );
    Cajita::g2p( *scalar_grid_field, *scalar_halo, points, num_point,
                 Cajita::Spline<1>(), scalar_gradient_g2p );
    Kokkos::deep_copy( vector_point_host, vector_point_field );

    // Interpolate a vector grid divergence to the points.
    Kokkos::deep_copy( scalar_point_field, 0.0 );
    auto vector_div_g2p =
        Cajita::createVectorDivergenceG2P( scalar_point_field, -0.5 );
    Cajita::g2p( *vector_grid_field, *vector_halo, points, num_point,
                 Cajita::Spline<1>(), vector_div_g2p );
    Kokkos::deep_copy( scalar_point_host, scalar_point_field );
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        interpolationExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
