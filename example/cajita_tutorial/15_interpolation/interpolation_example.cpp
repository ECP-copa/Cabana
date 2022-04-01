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

/******************************************************************************
 * User-defined thread-level functors may be used instead of the built-in
 * functors provided. Here, the definition of P2GExampleFunctor directly uses
 * the thread-level interpolation interface to perform some basic Cajita::P2G::
 * interpolations.
 * *****************************************************************************/
template <class ScalarView, class VectorView>
struct P2GExampleFunctor
{
    /* Value types. */
    using scalar_value_type = typename ScalarView::value_type;
    using vector_value_type = typename VectorView::value_type;

    /* Spline evaluation locations. */
    ScalarView _x;
    VectorView _y;

    /* Here, the constructor accepts both a scalar and a vector View of the
     * particle data to interpolate
     */
    P2GExampleFunctor( const ScalarView& x, const VectorView& y )
        : _x( x )
        , _y( y )
    {
        static_assert( 1 == ScalarView::Rank, "First View must be of scalars" );
        static_assert( 2 == VectorView::Rank,
                       "Second View must be of vectors" );
    }

    /* Apply spline interpolation. */
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        /* Access the point data */
        scalar_value_type scalar_point_data = _x( p );
        vector_value_type vector_point_data[2];

        /* Thread-local manipulations may be computed here. */
        for ( int d = 0; d < 2; ++d )
        {
            vector_point_data[d] = 3.0 * _y( p, d );
        }
        scalar_point_data *= 2.0;

        /* Finally, call the thread-level value spline interpolations. */
        Cajita::P2G::value( scalar_point_data, sd, view );
        Cajita::P2G::divergence( vector_point_data, sd, view );
    }
};

template <class ScalarView, class TensorView>
struct G2PExampleFunctor
{
    /* Value types. */
    using scalar_value_type = typename ScalarView::value_type;
    using tensor_value_type = typename TensorView::value_type;

    /* Spline evaluation locations. */
    ScalarView _x;
    TensorView _t;

    /* Here, the constructor accepts both a scalar and a tensor View of the
     * particle data to interpolate
     */
    G2PExampleFunctor( const ScalarView& x, const TensorView& t )
        : _x( x )
        , _t( t )
    {
        static_assert( 1 == ScalarView::Rank, "First View must be of scalars" );
        static_assert( 3 == TensorView::Rank,
                       "Second View must be of tensors" );
    }

    /* Apply spline interpolation. */
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        scalar_value_type scalar_result;
        Cajita::G2P::divergence( view, sd, scalar_result );
        _x( p ) += scalar_result * 2.0;

        tensor_value_type tensor_result[2][2];
        Cajita::G2P::gradient( view, sd, tensor_result );

        /* Thread-local manipulations may be computed here. */
        for ( int i = 0; i < 2; ++i )
            for ( int j = 0; j < 2; ++j )
            {
                _t( p, i, j ) = tensor_result[i][j];
            }
    }
};

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
     * the data being interpolated. Of note, a thread-level interface provides
     * methods to perform interpolations for a single particle datum to
     * neighboring mesh entitites.
     *
     * Cajita also provides a convenience interface for defining field-based P2G
     * or G2P operators, by wrapping the thread-level interpolation methods
     * with Kokkos::parallel_for loops over all particles in Cajita::p2g():
     *
     *  p2g( functor, points, num_points, SplineType, halo, grid_field )
     *
     *  where
     *
     *  functor : A thread-level functor object that has the following
     * signature:
     * | void operator() ( &spline, p, &grid_view )
     * |   spline : const reference to Cajita::SplineData<> object
     * |   p : thread particle index
     * |   grid_view : const reference to a grid array ScatterView
     *
     *  points : A Kokkos::View storing particle positions.
     *  num_points : The number of particles
     *  SplineType : A Cajita::Spline<SplineOrder>() type
     *  halo : A Cajita::Halo used in the final ScatterReduce from ghost regions
     *  grid_field : A Cajita::Array on which to perform the scatter
     */

    /***************************************************************************
     * Cajita provides a basic set of P2G  and G2P functors and corresponding
     * creation routines. The following examples demonstrate the use of these
     * built-in functors in combination with Cajita::p2g() and Cajita::g2p() to
     * perform interpolations of particle data of various ranks.
     * *************************************************************************/

    // Initialize the particle data fields.
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
     * User-defined thread-level functors may be used instead of the built-in
     * functors provided. Here, the definition of P2GExampleFunctor directly
     * uses the thread-level interpolation interface to perform basic
     * Cajita::P2G:: interpolations.
     *
     * The P2GExampleFunctor is initialized with both a scalar point field and a
     * vector point field.
     *
     * Whereas this example still passes the user-defined functor to
     * Cajita::p2g(), more advanced usages with kernel fusion and multiple
     * aggregated fields will be considered in another example.
     * ************************************************************************/

    P2GExampleFunctor<Kokkos::View<double*, ExecutionSpace>,
                      Kokkos::View<double* [2], ExecutionSpace>>
        example_p2g { scalar_point_field, vector_point_field };
    Cajita::p2g( example_p2g, points, num_point, Cajita::Spline<1>(),
                 *scalar_halo, *scalar_grid_field );

    /***************************************************************************
     * G2P
     **************************************************************************/
    /*
     * In addition to P2G, The Cajita::G2P namespace contains several
     * methods for interpolating data from the grid to particles. These
     * interpolations are inherently gather operations for particle-based
     * threading (multiple grid values are gathered to a single point).
     *
     * Here we again focus on the Cajita::g2p() interface to interpolate
     * from all grid points to particles.
     */

    /***************************************************************************
     * Cajita also provides a basic set of G2P functors and corresponding
     * creation routines. The following examples demonstrate the use of these
     * built-in functors in combination with Cajita::g2p() to perform
     * interpolations of grid data of various ranks.
     * *************************************************************************/

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

    G2PExampleFunctor<Kokkos::View<double*, ExecutionSpace>,
                      Kokkos::View<double* [2][2], ExecutionSpace>>
        example_g2p{ scalar_point_field, tensor_point_field };
    Cajita::g2p( *vector_grid_field, *vector_halo, points, num_point,
                 Cajita::Spline<1>(), example_g2p );
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
