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

#include <Cabana_Grid.hpp>
#include <Kokkos_Core.hpp>

#include <iostream>

/******************************************************************************
 * User-defined thread-level functors may be used instead of the built-in
 * functors provided. Here, the definition of P2GExampleFunctor directly uses
 * the thread-level interpolation interface to perform some basic
 * Cabana::Grid::P2G:: interpolations.
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
        static_assert( 1 == ScalarView::rank, "First View must be of scalars" );
        static_assert( 2 == VectorView::rank,
                       "Second View must be of vectors" );
    }

    /* Apply spline interpolation. */
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        /* Access the point data */
        scalar_value_type scalar_particle_data = _x( p );
        vector_value_type vector_particle_data[2];

        /* Thread-local manipulations may be computed here. */
        for ( int d = 0; d < 2; ++d )
        {
            vector_particle_data[d] = 3.0 * _y( p, d );
        }
        scalar_particle_data *= 2.0;

        /* Finally, call the thread-level value spline interpolations. */
        Cabana::Grid::P2G::value( scalar_particle_data, sd, view );
        Cabana::Grid::P2G::divergence( vector_particle_data, sd, view );
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
        static_assert( 1 == ScalarView::rank, "First View must be of scalars" );
        static_assert( 3 == TensorView::rank,
                       "Second View must be of tensors" );
    }

    /* Apply spline interpolation. */
    template <class SplineDataType, class GridViewType>
    KOKKOS_INLINE_FUNCTION void operator()( const SplineDataType& sd,
                                            const int p,
                                            const GridViewType& view ) const
    {
        scalar_value_type scalar_result;
        Cabana::Grid::G2P::divergence( view, sd, scalar_result );
        _x( p ) += scalar_result * 2.0;

        tensor_value_type tensor_result[2][2];
        Cabana::Grid::G2P::gradient( view, sd, tensor_result );

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
      Cabana::Grid provides various particle-to-grid (p2g) and grid-to-particle
      (g2p) interpolation methods, based on the rank of the interpolated field.
    */
    std::cout << "Cabana::Grid Interpolation Example\n" << std::endl;

    /*
      First, we need some setup to demonstrate the use of Cabana::Grid
      interpolation. This includes the creation of a simple uniform mesh and
      various fields on particles and the mesh.

      This example is all in 2D.
    */
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using MemorySpace = Kokkos::HostSpace;

    // Create the global mesh.
    std::array<double, 2> low_corner = { -1.2, 0.1 };
    std::array<double, 2> high_corner = { -0.2, 9.5 };
    double cell_size = 0.1;
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, cell_size );

    // Create the global grid.
    Cabana::Grid::DimBlockPartitioner<2> partitioner;
    std::array<bool, 2> is_dim_periodic = { true, true };
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create a  grid local_grid.
    int halo_width = 1;
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );
    auto local_mesh = Cabana::Grid::createLocalMesh<MemorySpace>( *local_grid );

    // Create a point in the center of every cell.
    auto cell_space = local_grid->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
    int num_particles = cell_space.size();
    Kokkos::View<double* [2], MemorySpace> particle_positions(
        Kokkos::ViewAllocateWithoutInitializing( "positions" ), num_particles );
    Kokkos::parallel_for(
        "fill_positions", createExecutionPolicy( cell_space, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            int pi = i - halo_width;
            int pj = j - halo_width;
            int pid = pi + cell_space.extent( Cabana::Grid::Dim::I ) * pj;
            int idx[2] = { i, j };
            double x[2];
            local_mesh.coordinates( Cabana::Grid::Cell(), idx, x );
            particle_positions( pid, Cabana::Grid::Dim::I ) =
                x[Cabana::Grid::Dim::I];
            particle_positions( pid, Cabana::Grid::Dim::J ) =
                x[Cabana::Grid::Dim::J];
        } );

    // Next, we use Cabana::Grid functionality to create grid data fields.
    // Create a scalar field on the grid. See the Array tutorial example for
    // information on these functions.
    auto scalar_layout =
        Cabana::Grid::createArrayLayout( local_grid, 1, Cabana::Grid::Node() );
    auto scalar_grid_field = Cabana::Grid::createArray<double, MemorySpace>(
        "scalar_grid_field", scalar_layout );

    // Create a halo for scatter operations. This concept is discussed in more
    // detail in the Halo tutorial example.
    auto scalar_halo = Cabana::Grid::createHalo(
        Cabana::Grid::NodeHaloPattern<2>(), halo_width, *scalar_grid_field );

    // Create a vector field on the grid.
    auto vector_layout =
        Cabana::Grid::createArrayLayout( local_grid, 2, Cabana::Grid::Node() );
    auto vector_grid_field = Cabana::Grid::createArray<double, MemorySpace>(
        "vector_grid_field", vector_layout );
    auto vector_halo = Cabana::Grid::createHalo(
        Cabana::Grid::NodeHaloPattern<2>(), halo_width, *vector_grid_field );

    // Simple Kokkos::Views may be used to represent particle data.
    // Create a scalar point field.
    Kokkos::View<double*, MemorySpace> scalar_particle_field(
        Kokkos::ViewAllocateWithoutInitializing( "particle_scalar" ),
        num_particles );

    // Create a vector point field.
    Kokkos::View<double* [2], MemorySpace> vector_particle_field(
        Kokkos::ViewAllocateWithoutInitializing( "particle_vector" ),
        num_particles );

    // Create a tensor point field.
    Kokkos::View<double* [2][2], MemorySpace> tensor_particle_field(
        Kokkos::ViewAllocateWithoutInitializing( "particle_tensor" ),
        num_particles );

    /***************************************************************************
     * P2G
     **************************************************************************/
    /*
     * The Cabana::Grid::P2G namespace contains several methods for
     * interpolating data from particles to the grid. These interpolations are
     * inherently scatter operations for particle-based threading (a single
     * particle maps to several grid points), which requires an underlying
     * Kokkos::ScatterView for the data being interpolated. Of note, a
     * thread-level interface provides methods to perform interpolations for a
     * single particle datum to neighboring mesh entities.
     *
     * Cabana::Grid also provides a convenience interface for defining
     * field-based P2G or G2P operators, by wrapping the thread-level
     * interpolation methods with Kokkos::parallel_for loops over all particles
     * in Cabana::Grid::p2g():
     *
     *  p2g( functor, particles, num_particles, SplineType, halo, grid_field )
     *
     *  where
     *
     *  functor : A thread-level functor object that has the following
     * signature:
     * | void operator() ( &spline, p, &grid_view )
     * |   spline : const reference to Cabana::Grid::SplineData<> object
     * |   p : thread particle index
     * |   grid_view : const reference to a grid array ScatterView
     *
     *  particle_positions : A Kokkos::View storing particle positions.
     *  num_particles : The number of particles
     *  SplineType : A Cabana::Grid::Spline<SplineOrder>() type
     *  halo : A Cabana::Grid::Halo used in the final ScatterReduce from ghost
     * regions grid_field : A Cabana::Grid::Array on which to perform the
     * scatter
     */

    std::cout << "P2G interpolations" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    // Initialize the particle data fields.
    Kokkos::deep_copy( scalar_particle_field, 3.5 );
    Kokkos::deep_copy( vector_particle_field, 3.5 );
    Kokkos::deep_copy( tensor_particle_field, 3.5 );

    // Reset the grid to zero.
    Cabana::Grid::ArrayOp::assign( *scalar_grid_field, 0.0,
                                   Cabana::Grid::Ghost() );
    auto scalar_view = scalar_grid_field->view();

    // Print out a random grid point before value interpolation.
    std::cout << "Individual grid point at (5, 5):\n\tbefore "
                 "p2g::value interpolation: "
              << scalar_view( 5, 5, 0 );

    /*
      Cabana::Grid provides a basic set of P2G and G2P functors and
      corresponding creation routines. The next four interpolations below
      demonstrate the use of these built-in functors in combination with the
      Cabana::Grid::p2g(), and later Cabana::Grid::g2p().
    */
    auto scalar_p2g =
        Cabana::Grid::createScalarValueP2G( scalar_particle_field, -0.5 );

    // Interpolate a scalar point value to the grid.
    Cabana::Grid::p2g( scalar_p2g, particle_positions, num_particles,
                       Cabana::Grid::Spline<1>(), *scalar_halo,
                       *scalar_grid_field );

    // Print out the same grid point after value interpolation.
    std::cout << "\n\tafter p2g::value interpolation: "
              << scalar_view( 5, 5, 0 ) << "\n\n";

    // Print out before gradient interpolation.
    auto vector_view = vector_grid_field->view();
    std::cout << "Individual grid point at (5, 5):\n\tbefore "
                 "p2g::gradient interpolation: <";
    std::cout << vector_view( 5, 5, 0 ) << ", " << vector_view( 5, 5, 1 )
              << ">";

    // Interpolate a scalar point gradient value to the grid.
    auto scalar_grad_p2g =
        Cabana::Grid::createScalarGradientP2G( scalar_particle_field, -0.5 );
    Cabana::Grid::p2g( scalar_grad_p2g, particle_positions, num_particles,
                       Cabana::Grid::Spline<1>(), *vector_halo,
                       *vector_grid_field );

    std::cout << "\n\tafter p2g::gradient interpolation: <";
    std::cout << vector_view( 5, 5, 0 ) << ", " << vector_view( 5, 5, 1 ) << ">"
              << "\n\n";

    // Reset the grid to zero and print before divergence interpolation.
    Cabana::Grid::ArrayOp::assign( *vector_grid_field, 0.0,
                                   Cabana::Grid::Ghost() );
    std::cout << "Individual grid point at (5, 5):\n\tbefore "
                 "p2g::divergence interpolation: <";
    std::cout << vector_view( 5, 5, 0 ) << ", " << vector_view( 5, 5, 1 )
              << ">";

    // Interpolate a tensor point divergence value to the grid.
    auto tensor_div_p2g =
        Cabana::Grid::createTensorDivergenceP2G( tensor_particle_field, -0.5 );
    Cabana::Grid::p2g( tensor_div_p2g, particle_positions, num_particles,
                       Cabana::Grid::Spline<1>(), *vector_halo,
                       *vector_grid_field );

    std::cout << "\n\tafter p2g::divergence interpolation: <";
    std::cout << vector_view( 5, 5, 0 ) << ", " << vector_view( 5, 5, 1 ) << ">"
              << "\n\n";

    // Reset the grid to zero and print before vector value interpolation.
    Cabana::Grid::ArrayOp::assign( *vector_grid_field, 0.0,
                                   Cabana::Grid::Ghost() );
    std::cout << "Individual grid point at (5, 5, 5):\n\tbefore "
                 "p2g::value interpolation: <";
    std::cout << vector_view( 5, 5, 0 ) << ", " << vector_view( 5, 5, 1 )
              << ">";

    // Interpolate a vector point value to the grid.
    auto vector_p2g =
        Cabana::Grid::createVectorValueP2G( vector_particle_field, -0.5 );
    Cabana::Grid::p2g( vector_p2g, particle_positions, num_particles,
                       Cabana::Grid::Spline<1>(), *vector_halo,
                       *vector_grid_field );

    std::cout << "\n\tafter p2g::value interpolation: <";
    std::cout << vector_view( 5, 5, 0 ) << ", " << vector_view( 5, 5, 1 ) << ">"
              << "\n\n";

    /***************************************************************************
     * User-defined thread-level functors may be used instead of the built-in
     * functors provided. Here, the definition of P2GExampleFunctor (defined
     * above) directly uses the thread-level interpolation interface to perform
     * basic Cabana::Grid::P2G interpolations.
     *
     * The P2GExampleFunctor is initialized with both a scalar point field and a
     * vector point field.
     *
     * Although this example still passes the user-defined functor to
     * Cabana::Grid::p2g(), more advanced usages with kernel fusion and multiple
     * aggregated fields will be considered in a separate example.
     * ************************************************************************/

    P2GExampleFunctor<Kokkos::View<double*, MemorySpace>,
                      Kokkos::View<double* [2], MemorySpace>>
        example_p2g { scalar_particle_field, vector_particle_field };
    Cabana::Grid::p2g( example_p2g, particle_positions, num_particles,
                       Cabana::Grid::Spline<1>(), *scalar_halo,
                       *scalar_grid_field );

    /***************************************************************************
     * G2P
     **************************************************************************/
    /*
     * In addition to P2G, The Cabana::Grid::G2P namespace contains several
     * methods for interpolating data from the grid to particles. These
     * interpolations are inherently gather operations for particle-based
     * threading (multiple grid values are gathered to a single point).
     *
     * Here we again focus on the Cabana::Grid::g2p() interface to interpolate
     * from all grid points to particles.
     */

    std::cout << "G2P interpolations" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    // Reset the particles to zero and print out a random particle before
    // interpolation.
    Kokkos::deep_copy( scalar_particle_field, 0.0 );
    std::cout << "Individual particle at (127):\n\tbefore "
                 "g2p::value interpolation: ";
    std::cout << scalar_particle_field( 127 );

    /*
      Just like the P2G examples above, Cabana::Grid also provides a basic set
      of G2P functors and corresponding creation routines. The next four
      examples demonstrate the use of these built-in functors in combination
      with Cabana::Grid::g2p() to perform interpolations.
    */
    auto scalar_value_g2p =
        Cabana::Grid::createScalarValueG2P( scalar_particle_field, -0.5 );

    // Interpolate a scalar grid value to the particles.
    Cabana::Grid::g2p( *scalar_grid_field, *scalar_halo, particle_positions,
                       num_particles, Cabana::Grid::Spline<1>(),
                       scalar_value_g2p );

    // Print out the same particle after value interpolation.
    std::cout << "\n\tafter g2p::value interpolation: "
              << scalar_particle_field( 127 ) << "\n\n";

    // Interpolate a vector grid value to the particles.
    Kokkos::deep_copy( vector_particle_field, 0.0 );
    std::cout << "Individual particle at (127):\n\tbefore "
                 "g2p::value interpolation: ";
    std::cout << "<" << vector_particle_field( 127, 0 ) << ", "
              << vector_particle_field( 127, 1 ) << ">";

    auto vector_value_g2p =
        Cabana::Grid::createVectorValueG2P( vector_particle_field, -0.5 );
    Cabana::Grid::g2p( *vector_grid_field, *vector_halo, particle_positions,
                       num_particles, Cabana::Grid::Spline<1>(),
                       vector_value_g2p );

    std::cout << "\n\tafter g2p::value interpolation: ";
    std::cout << "<" << vector_particle_field( 127, 0 ) << ", "
              << vector_particle_field( 127, 1 ) << ">"
              << "\n\n";

    // Interpolate a scalar grid gradient to the particles.
    Kokkos::deep_copy( vector_particle_field, 0.0 );
    std::cout << "Individual particle at (127):\n\tbefore "
                 "g2p::gradient interpolation: ";
    std::cout << "<" << vector_particle_field( 127, 0 ) << ", "
              << vector_particle_field( 127, 1 ) << ">";

    auto scalar_gradient_g2p =
        Cabana::Grid::createScalarGradientG2P( vector_particle_field, -0.5 );
    Cabana::Grid::g2p( *scalar_grid_field, *scalar_halo, particle_positions,
                       num_particles, Cabana::Grid::Spline<1>(),
                       scalar_gradient_g2p );

    std::cout << "\n\tafter g2p::gradient interpolation: ";
    std::cout << "<" << vector_particle_field( 127, 0 ) << ", "
              << vector_particle_field( 127, 1 ) << ">"
              << "\n\n";

    // Interpolate a vector grid divergence to the particles.
    Kokkos::deep_copy( scalar_particle_field, 0.0 );
    std::cout << "Individual particle at (127):\n\tbefore "
                 "g2p::divergence interpolation: ";
    std::cout << scalar_particle_field( 127 );

    auto vector_div_g2p =
        Cabana::Grid::createVectorDivergenceG2P( scalar_particle_field, -0.5 );
    Cabana::Grid::g2p( *vector_grid_field, *vector_halo, particle_positions,
                       num_particles, Cabana::Grid::Spline<1>(),
                       vector_div_g2p );

    std::cout << "\n\tafter g2p::divergence interpolation: "
              << scalar_particle_field( 127 ) << "\n\n";

    /***************************************************************************
     * User-defined thread-level functors can again be used instead, just as
     * above with P2G. This user-defined functor is again passed to the overall
     * interpolation function, but is also useful when combined in fused kernel
     * operations.
     * *************************************************************************/
    G2PExampleFunctor<Kokkos::View<double*, MemorySpace>,
                      Kokkos::View<double* [2][2], MemorySpace>>
        example_g2p { scalar_particle_field, tensor_particle_field };
    Cabana::Grid::g2p( *vector_grid_field, *vector_halo, particle_positions,
                       num_particles, Cabana::Grid::Spline<1>(), example_g2p );
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // MPI only needed to create the grid/mesh. Not intended to be run with
    // multiple ranks.
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        interpolationExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
