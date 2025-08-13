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

//---------------------------------------------------------------------------//
// Spline example.
//---------------------------------------------------------------------------//

void splineExample()
{
    /*
      Cabana::Grid provides B-Spline related interfaces for uniform grids
      operations. The spline orders are:
        0 - nearest grid point interpolation
        1 - linear
        2 - quadratic
        3 - cubic
    */
    std::cout << "Cabana::Grid Spline Example\n" << std::endl;

    /*
      First, we need some setup to facilitate the use of Cabana::Grid splines.

      This includes the creation of a simple uniform mesh and a single particle
      position.
    */

    // Create a 3-dimensional global uniform mesh
    // Global bounding box.
    double cell_size = 0.05;
    std::array<int, 3> global_num_cell = { 20, 20, 20 };
    std::array<double, 3> global_low_corner = { 0, 0, 0 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };

    // Create the global mesh
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Set up a uniform partitioner
    Cabana::Grid::DimBlockPartitioner<3> partitioner;

    // Create the global grid.
    std::array<bool, 3> periodic = { false, false, false };
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, periodic, partitioner );

    // Get the local grid.
    int halo_cell_width = 1;
    auto local_grid =
        Cabana::Grid::createLocalGrid( global_grid, halo_cell_width );

    // Get the local mesh for this rank.
    auto local_mesh =
        Cabana::Grid::createLocalMesh<Kokkos::HostSpace>( *local_grid );

    // Create a particle position vector.
    double xp[3] = { 0.5, 2, 1.5 };

    /********************************************************************/
    /*
      There are two ways to use Cabana::Grid splines in practice.

      In the first, one constructs an object of type:
        SplineData<Scalar, Order, NumSpaceDim, EntityType, Tags...> where
            Scalar is the underlying scalar type
            Order = spline order
            NumSpaceDim = number of spatial dimensions
            EntityType = geometric entity type over which the spline is defined
            Tags = variadic list of desired SplineDataMember tags.

      The available SplineDataMembers and their associated member names are
        SplinePhysicalCellSize = dx[d]
        SplineLogicalPosition = x[d]
        SplinePhysicalDistance = d[d]
        SplineWeightValues = w[d][num_knots]
        SplineWeightPhysicalGradients g[d][num_knots]

      where d indexes NumSpaceDim, and num_knots is SplineOrder + 1

      This design allows for customizing the relevant spline data for a
      particular application. If all Tags are omitted, all SplineDataMembers are
      included.
    */
    using value_type = double;
    constexpr int order = 2;     // quadratic B-spline
    constexpr int num_space = 3; // 3-dimensional

    // This spline data type will be defined on grid nodes in this example.
    using SD = Cabana::Grid::SplineData<value_type, order, num_space,
                                        Cabana::Grid::Node>;

    SD sd_n; // Default construction
    std::cout << "Spline order: " << SD::order << " (quadratic)" << std::endl;

    // Evaluate the spline data given a point position and the local mesh
    Cabana::Grid::evaluateSpline( local_mesh, xp, sd_n );

    // The spline data by default contains all available SplineDataMembers,
    // whose values were all updated from the previous call to evaluateSpline()
    std::cout << "Spline weights:" << std::endl;
    for ( int d = 0; d < num_space; ++d )
    {
        std::cout << "  Dim " << d << ": ";
        for ( int k = 0; k < SD::num_knot; ++k )
        {
            std::cout << sd_n.w[d][k] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Only a subset of specific SplineDataMember tags may be used instead of
    // the full set by default:
    using DataTags = Cabana::Grid::SplineDataMemberTypes<
        Cabana::Grid::SplinePhysicalCellSize,
        Cabana::Grid::SplineLogicalPosition,
        Cabana::Grid::SplinePhysicalDistance, Cabana::Grid::SplineWeightValues>;

    // This is may be useful in reducing memory
    // spline data type
    using SD_t = Cabana::Grid::SplineData<value_type, order, num_space,
                                          Cabana::Grid::Node, DataTags>;

    // Check members.
    static_assert( SD_t::has_physical_cell_size,
                   "spline data missing physical cell size" );
    static_assert( SD_t::has_logical_position,
                   "spline data missing logical position" );
    static_assert( SD_t::has_physical_distance,
                   "spline data missing physical distance" );
    static_assert( SD_t::has_weight_values,
                   "spline data missing weight values" );

    // For example, the following line gives a compile-time error since the
    // SplineWeightPhysicalGradients data tag was not included static_assert(
    // SD_t::has_weight_physical_gradients,
    //                "spline data missing weight physical gradients" );

    /*
      We can also use the Spline<Order> interface directly to evaluate
      specific spline data given basic information about the local grid.

      This approach is useful if one needs to write their own particle-grid
      operators. However, the built-in Cabana::Grid particle-grid operators in
      Cabana_Grid_Interpolation.hpp does not require the use of these
      interfaces except to construct the SplineData as above.
    */
    double rdx = 1.0 / cell_size;
    double values[3]; // num_knots for a 2nd-order spline
    double x0[3];     // Logical position in primal grid

    std::cout << "Spline weights (static interface):" << std::endl;
    for ( int d = 0; d < num_space; ++d )
    {
        // These interfaces are for a single dimension only.
        x0[d] = Cabana::Grid::Spline<2>::mapToLogicalGrid(
            xp[d], rdx, global_low_corner[d] );
        Cabana::Grid::Spline<2>::value( x0[d], values );
        std::cout << "  Dim " << d << ": ";
        for ( auto v : values )
        {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    // The offsets are the fixed logical space offsets of the knots,
    // which depends on the order of the spline.
    // For the quadratic B-spline, there are 3 knots with offsets -1, 0, 1:
    int offsets_quadratic[3];
    std::cout << "Spline offsets (static interface) (quadratic):" << std::endl;
    for ( int d = 0; d < num_space; ++d )
    {
        Cabana::Grid::Spline<2>::offsets( offsets_quadratic );
        std::cout << "  Dim " << d << ": ";
        for ( auto o : offsets_quadratic )
        {
            std::cout << o << " ";
        }
        std::cout << std::endl;
    }

    // For a cubic B-spline, there are 4 knots, with offsets -1, 0, 1, 2.
    // This number is available as a static constexpr member for declaring
    // arrays at compile-time.
    int offsets_cubic[Cabana::Grid::Spline<3>::num_knot];
    std::cout << "Spline offsets (static interface) (cubic):" << std::endl;
    for ( int d = 0; d < num_space; ++d )
    {
        Cabana::Grid::Spline<3>::offsets( offsets_cubic );
        std::cout << "  Dim " << d << ": ";
        for ( auto o : offsets_cubic )
        {
            std::cout << o << " ";
        }
        std::cout << std::endl;
    }

    // The stencil returns the entity indices for a given logical space position
    int stencil[3];
    std::cout << "Spline stencil (static interface):" << std::endl;
    for ( int d = 0; d < num_space; ++d )
    {
        Cabana::Grid::Spline<2>::stencil( x0[d], stencil );
        std::cout << "  Dim " << d << ": ";
        for ( auto si : stencil )
        {
            std::cout << si << " ";
        }
        std::cout << std::endl;
    }

    // The knot basis gradient values in physical units for a given logical
    // position.
    double grad[3];
    std::cout << "Spline gradient (static interface):" << std::endl;
    for ( int d = 0; d < num_space; ++d )
    {
        Cabana::Grid::Spline<2>::gradient( x0[d], rdx, grad );
        std::cout << "  Dim " << d << ": ";
        for ( auto g : grad )
        {
            std::cout << g << " ";
        }
        std::cout << std::endl;
    }

    /*
      Splines are used extensively within the interpolation example for
      particle-grid interactions.
    */
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

        splineExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
