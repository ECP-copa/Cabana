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

#include <array>
#include <iostream>

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <Cabana_Grid.hpp>

template <class ExecutionSpace, class LocalGridType, class ArrayType>
void updateBoundaries( ExecutionSpace exec_space, LocalGridType local_grid,
                       ArrayType& field )
{
    // Update the boundary on each face of the cube.
    for ( int d = 0; d < 3; d++ )
    {
        for ( int dir = -1; dir < 2; dir += 2 )
        {
            std::array<int, 3> plane = { 0, 0, 0 };
            plane[d] = dir;

            // Get the boundary indices for this plane (each one is a separate,
            // contiguous index space).
            auto boundary_space = local_grid->boundaryIndexSpace(
                Cabana::Grid::Own(), Cabana::Grid::Cell(), plane );

            Cabana::Grid::grid_parallel_for(
                "boundary_update", exec_space, boundary_space,
                KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                    // Neumann boundary condition example
                    field( i, j, k, 0 ) =
                        field( i - plane[0], j - plane[1], k - plane[2], 0 );
                } );
        }
    }
}

/*
  This micro-application solves the heat equation using finite differences.

  If not familiar with Cabana, it is recommended to go through the Core and
  Grid tutorials prior to this example.
*/
void finiteDifference()
{
    // Use the default execution (and memory) space - these can instead be
    // explicitly chosen.
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;

    // Define the system and create the global mesh
    double cell_size = 1e-5;
    std::array<double, 3> global_low_corner = { -2.5e-4, -2.5e-4, -2.5e-4 };
    std::array<double, 3> global_high_corner = { 2.5e-4, 2.5e-4, 0 };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Create the global grid for a non-periodic system.
    Cabana::Grid::DimBlockPartitioner<3> partitioner;
    std::array<bool, 3> periodic = { false, false, false };
    auto global_grid =
        createGlobalGrid( MPI_COMM_WORLD, global_mesh, periodic, partitioner );

    // Create a local grid and local mesh with halo region.
    // Note this halo width needs to be large enough for the stencil used below.
    unsigned halo_width = 1;
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, halo_width );
    auto local_mesh = Cabana::Grid::createLocalMesh<device_type>( *local_grid );

    // Create temperature array on the cells for finite difference calculations
    auto owned_space = local_grid->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );
    auto layout =
        createArrayLayout( global_grid, halo_width, 1, Cabana::Grid::Cell() );
    std::string name( "temperature" );
    auto T_array =
        Cabana::Grid::createArray<double, device_type>( name, layout );

    // Set all initial values to room temperature (K). This sets both the owned
    // and ghosted cell values.
    Cabana::Grid::ArrayOp::assign( *T_array, 300.0, Cabana::Grid::Ghost() );

    // Create the halo for grid MPI communication.
    auto T_halo =
        createHalo( Cabana::Grid::NodeHaloPattern<3>(), halo_width, *T_array );

    // Update initial physical and MPI processor boundaries.
    auto T = T_array->view();
    updateBoundaries( exec_space(), local_grid, T );
    T_halo->gather( exec_space(), *T_array );

    // Create array to store previous temperature for explicit time udpate.
    // Note that this should not be created from the original array - this must
    // point to different memory.
    auto T_prev_array =
        Cabana::Grid::createArray<double, device_type>( name, layout );
    auto T_prev = T_prev_array->view();

    // Time-related inputs.
    double dt = 1e-6;
    double end_time = 1e-3;
    int num_steps = static_cast<int>( end_time / dt );

    // Material property inputs.
    double density = 1000.0;
    double specific_heat = 1000.0;
    double thermal_conductivity = 10.0;
    // Derived inputs.
    double alpha = thermal_conductivity / ( density * specific_heat );
    double alpha_dt_dx2 = alpha * dt / ( cell_size * cell_size );
    double dt_rho_cp = dt / ( density * specific_heat );

    // Gaussian heat source parameters (sigma is the standard deviation of the
    // gaussian).
    double eta = 0.1;
    double power = 200.0;
    double sigma[3] = { 50e-6, 50e-6, 25e-6 };
    double sqrt2 = Kokkos::sqrt( 2.0 );
    double r[3] = { sigma[0] / sqrt2, sigma[1] / sqrt2, sigma[2] / sqrt2 };
    // Intensity of the heat source.
    double I = 2.0 * eta * power /
               ( M_PI * Kokkos::sqrt( M_PI ) * r[0] * r[1] * r[2] );

    // Timestep loop.
    int output_freq = 10;
    for ( int step = 0; step < num_steps; ++step )
    {
        if ( global_grid->blockId() == 0 && step % output_freq == 0 )
            std::cout << "Step " << step << " / " << num_steps << std::endl;

        // Store previous value for explicit update
        Kokkos::deep_copy( T_prev, T );

        // Solve heat conduction from point source with finite difference.
        Cabana::Grid::grid_parallel_for(
            "finite_difference", exec_space(), owned_space,
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                double loc[3];
                int idx[3] = { i, j, k };
                local_mesh.coordinates( Cabana::Grid::Cell(), idx, loc );

                double f = ( loc[0] * loc[0] / r[0] / r[0] ) +
                           ( loc[1] * loc[1] / r[1] / r[1] ) +
                           ( loc[2] * loc[2] / r[2] / r[2] );

                double Q = I * Kokkos::exp( -f ) * dt_rho_cp;

                double laplacian =
                    ( -6.0 * T_prev( i, j, k, 0 ) + T_prev( i - 1, j, k, 0 ) +
                      T_prev( i + 1, j, k, 0 ) + T_prev( i, j - 1, k, 0 ) +
                      T_prev( i, j + 1, k, 0 ) + T_prev( i, j, k - 1, 0 ) +
                      T_prev( i, j, k + 1, 0 ) ) *
                    alpha_dt_dx2;

                T( i, j, k, 0 ) += laplacian + Q;
            } );

        // Update the physical system boundary conditions.
        updateBoundaries( exec_space(), local_grid, T );

        // Exchange halo values on MPI boundaries.
        T_halo->gather( exec_space(), *T_array );
    }

    // Write the final state to file.
    Cabana::Grid::Experimental::BovWriter::writeTimeStep( 0, 0, *T_array );
}

// Main function which initializes/finalizes MPI and Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    finiteDifference();

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
