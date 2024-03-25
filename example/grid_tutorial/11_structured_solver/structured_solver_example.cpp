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

#include <mpi.h>

#include <array>
#include <iostream>
#include <vector>

//---------------------------------------------------------------------------//
// Structured Solver Example
//---------------------------------------------------------------------------//
void structuredSolverExample()
{
    /*
      In this example we will demonstrate building a Cabana::Grid Reference
      Conjugate Gradient Solver that solves a Poisson equation with designated
      solution tolerance,

           Laplacian( lhs ) = rhs,

      This is discretized at {i,j,k}

           Laplacian( lhs )_{i,j,k} = rhs_{i,j,k},

      which includes 7 stencils at current {i,j,k}

           { 0, 0, 0 }, { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 },
           { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 }
    */

    std::cout << "Cabana::Grid Structured Solver Example\n" << std::endl;

    /*
      As with all Cabana::Grid examples, we start by defining everything
      necessary to create the local grid.
    */
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

    // Create the global grid.
    double cell_size = 0.25;
    std::array<bool, 3> is_dim_periodic = { false, false, false };
    std::array<double, 3> global_low_corner = { -1.0, -2.0, -1.0 };
    std::array<double, 3> global_high_corner = { 1.0, 1.0, 0.5 };
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Create the global grid.
    Cabana::Grid::DimBlockPartitioner<3> partitioner;
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create a local grid.
    auto local_mesh = createLocalGrid( global_grid, 1 );
    auto owned_space = local_mesh->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Local() );

    /************************************************************************/

    // Create the RHS.
    auto vector_layout =
        Cabana::Grid::createArrayLayout( local_mesh, 1, Cabana::Grid::Cell() );
    auto rhs =
        Cabana::Grid::createArray<double, MemorySpace>( "rhs", vector_layout );
    Cabana::Grid::ArrayOp::assign( *rhs, 1.0, Cabana::Grid::Own() );

    // Create a 7-point 3d laplacian stencil.
    std::vector<std::array<int, 3>> stencil = {
        { 0, 0, 0 }, { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 },
        { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };

    // Create an array and initialize to zero.
    auto lhs =
        Cabana::Grid::createArray<double, MemorySpace>( "lhs", vector_layout );
    Cabana::Grid::ArrayOp::assign( *lhs, 0.0, Cabana::Grid::Own() );

    /*
      Now we create the solver. Cabana::Grid implements a conjugate gradient
      solver, but more options are available through an interface to HYPRE (see
      the next example).
    */
    auto ref_solver =
        Cabana::Grid::createReferenceConjugateGradient<double, MemorySpace>(
            *vector_layout );
    ref_solver->setMatrixStencil( stencil );
    const auto& ref_entries = ref_solver->getMatrixValues();
    auto matrix_view = ref_entries.view();

    auto global_space = local_mesh->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Cell(), Cabana::Grid::Global() );
    int ncell_i = global_grid->globalNumEntity( Cabana::Grid::Cell(),
                                                Cabana::Grid::Dim::I );
    int ncell_j = global_grid->globalNumEntity( Cabana::Grid::Cell(),
                                                Cabana::Grid::Dim::J );
    int ncell_k = global_grid->globalNumEntity( Cabana::Grid::Cell(),
                                                Cabana::Grid::Dim::K );

    // Fill out laplacian entries of reference solver. Entities on the system
    // boundary need to be initialized to zero.
    Kokkos::parallel_for(
        "fill_ref_entries",
        createExecutionPolicy( owned_space, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            int gi = i + global_space.min( Cabana::Grid::Dim::I ) -
                     owned_space.min( Cabana::Grid::Dim::I );
            int gj = j + global_space.min( Cabana::Grid::Dim::J ) -
                     owned_space.min( Cabana::Grid::Dim::J );
            int gk = k + global_space.min( Cabana::Grid::Dim::K ) -
                     owned_space.min( Cabana::Grid::Dim::K );
            matrix_view( i, j, k, 0 ) = 6.0;
            matrix_view( i, j, k, 1 ) = ( gi - 1 >= 0 ) ? -1.0 : 0.0;
            matrix_view( i, j, k, 2 ) = ( gi + 1 < ncell_i ) ? -1.0 : 0.0;
            matrix_view( i, j, k, 3 ) = ( gj - 1 >= 0 ) ? -1.0 : 0.0;
            matrix_view( i, j, k, 4 ) = ( gj + 1 < ncell_j ) ? -1.0 : 0.0;
            matrix_view( i, j, k, 5 ) = ( gk - 1 >= 0 ) ? -1.0 : 0.0;
            matrix_view( i, j, k, 6 ) = ( gk + 1 < ncell_k ) ? -1.0 : 0.0;
        } );

    std::vector<std::array<int, 3>> diag_stencil = { { 0, 0, 0 } };
    ref_solver->setPreconditionerStencil( diag_stencil );
    const auto& preconditioner_entries = ref_solver->getPreconditionerValues();
    auto preconditioner_view = preconditioner_entries.view();
    Kokkos::parallel_for(
        "fill_preconditioner_entries",
        createExecutionPolicy( owned_space, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            preconditioner_view( i, j, k, 0 ) = 1.0 / 6.0;
        } );

    // The desired tolerance must be set for each solve.
    ref_solver->setTolerance( 1.0e-11 );

    // Set the maximum iterations.
    ref_solver->setMaxIter( 2000 );

    /*
      The print level defines the information output during the solve:
        - 0: no solver output
        - 1: final solver output
        - 2: solver output each step
    */
    ref_solver->setPrintLevel( 2 );

    // Setup the problem - this is necessary before solving.
    ref_solver->setup();

    // Now solve the problem.
    ref_solver->solve( *rhs, *lhs );

    /*
      Setup the problem again. We would need to do this if we changed the matrix
      entries, but in this case we just leave it unchanged.
    */
    ref_solver->setup();

    // Reset to the same initial condition and solve the problem again.
    Cabana::Grid::ArrayOp::assign( *rhs, 2.0, Cabana::Grid::Own() );
    Cabana::Grid::ArrayOp::assign( *lhs, 0.0, Cabana::Grid::Own() );
    ref_solver->solve( *rhs, *lhs );
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

        structuredSolverExample();
    }
    MPI_Finalize();

    return 0;
}
//---------------------------------------------------------------------------//
