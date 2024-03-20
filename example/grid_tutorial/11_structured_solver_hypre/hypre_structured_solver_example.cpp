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
// HYPRE Structured Solver Example
//---------------------------------------------------------------------------//
void hypreStructuredSolverExample()
{
    /*
      In this example we will demonstrate building a HYPRE Structured Solver
      that solve a Poisson equation with designated solution tolerance,

           Laplacian( lhs ) = rhs,

      This is discretized at {i,j,k}

           Laplacian( lhs )_{i,j,k} = rhs_{i,j,k},

      which includes 7 stencils at current {i,j,k}

           { 0, 0, 0 }, { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 },
           { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 }

      You can try one of the following solver type and preconditioner type

        solver type : PCG, GMRES, BiCGSTAB, PFMG,
        preconditioner type : none, Diagonal, Jacobi
    */

    std::cout << "Cabana::Grid HYPRE Structured Solver Example\n" << std::endl;

    /*
      As with all Cabana::Grid examples, we start by defining everything from
      the global mesh to the local grid.
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

    // Create the LHS.
    auto lhs =
        Cabana::Grid::createArray<double, MemorySpace>( "lhs", vector_layout );
    Cabana::Grid::ArrayOp::assign( *lhs, 0.0, Cabana::Grid::Own() );

    // Create a solver.
    auto solver =
        Cabana::Grid::createHypreStructuredSolver<double, MemorySpace>(
            "PCG", *vector_layout );

    // Create a 7-point 3d laplacian stencil.
    std::vector<std::array<int, 3>> stencil = {
        { 0, 0, 0 }, { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 },
        { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };
    solver->setMatrixStencil( stencil );

    // Create the matrix entries. The stencil is defined over cells.
    auto matrix_entry_layout =
        Cabana::Grid::createArrayLayout( local_mesh, 7, Cabana::Grid::Cell() );
    auto matrix_entries = Cabana::Grid::createArray<double, MemorySpace>(
        "matrix_entries", matrix_entry_layout );
    auto entry_view = matrix_entries->view();
    Kokkos::parallel_for(
        "fill_matrix_entries",
        createExecutionPolicy( owned_space, ExecutionSpace() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            entry_view( i, j, k, 0 ) = 6.0;
            entry_view( i, j, k, 1 ) = -1.0;
            entry_view( i, j, k, 2 ) = -1.0;
            entry_view( i, j, k, 3 ) = -1.0;
            entry_view( i, j, k, 4 ) = -1.0;
            entry_view( i, j, k, 5 ) = -1.0;
            entry_view( i, j, k, 6 ) = -1.0;
        } );

    solver->setMatrixValues( *matrix_entries );

    solver->printMatrix( "Struct.mat" );

    // The desired tolerance must be set for each solve.
    solver->setTolerance( 1.0e-9 );

    // Set the maximum iterations.
    solver->setMaxIter( 2000 );

    /*
      The print level defines the information output from HYPRE during the solve
    */
    solver->setPrintLevel( 2 );

    /*
      Create a preconditioner - in this case we use Jacobi (other available
      options are shown above).
    */
    std::string precond_type = "Jacobi";
    auto preconditioner =
        Cabana::Grid::createHypreStructuredSolver<double, MemorySpace>(
            precond_type, *vector_layout, true );
    solver->setPreconditioner( preconditioner );

    // Setup the problem - this is necessary before solving.
    solver->setup();

    // Now solve the problem.
    solver->solve( *rhs, *lhs );

    /*
      Setup the problem again. We would need to do this if we changed the matrix
      entries, but in this case we just leave it unchanged.
    */
    solver->setup();

    // Reset to the same initial condition and solve the problem again.
    Cabana::Grid::ArrayOp::assign( *rhs, 2.0, Cabana::Grid::Own() );
    Cabana::Grid::ArrayOp::assign( *lhs, 0.0, Cabana::Grid::Own() );
    solver->solve( *rhs, *lhs );
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

        /*
          The hypre solver capabilities used by Cabana must be initialized and
          finalized. HYPRE_Init() initializes hypre. A call to HYPRE_Init()
          must be included before any hypre calls occur
        */
        HYPRE_Init();

        hypreStructuredSolverExample();

        /*
          The hypre solver capabilities used by Cabana must be initialized and
          finalized. HYPRE_Finalize() finalizes hypre. A call to
          HYPRE_Finalize() should not occur before all calls to hypre
          capabilities are finished. This call is placed outside the hypre
          solver function to ensure the hypre objects are out of scope
        */
        HYPRE_Finalize();
    }
    MPI_Finalize();

    return 0;
}
//---------------------------------------------------------------------------//
