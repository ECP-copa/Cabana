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

#include <mpi.h>

#include <array>
#include <vector>

//---------------------------------------------------------------------------//
// HYPRE Structured Solver Example
//---------------------------------------------------------------------------//
void hypreStructuredSolverExample()
{
    std::cout << "Cajita HYPRE Structured Solver Example\n" << std::endl;

    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

    // Create the global grid.
    double cell_size = 0.25;
    std::array<bool, 3> is_dim_periodic = { false, false, false };
    std::array<double, 3> global_low_corner = { -1.0, -2.0, -1.0 };
    std::array<double, 3> global_high_corner = { 1.0, 1.0, 0.5 };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Create the global grid.
    Cajita::DimBlockPartitioner<3> partitioner;
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    // Create a local grid.
    auto local_mesh = createLocalGrid( global_grid, 1 );
    auto owned_space = local_mesh->indexSpace( Cajita::Own(), Cajita::Cell(),
                                               Cajita::Local() );

    // Create the RHS.
    auto vector_layout = createArrayLayout( local_mesh, 1, Cajita::Cell() );
    auto rhs = Cajita::createArray<double, MemorySpace>( "rhs", vector_layout );
    Cajita::ArrayOp::assign( *rhs, 1.0, Cajita::Own() );

    // Create the LHS.
    auto lhs = Cajita::createArray<double, MemorySpace>( "lhs", vector_layout );
    Cajita::ArrayOp::assign( *lhs, 0.0, Cajita::Own() );

    // Create a solver.
    auto solver = Cajita::createHypreStructuredSolver<double, MemorySpace>(
        "PCG", *vector_layout );

    // Create a 7-point 3d laplacian stencil.
    std::vector<std::array<int, 3>> stencil = {
        { 0, 0, 0 }, { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 },
        { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };
    solver->setMatrixStencil( stencil );

    // Create the matrix entries. The stencil is defined over cells.
    auto matrix_entry_layout =
        createArrayLayout( local_mesh, 7, Cajita::Cell() );
    auto matrix_entries = Cajita::createArray<double, MemorySpace>(
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

    // Set the tolerance.
    solver->setTolerance( 1.0e-9 );

    // Set the maximum iterations.
    solver->setMaxIter( 2000 );

    // Set the print level.
    solver->setPrintLevel( 2 );

    // Create a preconditioner.
    std::string precond_type = "Jacobi";
    auto preconditioner =
        Cajita::createHypreStructuredSolver<double, MemorySpace>(
            precond_type, *vector_layout, true );
    solver->setPreconditioner( preconditioner );

    // Setup the problem.
    solver->setup();

    // Solve the problem.
    solver->solve( *rhs, *lhs );

    // Setup the problem again. We would need to do this if we
    // changed the matrix entries.
    solver->setup();

    // Solve the problem again
    Cajita::ArrayOp::assign( *rhs, 2.0, Cajita::Own() );
    Cajita::ArrayOp::assign( *lhs, 0.0, Cajita::Own() );
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

        hypreStructuredSolverExample();
    }
    MPI_Finalize();

    return 0;
}
//---------------------------------------------------------------------------//
