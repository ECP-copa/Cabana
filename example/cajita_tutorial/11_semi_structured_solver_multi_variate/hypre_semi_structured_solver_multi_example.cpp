/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
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
// HYPRE Semi-Structured Solver Example
//---------------------------------------------------------------------------//
void hypreSemiStructuredSolverExample()
{
    /*
      In this example we will demonstrate building a HYPRE semi-structured
      solver that solves 3, independent, Poisson equations with designated
      solution tolerance,

           Laplacian( lhs ) = rhs,

      This is discretized at {i,j,k}

           Laplacian( lhs )_{i,j,k} = rhs_{i,j,k},

      which includes 7 stencils at current {i,j,k}

           { 0, 0, 0 }, { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 },
           { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 }

      You can try one of the following solver type and preconditioner type

        solver type : PCG, GMRES, BiCGSTAB,
        preconditioner type : none, Diagonal
    */

    std::cout << "Cajita HYPRE Semi-Structured Solver Example\n" << std::endl;

    /*
      As with all Cajita examples, we start by defining everything from the
      global mesh to the local grid.
    */
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

    /************************************************************************/

    // Create the RHS.
    auto vector_layout = createArrayLayout( local_mesh, 3, Cajita::Cell() );
    auto rhs = Cajita::createArray<double, MemorySpace>( "rhs", vector_layout );
    Cajita::ArrayOp::assign( *rhs, 1.0, Cajita::Own() );

    // Create the LHS.
    auto lhs = Cajita::createArray<double, MemorySpace>( "lhs", vector_layout );
    Cajita::ArrayOp::assign( *lhs, 0.0, Cajita::Own() );

    // Create a solver.
    auto solver = Cajita::createHypreSemiStructuredSolver<double, MemorySpace>(
        "PCG", *vector_layout, false, 3 );

    // Create a 7-point 3d laplacian stencil.
    std::vector<std::array<int, 3>> stencil = {
        { 0, 0, 0 }, { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 },
        { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };

    solver->createMatrixStencil( 3, 0, 3, { 7, 0, 0 } );
    solver->createMatrixStencil( 3, 1, 3, { 0, 7, 0 } );
    solver->createMatrixStencil( 3, 2, 3, { 0, 0, 7 } );
    for ( int v = 0; v < 3; ++v )
    {
        solver->setMatrixStencil( stencil, v, v );
    }

    solver->setSolverGraph( 3 );

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

    solver->initializeHypreMatrix();

    for ( int v_h = 0; v_h < 3; ++v_h )
    {
        solver->setMatrixValues( *matrix_entries, v_h, v_h );
    }

    // The desired tolerance must be set for each solve.
    solver->setTolerance( 1.0e-9 );

    // Set the maximum iterations.
    solver->setMaxIter( 2000 );

    /*
      The print level defines the information output from HYPRE during the solve
    */
    solver->setPrintLevel( 2 );

    /*
      Create a preconditioner - in this case we use Diagonal
    */
    std::string precond_type = "Diagonal";
    auto preconditioner =
        Cajita::createHypreSemiStructuredSolver<double, MemorySpace>(
            precond_type, *vector_layout, true, 3 );
    solver->setPreconditioner( preconditioner );

    // Setup the problem - this is necessary before solving.
    solver->setup();

    // Now solve the problem.
    solver->solve( *rhs, *lhs, 3 );

    /*
      Setup the problem again. We would need to do this if we changed the matrix
      entries, but in this case we just leave it unchanged.
    */
    solver->setup();
    // Reset to the same initial condition and solve the problem again.
    Cajita::ArrayOp::assign( *rhs, 2.0, Cajita::Own() );
    Cajita::ArrayOp::assign( *lhs, 0.0, Cajita::Own() );
    solver->solve( *rhs, *lhs, 3 );
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
          finalized. HYPRE_Init() initializes hypre. A call to HYPRE_Init() must
          be included before any hypre calls occur
        */
        HYPRE_Init();

        hypreSemiStructuredSolverExample();

        /*
          The hypre solver capabilities used by Cabana must be initialized and
          finalized. HYPRE_Finalize() finalizes hypre. A call to
          HYPRE_Finalize() should not occur before all calls to hypre
          capabilites are finished.
        */
        HYPRE_Finalize();
    }
    MPI_Finalize();

    return 0;
}
//---------------------------------------------------------------------------//
