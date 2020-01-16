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

#include <Cajita_Types.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_UniformDimPartitioner.hpp>
#include <Cajita_Array.hpp>
#include <Cajita_ReferenceStructuredSolver.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <vector>
#include <array>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void poissonTest( const std::string& solver_type, const std::string& precond_type )
{
    // Create the global grid.
    double cell_size = 0.1;
    std::array<bool,3> is_dim_periodic = {false,false,false};
    std::array<double,3> global_low_corner = {-1.0, -2.0, -1.0 };
    std::array<double,3> global_high_corner = { 1.0, 1.0, 0.5 };
    auto global_mesh = createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Create the global grid.
    UniformDimPartitioner partitioner;
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         global_mesh,
                                         is_dim_periodic,
                                         partitioner );

    // Create a local grid.
    auto local_mesh = createLocalGrid( global_grid, 1 );
    auto owned_space = local_mesh->indexSpace(Own(),Cell(),Local());

    // Create the RHS.
    auto vector_layout = createArrayLayout( local_mesh, 1, Cell() );
    auto rhs = createArray<double,TEST_DEVICE>( "rhs", vector_layout );
    ArrayOp::assign( *rhs, 1.0, Own() );

    // Create the LHS.
    auto lhs = createArray<double,TEST_DEVICE>( "lhs", vector_layout );
    ArrayOp::assign( *lhs, 0.0, Own() );

    // Create a solver.
    auto solver =
        createReferenceConjugateGradient<double,TEST_DEVICE>( *vector_layout );

    // Create a 7-point 3d laplacian stencil.
    std::vector<std::array<int,3> > matrix_stencil =
        { {0,0,0}, {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1} };
    solver->setMatrixStencil( matrix_stencil );

    // Create the matrix entries. The stencil is defined over cells. If a stencil
    // reaches out of bounds then set the entry to zero.
    const auto& matrix_entries = solver->getMatrixValues();
    auto matrix_view = matrix_entries.view();
    auto global_space = local_mesh->indexSpace(Ghost(),Cell(),Global());
    Kokkos::parallel_for(
        "fill_matrix_entries",
        createExecutionPolicy( owned_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            matrix_view(i,j,k,0) = -6.0;
            matrix_view(i,j,k,1) =
                ( i + global_space.min(Dim::I) > 0 ) ? 1.0 : 0.0;
            matrix_view(i,j,k,2) =
                ( i + global_space.min(Dim::I) < global_space.max(Dim::I) - 1 ) ? 1.0 : 0.0;
            matrix_view(i,j,k,3) =
                ( j + global_space.min(Dim::J) > 0 ) ? 1.0 : 0.0;
            matrix_view(i,j,k,4) =
                ( j + global_space.min(Dim::J) < global_space.max(Dim::J) - 1 ) ? 1.0 : 0.0;
            matrix_view(i,j,k,5) =
                ( k + global_space.min(Dim::K) > 0 ) ? 1.0 : 0.0;
            matrix_view(i,j,k,6) =
                ( k + global_space.min(Dim::K) < global_space.max(Dim::K) - 1 ) ? 1.0 : 0.0;
        } );

    // Create a diagonal preconditioner.
    std::vector<std::array<int,3> > diag_stencil = { {0,0,0} };
    solver->setPreconditionerStencil( diag_stencil );
    const auto& preconditioner_entries = solver->getPreconditionerValues();
    auto preconditioner_view = preconditioner_entries.view();
    Kokkos::parallel_for(
        "fill_preconditioner_entries",
        createExecutionPolicy( owned_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            preconditioner_view(i,j,k,0) = -1.0 / 6.0;
        } );

    // Set the tolerance.
    solver->setTolerance( 1.0e-8 );

    // Set the maximum iterations.
    solver->setMaxIter( 2000 );

    // Set the print level.
    solver->setPrintLevel( 2 );

    // Solve the problem.
    solver->solve( *rhs, *lhs );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( structured_solver, pcg_diag_test )
{
    poissonTest( "PCG", "Diagonal" );
}

//---------------------------------------------------------------------------//

} // end namespace Test
