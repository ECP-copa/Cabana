/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita_Array.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_HypreStructuredSolver.hpp>
#include <Cajita_ReferenceStructuredSolver.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void poissonTest( const std::string &solver_type,
                  const std::string &precond_type )
{
    // Create the global grid.
    double cell_size = 0.1;
    std::array<bool, 3> is_dim_periodic = {false, false, false};
    std::array<double, 3> global_low_corner = {-1.0, -2.0, -1.0};
    std::array<double, 3> global_high_corner = {1.0, 1.0, 0.5};
    auto global_mesh = createUniformGlobalMesh( global_low_corner,
                                                global_high_corner, cell_size );

    // Create the global grid.
    UniformDimPartitioner partitioner;
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid.
    auto local_mesh = createLocalGrid( global_grid, 1 );
    auto owned_space = local_mesh->indexSpace( Own(), Cell(), Local() );

    // Create the RHS.
    auto vector_layout = createArrayLayout( local_mesh, 1, Cell() );
    auto rhs = createArray<double, TEST_DEVICE>( "rhs", vector_layout );
    ArrayOp::assign( *rhs, 1.0, Own() );

    // Create the LHS.
    auto lhs = createArray<double, TEST_DEVICE>( "lhs", vector_layout );
    ArrayOp::assign( *lhs, 0.0, Own() );

    // Create a solver.
    auto solver = createHypreStructuredSolver<double, TEST_DEVICE>(
        solver_type, *vector_layout );

    // Create a 7-point 3d laplacian stencil.
    std::vector<std::array<int, 3>> stencil = {
        {0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
        {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
    solver->setMatrixStencil( stencil );

    // Create the matrix entries. The stencil is defined over cells.
    auto matrix_entry_layout = createArrayLayout( local_mesh, 7, Cell() );
    auto matrix_entries = createArray<double, TEST_DEVICE>(
        "matrix_entries", matrix_entry_layout );
    auto entry_view = matrix_entries->view();
    Kokkos::parallel_for(
        "fill_matrix_entries",
        createExecutionPolicy( owned_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            entry_view( i, j, k, 0 ) = -6.0;
            entry_view( i, j, k, 1 ) = 1.0;
            entry_view( i, j, k, 2 ) = 1.0;
            entry_view( i, j, k, 3 ) = 1.0;
            entry_view( i, j, k, 4 ) = 1.0;
            entry_view( i, j, k, 5 ) = 1.0;
            entry_view( i, j, k, 6 ) = 1.0;
        } );

    solver->setMatrixValues( *matrix_entries );

    // Set the tolerance.
    solver->setTolerance( 1.0e-8 );

    // Set the maximum iterations.
    solver->setMaxIter( 2000 );

    // Set the print level.
    solver->setPrintLevel( 2 );

    // Create a preconditioner.
    if ( "none" != precond_type )
    {
        auto preconditioner = createHypreStructuredSolver<double, TEST_DEVICE>(
            precond_type, *vector_layout, true );
        solver->setPreconditioner( preconditioner );
    }

    // Setup the problem.
    solver->setup();

    // Solve the problem.
    solver->solve( *rhs, *lhs );

    // Create a solver reference for comparison.
    auto lhs_ref = createArray<double, TEST_DEVICE>( "lhs_ref", vector_layout );
    ArrayOp::assign( *lhs_ref, 0.0, Own() );

    auto ref_solver =
        createReferenceConjugateGradient<double, TEST_DEVICE>( *vector_layout );
    ref_solver->setMatrixStencil( stencil );
    const auto &ref_entries = ref_solver->getMatrixValues();
    auto matrix_view = ref_entries.view();
    auto global_space = local_mesh->indexSpace( Ghost(), Cell(), Global() );
    Kokkos::parallel_for(
        "fill_ref_entries",
        createExecutionPolicy( owned_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            matrix_view( i, j, k, 0 ) = -6.0;
            matrix_view( i, j, k, 1 ) =
                ( i + global_space.min( Dim::I ) > 0 ) ? 1.0 : 0.0;
            matrix_view( i, j, k, 2 ) = ( i + global_space.min( Dim::I ) <
                                          global_space.max( Dim::I ) - 1 )
                                            ? 1.0
                                            : 0.0;
            matrix_view( i, j, k, 3 ) =
                ( j + global_space.min( Dim::J ) > 0 ) ? 1.0 : 0.0;
            matrix_view( i, j, k, 4 ) = ( j + global_space.min( Dim::J ) <
                                          global_space.max( Dim::J ) - 1 )
                                            ? 1.0
                                            : 0.0;
            matrix_view( i, j, k, 5 ) =
                ( k + global_space.min( Dim::K ) > 0 ) ? 1.0 : 0.0;
            matrix_view( i, j, k, 6 ) = ( k + global_space.min( Dim::K ) <
                                          global_space.max( Dim::K ) - 1 )
                                            ? 1.0
                                            : 0.0;
        } );

    std::vector<std::array<int, 3>> diag_stencil = {{0, 0, 0}};
    ref_solver->setPreconditionerStencil( diag_stencil );
    const auto &preconditioner_entries = ref_solver->getPreconditionerValues();
    auto preconditioner_view = preconditioner_entries.view();
    Kokkos::parallel_for(
        "fill_preconditioner_entries",
        createExecutionPolicy( owned_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            preconditioner_view( i, j, k, 0 ) = -1.0 / 6.0;
        } );

    ref_solver->setTolerance( 1.0e-12 );
    ref_solver->setPrintLevel( 1 );
    ref_solver->setup();
    ref_solver->solve( *rhs, *lhs_ref );

    // Check the results.
    double epsilon = 1.0e-3;
    auto lhs_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), lhs->view() );
    auto lhs_ref_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), lhs_ref->view() );
    for ( int i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( int j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
            for ( int k = owned_space.min( Dim::K );
                  k < owned_space.max( Dim::K ); ++k )
                EXPECT_NEAR( lhs_host( i, j, k, 0 ), lhs_ref_host( i, j, k, 0 ),
                             epsilon );

    // Setup the problem again. We would need to do this if we changed the
    // matrix entries.
    solver->setup();

    // Solve the problem again
    ArrayOp::assign( *rhs, 2.0, Own() );
    ArrayOp::assign( *lhs, 0.0, Own() );
    solver->solve( *rhs, *lhs );

    // Compute another reference solution.
    ArrayOp::assign( *lhs_ref, 0.0, Own() );
    ref_solver->solve( *rhs, *lhs_ref );

    // Check the results again
    lhs_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), lhs->view() );
    lhs_ref_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(),
                                                        lhs_ref->view() );
    for ( int i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( int j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
            for ( int k = owned_space.min( Dim::K );
                  k < owned_space.max( Dim::K ); ++k )
                EXPECT_NEAR( lhs_host( i, j, k, 0 ), lhs_ref_host( i, j, k, 0 ),
                             epsilon );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( structured_solver, pcg_none_test ) { poissonTest( "PCG", "none" ); }

TEST( structured_solver, gmres_none_test ) { poissonTest( "GMRES", "none" ); }

TEST( structured_solver, bicgstab_none_test )
{
    poissonTest( "BiCGSTAB", "none" );
}

TEST( structured_solver, pfmg_none_test ) { poissonTest( "PFMG", "none" ); }

TEST( structured_solver, pcg_diag_test ) { poissonTest( "PCG", "Diagonal" ); }

TEST( structured_solver, gmres_diag_test )
{
    poissonTest( "GMRES", "Diagonal" );
}

TEST( structured_solver, bicgstab_diag_test )
{
    poissonTest( "BiCGSTAB", "Diagonal" );
}

TEST( structured_solver, pcg_jacobi_test ) { poissonTest( "PCG", "Jacobi" ); }

TEST( structured_solver, gmres_jacobi_test )
{
    poissonTest( "GMRES", "Jacobi" );
}

TEST( structured_solver, bicgstab_jacobi_test )
{
    poissonTest( "BiCGSTAB", "Jacobi" );
}

//---------------------------------------------------------------------------//

} // end namespace Test
