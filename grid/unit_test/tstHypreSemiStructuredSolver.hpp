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

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_HypreSemiStructuredSolver.hpp>
#include <Cabana_Grid_HypreStructuredSolver.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_ReferenceStructuredSolver.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace Cabana::Grid;

namespace Test
{

//---------------------------------------------------------------------------//
// FIXME: Only run test if HYPRE is compatible with the memory space. This
// is currently written in this structure because HYPRE only has
// compile-time switches for backends and hence only one can be used at a
// time. Once they have a run-time switch we can use that instead.
template <class MemorySpace>
std::enable_if_t<!HypreIsCompatibleWithMemorySpace<MemorySpace>::value, void>
poissonTest( const std::string&, const std::string&, MemorySpace )
{
}

template <class MemorySpace>
std::enable_if_t<HypreIsCompatibleWithMemorySpace<MemorySpace>::value, void>
poissonTest( const std::string& solver_type, const std::string& precond_type,
             MemorySpace )
{
    // Create the global grid.
    double cell_size = 0.25;
    std::array<bool, 3> is_dim_periodic = { false, false, false };
    std::array<double, 3> global_low_corner = { -1.0, -2.0, -1.0 };
    std::array<double, 3> global_high_corner = { 1.0, 1.0, 0.5 };
    auto global_mesh = createUniformGlobalMesh( global_low_corner,
                                                global_high_corner, cell_size );

    // Create the global grid.
    DimBlockPartitioner<3> partitioner;
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid.
    auto local_mesh = createLocalGrid( global_grid, 1 );
    auto owned_space = local_mesh->indexSpace( Own(), Cell(), Local() );

    // Create the RHS.
    auto vector_layout = createArrayLayout( local_mesh, 1, Cell() );
    auto rhs = createArray<double, MemorySpace>( "rhs", vector_layout );
    ArrayOp::assign( *rhs, 1.0, Own() );

    // Create the LHS.
    auto lhs = createArray<double, MemorySpace>( "lhs", vector_layout );
    ArrayOp::assign( *lhs, 0.0, Own() );

    HYPRE_Init();

    // Create a solver.
    auto solver = createHypreSemiStructuredSolver<double, MemorySpace>(
        solver_type, *vector_layout, false, 1 );

    // Create a 7-point 3d laplacian stencil.
    std::vector<std::array<int, 3>> stencil = {
        { 0, 0, 0 }, { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 },
        { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };
    solver->createMatrixStencil( 3, 0, 1, { 7 } );
    solver->setMatrixStencil( stencil, 0, 0 );

    solver->setSolverGraph( 1 );

    // Create the matrix entries. The stencil is defined over cells.
    auto matrix_entry_layout = createArrayLayout( local_mesh, 7, Cell() );
    auto matrix_entries = createArray<double, MemorySpace>(
        "matrix_entries", matrix_entry_layout );
    auto entry_view = matrix_entries->view();
    Kokkos::parallel_for(
        "fill_matrix_entries",
        createExecutionPolicy( owned_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            entry_view( i, j, k, 0 ) = 6.0;
            entry_view( i, j, k, 1 ) = -1.0;
            entry_view( i, j, k, 2 ) = -1.0;
            entry_view( i, j, k, 3 ) = -1.0;
            entry_view( i, j, k, 4 ) = -1.0;
            entry_view( i, j, k, 5 ) = -1.0;
            entry_view( i, j, k, 6 ) = -1.0;
        } );

    solver->setMatrixValues( *matrix_entries, 0, 0 );

    // Set the tolerance.
    solver->setTolerance( 1.0e-9 );

    // Set the maximum iterations.
    solver->setMaxIter( 2000 );

    // Set the print level.
    solver->setPrintLevel( 2 );

    // Create a preconditioner.
    if ( "none" != precond_type )
    {
        auto preconditioner =
            createHypreSemiStructuredSolver<double, MemorySpace>(
                precond_type, *vector_layout, true );
        solver->setPreconditioner( preconditioner );
    }

    // Setup the problem.
    solver->setup();

    // Solve the problem.
    solver->solve( *rhs, *lhs, 1 );

    // Create a solver reference for comparison.
    auto lhs_ref = createArray<double, MemorySpace>( "lhs_ref", vector_layout );
    ArrayOp::assign( *lhs_ref, 0.0, Own() );

    auto ref_solver =
        createReferenceConjugateGradient<double, MemorySpace>( *vector_layout );
    ref_solver->setMatrixStencil( stencil );
    const auto& ref_entries = ref_solver->getMatrixValues();
    auto matrix_view = ref_entries.view();
    auto global_space = local_mesh->indexSpace( Own(), Cell(), Global() );
    int ncell_i = global_grid->globalNumEntity( Cell(), Dim::I );
    int ncell_j = global_grid->globalNumEntity( Cell(), Dim::J );
    int ncell_k = global_grid->globalNumEntity( Cell(), Dim::K );
    Kokkos::parallel_for(
        "fill_ref_entries",
        createExecutionPolicy( owned_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            int gi = i + global_space.min( Dim::I ) - owned_space.min( Dim::I );
            int gj = j + global_space.min( Dim::J ) - owned_space.min( Dim::J );
            int gk = k + global_space.min( Dim::K ) - owned_space.min( Dim::K );
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
        createExecutionPolicy( owned_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            preconditioner_view( i, j, k, 0 ) = 1.0 / 6.0;
        } );

    ref_solver->setTolerance( 1.0e-11 );
    ref_solver->setPrintLevel( 1 );
    ref_solver->setup();
    ref_solver->solve( *rhs, *lhs_ref );

    // Check the results.
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
                EXPECT_FLOAT_EQ( lhs_host( i, j, k, 0 ),
                                 lhs_ref_host( i, j, k, 0 ) );

    // Setup the problem again. We would need to do this if we changed the
    // matrix entries.
    solver->setup();

    // Solve the problem again
    ArrayOp::assign( *rhs, 2.0, Own() );
    ArrayOp::assign( *lhs, 0.0, Own() );
    solver->solve( *rhs, *lhs, 1 );

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
                EXPECT_FLOAT_EQ( lhs_host( i, j, k, 0 ),
                                 lhs_ref_host( i, j, k, 0 ) );

    HYPRE_Finalize();
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( semi_structured_solver, pcg_none_test )
{
    poissonTest( "PCG", "none", TEST_MEMSPACE{} );
}

TEST( semi_structured_solver, gmres_none_test )
{
    poissonTest( "GMRES", "none", TEST_MEMSPACE{} );
}

TEST( semi_structured_solver, bicgstab_none_test )
{
    poissonTest( "BiCGSTAB", "none", TEST_MEMSPACE{} );
}

TEST( semi_structured_solver, pcg_diag_test )
{
    poissonTest( "PCG", "Diagonal", TEST_MEMSPACE{} );
}

TEST( semi_structured_solver, gmres_diag_test )
{
    poissonTest( "GMRES", "Diagonal", TEST_MEMSPACE{} );
}

TEST( semi_structured_solver, bicgstab_diag_test )
{
    poissonTest( "BiCGSTAB", "Diagonal", TEST_MEMSPACE{} );
}

//---------------------------------------------------------------------------//

} // end namespace Test
