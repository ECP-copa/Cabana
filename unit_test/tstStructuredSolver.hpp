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
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_UniformDimPartitioner.hpp>
#include <Cajita_Array.hpp>
#include <Cajita_StructuredSolver.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <vector>
#include <array>

using namespace Cajita;

namespace Test
{

//---------------------------------------------------------------------------//
void poissonTest()
{
    // Let MPI compute the partitioning for this test.
    UniformDimPartitioner partitioner;

    // Create the global grid.
    double cell_size = 0.1;
    std::vector<bool> is_dim_periodic = {true,true,true};
    std::vector<double> global_low_corner = {-1.0, -1.0, -1.0 };
    std::vector<double> global_high_corner = { 1.0, 1.0, 1.0 };
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD,
                                         partitioner,
                                         is_dim_periodic,
                                         global_low_corner,
                                         global_high_corner,
                                         cell_size );

    // Create a block.
    auto block = createBlock( global_grid, 0 );
    auto owned_space = block->indexSpace(Own(),Cell(),Local());

    // Create the RHS.
    auto vector_layout = createArrayLayout( block, 1, Cell() );
    auto rhs = createArray<double,TEST_DEVICE>( "rhs", vector_layout );
    ArrayOp::assign( *rhs, 1.0, Own() );

    // Create the LHS.
    auto lhs = createArray<double,TEST_DEVICE>( "lhs", vector_layout );
    ArrayOp::assign( *lhs, 0.0, Own() );

    // Create a solver.
    auto solver =
        createStructuredSolver( "PCG", *vector_layout, TEST_DEVICE() );

    // Create a 7-point 3d laplacian stencil.
    std::vector<std::array<int,3> > stencil =
        { {0,0,0}, {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1} };
    solver->setMatrixStencil( stencil );

    // Create the matrix entries. The stencil is defined over cells.
    auto matrix_entry_layout = createArrayLayout( block, 7, Cell() );
    auto matrix_entries = createArray<double,TEST_DEVICE>(
        "matrix_entries", matrix_entry_layout );
    auto entry_view = matrix_entries->view();
    Kokkos::parallel_for(
        "fill_matrix_entries",
        createExecutionPolicy( owned_space, TEST_EXECSPACE() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            entry_view(i,j,k,0) = -6.0;
            entry_view(i,j,k,1) = 0.0;
            entry_view(i,j,k,2) = 0.0;
            entry_view(i,j,k,3) = 0.0;
            entry_view(i,j,k,4) = 0.0;
            entry_view(i,j,k,5) = 0.0;
            entry_view(i,j,k,6) = 0.0;
        } );
    solver->setMatrixValues( *matrix_entries );

    // Set the tolerance.
    solver->setTolerance( 1.0e-4 );

    // Set the print level.
    solver->setPrintLevel( 2 );

    // Setup the problem.
    solver->setup();

    // Solve the problem.
    solver->solve( *rhs, *lhs );

    // Check the results.
    auto lhs_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), lhs->view() );
    for ( int i = owned_space.min(Dim::I);
          i < owned_space.max(Dim::I);
          ++i )
        for ( int j = owned_space.min(Dim::J);
              j < owned_space.max(Dim::J);
              ++j )
            for ( int k = owned_space.min(Dim::K);
                  k < owned_space.max(Dim::K);
                  ++k )
                EXPECT_EQ( lhs_host(i,j,k,0), -1.0 / 6.0 );

    // Setup the problem again. We would need to do this if we changed the
    // matrix entries.
    solver->setup();

    // Solve the problem again
    ArrayOp::assign( *rhs, 2.0, Own() );
    ArrayOp::assign( *lhs, 0.0, Own() );
    solver->solve( *rhs, *lhs );

    // Check the results again
    lhs_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), lhs->view() );
    for ( int i = owned_space.min(Dim::I);
          i < owned_space.max(Dim::I);
          ++i )
        for ( int j = owned_space.min(Dim::J);
              j < owned_space.max(Dim::J);
              ++j )
            for ( int k = owned_space.min(Dim::K);
                  k < owned_space.max(Dim::K);
                  ++k )
                EXPECT_EQ( lhs_host(i,j,k,0), -1.0 / 3.0 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( structured_solver, poisson_test )
{
    poissonTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
