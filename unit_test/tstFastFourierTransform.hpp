/*******************************************n*********************************
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
#include <Cajita_IndexSpace.hpp>
#include <Cajita_FastFourierTransform.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <vector>
#include <array>
#include <type_traits>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void memoryTest()
{
    auto mtype = HeffteMemoryTraits<TEST_MEMSPACE>::value;
    HEFFTE_NS::Memory fft_mem;
    fft_mem.memory_type = mtype;
    int size = 12;
    int nbytes = size * sizeof(double);
    double* ptr = (double*) fft_mem.smalloc( nbytes, mtype );
    EXPECT_NE( ptr, nullptr );
    fft_mem.sfree( ptr, mtype );
}

//---------------------------------------------------------------------------//
void forwardReverseTest()
{
    // Create the global mesh.
    double cell_size = 0.1;
    std::array<bool,3> is_dim_periodic = {true,true,true};
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
    auto local_mesh = createLocalGrid( global_grid, 0 );
    auto owned_space = local_mesh->indexSpace(Own(),Cell(),Local());

    // Create the LHS.
    auto vector_layout = createArrayLayout( local_mesh, 1, Cell() );
    auto lhs = createArray<Kokkos::complex<double>,TEST_DEVICE>( "lhs", vector_layout );
    auto lhs_view = lhs->view();
    Kokkos::parallel_for(
        "fill_lhs",
        createExecutionPolicy(owned_space,TEST_EXECSPACE()),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            lhs_view(i,j,k,0).real( 1.1 );
            lhs_view(i,j,k,0).imag( -2.5 );
        });

    // Create an FFT
    auto fft = createFastFourierTransform<double,TEST_DEVICE>(
        *vector_layout, FastFourierTransformParams{}.setCollectiveType( 2 ).setExchangeType( 0 ).setPackType( 2 ).setScalingType( 1 ) );

    // Forward transform
    fft->forward( *lhs );

    // Reverse transform
    fft->reverse( *lhs );

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
            {
                EXPECT_EQ( lhs_host(i,j,k,0).real(), 1.1 );
                EXPECT_EQ( lhs_host(i,j,k,0).imag(), -2.5 );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( fast_fourier_transform, memory_test )
{
    memoryTest();
}

//---------------------------------------------------------------------------//
TEST( fast_fourier_transform, forward_reverse_test )
{
    forwardReverseTest();
}

//---------------------------------------------------------------------------//

} // end namespace Test
