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
#include <Cajita_FastFourierTransform.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <type_traits>
#include <vector>

using namespace Cajita;

namespace Test
{
//---------------------------------------------------------------------------//
void memoryTest()
{
    auto mtype = Experimental::HeffteMemoryTraits<TEST_MEMSPACE>::value;
    HEFFTE::Memory fft_mem;
    fft_mem.memory_type = mtype;
    int size = 12;
    int nbytes = size * sizeof( double );
    double* ptr = (double*)fft_mem.smalloc( nbytes, mtype );
    EXPECT_NE( ptr, nullptr );
    fft_mem.sfree( ptr, mtype );
}

//---------------------------------------------------------------------------//
void forwardReverseTest()
{
    // Create the global mesh.
    double cell_size = 0.1;
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    std::array<double, 3> global_low_corner = { -1.0, -2.0, -1.0 };
    std::array<double, 3> global_high_corner = { 1.0, 1.0, 0.5 };
    auto global_mesh = createUniformGlobalMesh( global_low_corner,
                                                global_high_corner, cell_size );

    // Create the global grid.
    UniformDimPartitioner partitioner;
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid.
    auto local_grid = createLocalGrid( global_grid, 0 );
    auto owned_space = local_grid->indexSpace( Own(), Cell(), Local() );
    auto ghosted_space = local_grid->indexSpace( Ghost(), Cell(), Local() );

    // Create a random vector to transform..
    auto vector_layout = createArrayLayout( local_grid, 1, Cell() );
    auto lhs = createArray<Kokkos::complex<double>, TEST_DEVICE>(
        "lhs", vector_layout );
    auto lhs_view = lhs->view();
    auto lhs_host = createArray<Kokkos::complex<double>,
                                typename decltype( lhs_view )::array_layout,
                                Kokkos::HostSpace>( "lhs_host", vector_layout );
    auto lhs_host_view = lhs_host->view();
    uint64_t seed =
        global_grid->blockId() + ( 19383747 % ( global_grid->blockId() + 1 ) );
    using rnd_type = Kokkos::Random_XorShift64_Pool<Kokkos::HostSpace>;
    rnd_type pool;
    pool.init( seed, ghosted_space.size() );
    for ( int i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( int j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
            for ( int k = owned_space.min( Dim::K );
                  k < owned_space.max( Dim::K ); ++k )
            {
                auto rand = pool.get_state( i + j + k );
                lhs_host_view( i, j, k, 0 ).real() =
                    Kokkos::rand<decltype( rand ), double>::draw( rand, 0.0,
                                                                  1.0 );
                lhs_host_view( i, j, k, 0 ).imag() =
                    Kokkos::rand<decltype( rand ), double>::draw( rand, 0.0,
                                                                  1.0 );
            }

    // Copy to the device.
    Kokkos::deep_copy( lhs_view, lhs_host_view );

    // Create an FFT // ! Old version
    // auto fft = Experimental::createFastFourierTransform<double, TEST_DEVICE>(
    //     *vector_layout, Experimental::FastFourierTransformParams{}
    //                         .setCollectiveType( 2 )
    //                         .setExchangeType( 0 )
    //                         .setPackType( 2 )       // ! Now there is only a single Pack kernel
    //                         .setScalingType( 1 ) ); // ! Now scale is input for the compute kernel

    //* New heFFTe version for create FFT plans
    
    //* Define the FFT backend 
    auto backend_tag = heffte::backend::cufft;  //* can also be backend::fftw, backend::mkl

    //* Instantiate a set of default parameters according to the backend type
    heffte::plan_options params = heffte::default_options<backend_tag>();
    
    //* Choose the desired options
    //* 1. MPI communication
    params.use_alltoall = true;
    // params.use_alltoall = false;  //* MPI point-to-point

    //* 2. Data exchange type  
    params.use_pencils = true;  //* Pencil decomposition
    // params.use_pencils = false; //* Slab decomposition

    //* 3. Data handling
    params.use_reorder = true;  //* Use data in contiguous memory (requires tensor transposition)
    // params.use_reorder = false; //* Use strided data (does not require tensor transposition)

    auto fft = Experimental::createFastFourierTransform<backend_tag, double, TEST_DEVICE>(
        *vector_layout, params);                            

    // Forward transform
    fft->forward( *lhs );

    // Reverse transform
    fft->reverse( *lhs );

    // Check the results.
    auto lhs_result =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), lhs->view() );
    for ( int i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( int j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
            for ( int k = owned_space.min( Dim::K );
                  k < owned_space.max( Dim::K ); ++k )
            {
                EXPECT_FLOAT_EQ( lhs_host_view( i, j, k, 0 ).real(),
                                 lhs_result( i, j, k, 0 ).real() );
                EXPECT_FLOAT_EQ( lhs_host_view( i, j, k, 0 ).imag(),
                                 lhs_result( i, j, k, 0 ).imag() );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
//! There should not be any bug with the modifications made, please contact heFFTe team otherwise.
// NOTE: This test exposes the GPU FFT memory bug in HEFFTE. Re-enable this
// when we enable GPU FFTs to test the bug.
// TEST( fast_fourier_transform, memory_test )
// {
//     memoryTest();
// }

//---------------------------------------------------------------------------//
TEST( fast_fourier_transform, forward_reverse_test ) { forwardReverseTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
