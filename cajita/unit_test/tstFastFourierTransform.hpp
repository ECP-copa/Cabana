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
template <class HostBackendType>
void forwardReverseTest( bool use_default, bool use_params )
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
    auto vector_layout = createArrayLayout( local_grid, 2, Cell() );
    auto lhs = createArray<double, TEST_DEVICE>( "lhs", vector_layout );
    auto lhs_view = lhs->view();
    auto lhs_host =
        createArray<double, typename decltype( lhs_view )::array_layout,
                    Kokkos::HostSpace>( "lhs_host", vector_layout );
    auto lhs_host_view = Kokkos::create_mirror_view( lhs_view );
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
                lhs_host_view( i, j, k, 0 ) =
                    Kokkos::rand<decltype( rand ), double>::draw( rand, 0.0,
                                                                  1.0 );
                lhs_host_view( i, j, k, 1 ) =
                    Kokkos::rand<decltype( rand ), double>::draw( rand, 0.0,
                                                                  1.0 );
            }

    // Copy to the device.
    Kokkos::deep_copy( lhs_view, lhs_host_view );

    // Create FFT options
    Experimental::FastFourierTransformParams params;

    // Set MPI communication
    params.setAllToAll( true );

    // Set data exchange type (false uses slab decomposition)
    params.setPencils( true );

    // Set data handling (true uses contiguous memory and requires tensor
    // transposition; false uses strided data with no transposition)
    params.setReorder( true );

    if ( use_default && use_params )
    {
        auto fft =
            Experimental::createHeffteFastFourierTransform<double, TEST_DEVICE>(
                *vector_layout, params );
        // Forward transform
        fft->forward( *lhs, Experimental::FFTScaleFull() );
        // Reverse transform
        fft->reverse( *lhs, Experimental::FFTScaleNone() );
    }
    else if ( use_default )
    {
        auto fft =
            Experimental::createHeffteFastFourierTransform<double, TEST_DEVICE>(
                *vector_layout );
        fft->forward( *lhs, Experimental::FFTScaleFull() );
        fft->reverse( *lhs, Experimental::FFTScaleNone() );
    }
#if !defined( KOKKOS_ENABLE_CUDA ) && !defined( KOKKOS_ENABLE_HIP )
    else if ( use_params )
    {
        auto fft = Experimental::createHeffteFastFourierTransform<
            double, TEST_DEVICE, HostBackendType>( *vector_layout, params );
        fft->forward( *lhs, Experimental::FFTScaleFull() );
        fft->reverse( *lhs, Experimental::FFTScaleNone() );
    }
    else
    {
        auto fft = Experimental::createHeffteFastFourierTransform<
            double, TEST_DEVICE, HostBackendType>( *vector_layout );
        fft->forward( *lhs, Experimental::FFTScaleFull() );
        fft->reverse( *lhs, Experimental::FFTScaleNone() );
    }
#endif

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
                EXPECT_FLOAT_EQ( lhs_host_view( i, j, k, 0 ),
                                 lhs_result( i, j, k, 0 ) );
                EXPECT_FLOAT_EQ( lhs_host_view( i, j, k, 1 ),
                                 lhs_result( i, j, k, 1 ) );
            }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( fast_fourier_transform, forward_reverse_test )
{
    // Dummy template argument.
    forwardReverseTest<Experimental::FFTBackendFFTW>( true, true );
    forwardReverseTest<Experimental::FFTBackendFFTW>( true, false );

#ifdef Heffte_ENABLE_FFTW
    forwardReverseTest<Experimental::FFTBackendFFTW>( false, true );
    forwardReverseTest<Experimental::FFTBackendFFTW>( false, false );
#endif
#ifdef Heffte_ENABLE_MKL
    forwardReverseTest<Experimental::FFTBackendMKL>( false, true );
    forwardReverseTest<Experimental::FFTBackendMKL>( false, false );
#endif
}

//---------------------------------------------------------------------------//

} // end namespace Test
