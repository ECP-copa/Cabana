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
#include <Cabana_Grid_FastFourierTransform.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

#include <array>
#include <type_traits>
#include <vector>

using namespace Cabana::Grid;

namespace Test
{

template <class HostBackendType, class ArrayLayoutType, class ArrayType>
void calculateFFT( bool use_default, bool use_params,
                   const ArrayLayoutType vector_layout, ArrayType& lhs )
{
    // Create FFT options
    Experimental::FastFourierTransformParams params;

    // Set MPI communication
    params.setAlltoAll( Cabana::Grid::Experimental::FFTCommPattern::alltoallv );

    // Set data exchange type (false uses slab decomposition)
    params.setPencils( true );

    // Set data handling (true uses contiguous memory and requires tensor
    // transposition; false uses strided data with no transposition)
    params.setReorder( true );

    if ( use_default && use_params )
    {
        auto fft = Experimental::createHeffteFastFourierTransform<
            double, TEST_MEMSPACE>( *vector_layout, params );
        // Forward transform
        fft->forward( *lhs, Experimental::FFTScaleFull() );
        // Reverse transform
        fft->reverse( *lhs, Experimental::FFTScaleNone() );
    }
    else if ( use_default )
    {
        auto fft = Experimental::createHeffteFastFourierTransform<
            double, TEST_MEMSPACE>( *vector_layout );
        fft->forward( *lhs, Experimental::FFTScaleFull() );
        fft->reverse( *lhs, Experimental::FFTScaleNone() );
    }
#if !defined( KOKKOS_ENABLE_CUDA ) && !defined( KOKKOS_ENABLE_HIP ) &&         \
    !defined( KOKKOS_ENABLE_SYCL )
    else if ( use_params )
    {
        auto fft = Experimental::createHeffteFastFourierTransform<
            double, TEST_MEMSPACE, HostBackendType>( *vector_layout, params );
        fft->forward( *lhs, Experimental::FFTScaleFull() );
        fft->reverse( *lhs, Experimental::FFTScaleNone() );
    }
    else
    {
        auto fft = Experimental::createHeffteFastFourierTransform<
            double, TEST_MEMSPACE, HostBackendType>( *vector_layout );
        fft->forward( *lhs, Experimental::FFTScaleFull() );
        fft->reverse( *lhs, Experimental::FFTScaleNone() );
    }
#endif
}

//---------------------------------------------------------------------------//
template <class HostBackendType>
void forwardReverseTest3d( bool use_default, bool use_params )
{
    // Create the global mesh.
    double cell_size = 0.1;
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    std::array<double, 3> global_low_corner = { -1.0, -2.0, -1.0 };
    std::array<double, 3> global_high_corner = { 1.0, 1.0, 0.5 };
    auto global_mesh = createUniformGlobalMesh( global_low_corner,
                                                global_high_corner, cell_size );

    // Create the global grid.
    DimBlockPartitioner<3> partitioner;
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid.
    auto local_grid = createLocalGrid( global_grid, 0 );
    auto owned_space = local_grid->indexSpace( Own(), Cell(), Local() );
    auto ghosted_space = local_grid->indexSpace( Ghost(), Cell(), Local() );

    // Create a random vector to transform..
    auto vector_layout = createArrayLayout( local_grid, 2, Cell() );
    auto lhs = createArray<double, TEST_MEMSPACE>( "lhs", vector_layout );
    auto lhs_view = lhs->view();
    auto lhs_host =
        createArray<double, typename decltype( lhs_view )::array_layout,
                    Kokkos::HostSpace>( "lhs_host", vector_layout );
    auto lhs_host_view = Kokkos::create_mirror_view( lhs_view );
    uint64_t seed =
        global_grid->blockId() + ( 19383747 % ( global_grid->blockId() + 1 ) );
    using rnd_type = Kokkos::Random_XorShift64_Pool<Kokkos::HostSpace>;
    // FIXME: remove when 4.7 required
#if ( KOKKOS_VERSION < 40700 )
    rnd_type pool;
    pool.init( seed, ghosted_space.size() );
#else
    rnd_type pool( seed, ghosted_space.size() );
#endif
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

    // Calculate forward and reverse FFT.
    calculateFFT<HostBackendType>( use_default, use_params, vector_layout,
                                   lhs );

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
template <class HostBackendType>
void forwardReverseTest2d( bool use_default, bool use_params )
{
    // Create the global mesh.
    double cell_size = 0.1;
    std::array<bool, 2> is_dim_periodic = { true, true };
    std::array<double, 2> global_low_corner = { -1.0, -2.0 };
    std::array<double, 2> global_high_corner = { 1.0, 0.5 };
    auto global_mesh = createUniformGlobalMesh( global_low_corner,
                                                global_high_corner, cell_size );

    // Create the global grid.
    DimBlockPartitioner<2> partitioner;
    auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                         is_dim_periodic, partitioner );

    // Create a local grid.
    auto local_grid = createLocalGrid( global_grid, 0 );
    auto owned_space = local_grid->indexSpace( Own(), Cell(), Local() );
    auto ghosted_space = local_grid->indexSpace( Ghost(), Cell(), Local() );

    // Create a random vector to transform..
    auto vector_layout = createArrayLayout( local_grid, 2, Cell() );
    auto lhs = createArray<double, TEST_MEMSPACE>( "lhs", vector_layout );
    auto lhs_view = lhs->view();
    auto lhs_host =
        createArray<double, typename decltype( lhs_view )::array_layout,
                    Kokkos::HostSpace>( "lhs_host", vector_layout );
    auto lhs_host_view = Kokkos::create_mirror_view( lhs_view );
    uint64_t seed =
        global_grid->blockId() + ( 19383747 % ( global_grid->blockId() + 1 ) );
    using rnd_type = Kokkos::Random_XorShift64_Pool<Kokkos::HostSpace>;
    // FIXME: remove when 4.7 required
#if ( KOKKOS_VERSION < 40700 )
    rnd_type pool;
    pool.init( seed, ghosted_space.size() );
#else
    rnd_type pool( seed, ghosted_space.size() );
#endif
    for ( int i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( int j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
        {
            auto rand = pool.get_state( i + j );
            lhs_host_view( i, j, 0 ) =
                Kokkos::rand<decltype( rand ), double>::draw( rand, 0.0, 1.0 );
            lhs_host_view( i, j, 1 ) =
                Kokkos::rand<decltype( rand ), double>::draw( rand, 0.0, 1.0 );
        }

    // Copy to the device.
    Kokkos::deep_copy( lhs_view, lhs_host_view );

    // Calculate forward and reverse FFT.
    calculateFFT<HostBackendType>( use_default, use_params, vector_layout,
                                   lhs );

    // Check the results.
    auto lhs_result =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), lhs->view() );
    for ( int i = owned_space.min( Dim::I ); i < owned_space.max( Dim::I );
          ++i )
        for ( int j = owned_space.min( Dim::J ); j < owned_space.max( Dim::J );
              ++j )
        {
            EXPECT_FLOAT_EQ( lhs_host_view( i, j, 0 ), lhs_result( i, j, 0 ) );
            EXPECT_FLOAT_EQ( lhs_host_view( i, j, 1 ), lhs_result( i, j, 1 ) );
        }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( FastFourier, ForwardReverse3d )
{
    // Dummy template argument.
    forwardReverseTest3d<Experimental::Impl::FFTBackendDefault>( true, true );
    forwardReverseTest3d<Experimental::Impl::FFTBackendDefault>( true, false );

#ifdef Heffte_ENABLE_FFTW
    forwardReverseTest3d<Experimental::FFTBackendFFTW>( false, true );
    forwardReverseTest3d<Experimental::FFTBackendFFTW>( false, false );
#endif
#ifdef Heffte_ENABLE_MKL
    forwardReverseTest3d<Experimental::FFTBackendMKL>( false, true );
    forwardReverseTest3d<Experimental::FFTBackendMKL>( false, false );
#endif
}

TEST( FastFourier, ForwardReverse2d )
{
    // Dummy template argument.
    forwardReverseTest2d<Experimental::Impl::FFTBackendDefault>( true, true );
    forwardReverseTest2d<Experimental::Impl::FFTBackendDefault>( true, false );

#ifdef Heffte_ENABLE_FFTW
    forwardReverseTest2d<Experimental::FFTBackendFFTW>( false, true );
    forwardReverseTest2d<Experimental::FFTBackendFFTW>( false, false );
#endif
#ifdef Heffte_ENABLE_MKL
    forwardReverseTest2d<Experimental::FFTBackendMKL>( false, true );
    forwardReverseTest2d<Experimental::FFTBackendMKL>( false, false );
#endif
}
//---------------------------------------------------------------------------//

} // end namespace Test
