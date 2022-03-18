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
#include <iostream>
#include <random>

//---------------------------------------------------------------------------//
// heFFTe Fast Fourier Transform example.
//---------------------------------------------------------------------------//
void fastFourierTransformHeffteExample()
{
    /*
      This example shows how to perform a fast Fourier transform with Cajita.
      The current Cajita FFTs take advantage of the heFFTe library, with plans
      to support more FFT libraries.

      The basic steps for performing FFTs in Cajita are:
      1. Create a mesh
      2. Create the complex data vector you would like to transform.
      3. Create the FFT and set any options you would like (detailed more below)
    */
    std::cout << "Cajita heFFTE Fast Fourier Transform Example\n" << std::endl;

    /*
       Create the global mesh.
       In this example we create a uniform 3D periodic mesh, which is not
       symmetric. Details on creating a global mesh, grid, and partitioner can
       be found in previous Cajita examples

       Declare the device memory and execution space to use
       In this example, we use the host space for execution with the default
       host execution space.
    */
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    double cell_size = 0.5;
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    std::array<double, 3> global_low_corner = { -1.0, -2.0, -1.0 };
    std::array<double, 3> global_high_corner = { 1.0, 1.0, 0.5 };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Create the global grid.
    Cajita::DimBlockPartitioner<3> partitioner;
    auto global_grid = Cajita::createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

    // Create a local grid.
    auto local_grid = Cajita::createLocalGrid( global_grid, 0 );
    auto owned_space = local_grid->indexSpace( Cajita::Own(), Cajita::Cell(),
                                               Cajita::Local() );

    /*
       Create a random vector of complex data on which we will perform FFTs.

       Note that for every position there are two values. These correspond to
       the real (index 0) and imaginary (index 1) parts.

       Also note that this is done on the host with standard library random
       generation, but could also be done using Kokkos::Random
    */
    auto vector_layout =
        Cajita::createArrayLayout( local_grid, 2, Cajita::Cell() );
    auto lhs = Cajita::createArray<double, DeviceType>( "lhs", vector_layout );
    auto lhs_view = lhs->view();
    const uint64_t seed =
        global_grid->blockId() + ( 19383747 % ( global_grid->blockId() + 1 ) );
    std::mt19937 gen( seed );
    std::uniform_real_distribution<> dis( 0.0, 1.0 );
    for ( int i = owned_space.min( Cajita::Dim::I );
          i < owned_space.max( Cajita::Dim::I ); ++i )
        for ( int j = owned_space.min( Cajita::Dim::J );
              j < owned_space.max( Cajita::Dim::J ); ++j )
            for ( int k = owned_space.min( Cajita::Dim::K );
                  k < owned_space.max( Cajita::Dim::K ); ++k )
            {
                lhs_view( i, j, k, 0 ) = dis( gen );
                lhs_view( i, j, k, 1 ) = dis( gen );
            }

    // Copy this initial state to compare later.
    auto lhs_init = Cajita::ArrayOp::clone( *lhs );
    Cajita::ArrayOp::copy( *lhs_init, *lhs, Cajita::Own() );
    auto lhs_init_view = lhs_init->view();

    /*
      Create FFT options, which are set first, then passed to the FFT
      constructor.

      These options reflect those exposed by the heFFTe API and could grow or
      changed based on additional FFT library support.
    */
    Cajita::Experimental::FastFourierTransformParams params;

    // Set communication to use all-to-all MPI communication.
    // Set this option to false for point-to-point communication
    params.setAllToAll( true );

    // Set data exchange type to use pencil decomposition
    // Set this option to false to use slab decomposition
    params.setPencils( true );

    // Set data handling (true uses contiguous memory and requires tensor
    // transposition; false uses strided data with no transposition)
    params.setReorder( true );

    /*
      The three options set above are actually choosing the default
      parameters for Cajita FFTs, and are set just to provide an example.
    */

    /*
       Create the FFT with the set parameters

       Cajita (through heFFTe) has the following FFT backends available:
        - FFTW   (default for host execution)
        - mkl    (optional host execution option)
        - cufft  (default with Cuda execution)
        - rocfft (default with HIP execution)

       In this example we use the default FFT backend type based on the
       execution space and enabled heFFTe backends, but you could explicitly
       set an appropriate backend by adding an additional template parameter
       (Cajita::Experimental::FFTBackendMKL in this case) to the constructor.
    */
    auto fft = Cajita::Experimental::createHeffteFastFourierTransform<
        double, DeviceType>( *vector_layout, params );

    /*
      Now forward or reverse transforms can be performed via
      fft->forward, or fft->reverse, respectively.

      FFTs are often scaled, since the result of a forward and reverse
      transform in sequence gives the same data but with each value multiplied
      by N (where N is the number of values) Cajita FFTs provide (through
      heFFTe) the following scaling options:
      full scaling (divide by N) -> Experimental::FFTScaleFull()
      no scaling -> Experimental::FFTScaleNone()
      symmetric scaling (divide by sqrt(N)) -> Experimental::FFTScaleSymmetric()
      Here we use a common choice to use full scaling, followed by no scaling.
    */

    // Perform a forward transform with full scaling
    fft->forward( *lhs, Cajita::Experimental::FFTScaleFull() );
    // Perform a reverse transform with no scaling
    fft->reverse( *lhs, Cajita::Experimental::FFTScaleNone() );

    /*
      Print the results. Given the scaling choices, we expect the results to be
      equal to the original values.
    */
    std::cout << "Complex pairs:" << std::endl;
    lhs_view = lhs->view();
    for ( int i = owned_space.min( Cajita::Dim::I );
          i < owned_space.max( Cajita::Dim::I ); ++i )
        for ( int j = owned_space.min( Cajita::Dim::J );
              j < owned_space.max( Cajita::Dim::J ); ++j )
            for ( int k = owned_space.min( Cajita::Dim::K );
                  k < owned_space.max( Cajita::Dim::K ); ++k )
            {
                std::cout << "index: (" << i << ", " << j << ", " << k << ") "
                          << "initial: (" << lhs_init_view( i, j, k, 0 ) << ", "
                          << lhs_init_view( i, j, k, 1 ) << ")  result: ("
                          << lhs_view( i, j, k, 0 ) << ", "
                          << lhs_view( i, j, k, 1 ) << ")" << std::endl;
            }
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

        fastFourierTransformHeffteExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
