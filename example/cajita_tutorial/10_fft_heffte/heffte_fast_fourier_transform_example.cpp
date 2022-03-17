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
// Fast Fourier Transform example.
//---------------------------------------------------------------------------//
void FastFourierTransformExample()
{
    /*
      This example shows how to perform a fast fourier transform with Cajita.
      The current Cajita FFTs take advantage of the heFFTe library, though
      in the future we plan to support more FFT libraries.

      The basic steps for performing FFTs in Cajita are:

      1. Create a mesh

      2. Create the complex data vector you would like to transform.

      3. Create the FFT and set any options you would like (detailed more below)
*/

    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    if ( comm_rank == 0 )
    {
        std::cout << "Cajita Fast Fourier Transform Example\n" << std::endl;
        std::cout << "    (intended to be run with MPI)\n" << std::endl;
    }

    /*
       Create the global mesh.
       In this example we create a uniform 3D periodic mesh, which is not
       symmetric. Details on creating a global mesh, grid, and partitioner can
       be found in previous Cajita examples

       Declare the device memory and execution space to use
       In this example, we use the host space for execution with OpenMP
    */
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Declare which FFT backend to use.
       Cajita (through heFFTe) has the following backend FFT libraries
       available:
       FFTW (default for host execution)
       cufft (default with Cuda execution)
       mkl   (default with MKL execution)
       rocfft (default with HIP execution)
       In this example we use the default FFT backend type based on the
       execution space and enabled heFFTe backends, but you could explicitly
       set an appropriate backend such as Cajita::Experimental::FFTBackendFFTW
    */
    using FFTBackendType = Cajita::Experimental::Impl::FFTBackendDefault;

    double cell_size = 0.1;
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

    // Create a random vector to transform.
    // This is the vector of complex data on which we will perform FFTs
    // Note that for every position there are two values. These correspond
    // to the real (index 0) and imaginary (index 1) parts
    auto vector_layout =
        Cajita::createArrayLayout( local_grid, 2, Cajita::Cell() );
    auto lhs = Cajita::createArray<double, DeviceType>( "lhs", vector_layout );
    auto lhs_view = lhs->view();
    auto lhs_host =
        Cajita::createArray<double, typename decltype( lhs_view )::array_layout,
                            Kokkos::HostSpace>( "lhs_host", vector_layout );
    auto lhs_host_view = Kokkos::create_mirror_view( lhs_view );
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
                lhs_host_view( i, j, k, 0 ) = dis( gen );
                lhs_host_view( i, j, k, 1 ) = dis( gen );
            }

    // A deep_copy to the device would be needed
    // if we were using a device instead. (shown here as example)
    // Kokkos::deep_copy( lhs_view, lhs_host_view );

    // Create FFT options
    // FFT options are set first, then passed to the FFT constructor.
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

    // The three options set above are actually choosing the default
    // parameters for Cajita FFTs, and are set just to provide an example.

    // Create the FFT with the set parameters
    auto fft = Cajita::Experimental::createHeffteFastFourierTransform<
        double, DeviceType, FFTBackendType>( *vector_layout, params );

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

    // Print the results
    // Given the scaling choices, we expect the results to be equal to the
    // original values
    auto lhs_result =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), lhs->view() );
    for ( int i = owned_space.min( Cajita::Dim::I );
          i < owned_space.max( Cajita::Dim::I ); ++i )
        for ( int j = owned_space.min( Cajita::Dim::J );
              j < owned_space.max( Cajita::Dim::J ); ++j )
            for ( int k = owned_space.min( Cajita::Dim::K );
                  k < owned_space.max( Cajita::Dim::K ); ++k )
            {
                if ( comm_rank == 0 )
                {
                    std::cout << "position: (" << i << ", " << j << ", " << k
                              << ") "
                              << "original complex pair: ("
                              << lhs_host_view( i, j, k, 0 ) << ", "
                              << lhs_host_view( i, j, k, 1 ) << ")  result: ("
                              << lhs_result( i, j, k, 0 ) << ", "
                              << lhs_result( i, j, k, 1 ) << ")" << std::endl;
                }
            }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    Kokkos::ScopeGuard scope_guard( argc, argv );

    FastFourierTransformExample();

    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
