/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Core.hpp>

#include <cuda_runtime.h>

#include <iostream>

//---------------------------------------------------------------------------//
// Global Cuda function for initializing AoSoA data via the SoA accessor.
//
// Because the slice is declared to be atomic we can safely sum into it from
// all threads.
//---------------------------------------------------------------------------//
template <class AtomicSlice>
__global__ void atomicThreadSum( AtomicSlice slice )
{
    /* Every thread writes to the slice atomically. */
    slice( 0 ) += 1.0;
}

//---------------------------------------------------------------------------//
// Atomic slice example using cuda.
//---------------------------------------------------------------------------//
void atomicSliceExample()
{
    /*
      Slices have optional memory traits which define the type of data access
      used when manipulating the slice. Upon construction via an AoSoA a slice
      has default access traits. Other traits may be assigned by creating a
      slice of a new type. In this example we will demonstrate using atomic
      operations with slices.
    */

    /*
      Declare the AoSoA parameters.
    */
    using DataTypes = Cabana::MemberTypes<double>;
    const int VectorLength = 32;
    using MemorySpace = Kokkos::CudaUVMSpace;
    using ExecutionSpace = Kokkos::Cuda;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Create the AoSoA. Just put a single value to demonstrate the atomic.
    */
    int num_tuple = 1;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "A", num_tuple );

    /*
      Create a slice over the single value and assign it to zero.
     */
    auto slice = Cabana::slice<0>( aosoa );
    slice( 0 ) = 0.0;

    /*
      Now create a version of the slice with atomic data access traits. We do
      this by simply assigning to an atomic slice type - the assignment is a
      shallow and unmanaged copy. When both 1D and 2D data accesses are used
      reads and writes are atomic.
    */
    decltype( slice )::atomic_access_slice atomic_slice = slice;

    /*
      Now do a parallel sum atomically on the atomic slice. Launch a bunch of
      threads and have each thread add to the atomic slice.
    */
    int num_cuda_block = 1;
    int cuda_block_size = 256;
    atomicThreadSum<<<num_cuda_block, cuda_block_size>>>( atomic_slice );

    /*
      We are using UVM so synchronize the device to ensure the kernel finishes
      before accessing the memory on the host.
    */
    cudaDeviceSynchronize();

    /*
      Print out the slice result - it should be equal to the number of CUDA
      threads. Note we can use the original slice as the atomic slice is just
      an alias of this with different memory access traits.
     */
    std::cout << "Atomic add results" << std::endl;
    std::cout << "Num CUDA threads = " << num_cuda_block * cuda_block_size
              << std::endl;
    std::cout << "Slice value =      " << slice( 0 ) << std::endl;
    std::cout << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    atomicSliceExample();

    return 0;
}

//---------------------------------------------------------------------------//
