/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
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
// If you are comparing this code to the other AoSoA examples note that the
// identical syntax is use here - the interface is portable accross
// programming models.
//
// Also note that we are passing the slices by value to the global CUDA kernel
// - the data was allocated with device-accessible memory and the slice copy
// constructor/assignment operator is a shallow copy of this
// memory. Therefore, passing the slice by value to this kernel simply copies
// the address to the device-accessible slice memory to be used in this kernel
// - not the memory itself.
//---------------------------------------------------------------------------//
template<class Slice0, class Slice1, class Slice2>
__global__
void initializeData( Slice0 slice_0, Slice1 slice_1, Slice2 slice_2 )
{
    /*
      Get the indices to operate on from the thread data. The SoA index is the
      block index and the array index within the SoA is the thread index in
      the block.
     */
    auto s = blockIdx.x;
    auto a = threadIdx.x;

    /*
      Only operate on the data associated with this thread if it is
      valid. Note here that we are using the aosoa function `arraySize()`
      to determine how many tuples are in the current SoA. Because the
      number of tuples in an AoSoA may not be evenly divisible by the
      vector length of the SoAs, the last SoA may not be completely full.
    */
    if ( a < slice_0.arraySize(s) )
    {
        /*
          Next loop over the values in the vector index of each tuple - this is
          the second tuple index. Assign values the same way did in the SoA
          example. Note that the data via the SoA interface is also accessible on
          device.
        */
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                slice_0.access(s,a,i,j) = (double) (a + i + j);

        for ( int i = 0; i < 4; ++i )
            slice_1.access(s,a,i) = (float) (a + i);

        slice_2.access(s,a) = a + 1234;
    }
}

//---------------------------------------------------------------------------//
// AoSoA example cuda.
//---------------------------------------------------------------------------//
void aosoaExample()
{
    /*
      Slices are a mechanism to access a tuple member across all tuples in an
      AoSoA as if it were one large multidimensional array. In this basic
      example we will demonstrate the way in which a user can generate a slice
      and access it's data using CUDA.

      Slices, like the AoSoA from which they are derived, can be accessed via
      1-dimensional and 2-dimensional indices with the later exposing
      vectorizable loops over the inner elements of each SoA in the data
      structure.
    */

    /*
       Start by declaring the types in our tuples will store. Store a rank-2
       array of doubles, a rank-1 array of floats, and a single integer in
       each tuple.
    */
    using DataTypes = Cabana::MemberTypes<double[3][3],
                                          float[4],
                                          int>;

    /*
      Next declare the vector length of our SoAs. This is how many tuples the
      SoAs will contain. A reasonable number for performance should be some
      multiple of the vector length on the machine you are using.
    */
    const int VectorLength = 32;

    /*
      Finally declare the memory space in which the AoSoA will be
      allocated. In this example we are writing basic loops that will execute
      on an NVIDIA GPU using the CUDA runtime. The CudaUVMSpace allocates
      memory in managed GPU memory via `cudaMallocManaged`. This memory is
      automatically paged between host and device depending on the context in
      which the memory is accessed.
    */
    using MemorySpace = Cabana::CudaUVMSpace;

    /*
       Create the AoSoA. We define how many tuples the aosoa will
       contain. Note that if the number of tuples is not evenly divisible by
       the vector length then the last SoA in the AoSoA will not be entirely
       full (although its memory will still be allocated).
    */
    int num_tuple = 5;
    Cabana::AoSoA<DataTypes,MemorySpace,VectorLength> aosoa( num_tuple );

    /*
      Create a slice over each tuple member in the AoSoA. An integer template
      parameter is used to indicate which member to slice.
     */
    auto slice_0 = aosoa.slice<0>();
    auto slice_1 = aosoa.slice<1>();
    auto slice_2 = aosoa.slice<2>();

    /*
      Initialize the aosoa data using the slices. One benefit of using the
      slice approach over the AoSoA tuple and SoA access approach is that
      kernels (such as the one above) can be written assuming they are
      receiving general slices of data which may or may not come from multiple
      AoSoA objects. The syntax is therefore simpler because the kernel writer
      no longer needs to know the compile-time constant integer corresponding
      to the member index of each data member.

      The parallelization strategy with CUDA focuses on warp-level operations
      with the vector length of the SoAs typically being set equivalent to or
      a multiple of the warp size on an NVIDIA card (typically 32 threads) and
      then threading over that data set.

      We achieve this by carefully setting the dimensions of the thread blocks
      to correspond to the AoSoA data layout:
    */
    int num_cuda_block = aosoa.numSoA();
    int cuda_block_size = VectorLength;
    initializeData<<<num_cuda_block,cuda_block_size>>>(
        slice_0, slice_1, slice_2 );

    /*
      We are using UVM so synchronize the device to ensure the kernel finishes
      before accessing the memory on the host.
    */
    cudaDeviceSynchronize();

    /*
       Now let's read the data we just wrote but this time we will access the
       data tuple-by-tuple using 1-dimensional indexing rather than using the
       2-dimensional.

       Note here that because we used Cuda UVM we can read/write the AoSoA
       data directly on the host with the cost of paging the data from the
       host to the device.

       Note that the slice data access syntax in 1D uses `operator()`.
     */
    for ( int t = 0; t < num_tuple; ++t )
    {
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                std::cout << "Tuple " << t
                          << ", member 0 element (" << i << "," << j << "): "
                          << slice_0(t,i,j) << std::endl;

        for ( int i = 0; i < 4; ++i )
            std::cout << "Tuple " << t
                      << ", member 1 element (" << i << "): "
                      << slice_1(t,i) << std::endl;

        std::cout << "Tuple " << t
                  << ", member 2: " << slice_2(t) << std::endl;
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Cabana::initialize(argc,argv);

    aosoaExample();

    Cabana::finalize();

    return 0;
}

//---------------------------------------------------------------------------//
