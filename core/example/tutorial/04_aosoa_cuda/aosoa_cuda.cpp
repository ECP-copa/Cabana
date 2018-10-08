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
// Also note that we are passing the AoSoA by value to the global CUDA kernel
// - the data was allocated with device-accessible memory and the AoSoA copy
// constructor/assignment operator is a shallow copy of this
// memory. Therefore, passing the AoSoA by value to this kernel simply copies
// the address to the device-accessible AoSoA memory to be used in this kernel
// - not the memory itself.
//---------------------------------------------------------------------------//
template<class AoSoA_t>
__global__ void initializeData( AoSoA_t aosoa )
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
    if ( a < aosoa.arraySize(s) )
    {
        /*
          Get a reference the SoA we are working on. The aosoa access()
          function gives us a direct reference to the underlying SoA data. We
          use a non-const reference here so that our changes are reflected in
          the data structure. We use auto here for simplicity but the return
          type is Cabana::SoA<MemberTypes,VectorLength>&.
        */
        auto& soa = aosoa.access(s);

        /*
          Next loop over the values in the vector index of each tuple - this is
          the second tuple index. Assign values the same way did in the SoA
          example. Note that the data via the SoA interface is also accessible on
          device.
        */
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                soa.get<0>(a,i,j) = (double) (a + i + j);

        for ( int i = 0; i < 4; ++i )
            soa.get<1>(a,i) = (float) (a + i);

        soa.get<2>(a) = a + 1234;
    }
}

//---------------------------------------------------------------------------//
// AoSoA example cuda.
//---------------------------------------------------------------------------//
void aosoaExample()
{
    /*
      Cabana array-of-structs-of-arrays (AoSoAs) is simply a list of Cabana
      SoAs. A Cabana AoSoA provides a convenient interface to create a
      manipulate these lists. In addition to needing a definition of the tuple
      types using Cabana::MemberTypes as well as defining the vector length
      for the SoAs, we now also need to declare where we want this list of
      SoAs allocated.

      In Cabana we do this with memory spaces (if you are familiar with Kokkos
      memory spaces we borrow directly from their concept). A memory space
      simply indicates what type of function is used to allocate and free memory
      (e.g. malloc() and free() or cudaMalloc() and cudaFree()). Depending on
      the type of memory space used the AoSoA data can be used with different
      types of accelerators or programming models.
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
       Print size data. In this case we have created an AoSoA with 5
       tuples. Because a vector length of 32 is used, a total memory capacity
       for 32 tuples will be allocated in 1 SoA.
    */
    std::cout << "aosoa.size() = " << aosoa.size() << std::endl;
    std::cout << "aosoa.capacity() = " << aosoa.capacity() << std::endl;
    std::cout << "aosoa.numSoA() = " << aosoa.numSoA() << std::endl;

    /*
      There are a variety of ways in which one can access the data within an
      AoSoA in stride-1 vector length sized loops over tuple indices which can
      often have a performance benefit. First we will look at getting
      individual SoA's which will introduce the important concept of
      2-dimensional tuple indices.

      The parallelization strategy with CUDA focuses on warp-level operations
      with the vector length of the SoAs typically being set equivalent to or
      a multiple of the warp size on an NVIDIA card (typically 32 threads) and
      then threading over that data set.

      We achieve this by carefully setting the dimensions of the thread blocks
      to correspond to the AoSoA data layout:
    */
    int num_cuda_block = aosoa.numSoA();
    int cuda_block_size = VectorLength;
    initializeData<<<num_cuda_block,cuda_block_size>>>( aosoa );

    /*
      We are using UVM so synchronize the device to ensure the kernel finishes
      before accessing the memory on the host.
    */
    cudaDeviceSynchronize();

    /*
       Now let's read the data we just wrote but this time we will access the
       data tuple-by-tuple using 1-dimensional indexing rather than using the
       2-dimensional indexing with SoAs.

       Note here that because we used Cuda UVM we can read/write the AoSoA
       data directly on the host with the cost of paging the data from the
       host to the device.
     */
    for ( int t = 0; t < num_tuple; ++t )
    {
        /*
           Get the tuple. Note that this is a copy of the data, not a
           reference. We use auto here for simplicity but the return type is
           Cabana::Tuple<MemberTypes>.
        */
        auto tp = aosoa.getTuple( t );

        /*
          Next read out the data the copy stored in the tuple.
        */
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                std::cout << "Tuple " << t
                          << ", member 0 element (" << i << "," << j << "): "
                          << tp.get<0>(i,j) << std::endl;

        for ( int i = 0; i < 4; ++i )
            std::cout << "Tuple " << t
                      << ", member 1 element (" << i << "): "
                      << tp.get<1>(i) << std::endl;

        std::cout << "Tuple " << t
                  << ", member 2: " << tp.get<2>() << std::endl;
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
