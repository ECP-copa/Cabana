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

#include <iostream>

//---------------------------------------------------------------------------//
// Slice example.
//---------------------------------------------------------------------------//
void sliceExample()
{
    /*
      Slices are a mechanism to access a tuple member across all tuples in an
      AoSoA as if it were one large multidimensional array. In this basic
      example we will demonstrate the way in which a user can generate a slice
      and access it's data.

      Slices, like the AoSoA from which they are derived, can be accessed via
      1-dimensional and 2-dimensional indices with the latter exposing
      vectorizable loops over the inner elements of each SoA in the data
      structure.

      A slice does not copy the AoSoA data - rather it simply points to the
      data as an unmanaged shallow copy. Because there is no reference
      counting associated with a slice, the user must take care not to delete
      the AoSoA from which the slice is derived before they are done using the
      slice. In addition, if the slice memory is changed in any way (i.e. from
      resizing, changing the capacity, or any other types of changes to the
      allocation) the slice is no longer valid and must be recreated from the
      new AoSoA data.
    */

    /*
       Start by declaring the types our tuples will store. Store a rank-2
       array of doubles, a rank-1 array of floats, and a single integer in
       each tuple.
    */
    using DataTypes = Cabana::MemberTypes<double[3][3], float[4], int>;

    /*
      Next declare the vector length of our SoAs. This is how many tuples the
      SoAs will contain. A reasonable number for performance should be some
      multiple of the vector length on the machine you are using.
    */
    const int VectorLength = 4;

    /*
      Finally declare the memory space in which the AoSoA will be allocated
      and the execution space in which kernels will execute. In this example
      we are writing basic loops that will execute on the CPU. The HostSpace
      allocates memory in standard CPU RAM.

      Kokkos also supports execution on NVIDIA GPUs. To create an AoSoA
      allocated with CUDA Unified Virtual Memory (UVM) use
      `Kokkos::CudaUVMSpace` instead of `Kokkos::HostSpace`. The CudaUVMSpace
      allocates memory in managed GPU memory via `cudaMallocManaged`. This
      memory is automatically paged between host and device depending on the
      context in which the memory is accessed.
    */
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Create the AoSoA. We define how many tuples the aosoa will
       contain. Note that if the number of tuples is not evenly divisible by
       the vector length then the last SoA in the AoSoA will not be entirely
       full (although its memory will still be allocated).
    */
    int num_tuple = 5;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "my_aosoa",
                                                              num_tuple );

    /*
      Create a slice over each tuple member in the AoSoA. An integer template
      parameter is used to indicate which member to slice. A slice object
      simply wraps the data associated with an AoSoA member in a more
      conventient accessor structure. A slice therefore has the same memory
      space as the AoSoA from which it was derived. Slices may optionally be
      assigned a label. This label is not included in the memory tracker
      because slices are unmanaged memory but may still be used for diagnostic
      purposes.
    */
    auto slice_0 = Cabana::slice<0>( aosoa, "my_slice_0" );
    auto slice_1 = Cabana::slice<1>( aosoa, "my_slice_1" );
    auto slice_2 = Cabana::slice<2>( aosoa, "my_slice_2" );

    /*
      Let's initialize the data using the 2D indexing scheme. Slice data can
      be accessed in 2D using the `access()` function. Note that both the SoA
      index and the array index are passed to this function.

      Also note that the slice object has a similar interface to the AoSoA for
      accessing the total number of tuples in the data structure and the array
      sizes.
    */
    for ( std::size_t s = 0; s < slice_0.numSoA(); ++s )
    {
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                for ( std::size_t a = 0; a < slice_0.arraySize( s ); ++a )
                    slice_0.access( s, a, i, j ) = 1.0 * ( a + i + j );

        for ( int i = 0; i < 4; ++i )
            for ( std::size_t a = 0; a < slice_0.arraySize( s ); ++a )
                slice_1.access( s, a, i ) = 1.0 * ( a + i );

        for ( std::size_t a = 0; a < slice_0.arraySize( s ); ++a )
            slice_2.access( s, a ) = a + 1234;
    }

    /*
       Now let's read the data we just wrote but this time we will access the
       data tuple-by-tuple using 1-dimensional indexing rather than using the
       2-dimensional indexing.

       As with the AoSoA, the upside to this approach is that the indexing is
       easy. The downside is that we lose the stride-1 vector length loops
       over the array index.

       Note that the slice data access syntax in 1D uses `operator()`.
     */
    for ( int t = 0; t < num_tuple; ++t )
    {
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                std::cout << "Tuple " << t << ", member 0 element (" << i << ","
                          << j << "): " << slice_0( t, i, j ) << std::endl;

        for ( int i = 0; i < 4; ++i )
            std::cout << "Tuple " << t << ", member 1 element (" << i
                      << "): " << slice_1( t, i ) << std::endl;

        std::cout << "Tuple " << t << ", member 2: " << slice_2( t )
                  << std::endl;
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    sliceExample();

    return 0;
}

//---------------------------------------------------------------------------//
