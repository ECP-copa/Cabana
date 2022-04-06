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
// AoSoA example.
//---------------------------------------------------------------------------//
void aosoaExample()
{
    /*
      Cabana array-of-structs-of-arrays (AoSoAs) is simply a list of Cabana
      SoAs. A Cabana AoSoA provides a convenient interface to create and
      manipulate these lists. In addition to needing a definition of the tuple
      types using Cabana::MemberTypes, as well as defining the vector length
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

      Kokkos also supports execution on NVIDIA GPUs. For example, to create an
      AoSoA allocated with CUDA Unified Virtual Memory (UVM) use
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
       full (although its memory will still be allocated). The AoSoA label
       allows one to track the managed memory in an AoSoA through the Kokkos
       allocation tracker.
    */
    int num_tuple = 5;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "my_aosoa",
                                                              num_tuple );

    /*
       Print the label and size data. In this case we have created an AoSoA
       with 5 tuples. Because a vector length of 4 is used, a total memory
       capacity for 8 tuples will be allocated in 2 SoAs.
    */
    std::cout << "aosoa.label() = " << aosoa.label() << std::endl;
    std::cout << "aosoa.size() = " << aosoa.size() << std::endl;
    std::cout << "aosoa.capacity() = " << aosoa.capacity() << std::endl;
    std::cout << "aosoa.numSoA() = " << aosoa.numSoA() << std::endl;

    /*
      There are a variety of ways in which one can access the data within an
      AoSoA in stride-1 vector length sized loops over tuple indices which can
      often have a performance benefit. First we will look at getting
      individual SoA's which will introduce the important concept of
      2-dimensional tuple indices. Start by looping over the SoA's. The SoA
      index is the first tuple index:
    */
    for ( std::size_t s = 0; s < aosoa.numSoA(); ++s )
    {
        /*
           Get a reference the SoA we are working on. The aosoa access()
           function gives us a direct reference to the underlying SoA data. We
           use a non-const reference here so that our changes are reflected in
           the data structure. We use auto here for simplicity but the return
           type is Cabana::SoA<MemberTypes,VectorLength>&.
        */
        auto& soa = aosoa.access( s );

        /*
          Next loop over the values in the vector index of each tuple - this
          is the second tuple index. Note here that we are using the aosoa
          function `arraySize()` to determine how many tuples are in the
          current SoA. Because the number of tuples in an AoSoA may not be
          evenly divisible by the vector length of the SoAs, the last SoA may
          not be completely full. Assign values the same way did in the SoA
          example.
        */
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                for ( std::size_t a = 0; a < aosoa.arraySize( s ); ++a )
                    Cabana::get<0>( soa, a, i, j ) = 1.0 * ( a + i + j );

        for ( int i = 0; i < 4; ++i )
            for ( std::size_t a = 0; a < aosoa.arraySize( s ); ++a )
                Cabana::get<1>( soa, a, i ) = 1.0 * ( a + i );

        for ( std::size_t a = 0; a < aosoa.arraySize( s ); ++a )
            Cabana::get<2>( soa, a ) = a + 1234;
    }

    /*
       Now let's read the data we just wrote but this time we will access the
       data tuple-by-tuple using 1-dimensional indexing rather than using the
       2-dimensional indexing with SoAs.

       The upside to this approach is that the indexing is easy. The downsides
       are that we lose the stride-1 vector length loops over the tuple index
       and by extracting an individual tuple, we are making a copy of the data
       rather than getting a reference.
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
                std::cout << "Tuple " << t << ", member 0 element (" << i << ","
                          << j << "): " << Cabana::get<0>( tp, i, j )
                          << std::endl;

        for ( int i = 0; i < 4; ++i )
            std::cout << "Tuple " << t << ", member 1 element (" << i
                      << "): " << Cabana::get<1>( tp, i ) << std::endl;

        std::cout << "Tuple " << t << ", member 2: " << Cabana::get<2>( tp )
                  << std::endl;
    }

    /*
      We can set data using a tuple too. Let's assign the tuple at 1D index 3
      some new data. First create the new tuple:
    */
    Cabana::Tuple<DataTypes> foo;

    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            Cabana::get<0>( foo, i, j ) = 1.1;

    for ( int i = 0; i < 4; ++i )
        Cabana::get<1>( foo, i ) = 2.2;

    Cabana::get<2>( foo ) = 3;

    /* Now assign it's data by copying it to the AoSoA at 1D index 3. */
    aosoa.setTuple( 3, foo );

    /*
      Now print using the 2D indexing to be sure we changed the data with the
      assignment.
    */
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            std::cout << "Updated tuple member 0 element (" << i << "," << j
                      << "): " << Cabana::get<0>( aosoa.access( 0 ), 3, i, j )
                      << std::endl;

    for ( int i = 0; i < 4; ++i )
        std::cout << "Update tuple member 1 (" << i
                  << "): " << Cabana::get<1>( aosoa.access( 0 ), 3, i )
                  << std::endl;

    std::cout << "Updated tuple member 2: "
              << Cabana::get<2>( aosoa.access( 0 ), 3 ) << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    aosoaExample();

    return 0;
}

//---------------------------------------------------------------------------//
