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
// Atomic slice example.
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
    const int VectorLength = 8;
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::OpenMP;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Create the AoSoA. Just put a single value to demonstrate the atomic.
    */
    int num_tuple = 1;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "X", num_tuple );

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
      Because the slice is declared to be atomic we can safely sum into it
      from all threads. Loop over the data a number of times in parallel and
      do an atomic sum.
    */
    int num_loop = 256;
#pragma omp parallel for
    for ( int s = 0; s < num_loop; ++s )
        atomic_slice( 0 ) += 1.0;

    /*
      Print out the slice result - it should be equal to the number of
      parallel loop ierations. Note we can use the original slice as the
      atomic slice is just an alias of this with different memory access
      traits.
     */
    std::cout << "Atomic add results" << std::endl;
    std::cout << "Parallel loop iterations = " << num_loop << std::endl;
    std::cout << "Slice value =              " << slice( 0 ) << std::endl;
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
