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

#include <Cabana_Core.hpp>

#include <iostream>

/* Define the inner vector length of SOA */
#include "veclen.h"

//---------------------------------------------------------------------------//
// AoSoA example.
//---------------------------------------------------------------------------//
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
using DataTypes = Cabana::MemberTypes<double[3][3], float[4], int>;

// This is the coresponding struct_of_array defined by SOA (using DataTypes)
struct local_data_struct_t
{
    double d0[3][3][VECLEN];
    double d1[4][VECLEN];
    int d2[VECLEN];
};

/*
  Next declare the vector length of our SoAs. This is how many tuples the
  SoAs will contain. A reasonable number for performance should be some
  multiple of the vector length on the machine you are using.
*/

/*
  Finally declare the memory space in which the AoSoA will be
  allocated. In this example we are writing basic loops that will execute
  on the CPU. The HostSpace allocates memory in standard CPU RAM.

  Kokkos also supports execution on NVIDIA GPUs. For example, to create an
  AoSoA allocated with CUDA Unified Virtual Memory (UVM) use
  `Kokkos::CudaUVMSpace` instead of `Kokkos::HostSpace`. The CudaUVMSpace
  allocates memory in managed GPU memory via `cudaMallocManaged`. This
  memory is automatically paged between host and device depending on the
  context in which the memory is accessed.
*/
using MemorySpace = Kokkos::HostSpace;

/*
  Create the AoSoA. We define how many tuples the aosoa will
  contain. Note that if the number of tuples is not evenly divisible by
  the vector length then the last SoA in the AoSoA will not be entirely
  full (although its memory will still be allocated).
*/

using AosoaTYPE = Cabana::AoSoA<DataTypes, MemorySpace, VECLEN>;

/* Declare functions that will be mixed with Fortran */
extern "C"
{
    void aosoaExample( local_data_struct_t *,
                       int ); // written in Fortran; called by C++
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char *argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    /*
      Print size data. In this case we have created an AoSoA with 5
      tuples. Because a vector length of 4 is used, a total memory capacity
      for 8 tuples will be allocated in 2 SoAs.
    */

    int num_element = 5;

    /* Create a pointer of AosoaType */
    AosoaTYPE *aosoa = new AosoaTYPE( "aosoa", num_element );

    std::cout << "aosoa.size() = " << aosoa->size() << std::endl;
    std::cout << "aosoa.capacity() = " << aosoa->capacity() << std::endl;
    std::cout << "aosoa.numSoA() = " << aosoa->numSoA() << std::endl;

    /* In calling the Fortran subroutine, we cast aosoa to conventional struct,
       and pass it to Fortran */
    aosoaExample( (local_data_struct_t *)( aosoa->data() ), num_element );

    delete aosoa;

    return 0;
}

//---------------------------------------------------------------------------//
