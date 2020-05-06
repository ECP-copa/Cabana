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
// SoA example.
//---------------------------------------------------------------------------//
/*
      Cabana structs-of-arrays (SoAs) are groups of Tuples with the data
      specified in an order such that the array (or tuple) index is stride-1
      in memory. This results in each dimension of each multidimensional tuple
      member being stored in a contiguous memory block the size of the number
      of tuples. The number of tuples in an SoA is defined as the "vector
      length" - this defines the size of the array.

      For example, consider declaring the following member type to store two
      arrays of doubles in each tuple:

          using types = Cabana::MemberTypes<double[3][2],double[4],float>;

      and vector length:

          const int VECLEN = 8;

      Declaring a Cabana SoA as Cabana::SoA<types,VECLEN> gives the equivalent
      data layout:

          struct MyEquivalentSoA
          {
              double d0[3][2][VECLEN];
              double d1[4][VECLEN];
              float d2[VECLEN];
          };

      Note: The data in this struct definition with an equivalent memory
      layout is stride-1 in the array index.

      Note: When defined as a compile-time argument the vector length must be
      a power of 2. Vector lengths that are not a power of 2 will emit a
      compiler error.

      Note: The members in an SoA-equivalent struct are in the same order as
      they are declared in Cabana::MemberTypes.
*/

/* Start by declaring the types in our tuples will store. Store a rank-2
       array of doubles, a rank-1 array of floats, and a single integer in
       each tuple.
*/
using DataTypes = Cabana::MemberTypes<double[3][3], float[4], int>;

/*
      Next declare the vector length of our SoA. This is how many tuples the
      SoA will contain. A reasonable number for performance should be some
      multiple of the vector length on the machine you are using.
*/

/* Create the SoA. */
using SoaTYPE = Cabana::SoA<DataTypes, VECLEN>;

/* Create a pointer of SoaType, which will be used in Fortran */
SoaTYPE *particle = new SoaTYPE;

/*  Declare functions that will be mixed with Fortran */
extern "C"
{
    void soaExample( SoaTYPE * ); // written in Fortan; called by C++
    void delete_soa();            // written in C++; called by Fortan
}

void delete_soa() { delete particle; }

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char *argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    /* Call the Fortran subroutine */
    soaExample( particle );

    return 0;
}

//---------------------------------------------------------------------------//
