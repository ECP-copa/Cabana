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
// SoA example.
//---------------------------------------------------------------------------//
void soaExample()
{
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

          const int veclen = 8;

      Declaring a Cabana SoA as Cabana::SoA<types,veclen> gives the equivalent
      data layout:

          struct MyEquivalentSoA
          {
              double d0[3][2][veclen];
              double d1[4][veclen];
              float d2[veclen];
          };

      Note: The data in this struct definition with an equivalent memory
      layout is stride-1 in the array index.

      Note: When defined as a compile-time argument the vector length must be
      a power of 2. Vector lengths that are not a power of 2 will emit a
      compiler error.

      Note: The members in an SoA-equivalent struct are in the same order as
      they are declared in Cabana::MemberTypes.
    */

    /* Start by declaring the types our tuples will store. Store a rank-2
       array of doubles, a rank-1 array of floats, and a single integer in
       each tuple.
    */
    using DataTypes = Cabana::MemberTypes<double[3][3], float[4], int>;

    /*
      Next declare the vector length of our SoA. This is how many tuples the
      SoA will contain. A reasonable number for performance should be some
      multiple of the vector length on the machine you are using.
    */
    const int VectorLength = 4;

    /* Create the SoA. */
    Cabana::SoA<DataTypes, VectorLength> soa;

    /* Assign data to the SoA values using the multidimensional data
       accessors. Each SoA element is accessed via a `get<>` function who's
       integer template parameter indicates which member to access in a nearly
       identical fashion to Cabana::Tuple. In this case:

       *: get<0> will return the values for the tuple member of type
          `double[3][3]` (i.e. the zero member)

       *: get<1> will return the values for the tuple member of type
          `float[4]` (i.e. the first member)

       *: get<2> will return the values for the tuple member of type
          `int` (i.e. the second member)

       In the code that follows below, note the additional introduction of an
       extra index over the vector length of the SoA. This is the first index
       in all get accessors. The loop over this extra index is moved to the
       inside of the other loops to promote stride-1 data accesses.
    */
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            for ( int a = 0; a < VectorLength; ++a )
                Cabana::get<0>( soa, a, i, j ) = 1.0 * ( a + i + j );

    for ( int i = 0; i < 4; ++i )
        for ( int a = 0; a < VectorLength; ++a )
            Cabana::get<1>( soa, a, i ) = 1.0 * ( a + i );

    for ( int a = 0; a < VectorLength; ++a )
        Cabana::get<2>( soa, a ) = a + 1234;

    /* Read back the tuple data using the same multidimensional accessors */
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            for ( int a = 0; a < VectorLength; ++a )
                std::cout << "Tuple member 0, tuple index " << a
                          << ", element (" << i << "," << j
                          << "): " << Cabana::get<0>( soa, a, i, j )
                          << std::endl;

    for ( int i = 0; i < 4; ++i )
        for ( int a = 0; a < VectorLength; ++a )
            std::cout << "Tuple member 1, tuple index " << a << ", element ("
                      << i << "): " << Cabana::get<1>( soa, a, i ) << std::endl;

    for ( int a = 0; a < VectorLength; ++a )
        std::cout << "Tuple member 2, tuple index " << a << ": "
                  << Cabana::get<2>( soa, a ) << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    soaExample();

    return 0;
}

//---------------------------------------------------------------------------//
