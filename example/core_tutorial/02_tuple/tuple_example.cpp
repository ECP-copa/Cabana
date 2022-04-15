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
// Tuple example.
//---------------------------------------------------------------------------//
void tupleExample()
{
    /*
      Cabana tuples are similar to C++11 tuples in that they are ordered lists
      of elements of different types specified at compile time. Some aspects
      of Cabana::Tuple that are different from std::tuple:

      *: Cabana tuples must consist of data of trivial types and classes. A
         trivial type or class is one which:

         1) has contiguous storage (i.e. is trivially copyable)

         2) only supports static default initialization (trivially default
            constructible), either cv-qualified or not.

         Trivial types include scalar types, trivial classes, and arrays of any
         such types. A trivial class is a class (defined with class, struct or
         union) that is both trivially default constructible and trivially
         copyable, which implies that:

         1) it uses the implicitly defined default copy and move constructors,
            copy and move assignments, and destructor

         2) it has no virtual members

         3) it has no non-static data members with brace- or equal- initializers

         4) its base class and non-static data members (if any) are themselves
            also trivial types

      *: The types in a Cabana tuple are defined via a type parameter
         pack using the Cabana::MemberTypes class

      *: Cabana tuples are designed to store and access multidimensional array
         data with accessors provided to obtain references to individual array
         values within the tuple.
    */

    /* Start by declaring the types in our tuple will store. Store a rank-2
       array of doubles, a rank-1 array of floats, and a single integer in
       each tuple.

    */
    using DataTypes = Cabana::MemberTypes<double[3][3], float[4], int>;

    /*
       Create the tuple. This tuple is identical to:

       struct Foo
       {
           double d0[3][3];
           float d1[4];
           int d2;
       };


       Note that ata members in Cabana tuples are stored in the same order in
       which they are declared.
     */
    Cabana::Tuple<DataTypes> tp;

    /* Assign data to the tuple values using the multidimensional data
       accessors. Each tuple element is accessed via a `get<>` function who's
       integer template parameter indicates which member to access. In this
       case:

       *: get<0> will return the values for the tuple member of type
          `double[3][3]` (i.e. the zero member)

       *: get<1> will return the values for the tuple member of type
          `float[4]` (i.e. the first member)

       *: get<2> will return the values for the tuple member of type
          `int` (i.e. the second member)
    */
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            Cabana::get<0>( tp, i, j ) = 1.0 * ( i + j );

    for ( int i = 0; i < 4; ++i )
        Cabana::get<1>( tp, i ) = 1.0 * i;

    Cabana::get<2>( tp ) = 1234;

    /* Read back the tuple data using the same multidimensional accessors */
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            std::cout << "Tuple member 0 element (" << i << "," << j
                      << "): " << Cabana::get<0>( tp, i, j ) << std::endl;

    for ( int i = 0; i < 4; ++i )
        std::cout << "Tuple member 1 element (" << i
                  << "): " << Cabana::get<1>( tp, i ) << std::endl;

    std::cout << "Tuple member 2: " << Cabana::get<2>( tp ) << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    tupleExample();

    return 0;
}

//---------------------------------------------------------------------------//
