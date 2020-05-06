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

//---------------------------------------------------------------------------//
// Tuple example.
//---------------------------------------------------------------------------//

/*
      Cabana tuples are similar to C++11 tuples in that they are ordered lists
      of elements of different types specified at compile time. Some aspects
      of Cabana::Tuple that are different from std::tuple:

      *: Cabana tuples must consist of data of trivial types and classes. A
         trivial type or class is one who's

         1) storage is contiguous (i.e. trivially copyable)

         2) which only supports static default initialization (trivially
            default constructible), either cv-qualified or not.

         Trivial types include scalar types, trivial classes and arrays of any
         such types. A trivial class is a class (defined with class, struct or
         union) that is both trivially default constructible and trivially
         copyable, which implies that:

         1) uses the implicitly defined default copy and move constructors,
            copy and move assignments, and destructor

         2) has no virtual members

         3) has no non-static data members with brace- or equal- initializers

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

     struct local_particle_struct_t
     {
           double d0[3][3];
           float d1[4];
           int d2;
     };


       Note that ata members in Cabana tuples are stored in the same order in
       which they are declared.
*/
using TupleType = Cabana::Tuple<DataTypes>;

/*
  Create a pointer of TupleType, which will be used in Fortran
 */
TupleType *particle = new TupleType;

/*
  Declare functions that will be mixed with Fortran
 */
extern "C"
{
    void tupleExample( TupleType * ); // written in Fortan; called by C++
    void delete_tuple();              // written in C++; called by Fortan
}

void delete_tuple() { delete particle; }

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char *argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    /* Call the Fortran subroutine */
    tupleExample( particle );

    return 0;
}

//---------------------------------------------------------------------------//
