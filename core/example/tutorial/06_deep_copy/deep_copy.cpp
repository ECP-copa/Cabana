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

#include <iostream>

//---------------------------------------------------------------------------//
// Deep copy example.
//---------------------------------------------------------------------------//
void deepCopyExample()
{
    /*
      The AoSoA object allocates memory in a given memory space with a layout
      specified by the SoA objects it contains - mainly the data types in the
      SoAs and the vector length of the SoA. When we call the copy constructor
      or assignment operator of an AoSoA, a reference counted and shallow copy
      of that data is performed.

      In many cases we may want to actually copy the contents of an AoSoA into
      a new AoSoA which has the same member types. This new AoSoA may also
      have a different memory space or vector length than the original SoA. We
      refer to the actually copying of data between data structures as a deep
      copy. This example will demonstrate performing this deep copy.
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
      Declare the vector length and memory space parameters of the source
      AoSoA.
    */
    const int SrcVectorLength = 8;
    using SrcMemorySpace = Cabana::HostSpace;

    /*
      Declare the vector length and memory space parameters of the destination
      AoSoA.
    */
    const int DstVectorLength = 32;
    using DstMemorySpace = Cabana::CudaUVMSpace;

    /*
       Create the source and destination AoSoAs.
    */
    int num_tuple = 5;
    Cabana::AoSoA<DataTypes,SrcMemorySpace,SrcVectorLength> src_aosoa( num_tuple );
    Cabana::AoSoA<DataTypes,DstMemorySpace,DstVectorLength> dst_aosoa( num_tuple );

    /*
      Put some data in the source AoSoA.
    */
    for ( int s = 0; s < src_aosoa.numSoA(); ++s )
    {
        auto& soa = src_aosoa.access(s);

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                for ( int a = 0; a < src_aosoa.arraySize(s); ++a )
                    soa.get<0>(a,i,j) = 1.0 * (a + i + j);

        for ( int i = 0; i < 4; ++i )
            for ( int a = 0; a < src_aosoa.arraySize(s); ++a )
                soa.get<1>(a,i) = 1.0 * (a + i);

        for ( int a = 0; a < src_aosoa.arraySize(s); ++a )
            soa.get<2>(a) = a + 1234;
    }

    /*
      Deep copy the data from the source to the destination. A deep copy is
      possible if the AoSoAs have the same member types and they are the same
      size (i.e. they have the same number of tuples). As is the case here,
      they are allowed to have different memory spaces and vector lengths.
     */
    Cabana::deep_copy( dst_aosoa, src_aosoa );

    /*
       Now let's read the data from the destination that we just deep copied.
     */
    for ( int t = 0; t < num_tuple; ++t )
    {
        auto tp = dst_aosoa.getTuple( t );

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

    deepCopyExample();

    Cabana::finalize();

    return 0;
}

//---------------------------------------------------------------------------//
