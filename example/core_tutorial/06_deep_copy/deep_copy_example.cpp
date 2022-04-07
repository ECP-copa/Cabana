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
// Deep copy example.
//---------------------------------------------------------------------------//
void deepCopyExample()
{
    /*
      The AoSoA object allocates memory in a given memory space with a layout
      specified by the SoA objects it contains - mainly the data types in the
      SoAs and the vector length of the SoA. When we call the copy constructor
      or assignment operator of an AoSoA, a reference counted, shallow copy
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
    using DataTypes = Cabana::MemberTypes<double[3][3], float[4], int>;

    /*
      Declare the vector length and memory space parameters of the source
      AoSoA.
    */
    const int SrcVectorLength = 8;
    using SrcMemorySpace = Kokkos::HostSpace;
    using SrcExecutionSpace = Kokkos::Serial;
    using SrcDevice = Kokkos::Device<SrcExecutionSpace, SrcMemorySpace>;

    /*
      Declare the vector length and memory space parameters of the destination
      AoSoA.
    */
    const int DstVectorLength = 32;
    using DstMemorySpace = Kokkos::CudaUVMSpace;
    using DstExecutionSpace = Kokkos::Cuda;
    using DstDevice = Kokkos::Device<DstExecutionSpace, DstMemorySpace>;

    /*
       Create the source and destination AoSoAs.
    */
    int num_tuple = 5;
    Cabana::AoSoA<DataTypes, SrcDevice, SrcVectorLength> src_aosoa( "src",
                                                                    num_tuple );
    Cabana::AoSoA<DataTypes, DstDevice, DstVectorLength> dst_aosoa( "dst",
                                                                    num_tuple );

    /*
      Put some data in the source AoSoA.
    */
    for ( std::size_t s = 0; s < src_aosoa.numSoA(); ++s )
    {
        auto& soa = src_aosoa.access( s );

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                for ( unsigned a = 0; a < src_aosoa.arraySize( s ); ++a )
                    Cabana::get<0>( soa, a, i, j ) = 1.0 * ( a + i + j );

        for ( int i = 0; i < 4; ++i )
            for ( unsigned a = 0; a < src_aosoa.arraySize( s ); ++a )
                Cabana::get<1>( soa, a, i ) = 1.0 * ( a + i );

        for ( unsigned a = 0; a < src_aosoa.arraySize( s ); ++a )
            Cabana::get<2>( soa, a ) = a + 1234;
    }

    /*
      Deep copy the data from the source to the destination. A deep copy is
      possible if the source and destination AoSoAs have the same member types
      and they are the same size (i.e. they have the same number of
      tuples). As is the case here, they are allowed to have different memory
      spaces and vector lengths.
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
      In applications that utilize multiple memory spaces (e.g. those that use
      heterogeneous architectures) we can also create an AoSoA that mirrors
      another AoSoA. A mirror is a data structure that has an identical data
      layout but may be allocated in another memory space. A good example of a
      use for this type of capability is easily managing copies between a GPU
      and a CPU.

      Given that the AoSoA we created above is on the GPU we can easily create
      another identical AoSoA containing the same contents but that is
      allocated in a different memory space allowing for easy transfer back to
      the device:
     */
    auto dst_aosoa_host =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), dst_aosoa );

    /*
       Note that this view is now in the provided host space. If we were to
       use the same memory space as dst_aosoa was created in we would
       have received the same dst_aosoa back - no memory allocations or copies
       would have occured. This is particularly useful for writing code that
       will run on both heterogenous and homogenous architectures where the
       heterogenous case requires an allocation and a copy while the
       homogenous case does not.
     */

    /*
      Deep copy can also be performed on a slice-by-slice basis.
     */
    auto src_slice_0 = Cabana::slice<0>( src_aosoa );
    auto dst_slice_0 = Cabana::slice<0>( dst_aosoa );
    Cabana::deep_copy( dst_slice_0, src_slice_0 );
    auto src_slice_1 = Cabana::slice<1>( src_aosoa );
    auto dst_slice_1 = Cabana::slice<1>( dst_aosoa );
    Cabana::deep_copy( dst_slice_1, src_slice_1 );
    auto src_slice_2 = Cabana::slice<2>( src_aosoa );
    auto dst_slice_2 = Cabana::slice<2>( dst_aosoa );
    Cabana::deep_copy( dst_slice_2, src_slice_2 );

    /*
      One can also assign scalar values to every element in a slice. The slice
      can be in any memory space.
     */
    Cabana::deep_copy( src_slice_0, 3.4 );
    Cabana::deep_copy( src_slice_1, 2.22 );
    Cabana::deep_copy( src_slice_2, 12 );

    /*
      Or one can initialize each tuple in an AoSoA with the values of a given
      tuple. The AoSoA can be in any memory space.
     */
    Cabana::Tuple<DataTypes> tp;
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            Cabana::get<0>( tp, i, j ) = 1.0;
    for ( int i = 0; i < 4; ++i )
        Cabana::get<1>( tp, i ) = 3.23;
    Cabana::get<2>( tp ) = 39;
    Cabana::deep_copy( dst_aosoa, tp );
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    deepCopyExample();

    return 0;
}

//---------------------------------------------------------------------------//
