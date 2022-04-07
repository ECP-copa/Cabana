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
// Sorting example.
//---------------------------------------------------------------------------//
void sortingExample()
{
    /*
      In many algorithms we will want to bin or sort AoSoA data to improve
      computational performance. Binning and sorting can be achieved by using
      a user-defined comparator function or by using an auxiliary set of key
      values (such as a slice) whose sorted/binned order will define the order
      of the AoSoA. In this example we will demonstrate sorting an AoSoA with a
      slice of data as the keys.
    */

    /*
       Start by declaring the types in our tuples will store. We will use the
       integer as the sorting key in this example.
    */
    using DataTypes = Cabana::MemberTypes<double, int>;

    /*
      Next declare the data layout of the AoSoA. We use the host space here
      for the purposes of this example, but all memory spaces, vector lengths,
      and member type configurations are compatible with sorting.
    */
    const int VectorLength = 4;
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Create the AoSoA.
    */
    int num_tuple = 5;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "my_aosoa",
                                                              num_tuple );

    /*
      Fill the AoSoA with data. The integer member of the AoSoA will be
      created in DESCENDING ORDER. When sorted, the values in the AoSoA should
      then be reversed.
    */
    int forward_index_counter = 0;
    int reverse_index_counter = 100;
    for ( std::size_t s = 0; s < aosoa.numSoA(); ++s )
    {
        auto& soa = aosoa.access( s );

        // ASCENDING ORDER!
        for ( std::size_t a = 0; a < aosoa.arraySize( s ); ++a )
        {
            Cabana::get<0>( soa, a ) = forward_index_counter;
            ++forward_index_counter;
        }

        // DESCENDING ORDER!
        for ( std::size_t a = 0; a < aosoa.arraySize( s ); ++a )
        {
            Cabana::get<1>( soa, a ) = reverse_index_counter;
            --reverse_index_counter;
        }
    }

    /*
      Create the sorting data for the AoSoA based on using the integer member
      as the sorting key. The sorting data is a basic device-accessible data
      structure that describes how the data is to be reordered. In the case of
      binning, this data structure also describes how the data is binned. For
      some algorithms, this data structure may be all that is necessary if the
      data does not have to be physically reordered.

      We are using auto for the sort_data return value here for convenience,
      but the actual return type is Cabana::BinningData<MemorySpace>. Note
      that the binning data is templated on a memory space as it creates and
      stores data in the same memory space as the AoSoA.
     */
    auto keys = Cabana::slice<1>( aosoa );
    auto sort_data = Cabana::sortByKey( keys );

    /*
      Now actually permute the AoSoA (i.e. reorder the data).
    */
    Cabana::permute( sort_data, aosoa );

    /*
       Now let's read the data we just sorted. The integer member should
       appear in ASCENDING ORDER to indicate the sorting has been
       completed. The rest of the data will also have been sorted as well -
       all data in each tuple is permuted.
     */
    for ( int t = 0; t < num_tuple; ++t )
    {
        auto tp = aosoa.getTuple( t );

        // Should now be in DESCENDING ORDER!
        std::cout << "Tuple " << t << ", member 0: " << Cabana::get<0>( tp )
                  << std::endl;

        // Should now be in ASCENDING ORDER!
        std::cout << "Tuple " << t << ", member 1: " << Cabana::get<1>( tp )
                  << std::endl;
    }
    std::cout << std::endl;

    /*
      Now let's demonstrate binning. Binning groups keys in ascending order
      (if not custom comparator) in a fixed number of bins. The width of the
      bins (i.e. the largest and smallest values that will qualify as members
      of the bin) are computed based on a user-assigned number of bins and the
      minimum and maximum key values.

      Start by creating some new keys. We will manually create an alternating
      key pattern to demonstrate the effects of binning:
    */
    keys( 0 ) = 100;
    keys( 1 ) = 200;
    keys( 2 ) = 100;
    keys( 3 ) = 200;
    keys( 4 ) = 100;

    /*
      Now create the binning data. In this case let's create two bins: one
      for the 100's and one for the 200's.

      We are using auto for the bin_data return value here for convenience,
      but the actual return type is Cabana::BinningData<MemorySpace>. Note
      that the binning data is templated on a memory space as it creates and
      stores data in the same memory space as the AoSoA.
    */
    int num_bin = 2;
    auto bin_data = Cabana::binByKey( keys, num_bin );

    /*
      Now permute the data. This time the data should be grouped by key
      value.
    */
    Cabana::permute( bin_data, aosoa );

    /*
       Now let's read the data we just sorted. The integer member should
       be ordered in binned groups.
     */
    for ( int t = 0; t < num_tuple; ++t )
    {
        auto tp = aosoa.getTuple( t );

        std::cout << "Tuple " << t << ", member 0: " << Cabana::get<0>( tp )
                  << std::endl;

        std::cout << "Tuple " << t << ", member 1: " << Cabana::get<1>( tp )
                  << std::endl;
    }
    std::cout << std::endl;

    /*
      We can also interrogate the binning data itself - this is useful in many
      cases where we may want to know how the data is binned and use that in
      an algorithm, but not actually permute the data.

      Using the binning data we just created let's see how many bins there are
      - we asked for 2 so we should get at least 2. This number may be
      slightly larger due to integer arithmetic. In this case it is 3:
     */
    std::cout << "bin_data.numBin() = " << bin_data.numBin() << std::endl;

    /*
      Now let's get the number of tuples that are in each bin - this should
      correspond to our results above: 3 in the 100's bin and 2 in the 200's
      bin:
     */
    std::cout << "Bin 0 size = " << bin_data.binSize( 0 ) << std::endl;
    std::cout << "Bin 1 size = " << bin_data.binSize( 1 ) << std::endl;
    std::cout << "Bin 2 size = " << bin_data.binSize( 2 ) << std::endl;

    /*
      Finally let's get the local ids of the tuples that are in each
      bin. The new order of tuple ids stored in the permutation vector is
      grouped by bin. The offset array in the binning data tells us where each
      bin's group of ids starts:
    */
    for ( int b = 0; b < bin_data.numBin(); ++b )
    {
        std::cout << "Bin " << b << " ids: ";
        int offset = bin_data.binOffset( b );
        for ( int i = 0; i < bin_data.binSize( b ); ++i )
            std::cout << bin_data.permutation( offset + i ) << " ";
        std::cout << std::endl;
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    sortingExample();

    return 0;
}

//---------------------------------------------------------------------------//
