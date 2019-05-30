/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Core.hpp>
#include <Cabana_Sort.hpp>

#include <iostream>

//---------------------------------------------------------------------------//
// parallel for using an unmanaged view.
//---------------------------------------------------------------------------//
void parallelForExample()
{
    /* Declare general run parameters */
    using MemorySpace = Kokkos::HostSpace;
    const int VectorLength = 8;

    /*
     * Declare Underlying data type
    */

    // Regular class form
    class Data {
        public:
            double a[VectorLength];
            int    b[VectorLength];
    };

    // Cabana MemberTypes form:
    using DataTypes = Cabana::MemberTypes<double, int>;

    /*
     * Create the AoSoA.
    */
    const int num_tuple = 128;
    const int num_soa = 128/8; // Be careful with the int division..

    Data* local_data = new Data[num_tuple];

    // This is equivalent to a Cabana AoSoA of:
      // using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
      // using DeviceType = Kokkos::Device<ExecutionSpace,MemorySpace>;
      // Cabana::AoSoA<DataTypes,DeviceType,VectorLength> aosoa( "my_aosoa", num_tuple );

    /*
     * Populate user data.
    */
    for (int i = 0; i < num_soa; i++)
    {
        for (int j = 0; j < VectorLength; j++)
        {
            local_data[i].a[j] = i;
            local_data[i].b[j] = j;
        }
    }

    /*
    * Convert the user allocated memory into an unmanaged AoSoA
    */

    // Define the underlying AoSoA type to hold the unmanaged data
    using AoSoA_t = Cabana::AoSoA<
        DataTypes,
        MemorySpace,
        VectorLength,
        Kokkos::MemoryUnmanaged
    >;

    // Do convert
    auto ptr = reinterpret_cast<typename AoSoA_t::soa_type*>(local_data);
    AoSoA_t aosoa( ptr, num_soa, num_tuple );

    /*
    * Inspect Data In the Unmanaged AoSoA
    */
    auto slice_a = Cabana::slice<0>(aosoa); // a = 0
    auto slice_b = Cabana::slice<1>(aosoa); // b = 1

    // Look at the data in the AosoA
    for (int i = 0; i < num_tuple; i++)
    {
        if ( slice_b(i) != (i % VectorLength) )
        {
            // Unexpected Value
            printf("%d: Unexpected %d != %d \n", __LINE__, slice_b(i), i % VectorLength);
        }
    }

    /*
    * Use the view in an algorithm, like sort
    */
    auto binning_data = Cabana::sortByKey( slice_b );
    Cabana::permute( binning_data, aosoa );

    // Check it worked
    for (int i = 1; i < num_tuple; i++)
    {
        // We expect it to be monotonically increasing
        if (!(slice_b(i-1) <= slice_b(i)))
        {
            // Unexpected value
            printf("%d: Unexpected %d vs %d \n", __LINE__, slice_b(i-1) , slice_b(i) );
        }
    }

    // Clean up local data
    delete local_data;

}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    parallelForExample();

    return 0;
}

//---------------------------------------------------------------------------//
