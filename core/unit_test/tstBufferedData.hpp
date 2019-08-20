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

#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_BufferedAoSoA.hpp>
#include <Cabana_BufferedFor.hpp>

#include <gtest/gtest.h>

namespace Test
{
    void testBufferedDataCreation()
    {
        // Create an AoSoA
        const int vector_length = 16;

        // Data dimensions.
        const int dim_1 = 3;
        const int dim_2 = 2;
        const int dim_3 = 4;

        // Declare data types.
        using DataTypes = Cabana::MemberTypes<
            float[dim_1][dim_2][dim_3],
            int,
            double[dim_1],
            double[dim_1][dim_2]
        >;

        // Declare the AoSoA type.
        using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE,vector_length>;
        std::string label = "sample_aosoa";
        int num_data = 64;
        AoSoA_t aosoa( label, num_data );

        // Start by only buffering over one AoSoA at a time for stress test
        const int max_buffered_tuples = vector_length;

        // emulate a minimum of triple buffering?
        const int buffer_count = 3;

        // Hard code into OpenMP space for now
        using target_exec_space = Kokkos::OpenMP;

        // Init the AoSoA data
        auto mirror = Cabana::create_mirror_view_and_copy(
                Kokkos::HostSpace(), aosoa );
        auto mirror_slice_0 = Cabana::slice<0>(mirror);
        auto mirror_slice_1 = Cabana::slice<1>(mirror);
        auto mirror_slice_2 = Cabana::slice<2>(mirror);
        auto mirror_slice_3 = Cabana::slice<3>(mirror);

        // Initialize data with the rank accessors.
        float fval = 3.4;
        double dval = 1.23;
        int ival = 1;
        for ( std::size_t idx = 0; idx != aosoa.size(); ++idx )
        {
            // Member 0.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    for ( int k = 0; k < dim_3; ++k )
                        mirror_slice_0( idx, i, j, k ) = fval * (i+j+k);

            // Member 1.
            mirror_slice_1( idx ) = ival;

            // Member 2.
            for ( int i = 0; i < dim_1; ++i )
                mirror_slice_2( idx, i ) = dval * i;

            // Member 3.
            for ( int i = 0; i < dim_1; ++i )
                for ( int j = 0; j < dim_2; ++j )
                    mirror_slice_3( idx, i, j ) = dval * (i+j);
        }
        Cabana::deep_copy( aosoa, mirror );

        // Feed it into the buffer
        Cabana::BufferedAoSoA<
            max_buffered_tuples,
            buffer_count,
            target_exec_space,
            AoSoA_t
        > buffered_aosoa(aosoa);

        int num_buffers = buffered_aosoa.get_buffer_count();

        // Add call to safe in-loop handle
        //auto access_handler = test_buffer.access_old();

        // TODO: are the fences needed before we start?

        // Reset values so the outcome differs
        fval = 4.4;
        dval = 2.23;
        ival = 2;

        std::cout << "Calling buffered for using data from " << aosoa.data() << std::endl;

        // Overwrite the data in a buffered way
        Cabana::buffered_parallel_for(
            Kokkos::RangePolicy<TEST_EXECSPACE>(0,aosoa.size()),
            buffered_aosoa,
            KOKKOS_LAMBDA( const int s, const int a )
            {
                // We have to call access and slice in the loop

                // We have to be really careful about how this access is
                // captured in the loop on GPU, and follow how ScatterView does
                // it safely. The `buffered_aosoa` may get captured by
                // reference, and then not be valid in a GPU context
                auto buffered_access = buffered_aosoa.access();

                std::cout << "The underlying aosoa lives at " << &(buffered_access.aosoa) << " and has size " << buffered_access.aosoa.size() << std::endl;
                std::cout << "The data of the accesed aosoa lives at " << (buffered_access.aosoa.data()) << std::endl;
                std::cout << "The data of the buffered aosoa lives at " << (buffered_aosoa.internal_buffers[0].data()) << " and has size " << buffered_aosoa.internal_buffers[0].size() << std::endl;

                auto slice_0 = Cabana::slice<0>(buffered_access.aosoa);
                auto slice_1 = Cabana::slice<1>(buffered_access.aosoa);
                auto slice_2 = Cabana::slice<2>(buffered_access.aosoa);
                auto slice_3 = Cabana::slice<3>(buffered_access.aosoa);

                // Member 0.
                for ( int i = 0; i < dim_1; ++i )
                    for ( int j = 0; j < dim_2; ++j )
                        for ( int k = 0; k < dim_3; ++k )
                            slice_0.access( s, a, i, j, k ) = fval * (i+j+k) / 2.0;

                // Member 1.
                slice_1.access( s, a ) = ival / 2.0;

                // Member 2.
                for ( int i = 0; i < dim_1; ++i )
                    slice_2.access( s, a, i ) = dval * i / 2.0;

                // Member 3.
                for ( int i = 0; i < dim_1; ++i )
                    for ( int j = 0; j < dim_2; ++j )
                        slice_3.access( s, a, i, j ) = dval * (i+j) / 2.0;
            },
            "test buffered for"
        );

        // TODO: test the data values
        EXPECT_EQ( aosoa.size(), 10 );

    }

    //---------------------------------------------------------------------------//
    // RUN TESTS
    //---------------------------------------------------------------------------//
    TEST( TEST_CATEGORY, bufferedData_test )
    {
        testBufferedDataCreation();
    }

} // namesapce

