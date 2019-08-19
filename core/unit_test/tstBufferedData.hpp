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
#include <Cabana_BufferedData.hpp>

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
        using DataTypes =
            Cabana::MemberTypes<float[dim_1][dim_2][dim_3],
            int,
            double[dim_1],
            double[dim_1][dim_2]
                >;

        // Declare the AoSoA type.
        using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE,vector_length>;
        std::string label = "sample_aosoa";
        AoSoA_t aosoa( label );

        // Start by only buffering over one AoSoA at a time for stress test
        const int max_buffered_tuples = vector_length;

        // emulate a minimum of triple buffering?
        const int buffer_count = 3;

        // Hard code into OpenMP space for now
        using target_exec_space = Kokkos::OpenMP;

        // Feed it into the buffer
        Cabana::DataBuffer<
            max_buffered_tuples,
            buffer_count,
            target_exec_space,
            AoSoA_t
        > test_buffer(aosoa);

        int num_buffers = test_buffer.get_buffer_count();

        // Add call to safe in-loop handle
        auto access_handler = test_buffer.access();

        // Test some of the copying/execution logic
        // TODO: this should be pushed down into the class
        test_buffer.load_next_buffer();

        auto data_handle = test_buffer.access_buffer();
    }

    // TODO: add a test to test the auto creation doing execution

} // namesapce

