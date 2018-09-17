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

#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Parallel.hpp>
#include <Cabana_ExecutionPolicy.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <chrono>

// Performance test function.
void perfTest()
{
    // Declare the execution and memory spaces.
    using MemorySpace = Cabana::CudaUVMSpace;
    using ExecutionSpace = Kokkos::Cuda;

    // Declare the inner array layout.
    const int vector_length = 32;

    // Declare the parallel for algorithm tag.
    using parallel_algorithm_tag = Cabana::StructAndArrayParallelTag;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<double[3][3], // M1
                                          double[3][3], // M2
                                          double[3],    // V1
                                          double[3],    // V2
                                          double[3],    // RESULT
                                          double,       // S1
                                          double>;      // S2

    // Enumerate the types for convenience.
    enum MyTypes { M1 = 0,
                   M2,
                   V1,
                   V2,
                   RESULT,
                   S1,
                   S2 };

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes,MemorySpace,vector_length>;

    // Set the total problem size.
    std::size_t num_data = 1e7;

    // Create an Array-of-Structs-of-Arrays.
    AoSoA_t aosoa( num_data );

    // Make some slices.
    auto m1 = aosoa.slice<M1>();
    auto m2 = aosoa.slice<M2>();
    auto v1 = aosoa.slice<V1>();
    auto v2 = aosoa.slice<V2>();
    auto result = aosoa.slice<RESULT>();
    auto s1 = aosoa.slice<S1>();
    auto s2 = aosoa.slice<S2>();

    // Create an execution policy over the entire AoSoA.
    Cabana::RangePolicy<vector_length,ExecutionSpace> range_policy( aosoa );

    // Initialization functor.
    auto init_func = KOKKOS_LAMBDA( const int idx )
    {
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                m1(idx,i,j) = 1.0;

        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                m2(idx,i,j) = 1.0;

        for ( int i = 0; i < 3; ++i )
            v1(idx,i) = 1.0;

        for ( int i = 0; i < 3; ++i )
            v2(idx,i) = 1.0;

        s1(idx) = 1.0;
        s2(idx) = 1.0;
    };

    // Initialize.
    Cabana::parallel_for( range_policy, init_func, parallel_algorithm_tag() );

    // Create a work functor:
    // m3 = m1 * m2
    // m1 * v1 / s1 + m2 * v2 / s2 + dot(v1,v2) * v1 + m3 * v2
    auto work_func = KOKKOS_LAMBDA( const int idx )
    {
        double t1[3] = {0.0, 0.0, 0.0};
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                t1[i] += m1(idx,i,j) * v1(idx,j);

        double t2[3] = {0.0, 0.0, 0.0};
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                t2[i] += m2(idx,i,j) * v2(idx,j);

        double dotv1v2 = 0.0;
        for ( int i = 0; i < 3; ++i )
            dotv1v2 += v1(idx,i) * v2(idx,i);

        double m3[3][3] = {{0.0, 0.0, 0.0},
                           {0.0, 0.0, 0.0},
                           {0.0, 0.0, 0.0}};
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                for ( int k = 0; k < 3; ++k )
                    m3[i][j] += m1(idx,i,k) * m2(idx,k,j);

        double t3[3] = {0.0, 0.0, 0.0};
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                t3[i] += m3[i][j] * v2(idx,j);

        for ( int i = 0; i < 3; ++i )
            result(idx,i) = t1[i] / s1(idx) +
                            t2[i] / s2(idx) +
                            dotv1v2 * v1(idx,i) +
                            t3[i];
    };

    // Do work.
    auto start_time = std::chrono::high_resolution_clock::now();
    Cabana::parallel_for( range_policy, work_func, parallel_algorithm_tag() );
    auto end_time = std::chrono::high_resolution_clock::now();

    auto elapsed_time = end_time - start_time;
    auto ms_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_time);
    std::cout << "Run time: " << ms_elapsed.count() << "ms" << std::endl;
}

int main( int argc, char* argv[] )
{
    // Initialize the kokkos runtime.
    Kokkos::initialize( argc, argv );

    // Run the test.
    perfTest();

    // Finalize.
    Kokkos::finalize();
    return 0;
}
