/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "../Cabana_BenchmarkUtils.hpp"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

//---------------------------------------------------------------------------//
// Mixed-member AoSoA benchmark.
template <class Device>
void performanceTestMixed( std::ostream& stream, const std::string& test_prefix,
                           const std::vector<int>& problem_sizes )
{
    using exec_space = typename Device::execution_space;
    using memory_space = typename Device::memory_space;

    constexpr int num_run = 20;

    using member_types =
        Cabana::MemberTypes<double[3][3], double[3], double, int>;
    using aosoa_type = Cabana::AoSoA<member_types, memory_space>;

    int num_problem_size = problem_sizes.size();
    std::vector<aosoa_type> aosoas( num_problem_size );
    std::vector<int> psizes;

    // Create and initialize.
    for ( int p = 0; p < num_problem_size; ++p )
    {
        int num_p = problem_sizes[p];
        psizes.push_back( num_p );
        aosoas[p].resize( num_p );

        auto m33 = Cabana::slice<0>( aosoas[p], "mat33" );
        auto v3 = Cabana::slice<1>( aosoas[p], "vec3" );
        auto s0 = Cabana::slice<2>( aosoas[p], "scalar" );
        auto i0 = Cabana::slice<3>( aosoas[p], "int_scalar" );

        Kokkos::parallel_for(
            "init_mixed", Kokkos::RangePolicy<exec_space>( 0, num_p ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int d0 = 0; d0 < 3; ++d0 )
                    for ( int d1 = 0; d1 < 3; ++d1 )
                        m33( i, d0, d1 ) = 1.0 + d0 + d1;

                for ( int d = 0; d < 3; ++d )
                    v3( i, d ) = 2.0 + d;

                s0( i ) = 3.0;
                i0( i ) = 4;
            } );
        Kokkos::fence();
    }

    Cabana::Benchmark::Timer mixed_mat33_timer( test_prefix + "mixed_mat33",
                                                num_problem_size );
    Cabana::Benchmark::Timer mixed_vec3_timer( test_prefix + "mixed_vec3",
                                               num_problem_size );
    Cabana::Benchmark::Timer mixed_scalar_timer( test_prefix + "mixed_scalar",
                                                 num_problem_size );
    Cabana::Benchmark::Timer mixed_int_timer( test_prefix + "mixed_int",
                                              num_problem_size );
    Cabana::Benchmark::Timer mixed_all_timer( test_prefix + "mixed_all",
                                              num_problem_size );

    for ( int p = 0; p < num_problem_size; ++p )
    {
        int num_p = problem_sizes[p];

        std::cout << "Running mixed AoSoA slice benchmark for " << num_p
                  << " tuples on " << test_prefix << std::endl;

        auto m33 = Cabana::slice<0>( aosoas[p], "mat33" );
        auto v3 = Cabana::slice<1>( aosoas[p], "vec3" );
        auto s0 = Cabana::slice<2>( aosoas[p], "scalar" );
        auto i0 = Cabana::slice<3>( aosoas[p], "int_scalar" );

        for ( int t = 0; t < num_run; ++t )
        {
            mixed_mat33_timer.start( p );
            Kokkos::parallel_for(
                "bench_mixed_mat33",
                Kokkos::RangePolicy<exec_space>( 0, num_p ),
                KOKKOS_LAMBDA( const int i ) {
                    for ( int d0 = 0; d0 < 3; ++d0 )
                        for ( int d1 = 0; d1 < 3; ++d1 )
                            m33( i, d0, d1 ) += 1.0;
                } );
            Kokkos::fence();
            mixed_mat33_timer.stop( p );

            mixed_vec3_timer.start( p );
            Kokkos::parallel_for(
                "bench_mixed_vec3", Kokkos::RangePolicy<exec_space>( 0, num_p ),
                KOKKOS_LAMBDA( const int i ) {
                    for ( int d = 0; d < 3; ++d )
                        v3( i, d ) += 1.0;
                } );
            Kokkos::fence();
            mixed_vec3_timer.stop( p );

            mixed_scalar_timer.start( p );
            Kokkos::parallel_for(
                "bench_mixed_scalar",
                Kokkos::RangePolicy<exec_space>( 0, num_p ),
                KOKKOS_LAMBDA( const int i ) { s0( i ) += 1.0; } );
            Kokkos::fence();
            mixed_scalar_timer.stop( p );

            mixed_int_timer.start( p );
            Kokkos::parallel_for(
                "bench_mixed_int", Kokkos::RangePolicy<exec_space>( 0, num_p ),
                KOKKOS_LAMBDA( const int i ) { i0( i ) += 1; } );
            Kokkos::fence();
            mixed_int_timer.stop( p );

            mixed_all_timer.start( p );
            Kokkos::parallel_for(
                "bench_mixed_all", Kokkos::RangePolicy<exec_space>( 0, num_p ),
                KOKKOS_LAMBDA( const int i ) {
                    s0( i ) += 0.5;
                    for ( int d = 0; d < 3; ++d )
                        v3( i, d ) += s0( i );
                    for ( int d0 = 0; d0 < 3; ++d0 )
                        for ( int d1 = 0; d1 < 3; ++d1 )
                            m33( i, d0, d1 ) += v3( i, d0 ) + d1;
                    i0( i ) += static_cast<int>( m33( i, 0, 0 ) );
                } );
            Kokkos::fence();
            mixed_all_timer.stop( p );
        }
    }

    outputResults( stream, "problem_size", psizes, mixed_mat33_timer );
    outputResults( stream, "problem_size", psizes, mixed_vec3_timer );
    outputResults( stream, "problem_size", psizes, mixed_scalar_timer );
    outputResults( stream, "problem_size", psizes, mixed_int_timer );
    outputResults( stream, "problem_size", psizes, mixed_all_timer );
}

//---------------------------------------------------------------------------//
// Single-component AoSoA benchmark.
template <class Device>
void performanceTestSingle( std::ostream& stream,
                            const std::string& test_prefix,
                            const std::vector<int>& problem_sizes )
{
    using exec_space = typename Device::execution_space;
    using memory_space = typename Device::memory_space;

    constexpr int num_run = 20;

    using mat_member_types = Cabana::MemberTypes<double[3][3]>;
    using vec_member_types = Cabana::MemberTypes<double[3]>;
    using scalar_member_types = Cabana::MemberTypes<double>;
    using int_member_types = Cabana::MemberTypes<int>;

    using mat_aosoa_type = Cabana::AoSoA<mat_member_types, memory_space>;
    using vec_aosoa_type = Cabana::AoSoA<vec_member_types, memory_space>;
    using scalar_aosoa_type = Cabana::AoSoA<scalar_member_types, memory_space>;
    using int_aosoa_type = Cabana::AoSoA<int_member_types, memory_space>;

    int num_problem_size = problem_sizes.size();

    std::vector<mat_aosoa_type> mat_aosoas( num_problem_size );
    std::vector<vec_aosoa_type> vec_aosoas( num_problem_size );
    std::vector<scalar_aosoa_type> scalar_aosoas( num_problem_size );
    std::vector<int_aosoa_type> int_aosoas( num_problem_size );

    std::vector<int> psizes;

    // Create and initialize.
    for ( int p = 0; p < num_problem_size; ++p )
    {
        int num_p = problem_sizes[p];
        psizes.push_back( num_p );

        mat_aosoas[p].resize( num_p );
        vec_aosoas[p].resize( num_p );
        scalar_aosoas[p].resize( num_p );
        int_aosoas[p].resize( num_p );

        auto m33 = Cabana::slice<0>( mat_aosoas[p], "mat33" );
        auto v3 = Cabana::slice<0>( vec_aosoas[p], "vec3" );
        auto s0 = Cabana::slice<0>( scalar_aosoas[p], "scalar" );
        auto i0 = Cabana::slice<0>( int_aosoas[p], "int_scalar" );

        Kokkos::parallel_for(
            "init_single", Kokkos::RangePolicy<exec_space>( 0, num_p ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int d0 = 0; d0 < 3; ++d0 )
                    for ( int d1 = 0; d1 < 3; ++d1 )
                        m33( i, d0, d1 ) = 1.0 + d0 + d1;

                for ( int d = 0; d < 3; ++d )
                    v3( i, d ) = 2.0 + d;

                s0( i ) = 3.0;
                i0( i ) = 4;
            } );
        Kokkos::fence();
    }

    Cabana::Benchmark::Timer single_mat33_timer( test_prefix + "single_mat33",
                                                 num_problem_size );
    Cabana::Benchmark::Timer single_vec3_timer( test_prefix + "single_vec3",
                                                num_problem_size );
    Cabana::Benchmark::Timer single_scalar_timer( test_prefix + "single_scalar",
                                                  num_problem_size );
    Cabana::Benchmark::Timer single_int_timer( test_prefix + "single_int",
                                               num_problem_size );

    for ( int p = 0; p < num_problem_size; ++p )
    {
        int num_p = problem_sizes[p];

        std::cout << "Running single-component AoSoA benchmark for " << num_p
                  << " tuples on " << test_prefix << std::endl;

        auto m33 = Cabana::slice<0>( mat_aosoas[p], "mat33" );
        auto v3 = Cabana::slice<0>( vec_aosoas[p], "vec3" );
        auto s0 = Cabana::slice<0>( scalar_aosoas[p], "scalar" );
        auto i0 = Cabana::slice<0>( int_aosoas[p], "int_scalar" );

        for ( int t = 0; t < num_run; ++t )
        {
            single_mat33_timer.start( p );
            Kokkos::parallel_for(
                "bench_single_mat33",
                Kokkos::RangePolicy<exec_space>( 0, num_p ),
                KOKKOS_LAMBDA( const int i ) {
                    for ( int d0 = 0; d0 < 3; ++d0 )
                        for ( int d1 = 0; d1 < 3; ++d1 )
                            m33( i, d0, d1 ) += 1.0;
                } );
            Kokkos::fence();
            single_mat33_timer.stop( p );

            single_vec3_timer.start( p );
            Kokkos::parallel_for(
                "bench_single_vec3",
                Kokkos::RangePolicy<exec_space>( 0, num_p ),
                KOKKOS_LAMBDA( const int i ) {
                    for ( int d = 0; d < 3; ++d )
                        v3( i, d ) += 1.0;
                } );
            Kokkos::fence();
            single_vec3_timer.stop( p );

            single_scalar_timer.start( p );
            Kokkos::parallel_for(
                "bench_single_scalar",
                Kokkos::RangePolicy<exec_space>( 0, num_p ),
                KOKKOS_LAMBDA( const int i ) { s0( i ) += 1.0; } );
            Kokkos::fence();
            single_scalar_timer.stop( p );

            single_int_timer.start( p );
            Kokkos::parallel_for(
                "bench_single_int", Kokkos::RangePolicy<exec_space>( 0, num_p ),
                KOKKOS_LAMBDA( const int i ) { i0( i ) += 1; } );
            Kokkos::fence();
            single_int_timer.stop( p );
        }
    }

    outputResults( stream, "problem_size", psizes, single_mat33_timer );
    outputResults( stream, "problem_size", psizes, single_vec3_timer );
    outputResults( stream, "problem_size", psizes, single_scalar_timer );
    outputResults( stream, "problem_size", psizes, single_int_timer );
}

//---------------------------------------------------------------------------//
// main
int main( int argc, char* argv[] )
{
    Kokkos::initialize( argc, argv );
    {
        if ( argc < 2 )
            throw std::runtime_error(
                "Incorrect number of arguments.\n"
                "First argument - file name for output\n"
                "Optional second argument - run size (small or large)\n"
                "\n"
                "Example:\n"
                "./SlicePerformance test_results.txt\n" );

        std::string filename = argv[1];

        std::string run_type = "";
        if ( argc > 2 )
            run_type = argv[2];

        std::vector<int> problem_sizes = { 1000, 10000, 100000, 1000000 };
        std::vector<int> host_problem_sizes = problem_sizes;

        if ( run_type == "large" )
        {
            problem_sizes = { 10000, 100000, 1000000, 5000000, 10000000 };
            host_problem_sizes = { 10000, 100000, 1000000 };
        }

        std::fstream file;
        file.open( filename, std::fstream::out );

        using host_exec_space = Kokkos::DefaultHostExecutionSpace;
        using host_device_type = host_exec_space::device_type;
        using exec_space = Kokkos::DefaultExecutionSpace;
        using device_type = exec_space::device_type;

        if ( !std::is_same<device_type, host_device_type>{} )
        {
            performanceTestMixed<device_type>( file, "device_", problem_sizes );
            performanceTestSingle<device_type>( file, "device_",
                                                problem_sizes );
        }

        performanceTestMixed<host_device_type>( file, "host_",
                                                host_problem_sizes );
        performanceTestSingle<host_device_type>( file, "host_",
                                                 host_problem_sizes );

        file.close();
    }
    Kokkos::finalize();
    return 0;
}
