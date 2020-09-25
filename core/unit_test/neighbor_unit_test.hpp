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

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//
// List implementation.
template <class... Params>
struct TestNeighborList
{
    Kokkos::View<int *, Params...> counts;
    Kokkos::View<int **, Params...> neighbors;
};

template <class KokkosMemorySpace>
TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
createTestListHostCopy( const TestNeighborList<KokkosMemorySpace> &test_list )
{
    using data_layout = typename decltype( test_list.counts )::array_layout;
    TestNeighborList<data_layout, Kokkos::HostSpace> list_copy;
    Kokkos::resize( list_copy.counts, test_list.counts.extent( 0 ) );
    Kokkos::deep_copy( list_copy.counts, test_list.counts );
    Kokkos::resize( list_copy.neighbors, test_list.neighbors.extent( 0 ),
                    test_list.neighbors.extent( 1 ) );
    Kokkos::deep_copy( list_copy.neighbors, test_list.neighbors );
    return list_copy;
}

// Create a host copy of a list that implements the neighbor list interface.
template <class ListType>
TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
copyListToHost( const ListType &list, const int num_particle, const int max_n )
{
    TestNeighborList<TEST_MEMSPACE> list_copy;
    list_copy.counts =
        Kokkos::View<int *, TEST_MEMSPACE>( "counts", num_particle );
    list_copy.neighbors =
        Kokkos::View<int **, TEST_MEMSPACE>( "neighbors", num_particle, max_n );
    Kokkos::parallel_for(
        "copy list", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_particle ),
        KOKKOS_LAMBDA( const int p ) {
            list_copy.counts( p ) =
                Cabana::NeighborList<ListType>::numNeighbor( list, p );
            for ( int n = 0; n < list_copy.counts( p ); ++n )
                list_copy.neighbors( p, n ) =
                    Cabana::NeighborList<ListType>::getNeighbor( list, p, n );
        } );
    Kokkos::fence();
    return createTestListHostCopy( list_copy );
}

//---------------------------------------------------------------------------//
// Create particles.
Cabana::AoSoA<Cabana::MemberTypes<double[3]>, TEST_MEMSPACE>
createParticles( const int num_particle, const double box_min,
                 const double box_max )
{
    using DataTypes = Cabana::MemberTypes<double[3]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t aosoa( "aosoa", num_particle );

    auto position = Cabana::slice<0>( aosoa );
    using PoolType = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    using RandomType = Kokkos::Random_XorShift64<TEST_EXECSPACE>;
    PoolType pool( 342343901 );
    auto random_coord_op = KOKKOS_LAMBDA( const int p )
    {
        auto gen = pool.get_state();
        for ( int d = 0; d < 3; ++d )
            position( p, d ) =
                Kokkos::rand<RandomType, double>::draw( gen, box_min, box_max );
        pool.free_state( gen );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> exec_policy( 0, num_particle );
    Kokkos::parallel_for( exec_policy, random_coord_op );
    Kokkos::fence();
    return aosoa;
}

//---------------------------------------------------------------------------//
template <class PositionSlice>
TestNeighborList<TEST_MEMSPACE>
computeFullNeighborList( const PositionSlice &position,
                         const double neighborhood_radius )
{
    // Build a neighbor list with a brute force n^2 implementation. Count
    // first.
    TestNeighborList<TEST_MEMSPACE> list;
    int num_particle = position.size();
    double rsqr = neighborhood_radius * neighborhood_radius;
    list.counts = Kokkos::View<int *, TEST_MEMSPACE>( "test_neighbor_count",
                                                      num_particle );
    Kokkos::deep_copy( list.counts, 0 );
    auto count_op = KOKKOS_LAMBDA( const int i )
    {
        for ( int j = 0; j < num_particle; ++j )
        {
            if ( i != j )
            {
                double dsqr = 0.0;
                for ( int d = 0; d < 3; ++d )
                    dsqr += ( position( i, d ) - position( j, d ) ) *
                            ( position( i, d ) - position( j, d ) );
                if ( dsqr <= rsqr )
                    list.counts( i ) += 1;
            }
        }
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> exec_policy( 0, num_particle );
    Kokkos::parallel_for( exec_policy, count_op );
    Kokkos::fence();

    // Allocate.
    auto max_op = KOKKOS_LAMBDA( const int i, int &max_val )
    {
        if ( max_val < list.counts( i ) )
            max_val = list.counts( i );
    };
    int max_n;
    Kokkos::parallel_reduce( exec_policy, max_op, Kokkos::Max<int>( max_n ) );
    Kokkos::fence();
    list.neighbors = Kokkos::View<int **, TEST_MEMSPACE>( "test_neighbors",
                                                          num_particle, max_n );

    // Fill.
    auto fill_op = KOKKOS_LAMBDA( const int i )
    {
        int n_count = 0;
        for ( int j = 0; j < num_particle; ++j )
        {
            if ( i != j )
            {
                double dsqr = 0.0;
                for ( int d = 0; d < 3; ++d )
                    dsqr += ( position( i, d ) - position( j, d ) ) *
                            ( position( i, d ) - position( j, d ) );
                if ( dsqr <= rsqr )
                {
                    list.neighbors( i, n_count ) = j;
                    ++n_count;
                }
            }
        }
    };
    Kokkos::parallel_for( exec_policy, fill_op );
    Kokkos::fence();

    return list;
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType>
void checkFullNeighborList( const ListType &nlist,
                            const TestListType &N2_list_copy,
                            const int num_particle )
{
    // Create host neighbor list copy.
    auto list_copy = copyListToHost( nlist, N2_list_copy.neighbors.extent( 0 ),
                                     N2_list_copy.neighbors.extent( 1 ) );

    // Check the results.
    for ( int p = 0; p < num_particle; ++p )
    {
        // First check that the number of neighbors are the same.
        EXPECT_EQ( list_copy.counts( p ), N2_list_copy.counts( p ) );

        // Now extract the neighbors.
        std::vector<int> computed_neighbors( N2_list_copy.counts( p ) );
        std::vector<int> actual_neighbors( N2_list_copy.counts( p ) );
        for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
        {
            computed_neighbors[n] = list_copy.neighbors( p, n );
            actual_neighbors[n] = N2_list_copy.neighbors( p, n );
        }

        // Sort them because we have no guarantee of the order we will find
        // them in.
        std::sort( computed_neighbors.begin(), computed_neighbors.end() );
        std::sort( actual_neighbors.begin(), actual_neighbors.end() );

        // Now compare directly.
        for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
            EXPECT_EQ( computed_neighbors[n], actual_neighbors[n] );
    }
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType>
void checkHalfNeighborList( const ListType &nlist,
                            const TestListType &N2_list_copy,
                            const int num_particle )
{
    // Create host neighbor list copy.
    auto list_copy = copyListToHost( nlist, N2_list_copy.neighbors.extent( 0 ),
                                     N2_list_copy.neighbors.extent( 1 ) );

    // Check that the full list is twice the size of the half list.
    int half_size = 0;
    int full_size = 0;
    for ( int p = 0; p < num_particle; ++p )
    {
        half_size += list_copy.counts( p );
        full_size += N2_list_copy.counts( p );
    }
    EXPECT_EQ( full_size, 2 * half_size );

    // Check the half list by ensuring that a particle does not show up in the
    // neighbor list of its neighbors.
    for ( int p = 0; p < num_particle; ++p )
    {
        // Check each neighbor of p
        for ( int n = 0; n < list_copy.counts( p ); ++n )
        {
            // Get the id of the nth neighbor of p.
            auto p_n = list_copy.neighbors( p, n );

            // Check that p is not in the neighbor list of the nth neighbor of
            // p.
            for ( int m = 0; m < list_copy.counts( p_n ); ++m )
            {
                auto n_m = list_copy.neighbors( p_n, m );
                EXPECT_NE( n_m, p );
            }
        }
    }
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType>
void checkFullNeighborListPartialRange( const ListType &nlist,
                                        const TestListType N2_list_copy,
                                        const int num_particle,
                                        const int num_ignore )
{
    // Create host neighbor list copy.
    auto list_copy = copyListToHost( nlist, N2_list_copy.neighbors.extent( 0 ),
                                     N2_list_copy.neighbors.extent( 1 ) );

    // Check the results.
    for ( int p = 0; p < num_particle; ++p )
    {
        if ( p < num_ignore )
        {
            // First check that the number of neighbors are the same.
            EXPECT_EQ( list_copy.counts( p ), N2_list_copy.counts( p ) );

            // Now extract the neighbors.
            std::vector<int> computed_neighbors( N2_list_copy.counts( p ) );
            std::vector<int> actual_neighbors( N2_list_copy.counts( p ) );
            for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
            {
                computed_neighbors[n] = list_copy.neighbors( p, n );
                actual_neighbors[n] = N2_list_copy.neighbors( p, n );
            }

            // Sort them because we have no guarantee of the order we will find
            // them in.
            std::sort( computed_neighbors.begin(), computed_neighbors.end() );
            std::sort( actual_neighbors.begin(), actual_neighbors.end() );

            // Now compare directly.
            for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
                EXPECT_EQ( computed_neighbors[n], actual_neighbors[n] );
        }
        else
        {
            EXPECT_EQ( list_copy.counts( p ), 0 );
        }
    }
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType>
void checkFirstNeighborParallelFor( const ListType &nlist,
                                    const TestListType &N2_list_copy,
                                    const int num_particle )
{
    // Create Kokkos views for the write operation.
    using memory_space = typename TEST_MEMSPACE::memory_space;
    Kokkos::View<int *, Kokkos::HostSpace> N2_result( "N2_result",
                                                      num_particle );
    Kokkos::View<int *, memory_space> serial_result( "serial_result",
                                                     num_particle );
    Kokkos::View<int *, memory_space> team_result( "team_result",
                                                   num_particle );

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto serial_count_op = KOKKOS_LAMBDA( const int i, const int n )
    {
        Kokkos::atomic_add( &serial_result( i ), n );
    };
    auto team_count_op = KOKKOS_LAMBDA( const int i, const int n )
    {
        Kokkos::atomic_add( &team_result( i ), n );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );
    Cabana::neighbor_parallel_for( policy, serial_count_op, nlist,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(), "test_1st_serial" );
    Cabana::neighbor_parallel_for( policy, team_count_op, nlist,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::TeamOpTag(), "test_1st_team" );
    Kokkos::fence();

    // Use a full N^2 neighbor list to check against.
    for ( int p = 0; p < num_particle; ++p )
        for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
            N2_result( p ) += N2_list_copy.neighbors( p, n );

    // Check the result.
    auto serial_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), serial_result );
    auto team_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), team_result );
    for ( int p = 0; p < num_particle; ++p )
    {
        EXPECT_EQ( N2_result( p ), serial_mirror( p ) );
        EXPECT_EQ( N2_result( p ), team_mirror( p ) );
    }
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType>
void checkSecondNeighborParallelFor( const ListType &nlist,
                                     const TestListType &N2_list_copy,
                                     const int num_particle )
{
    // Create Kokkos views for the write operation.
    using memory_space = typename TEST_MEMSPACE::memory_space;
    Kokkos::View<int *, Kokkos::HostSpace> N2_result( "N2_result",
                                                      num_particle );
    Kokkos::View<int *, memory_space> serial_result( "serial_result",
                                                     num_particle );
    Kokkos::View<int *, memory_space> team_result( "team_result",
                                                   num_particle );
    Kokkos::View<int *, memory_space> vector_result( "vector_result",
                                                     num_particle );

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto serial_count_op =
        KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        Kokkos::atomic_add( &serial_result( i ), j );
        Kokkos::atomic_add( &serial_result( i ), k );
    };
    auto team_count_op = KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        Kokkos::atomic_add( &team_result( i ), j );
        Kokkos::atomic_add( &team_result( i ), k );
    };
    auto vector_count_op =
        KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        Kokkos::atomic_add( &vector_result( i ), j );
        Kokkos::atomic_add( &vector_result( i ), k );
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );
    Cabana::neighbor_parallel_for( policy, serial_count_op, nlist,
                                   Cabana::SecondNeighborsTag(),
                                   Cabana::SerialOpTag(), "test_2nd_serial" );
    Cabana::neighbor_parallel_for( policy, team_count_op, nlist,
                                   Cabana::SecondNeighborsTag(),
                                   Cabana::TeamOpTag(), "test_2nd_team" );
    Cabana::neighbor_parallel_for(
        policy, vector_count_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamVectorOpTag(), "test_2nd_vector" );
    Kokkos::fence();

    // Use a full N^2 neighbor list to check against.
    for ( int p = 0; p < num_particle; ++p )
        for ( int n = 0; n < N2_list_copy.counts( p ) - 1; ++n )
            for ( int a = n + 1; a < N2_list_copy.counts( p ); ++a )
            {
                N2_result( p ) += N2_list_copy.neighbors( p, n );
                N2_result( p ) += N2_list_copy.neighbors( p, a );
            }

    // Check the result.
    auto serial_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), serial_result );
    auto team_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), team_result );
    auto vector_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), vector_result );
    for ( int p = 0; p < num_particle; ++p )
    {
        EXPECT_EQ( N2_result( p ), serial_mirror( p ) );
        EXPECT_EQ( N2_result( p ), team_mirror( p ) );
        EXPECT_EQ( N2_result( p ), vector_mirror( p ) );
    }
}
//---------------------------------------------------------------------------//
template <class ListType, class TestListType, class AoSoAType>
void checkFirstNeighborParallelReduce( const ListType &nlist,
                                       const TestListType &N2_list_copy,
                                       const AoSoAType &aosoa )
{
    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto position = Cabana::slice<0>( aosoa );
    auto sum_op = KOKKOS_LAMBDA( const int i, const int n, double &sum )
    {
        sum += position( i, 0 ) + position( n, 0 );
    };

    int num_particle = position.size();
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );

    // Do the reductions.
    double serial_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::FirstNeighborsTag(),
        Cabana::SerialOpTag(), serial_sum, "test_reduce_serial" );
#ifndef KOKKOS_ENABLE_HIP // FIXME_HIP
    double team_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::FirstNeighborsTag(), Cabana::TeamOpTag(),
        team_sum, "test_reduce_team" );
#endif
    Kokkos::fence();

    // Get the expected result from N^2 list in serial.
    auto aosoa_mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto positions_mirror = Cabana::slice<0>( aosoa_mirror );
    double N2_sum = 0;
    for ( int p = 0; p < num_particle; ++p )
        for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
            N2_sum += positions_mirror( p, 0 ) +
                      positions_mirror( N2_list_copy.neighbors( p, n ), 0 );

    // Check the result.
    EXPECT_FLOAT_EQ( N2_sum, serial_sum );
#ifndef KOKKOS_ENABLE_HIP // FIXME_HIP
    EXPECT_FLOAT_EQ( N2_sum, team_sum );
#endif
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType, class AoSoAType>
void checkSecondNeighborParallelReduce( const ListType &nlist,
                                        const TestListType &N2_list_copy,
                                        const AoSoAType &aosoa )
{
    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto position = Cabana::slice<0>( aosoa );
    auto sum_op =
        KOKKOS_LAMBDA( const int i, const int n, const int a, double &sum )
    {
        sum += position( i, 0 ) + position( n, 0 ) + position( a, 0 );
    };

    int num_particle = position.size();
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );

    // Do the reductions.
    double serial_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::SerialOpTag(), serial_sum, "test_reduce_serial" );
#ifndef KOKKOS_ENABLE_HIP // FIXME_HIP
    double team_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamOpTag(), team_sum, "test_reduce_team" );
    double vector_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamVectorOpTag(), vector_sum, "test_reduce_vector" );
#endif
    Kokkos::fence();

    // Get the expected result from N^2 list in serial.
    auto aosoa_mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto positions_mirror = Cabana::slice<0>( aosoa_mirror );
    double N2_sum = 0;
    for ( int p = 0; p < num_particle; ++p )
        for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
            for ( int a = n + 1; a < N2_list_copy.counts( p ); ++a )
                N2_sum +=
                    positions_mirror( p, 0 ) +
                    positions_mirror( N2_list_copy.neighbors( p, n ), 0 ) +
                    positions_mirror( N2_list_copy.neighbors( p, a ), 0 );

    // Check the result.
    EXPECT_FLOAT_EQ( N2_sum, serial_sum );
#ifndef KOKKOS_ENABLE_HIP // FIXME_HIP
    EXPECT_FLOAT_EQ( N2_sum, team_sum );
    EXPECT_FLOAT_EQ( N2_sum, vector_sum );
#endif
}

//---------------------------------------------------------------------------//
// Default test settings.
struct NeighborListTestData
{
    int num_particle = 1e3;
    double test_radius = 2.32;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;

    double cell_size_ratio = 0.5;
    double grid_min[3] = { box_min, box_min, box_min };
    double grid_max[3] = { box_max, box_max, box_max };

    Cabana::AoSoA<Cabana::MemberTypes<double[3]>, TEST_MEMSPACE> aosoa;
    TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
        N2_list_copy;

    NeighborListTestData()
    {
        // Create the AoSoA and fill with random particle positions.
        aosoa = createParticles( num_particle, box_min, box_max );

        // Create a full N^2 neighbor list to check against.
        auto N2_list =
            computeFullNeighborList( Cabana::slice<0>( aosoa ), test_radius );
        N2_list_copy = createTestListHostCopy( N2_list );
    }
};

//---------------------------------------------------------------------------//
} // end namespace Test
