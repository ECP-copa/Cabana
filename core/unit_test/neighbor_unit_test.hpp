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

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>
#include <Cabana_ParticleInit.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//
// List implementation.
template <class... Params>
struct TestNeighborList
{
    Kokkos::View<int*, Params...> counts;
    Kokkos::View<int**, Params...> neighbors;
    int max;
    int total;
};

template <class KokkosMemorySpace>
TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
createTestListHostCopy( const TestNeighborList<KokkosMemorySpace>& test_list )
{
    using data_layout = typename decltype( test_list.counts )::array_layout;
    TestNeighborList<data_layout, Kokkos::HostSpace> list_copy;
    Kokkos::resize( list_copy.counts, test_list.counts.extent( 0 ) );
    Kokkos::deep_copy( list_copy.counts, test_list.counts );
    Kokkos::resize( list_copy.neighbors, test_list.neighbors.extent( 0 ),
                    test_list.neighbors.extent( 1 ) );
    Kokkos::deep_copy( list_copy.neighbors, test_list.neighbors );
    list_copy.total = test_list.total;
    list_copy.max = test_list.max;
    return list_copy;
}

// Create a host copy of a list that implements the neighbor list interface.
template <class ListType>
TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
copyListToHost( const ListType& list, const std::size_t num_particle,
                const std::size_t max_n )
{
    TestNeighborList<TEST_MEMSPACE> list_copy;
    list_copy.counts =
        Kokkos::View<int*, TEST_MEMSPACE>( "counts", num_particle );
    list_copy.neighbors =
        Kokkos::View<int**, TEST_MEMSPACE>( "neighbors", num_particle, max_n );
    Kokkos::Max<int> max_reduce( list_copy.max );
    // Use max here because every rank should return the same value.
    Kokkos::Max<int> total_reduce( list_copy.total );
    Kokkos::parallel_reduce(
        "copy list", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_particle ),
        KOKKOS_LAMBDA( const std::size_t p, int& max_val, int& total_val ) {
            list_copy.counts( p ) =
                Cabana::NeighborList<ListType>::numNeighbor( list, p );
            for ( int n = 0; n < list_copy.counts( p ); ++n )
                list_copy.neighbors( p, n ) =
                    Cabana::NeighborList<ListType>::getNeighbor( list, p, n );

            // Same for every particle, but we need to extract on device.
            max_val = Cabana::NeighborList<ListType>::maxNeighbor( list );
            total_val = Cabana::NeighborList<ListType>::totalNeighbor( list );
        },
        max_reduce, total_reduce );
    Kokkos::fence();
    return createTestListHostCopy( list_copy );
}

//---------------------------------------------------------------------------//
template <std::size_t Dim, class PositionSlice>
TestNeighborList<TEST_MEMSPACE>
computeFullNeighborList( const PositionSlice& position,
                         const double neighborhood_radius )
{
    // Build a neighbor list with a brute force n^2 implementation. Count
    // first.
    TestNeighborList<TEST_MEMSPACE> list;
    std::size_t num_particle = position.size();
    double rsqr = neighborhood_radius * neighborhood_radius;
    list.counts = Kokkos::View<int*, TEST_MEMSPACE>( "test_neighbor_count",
                                                     num_particle );
    Kokkos::deep_copy( list.counts, 0 );
    auto count_op = KOKKOS_LAMBDA( const std::size_t i )
    {
        for ( std::size_t j = 0; j < num_particle; ++j )
        {
            if ( i != j )
            {
                double dsqr = 0.0;
                for ( std::size_t d = 0; d < Dim; ++d )
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
    auto max_op =
        KOKKOS_LAMBDA( const std::size_t i, int& max_val, int& total_val )
    {
        if ( max_val < list.counts( i ) )
        {
            max_val = list.counts( i );
        }
        total_val += list.counts( i );
    };
    Kokkos::parallel_reduce( exec_policy, max_op, Kokkos::Max<int>( list.max ),
                             Kokkos::Sum<int>( list.total ) );
    Kokkos::fence();
    list.neighbors = Kokkos::View<int**, TEST_MEMSPACE>(
        "test_neighbors", num_particle, list.max );

    // Fill.
    auto fill_op = KOKKOS_LAMBDA( const std::size_t i )
    {
        int n_count = 0;
        for ( std::size_t j = 0; j < num_particle; ++j )
        {
            if ( i != j )
            {
                double dsqr = 0.0;
                for ( std::size_t d = 0; d < Dim; ++d )
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
void checkFullNeighborList( const ListType& nlist,
                            const TestListType& N2_list_copy,
                            const std::size_t num_particle )
{
    // Create host neighbor list copy.
    auto list_copy = copyListToHost( nlist, N2_list_copy.neighbors.extent( 0 ),
                                     N2_list_copy.neighbors.extent( 1 ) );

    // Check the results.
    for ( std::size_t p = 0; p < num_particle; ++p )
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

    // Check the total and max interfaces.
    EXPECT_EQ( N2_list_copy.max, list_copy.max );
    EXPECT_EQ( N2_list_copy.total, list_copy.total );
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType>
void checkHalfNeighborList( const ListType& nlist,
                            const TestListType& N2_list_copy,
                            const std::size_t num_particle )
{
    // Create host neighbor list copy.
    auto list_copy = copyListToHost( nlist, N2_list_copy.neighbors.extent( 0 ),
                                     N2_list_copy.neighbors.extent( 1 ) );

    // Check that the full list is twice the size of the half list.
    std::size_t half_size = 0;
    std::size_t full_size = 0;
    for ( std::size_t p = 0; p < num_particle; ++p )
    {
        half_size += list_copy.counts( p );
        full_size += N2_list_copy.counts( p );
    }
    EXPECT_EQ( full_size, 2 * half_size );

    // Check the half list by ensuring that a particle does not show up in the
    // neighbor list of its neighbors.
    for ( std::size_t p = 0; p < num_particle; ++p )
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

    // Check the total and max interfaces (only approximate for max).
    EXPECT_GE( N2_list_copy.max, list_copy.max );
    EXPECT_EQ( static_cast<int>( N2_list_copy.total / 2.0 ), list_copy.total );
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType>
void checkFullNeighborListPartialRange( const ListType& nlist,
                                        const TestListType N2_list_copy,
                                        const std::size_t num_particle,
                                        const std::size_t begin,
                                        const std::size_t end )
{
    // Create host neighbor list copy.
    auto list_copy = copyListToHost( nlist, N2_list_copy.neighbors.extent( 0 ),
                                     N2_list_copy.neighbors.extent( 1 ) );

    // Check the results.
    for ( std::size_t p = 0; p < num_particle; ++p )
    {
        if ( p >= begin && p < end )
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
template <class TestListType, class ViewType>
void checkFirstNeighborParallelFor( const TestListType& N2_list_copy,
                                    const ViewType& serial_result,
                                    const ViewType& team_result,
                                    const int multiplier,
                                    const std::size_t begin,
                                    const std::size_t end )
{
    std::size_t num_particle = serial_result.size();
    Kokkos::View<int*, Kokkos::HostSpace> N2_result( "N2_result",
                                                     num_particle );

    // Use a full N^2 neighbor list to check against.
    for ( std::size_t p = begin; p < end; ++p )
        for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
        {
            if ( N2_list_copy.neighbors( p, n ) >= static_cast<int>( begin ) &&
                 N2_list_copy.neighbors( p, n ) < static_cast<int>( end ) )
                N2_result( p ) += N2_list_copy.neighbors( p, n );
        }

    // Check the result.
    auto serial_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), serial_result );
    auto team_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), team_result );
    for ( std::size_t p = 0; p < num_particle; ++p )
    {
        if ( p >= begin && p < end )
        {
            EXPECT_EQ( N2_result( p ) * multiplier, serial_mirror( p ) );
            EXPECT_EQ( N2_result( p ) * multiplier, team_mirror( p ) );
        }
    }
}

template <class TestListType, class ViewType>
void checkSecondNeighborParallelFor( const TestListType& N2_list_copy,
                                     const ViewType& serial_result,
                                     const ViewType& team_result,
                                     const ViewType& vector_result,
                                     const int multiplier )
{
    std::size_t num_particle = serial_result.size();
    Kokkos::View<int*, Kokkos::HostSpace> N2_result( "N2_result",
                                                     num_particle );

    // Use a full N^2 neighbor list to check against.
    for ( std::size_t p = 0; p < num_particle; ++p )
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
    for ( std::size_t p = 0; p < num_particle; ++p )
    {
        EXPECT_EQ( N2_result( p ) * multiplier, serial_mirror( p ) );
        EXPECT_EQ( N2_result( p ) * multiplier, team_mirror( p ) );
        EXPECT_EQ( N2_result( p ) * multiplier, vector_mirror( p ) );
    }
}

template <class TestListType, class AoSoAType>
void checkFirstNeighborParallelReduce(
    const TestListType& N2_list_copy, const AoSoAType& aosoa,
    const double serial_sum, const double team_sum, const int multiplier,
    const std::size_t begin, const std::size_t end )
{
    std::size_t num_particle = aosoa.size();

    // Get the expected result from N^2 list in serial.
    auto aosoa_mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto positions_mirror = Cabana::slice<0>( aosoa_mirror );
    double N2_sum = 0;
    for ( std::size_t p = 0; p < num_particle; ++p )
        for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
            if ( p >= begin && p < end )
            {
                if ( N2_list_copy.neighbors( p, n ) >=
                         static_cast<int>( begin ) &&
                     N2_list_copy.neighbors( p, n ) < static_cast<int>( end ) )
                    N2_sum +=
                        positions_mirror( p, 0 ) +
                        positions_mirror( N2_list_copy.neighbors( p, n ), 0 );
            }

    // Check the result.
    EXPECT_FLOAT_EQ( N2_sum * multiplier, serial_sum );
    EXPECT_FLOAT_EQ( N2_sum * multiplier, team_sum );
}

template <class TestListType, class AoSoAType>
void checkSecondNeighborParallelReduce( const TestListType& N2_list_copy,
                                        const AoSoAType& aosoa,
                                        const double serial_sum,
                                        const double team_sum,
                                        const double vector_sum,
                                        const int multiplier )
{
    std::size_t num_particle = aosoa.size();

    // Get the expected result from N^2 list in serial.
    auto aosoa_mirror =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), aosoa );
    auto positions_mirror = Cabana::slice<0>( aosoa_mirror );
    double N2_sum = 0;
    for ( std::size_t p = 0; p < num_particle; ++p )
        for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
            for ( int a = n + 1; a < N2_list_copy.counts( p ); ++a )
                N2_sum +=
                    positions_mirror( p, 0 ) +
                    positions_mirror( N2_list_copy.neighbors( p, n ), 0 ) +
                    positions_mirror( N2_list_copy.neighbors( p, a ), 0 );

    // Check the result.
    EXPECT_FLOAT_EQ( N2_sum * multiplier, serial_sum );
    EXPECT_FLOAT_EQ( N2_sum * multiplier, team_sum );
    EXPECT_FLOAT_EQ( N2_sum * multiplier, vector_sum );
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType>
void checkFirstNeighborParallelForLambda( const ListType& nlist,
                                          const TestListType& N2_list_copy,
                                          const std::size_t num_particle )
{
    // Create Kokkos views for the write operation.
    using memory_space = typename TEST_MEMSPACE::memory_space;
    Kokkos::View<int*, Kokkos::HostSpace> N2_result( "N2_result",
                                                     num_particle );
    Kokkos::View<int*, memory_space> serial_result( "serial_result",
                                                    num_particle );
    Kokkos::View<int*, memory_space> team_result( "team_result", num_particle );

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto serial_count_op =
        KOKKOS_LAMBDA( const std::size_t i, const std::size_t n )
    {
        Kokkos::atomic_add( &serial_result( i ), n );
    };
    auto team_count_op =
        KOKKOS_LAMBDA( const std::size_t i, const std::size_t n )
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

    checkFirstNeighborParallelFor( N2_list_copy, serial_result, team_result, 1,
                                   0, num_particle );
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType>
void checkSecondNeighborParallelForLambda( const ListType& nlist,
                                           const TestListType& N2_list_copy,
                                           const std::size_t num_particle )
{
    // Create Kokkos views for the write operation.
    using memory_space = typename TEST_MEMSPACE::memory_space;
    Kokkos::View<int*, memory_space> serial_result( "serial_result",
                                                    num_particle );
    Kokkos::View<int*, memory_space> team_result( "team_result", num_particle );
    Kokkos::View<int*, memory_space> vector_result( "vector_result",
                                                    num_particle );

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto serial_count_op = KOKKOS_LAMBDA(
        const std::size_t i, const std::size_t j, const std::size_t k )
    {
        Kokkos::atomic_add( &serial_result( i ), j );
        Kokkos::atomic_add( &serial_result( i ), k );
    };
    auto team_count_op = KOKKOS_LAMBDA(
        const std::size_t i, const std::size_t j, const std::size_t k )
    {
        Kokkos::atomic_add( &team_result( i ), j );
        Kokkos::atomic_add( &team_result( i ), k );
    };
    auto vector_count_op = KOKKOS_LAMBDA(
        const std::size_t i, const std::size_t j, const std::size_t k )
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

    checkSecondNeighborParallelFor( N2_list_copy, serial_result, team_result,
                                    vector_result, 1 );
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType>
void checkSplitFirstNeighborParallelFor( const ListType& nlist,
                                         const TestListType& N2_list_copy,
                                         const std::size_t num_particle )
{
    // Create Kokkos views for the write operation.
    using memory_space = typename TEST_MEMSPACE::memory_space;
    Kokkos::View<int*, Kokkos::HostSpace> N2_result( "N2_result",
                                                     num_particle );
    Kokkos::View<int*, memory_space> serial_result( "serial_result",
                                                    num_particle );
    Kokkos::View<int*, memory_space> team_result( "team_result", num_particle );

    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );
    const auto range_begin = policy.begin();
    using team_policy_type =
        Kokkos::TeamPolicy<TEST_EXECSPACE, Kokkos::Schedule<Kokkos::Dynamic>>;
    team_policy_type team_policy( policy.end() - policy.begin(), Kokkos::AUTO );

    // Add the number of neighbors to the per particle counts.
    auto serial_neigh_op =
        KOKKOS_LAMBDA( const std::size_t i, const std::size_t n )
    {
        Kokkos::atomic_add( &serial_result( i ), n );
    };
    auto team_neigh_op =
        KOKKOS_LAMBDA( const std::size_t i, const std::size_t n )
    {
        Kokkos::atomic_add( &team_result( i ), n );
    };

    // Test the split neighbor iteration by adding a value from each central
    // particle and each neighbor (separately) and compare to N^2 counts.
    auto serial_central_op = KOKKOS_LAMBDA( const std::size_t i )
    {
        Kokkos::atomic_add( &serial_result( i ), i );

        Cabana::for_each_neighbor( i, serial_neigh_op, nlist,
                                   Cabana::FirstNeighborsTag() );
    };
    auto team_central_op =
        KOKKOS_LAMBDA( const typename team_policy_type::member_type& team )
    {
        const std::size_t i = team.league_rank() + range_begin;

        // Restrict central particle updates to once per team.
        Kokkos::single( Kokkos::PerTeam( team ),
                        [=]() { Kokkos::atomic_add( &team_result( i ), i ); } );

        Cabana::for_each_neighbor( i, team, team_neigh_op, nlist,
                                   Cabana::FirstNeighborsTag() );
    };

    Kokkos::parallel_for( "test_embedded_1st_serial", policy,
                          serial_central_op );
    Kokkos::parallel_for( "test_embedded_1st_team", team_policy,
                          team_central_op );
    Kokkos::fence();

    // Use a full N^2 neighbor list to check against.
    for ( std::size_t p = 0; p < num_particle; ++p )
    {
        N2_result( p ) += p;
        for ( int n = 0; n < N2_list_copy.counts( p ); ++n )
        {
            N2_result( p ) += N2_list_copy.neighbors( p, n );
        }
    }

    // Check the result.
    auto serial_mirror = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), serial_result );
    auto team_mirror =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), team_result );
    for ( std::size_t p = 0; p < num_particle; ++p )
    {
        EXPECT_EQ( N2_result( p ), serial_mirror( p ) );
        EXPECT_EQ( N2_result( p ), team_mirror( p ) );
    }
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType, class AoSoAType>
void checkFirstNeighborParallelReduceLambda( const ListType& nlist,
                                             const TestListType& N2_list_copy,
                                             const AoSoAType& aosoa )
{
    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto position = Cabana::slice<0>( aosoa );
    auto sum_op =
        KOKKOS_LAMBDA( const std::size_t i, const std::size_t n, double& sum )
    {
        sum += position( i, 0 ) + position( n, 0 );
    };

    std::size_t num_particle = position.size();
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );

    // Do the reductions.
    double serial_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::FirstNeighborsTag(),
        Cabana::SerialOpTag(), serial_sum, "test_reduce_serial" );
    double team_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::FirstNeighborsTag(), Cabana::TeamOpTag(),
        team_sum, "test_reduce_team" );
    Kokkos::fence();

    checkFirstNeighborParallelReduce( N2_list_copy, aosoa, serial_sum, team_sum,
                                      1, 0, num_particle );
}

//---------------------------------------------------------------------------//
template <class ListType, class TestListType, class AoSoAType>
void checkSecondNeighborParallelReduceLambda( const ListType& nlist,
                                              const TestListType& N2_list_copy,
                                              const AoSoAType& aosoa )
{
    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto position = Cabana::slice<0>( aosoa );
    auto sum_op = KOKKOS_LAMBDA( const std::size_t i, const std::size_t n,
                                 const std::size_t a, double& sum )
    {
        sum += position( i, 0 ) + position( n, 0 ) + position( a, 0 );
    };

    std::size_t num_particle = position.size();
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );

    // Do the reductions.
    double serial_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::SerialOpTag(), serial_sum, "test_reduce_serial" );
    double team_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamOpTag(), team_sum, "test_reduce_team" );
    double vector_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, sum_op, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamVectorOpTag(), vector_sum, "test_reduce_vector" );
    Kokkos::fence();

    checkSecondNeighborParallelReduce( N2_list_copy, aosoa, serial_sum,
                                       team_sum, vector_sum, 1 );
}

//---------------------------------------------------------------------------//
// Check parallel with functor (with and without work tag)

// Functor work tag for assigning double the value.
class DoubleValueWorkTag
{
};

template <class ViewType>
struct FirstNeighForOp
{
    ViewType _result;

    FirstNeighForOp( const std::size_t num_particle )
    {
        _result = ViewType( "result", num_particle );
    }

    // tagged version that assigns double the value.
    KOKKOS_INLINE_FUNCTION void operator()( const DoubleValueWorkTag&,
                                            const std::size_t i,
                                            const std::size_t n ) const
    {
        Kokkos::atomic_add( &_result( i ), 2 * n );
    }
    KOKKOS_INLINE_FUNCTION void operator()( const std::size_t i,
                                            const std::size_t n ) const
    {
        Kokkos::atomic_add( &_result( i ), n );
    }
};

template <class ListType, class TestListType>
void checkFirstNeighborParallelForFunctor( const ListType& nlist,
                                           const TestListType& N2_list_copy,
                                           const std::size_t num_particle,
                                           const bool use_tag )
{
    if ( use_tag )
    {
        Kokkos::RangePolicy<TEST_EXECSPACE, DoubleValueWorkTag> policy(
            0, num_particle );
        checkFirstNeighborParallelForFunctor( nlist, N2_list_copy, policy, 2,
                                              num_particle );
    }
    else
    {
        Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );
        checkFirstNeighborParallelForFunctor( nlist, N2_list_copy, policy, 1,
                                              num_particle );
    }
}

template <class ListType, class TestListType, class PolicyType>
void checkFirstNeighborParallelForFunctor( const ListType& nlist,
                                           const TestListType& N2_list_copy,
                                           const PolicyType policy,
                                           const int multiplier,
                                           const std::size_t num_particle )
{
    // Create Kokkos views for the write operation.
    using memory_space = typename TEST_MEMSPACE::memory_space;
    using view_type = Kokkos::View<int*, memory_space>;

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts using a functor.
    FirstNeighForOp<view_type> serial_functor( num_particle );
    FirstNeighForOp<view_type> team_functor( num_particle );

    Cabana::neighbor_parallel_for( policy, serial_functor, nlist,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(), "test_1st_serial" );
    Cabana::neighbor_parallel_for( policy, team_functor, nlist,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::TeamOpTag(), "test_1st_team" );
    Kokkos::fence();

    checkFirstNeighborParallelFor( N2_list_copy, serial_functor._result,
                                   team_functor._result, multiplier, 0,
                                   num_particle );
}

//---------------------------------------------------------------------------//
template <class ViewType>
struct SecondNeighForOp
{
    ViewType _result;

    SecondNeighForOp( const std::size_t num_particle )
    {
        _result = ViewType( "result", num_particle );
    }

    // tagged version that assigns double the value.
    KOKKOS_INLINE_FUNCTION void operator()( const DoubleValueWorkTag&,
                                            const std::size_t i,
                                            const std::size_t n,
                                            const std::size_t a ) const
    {
        Kokkos::atomic_add( &_result( i ), 2 * ( n + a ) );
    }
    KOKKOS_INLINE_FUNCTION void operator()( const std::size_t i,
                                            const std::size_t n,
                                            const std::size_t a ) const
    {
        Kokkos::atomic_add( &_result( i ), n + a );
    }
};

template <class ListType, class TestListType>
void checkSecondNeighborParallelForFunctor( const ListType& nlist,
                                            const TestListType& N2_list_copy,
                                            const std::size_t num_particle,
                                            const bool use_tag )
{
    if ( use_tag )
    {
        Kokkos::RangePolicy<TEST_EXECSPACE, DoubleValueWorkTag> policy(
            0, num_particle );
        checkSecondNeighborParallelForFunctor( nlist, N2_list_copy,
                                               num_particle, policy, 2 );
    }
    else
    {
        Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );
        checkSecondNeighborParallelForFunctor( nlist, N2_list_copy,
                                               num_particle, policy, 1 );
    }
}

template <class ListType, class TestListType, class PolicyType>
void checkSecondNeighborParallelForFunctor( const ListType& nlist,
                                            const TestListType& N2_list_copy,
                                            const std::size_t num_particle,
                                            const PolicyType policy,
                                            const int multiplier )
{
    // Create Kokkos views for the write operation.
    using memory_space = typename TEST_MEMSPACE::memory_space;
    using view_type = Kokkos::View<int*, memory_space>;

    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts using a functor.
    SecondNeighForOp<view_type> serial_functor( num_particle );
    SecondNeighForOp<view_type> team_functor( num_particle );
    SecondNeighForOp<view_type> vector_functor( num_particle );

    Cabana::neighbor_parallel_for( policy, serial_functor, nlist,
                                   Cabana::SecondNeighborsTag(),
                                   Cabana::SerialOpTag(), "test_2nd_serial" );
    Cabana::neighbor_parallel_for( policy, team_functor, nlist,
                                   Cabana::SecondNeighborsTag(),
                                   Cabana::TeamOpTag(), "test_2nd_team" );
    Cabana::neighbor_parallel_for(
        policy, vector_functor, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamVectorOpTag(), "test_2nd_vector" );

    Kokkos::fence();

    checkSecondNeighborParallelFor( N2_list_copy, serial_functor._result,
                                    team_functor._result,
                                    vector_functor._result, multiplier );
}

//---------------------------------------------------------------------------//
// Check parallel reductions with functor (with and without work tag)
template <class PositionSlice>
struct FirstNeighReduceOp
{
    PositionSlice _position;

    FirstNeighReduceOp( const PositionSlice position )
        : _position( position )
    {
    }

    // tagged version that assigns double the value.
    KOKKOS_INLINE_FUNCTION void operator()( const DoubleValueWorkTag&,
                                            const std::size_t i,
                                            const std::size_t n,
                                            double& sum ) const
    {
        sum += ( _position( i, 0 ) + _position( n, 0 ) ) * 2;
    }
    KOKKOS_INLINE_FUNCTION void
    operator()( const std::size_t i, const std::size_t n, double& sum ) const
    {
        sum += _position( i, 0 ) + _position( n, 0 );
    }
};

template <class ListType, class TestListType, class AoSoAType>
void checkFirstNeighborParallelReduceFunctor( const ListType& nlist,
                                              const TestListType& N2_list_copy,
                                              const AoSoAType aosoa,
                                              const bool use_tag )
{
    double num_particle = aosoa.size();
    if ( use_tag )
    {
        Kokkos::RangePolicy<TEST_EXECSPACE, DoubleValueWorkTag> policy(
            0, num_particle );
        checkFirstNeighborParallelReduceFunctor( nlist, N2_list_copy, aosoa,
                                                 policy, 2 );
    }
    else
    {
        Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );
        checkFirstNeighborParallelReduceFunctor( nlist, N2_list_copy, aosoa,
                                                 policy, 1 );
    }
}

template <class ListType, class TestListType, class AoSoAType, class PolicyType>
void checkFirstNeighborParallelReduceFunctor( const ListType& nlist,
                                              const TestListType& N2_list_copy,
                                              const AoSoAType& aosoa,
                                              const PolicyType policy,
                                              const int multiplier )
{
    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto position = Cabana::slice<0>( aosoa );
    using slice_type = typename AoSoAType::template member_slice_type<0>;

    FirstNeighReduceOp<slice_type> serial_sum_functor( position );
    FirstNeighReduceOp<slice_type> team_sum_functor( position );

    // Do the reductions.
    double serial_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, serial_sum_functor, nlist, Cabana::FirstNeighborsTag(),
        Cabana::SerialOpTag(), serial_sum, "test_reduce_serial" );
    double team_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, team_sum_functor, nlist, Cabana::FirstNeighborsTag(),
        Cabana::TeamOpTag(), team_sum, "test_reduce_team" );
    Kokkos::fence();

    checkFirstNeighborParallelReduce( N2_list_copy, aosoa, serial_sum, team_sum,
                                      multiplier, 0, aosoa.size() );
}

template <class PositionSlice>
struct SecondNeighReduceOp
{
    PositionSlice _position;

    SecondNeighReduceOp( const PositionSlice position )
        : _position( position )
    {
    }

    // tagged version that assigns double the value.
    KOKKOS_INLINE_FUNCTION void
    operator()( const DoubleValueWorkTag&, const std::size_t i,
                const std::size_t n, const std::size_t a, double& sum ) const
    {
        sum +=
            ( _position( i, 0 ) + _position( n, 0 ) + _position( a, 0 ) ) * 2;
    }
    KOKKOS_INLINE_FUNCTION void operator()( const std::size_t i,
                                            const std::size_t n,
                                            const std::size_t a,
                                            double& sum ) const
    {
        sum += _position( i, 0 ) + _position( n, 0 ) + _position( a, 0 );
    }
};

template <class ListType, class TestListType, class AoSoAType>
void checkSecondNeighborParallelReduceFunctor( const ListType& nlist,
                                               const TestListType& N2_list_copy,
                                               const AoSoAType aosoa,
                                               const bool use_tag )
{
    double num_particle = aosoa.size();
    if ( use_tag )
    {
        Kokkos::RangePolicy<TEST_EXECSPACE, DoubleValueWorkTag> policy(
            0, num_particle );
        checkSecondNeighborParallelReduceFunctor( nlist, N2_list_copy, aosoa,
                                                  policy, 2 );
    }
    else
    {
        Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, num_particle );
        checkSecondNeighborParallelReduceFunctor( nlist, N2_list_copy, aosoa,
                                                  policy, 1 );
    }
}

template <class ListType, class TestListType, class AoSoAType, class PolicyType>
void checkSecondNeighborParallelReduceFunctor( const ListType& nlist,
                                               const TestListType& N2_list_copy,
                                               const AoSoAType& aosoa,
                                               const PolicyType policy,
                                               const int multiplier )
{
    // Test the list parallel operation by adding a value from each neighbor
    // to the particle and compare to counts.
    auto position = Cabana::slice<0>( aosoa );
    using slice_type = typename AoSoAType::template member_slice_type<0>;

    SecondNeighReduceOp<slice_type> serial_sum_functor( position );
    SecondNeighReduceOp<slice_type> team_sum_functor( position );
    SecondNeighReduceOp<slice_type> vector_sum_functor( position );

    // Do the reductions.
    double serial_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, serial_sum_functor, nlist, Cabana::SecondNeighborsTag(),
        Cabana::SerialOpTag(), serial_sum, "test_reduce_serial" );
    double team_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, team_sum_functor, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamOpTag(), team_sum, "test_reduce_team" );
    double vector_sum = 0;
    Cabana::neighbor_parallel_reduce(
        policy, vector_sum_functor, nlist, Cabana::SecondNeighborsTag(),
        Cabana::TeamVectorOpTag(), vector_sum, "test_reduce_vector" );
    Kokkos::fence();

    checkSecondNeighborParallelReduce( N2_list_copy, aosoa, serial_sum,
                                       team_sum, vector_sum, multiplier );
}

//---------------------------------------------------------------------------//
// Default test settings.

template <std::size_t Dim>
struct NeighborListTestData;

template <>
struct NeighborListTestData<3>
{
    std::size_t num_particle = 300;
    std::size_t begin = 75;
    std::size_t end = 225;
    double test_radius = 2.32;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;

    double cell_size_ratio = 0.5;
    double grid_min[3] = { box_min, box_min, box_min };
    double grid_max[3] = { box_max, box_max, box_max };

    using DataTypes = Cabana::MemberTypes<double[3]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t aosoa;

    TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
        N2_list_copy;

    NeighborListTestData()
    {
        // Create the AoSoA and fill with random particle positions.
        aosoa = AoSoA_t( "random", num_particle );

#ifdef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
        using AoSoA_copy = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
        AoSoA_copy aosoa_copy( "aosoa", num_particle );
        auto positions = Cabana::slice<0>( aosoa_copy );
#else
        auto positions = Cabana::slice<0>( aosoa );
#endif

        Cabana::createParticles( Cabana::InitRandom(), positions,
                                 positions.size(), grid_min, grid_max );

#ifdef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
        Cabana::deep_copy( aosoa, aosoa_copy );
#endif

        // Create a full N^2 neighbor list to check against.
        auto N2_list = computeFullNeighborList<3>( positions, test_radius );
        N2_list_copy = createTestListHostCopy( N2_list );
    }
};

template <>
struct NeighborListTestData<2>
{
    std::size_t num_particle = 300;
    std::size_t begin = 75;
    std::size_t end = 225;
    double test_radius = 2.32;
    double box_min = -5.3 * test_radius;
    double box_max = 4.7 * test_radius;

    double cell_size_ratio = 0.5;
    double grid_min[2] = { box_min, box_min };
    double grid_max[2] = { box_max, box_max };

    using DataTypes = Cabana::MemberTypes<double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t aosoa;

    TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
        N2_list_copy;

    NeighborListTestData()
    {
        // Create the AoSoA and fill with random particle positions.
        aosoa = AoSoA_t( "random", num_particle );

#ifdef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
        using AoSoA_copy = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
        AoSoA_copy aosoa_copy( "aosoa", num_particle );
        auto positions = Cabana::slice<0>( aosoa_copy );
#else
        auto positions = Cabana::slice<0>( aosoa );
#endif

        Cabana::create<2>( Cabana::InitRandom(), positions, positions.size(),
                           grid_min, grid_max );

#ifdef KOKKOS_ENABLE_OPENMPTARGET // FIXME_OPENMPTARGET
        Cabana::deep_copy( aosoa, aosoa_copy );
#endif

        // Create a full N^2 neighbor list to check against.
        auto N2_list = computeFullNeighborList<2>( positions, test_radius );
        N2_list_copy = createTestListHostCopy( N2_list );
    }
};

//---------------------------------------------------------------------------//
// Default ordered test settings.
struct NeighborListTestDataOrdered
{
    std::size_t num_particle;
    std::size_t num_ignore = 100;
    double test_radius;
    double box_min = 0.0;
    double box_max = 5.0;

    double cell_size_ratio = 0.5;
    double grid_min[3] = { box_min, box_min, box_min };
    double grid_max[3] = { box_max, box_max, box_max };

    using DataTypes = Cabana::MemberTypes<double[3]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t aosoa;

    TestNeighborList<typename TEST_EXECSPACE::array_layout, Kokkos::HostSpace>
        N2_list_copy;

    NeighborListTestDataOrdered( const std::size_t particle_x,
                                 const std::size_t m = 3 )
    {
        num_particle = particle_x * particle_x * particle_x;
        double dx = ( grid_max[0] - grid_min[0] ) / particle_x;
        // Use a fixed ratio of cutoff / spacing (and include a floating point
        // tolerance).
        test_radius = m * dx + 1e-7;

        // Create the AoSoA and fill with ordered particle positions.
        aosoa = AoSoA_t( "ordered", num_particle );
        createParticles( particle_x, dx );

        // Create a full N^2 neighbor list to check against.
        auto positions = Cabana::slice<0>( aosoa );
        auto N2_list = computeFullNeighborList<3>( positions, test_radius );
        N2_list_copy = createTestListHostCopy( N2_list );
    }

    void createParticles( const std::size_t particle_x, const double dx )
    {
        auto positions = Cabana::slice<0>( aosoa );

        Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, positions.size() );
        Kokkos::parallel_for(
            "ordered_particles", policy, KOKKOS_LAMBDA( int pid ) {
                int i = pid / ( particle_x * particle_x );
                int j = ( pid / particle_x ) % particle_x;
                int k = pid % particle_x;
                positions( pid, 0 ) = dx / 2 + dx * i;
                positions( pid, 1 ) = dx / 2 + dx * j;
                positions( pid, 2 ) = dx / 2 + dx * k;
            } );
        Kokkos::fence();
    }
};

//---------------------------------------------------------------------------//
} // end namespace Test
