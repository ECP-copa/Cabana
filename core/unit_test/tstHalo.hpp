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
#include <Cabana_Halo.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <memory>
#include <vector>

namespace Test
{

struct UniqueTestTag
{
    int num_local;
    int num_send;
    int num_recv;
    UniqueTestTag()
    {
        int my_size = -1;
        MPI_Comm_size( MPI_COMM_WORLD, &my_size );
        num_local = 2 * my_size;
        num_send = my_size;
        num_recv = my_size;
    }
};
struct AllTestTag
{
    int num_local = 1;
    int num_send;
    int num_recv;
    AllTestTag()
    {
        int my_size = -1;
        MPI_Comm_size( MPI_COMM_WORLD, &my_size );
        num_send = my_size;
        num_recv = my_size;
    }
};

template <class BuildType, class CommType>
struct HaloData
{
    // Create an AoSoA of local data with space allocated for local data.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    using AoSoA_Host_t = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    AoSoA_t aosoa;

    HaloData( Cabana::Halo<TEST_MEMSPACE, BuildType, CommType> halo )
    {
        aosoa = AoSoA_t( "data", halo.numLocal() + halo.numGhost() );
    }

    AoSoA_t createData( const int my_rank, const int num_local )
    {
        auto slice_int = Cabana::slice<0>( aosoa );
        auto slice_dbl = Cabana::slice<1>( aosoa );

        // Fill the local data.
        auto fill_func = KOKKOS_LAMBDA( const int i )
        {
            slice_int( i ) = my_rank + 1;
            slice_dbl( i, 0 ) = my_rank + 1;
            slice_dbl( i, 1 ) = my_rank + 1.5;
        };
        Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_local );
        Kokkos::parallel_for( range_policy, fill_func );
        Kokkos::fence();
        return aosoa;
    }

    AoSoA_Host_t copyToHost()
    {
        // Deep copy to check after gather/scatter.
        AoSoA_Host_t aosoa_host( "data_host", aosoa.size() );
        Cabana::deep_copy( aosoa_host, aosoa );
        return aosoa_host;
    }
};

template <class CommType, class BuildType>
auto createHalo( UniqueTestTag, CommType, BuildType, const int use_topology,
                 const int my_size, const int num_local )
{
    // Export version:
    // Every rank will send ghosts to all other ranks. Send one element to
    // each rank including yourself. Interleave the sends. The resulting
    // communication plan has ghosts that have one unique destination.

    // Import version:
    // Every rank will import ghosts from all other ranks. Import one element
    // from each rank including yourself. Interleave the imports. The resulting
    // communication plan has ghosts that have one unique destination.
    std::shared_ptr<Cabana::Halo<TEST_MEMSPACE, BuildType, CommType>> halo;

    Kokkos::View<int*, Kokkos::HostSpace> ranks_host( "ranks", my_size );
    Kokkos::View<std::size_t*, Kokkos::HostSpace> ids_host( "ids", my_size );
    std::vector<int> neighbors( my_size );

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    for ( int n = 0; n < my_size; ++n )
    {
        neighbors[n] = n;
        ranks_host( n ) = n;
        ids_host( n ) = 2 * n + 1;
    }
    auto export_ranks =
        Kokkos::create_mirror_view_and_copy( TEST_MEMSPACE(), ranks_host );
    auto export_ids =
        Kokkos::create_mirror_view_and_copy( TEST_MEMSPACE(), ids_host );

    // Create the plan.
    if ( use_topology )
        halo =
            std::make_shared<Cabana::Halo<TEST_MEMSPACE, BuildType, CommType>>(
                MPI_COMM_WORLD, num_local, export_ids, export_ranks,
                neighbors );
    else
        halo =
            std::make_shared<Cabana::Halo<TEST_MEMSPACE, BuildType, CommType>>(
                MPI_COMM_WORLD, num_local, export_ids, export_ranks );

    return halo;
}

template <class CommType, class BuildType>
auto createHalo( AllTestTag, CommType, BuildType, const int use_topology,
                 const int my_size, const int num_local )
{
    // Export version:
    // Every rank will send its single data point as ghosts to all other
    // ranks. This will create collisions in the scatter as every rank will
    // have data for this rank in the summation.

    // Import version:
    // Every rank will import a single data point as a ghost from all other
    // ranks. This will create collisions in the scatter as every rank will
    // have data for this rank in the summation.
    std::shared_ptr<Cabana::Halo<TEST_MEMSPACE, BuildType, CommType>> halo;

    Kokkos::View<int*, Kokkos::HostSpace> ranks_host( "ranks", my_size );
    Kokkos::View<std::size_t*, TEST_MEMSPACE> ids( "ids", my_size );
    Kokkos::deep_copy( ids, 0 );
    std::vector<int> neighbors( my_size );
    for ( int n = 0; n < my_size; ++n )
    {
        neighbors[n] = n;
        ranks_host( n ) = n;
    }
    auto export_ranks =
        Kokkos::create_mirror_view_and_copy( TEST_MEMSPACE(), ranks_host );

    // Create the plan.
    if ( use_topology )
        halo =
            std::make_shared<Cabana::Halo<TEST_MEMSPACE, BuildType, CommType>>(
                MPI_COMM_WORLD, num_local, ids, export_ranks, neighbors );
    else
        halo =
            std::make_shared<Cabana::Halo<TEST_MEMSPACE, BuildType, CommType>>(
                MPI_COMM_WORLD, num_local, ids, export_ranks );

    return halo;
}

template <class AoSoAType>
void checkGatherAoSoA( UniqueTestTag, AoSoAType data_host, const int my_size,
                       const int my_rank, const int num_local )
{
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );

    // check that the local data didn't change.
    for ( int i = 0; i < my_size; ++i )
    {
        EXPECT_EQ( slice_int_host( 2 * i ), my_rank + 1 );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i, 0 ), my_rank + 1 );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i, 1 ), my_rank + 1.5 );

        EXPECT_EQ( slice_int_host( 2 * i + 1 ), my_rank + 1 );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i + 1, 0 ), my_rank + 1 );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i + 1, 1 ), my_rank + 1.5 );
    }

    // Check that we got one element from everyone.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host( i ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), my_rank + 1.5 );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host( i ), 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), 1.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host( i ), send_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), send_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), send_rank + 1.5 );
        }
    }
}

template <class BuildType, class AoSoAType>
void checkScatter( UniqueTestTag, AoSoAType data_host, const int my_size,
                   const int my_rank, const int num_local )
{
    static_assert( std::is_same_v<BuildType, Cabana::Export> ||
                       std::is_same_v<BuildType, Cabana::Import>,
                   "Cabana::Test::tstHalo::checkScatter: BuildType must be "
                   "either Cabana::Export or Cabana::Import." );

    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );

    if constexpr ( std::is_same_v<BuildType, Cabana::Export> )
    {
        // Export version. Check that the local data was updated. Every ghost
        // had a unique destination so the result should be doubled for those
        // elements that were ghosted.
        for ( int i = 0; i < my_size; ++i )
        {
            EXPECT_EQ( slice_int_host( 2 * i ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i, 0 ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i, 1 ), my_rank + 1.5 );

            EXPECT_EQ( slice_int_host( 2 * i + 1 ), 2 * ( my_rank + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i + 1, 0 ),
                              2 * ( my_rank + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i + 1, 1 ),
                              2 * ( my_rank + 1.5 ) );
        }
    }
    if constexpr ( std::is_same_v<BuildType, Cabana::Import> )
    {
        // Import test. Only one data piece should be updated with id my_rank *
        // 2 + 1
        for ( int i = 0; i < num_local; ++i )
        {
            if ( i != my_rank * 2 + 1 )
            {
                EXPECT_EQ( slice_int_host( i ), my_rank + 1 );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), my_rank + 1 );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), my_rank + 1.5 );
            }
            else
            {
                // This is the updated id, with value original value + (my_size
                // * original value).
                EXPECT_EQ( slice_int_host( i ),
                           ( my_rank + 1 ) * ( my_size + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ),
                                  ( my_rank + 1 ) * ( my_size + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                                  ( my_rank + 1.5 ) * ( my_size + 1 ) );
            }
        }
    }

    // Check that the ghost data didn't change.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host( i ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), my_rank + 1.5 );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host( i ), 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), 1.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host( i ), send_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), send_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), send_rank + 1.5 );
        }
    }
}

template <class BuildType, class AoSoAType>
void checkGatherSlice( UniqueTestTag, AoSoAType data_host, const int my_size,
                       const int my_rank, const int num_local )
{
    static_assert( std::is_same_v<BuildType, Cabana::Export> ||
                       std::is_same_v<BuildType, Cabana::Import>,
                   "Cabana::Test::tstHalo::checkGatherSlice: BuildType must be "
                   "either Cabana::Export or Cabana::Import." );

    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );

    if constexpr ( std::is_same_v<BuildType, Cabana::Export> )
    {
        // Check that the local data remained unchanged.
        for ( int i = 0; i < my_size; ++i )
        {
            EXPECT_EQ( slice_int_host( 2 * i ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i, 0 ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i, 1 ), my_rank + 1.5 );

            EXPECT_EQ( slice_int_host( 2 * i + 1 ), 2 * ( my_rank + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i + 1, 0 ),
                              2 * ( my_rank + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i + 1, 1 ),
                              2 * ( my_rank + 1.5 ) );
        }

        // Check that the ghost data was updated.
        for ( int i = num_local; i < num_local + my_size; ++i )
        {
            // Self sends are first.
            int send_rank = i - num_local;
            if ( send_rank == 0 )
            {
                EXPECT_EQ( slice_int_host( i ), 2 * ( my_rank + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), 2 * ( my_rank + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                                  2 * ( my_rank + 1.5 ) );
            }
            else if ( send_rank == my_rank )
            {
                EXPECT_EQ( slice_int_host( i ), 2 );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), 2 );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), 3 );
            }
            else
            {
                EXPECT_EQ( slice_int_host( i ), 2 * ( send_rank + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ),
                                  2 * ( send_rank + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                                  2 * ( send_rank + 1.5 ) );
            }
        }
    }
    if constexpr ( std::is_same_v<BuildType, Cabana::Import> )
    {
        // Import tests. Check that the local data remained unchanged.
        for ( int i = 0; i < num_local; ++i )
        {
            if ( i != my_rank * 2 + 1 )
            {
                EXPECT_EQ( slice_int_host( i ), my_rank + 1 );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), my_rank + 1 );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), my_rank + 1.5 );
            }
            else
            {
                // This is the updated id, with value original value + (my_size
                // * original value).
                EXPECT_EQ( slice_int_host( i ),
                           ( my_rank + 1 ) * ( my_size + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ),
                                  ( my_rank + 1 ) * ( my_size + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                                  ( my_rank + 1.5 ) * ( my_size + 1 ) );
            }
        }

        // Ghosted data should be multiplied by (my_size + 1)
        // for being gathered from itself plus all other ranks and then being
        // added to the existing value.
        for ( int i = num_local; i < num_local + my_size; ++i )
        {
            // Self sends are first.
            int send_rank = i - num_local;
            if ( send_rank == 0 )
            {
                EXPECT_EQ( slice_int_host( i ),
                           ( my_size + 1 ) * ( my_rank + 1 ) )
                    << "Rank " << my_rank << ", i: " << i << std::endl;
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ),
                                  ( my_size + 1 ) * ( my_rank + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                                  ( my_size + 1 ) * ( my_rank + 1.5 ) );
            }
            else if ( send_rank == my_rank )
            {
                EXPECT_EQ( slice_int_host( i ), ( my_size + 1 ) * 1 );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), ( my_size + 1 ) * 1 );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                                  ( my_size + 1 ) * 1.5 );
            }
            else
            {
                EXPECT_EQ( slice_int_host( i ),
                           ( my_size + 1 ) * ( send_rank + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ),
                                  ( my_size + 1 ) * ( send_rank + 1 ) );
                EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                                  ( my_size + 1 ) * ( send_rank + 1.5 ) );
            }
        }
    }
}

template <class AoSoAType>
void checkGatherAoSoA( AllTestTag, AoSoAType data_host, const int my_size,
                       const int my_rank, const int num_local )
{
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );

    // check that the local data didn't change.
    EXPECT_EQ( slice_int_host( 0 ), my_rank + 1 );
    EXPECT_DOUBLE_EQ( slice_dbl_host( 0, 0 ), my_rank + 1 );
    EXPECT_DOUBLE_EQ( slice_dbl_host( 0, 1 ), my_rank + 1.5 );

    // Check that we got one element from everyone.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host( i ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), my_rank + 1.5 );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host( i ), 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), 1.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host( i ), send_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), send_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), send_rank + 1.5 );
        }
    }
}

template <class BuildType, class AoSoAType>
void checkScatter( AllTestTag, AoSoAType data_host, const int my_size,
                   const int my_rank, const int num_local )
{
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );

    // Check that the local data was updated. Every ghost was sent to all of
    // the ranks so the result should be multiplied by the number of ranks.
    EXPECT_EQ( slice_int_host( 0 ), ( my_size + 1 ) * ( my_rank + 1 ) );
    EXPECT_DOUBLE_EQ( slice_dbl_host( 0, 0 ),
                      ( my_size + 1 ) * ( my_rank + 1 ) );
    EXPECT_DOUBLE_EQ( slice_dbl_host( 0, 1 ),
                      ( my_size + 1 ) * ( my_rank + 1.5 ) );

    // Check that the ghost data didn't change.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host( i ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), my_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), my_rank + 1.5 );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host( i ), 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), 1.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host( i ), send_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), send_rank + 1 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), send_rank + 1.5 );
        }
    }
}

template <class BuildTag, class AoSoAType>
void checkGatherSlice( AllTestTag, AoSoAType data_host, const int my_size,
                       const int my_rank, const int num_local )
{
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );

    // Check that the local data remained unchanged.
    EXPECT_EQ( slice_int_host( 0 ), ( my_size + 1 ) * ( my_rank + 1 ) );
    EXPECT_DOUBLE_EQ( slice_dbl_host( 0, 0 ),
                      ( my_size + 1 ) * ( my_rank + 1 ) );
    EXPECT_DOUBLE_EQ( slice_dbl_host( 0, 1 ),
                      ( my_size + 1 ) * ( my_rank + 1.5 ) );

    // Check that the ghost data was updated.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host( i ), ( my_size + 1 ) * ( my_rank + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ),
                              ( my_size + 1 ) * ( my_rank + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                              ( my_size + 1 ) * ( my_rank + 1.5 ) );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host( i ), ( my_size + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), ( my_size + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), ( my_size + 1 ) * 1.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host( i ),
                       ( my_size + 1 ) * ( send_rank + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ),
                              ( my_size + 1 ) * ( send_rank + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                              ( my_size + 1 ) * ( send_rank + 1.5 ) );
        }
    }
}

template <class CommData>
void checkSizeAndCapacity( CommData comm_data, const int num_send,
                           const int num_recv, const double overalloc )
{
    auto send_size = comm_data.sendSize();
    auto recv_size = comm_data.receiveSize();
    EXPECT_EQ( send_size, num_send );
    EXPECT_EQ( recv_size, num_recv );
    auto send_capacity = comm_data.sendCapacity();
    auto recv_capacity = comm_data.receiveCapacity();
    EXPECT_EQ( send_capacity, num_send * overalloc );
    EXPECT_EQ( recv_capacity, num_recv * overalloc );
}

//---------------------------------------------------------------------------//
// Gather/scatter test.
template <class TestTag, class CommType, class BuildType>
void testHalo( TestTag tag, CommType comm_space, BuildType build_type,
               const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Make a communication plan.
    int num_local = tag.num_local;
    auto halo = createHalo( tag, comm_space, build_type, use_topology, my_size,
                            num_local );

    // Check the plan.
    EXPECT_EQ( halo->numLocal(), num_local );
    EXPECT_EQ( halo->numGhost(), my_size );

    // Create particle data.
    HaloData<BuildType, CommType> halo_data( *halo );
    auto data = halo_data.createData( my_rank, num_local );

    // Gather by AoSoA.
    Cabana::gather( *halo, data );

    // Compare against original host data.
    auto data_host = halo_data.copyToHost();
    checkGatherAoSoA( tag, data_host, my_size, my_rank, num_local );

    // Scatter back the results,
    auto slice_int = Cabana::slice<0>( data );
    auto slice_dbl = Cabana::slice<1>( data );
    Cabana::scatter( *halo, slice_int );
    Cabana::scatter( *halo, slice_dbl );
    Cabana::deep_copy( data_host, data );
    checkScatter<BuildType>( tag, data_host, my_size, my_rank, num_local );

    // Gather again, this time with slices.
    Cabana::gather( *halo, slice_int );
    Cabana::gather( *halo, slice_dbl );
    Cabana::deep_copy( data_host, data );
    checkGatherSlice<BuildType>( tag, data_host, my_size, my_rank, num_local );
}

//---------------------------------------------------------------------------//
// Gather/scatter test with persistent buffers.
template <class CommType, class BuildType, class TestTag>
void testHaloBuffers( TestTag tag, CommType comm_space, BuildType build_type,
                      const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Make a communication plan.
    int num_local = tag.num_local;
    auto halo = createHalo( tag, comm_space, build_type, use_topology, my_size,
                            num_local );

    // Check the plan.
    EXPECT_EQ( halo->numLocal(), num_local );
    EXPECT_EQ( halo->numGhost(), my_size );

    // Create particle data.
    HaloData<BuildType, CommType> halo_data( *halo );
    auto data = halo_data.createData( my_rank, num_local );

    // Create send and receive buffers with an overallocation.
    double overalloc = 3.0; // large value since very little is communicated.
    auto gather = createGather( *halo, data, overalloc );

    // Check sizes and capacities.
    int num_send = tag.num_send;
    int num_recv = tag.num_recv;
    checkSizeAndCapacity( gather, num_send, num_recv, overalloc );

    // Gather by AoSoA using preallocated buffers.
    gather.apply();

    // Compare against original host data.
    auto data_host = halo_data.copyToHost();
    checkGatherAoSoA( tag, data_host, my_size, my_rank, num_local );

    // Scatter back the results, now with preallocated slice buffers.
    auto slice_int = Cabana::slice<0>( data );
    auto slice_dbl = Cabana::slice<1>( data );
    auto scatter_int = createScatter( *halo, slice_int, overalloc );
    auto scatter_dbl = createScatter( *halo, slice_dbl, overalloc );
    scatter_int.apply();
    scatter_dbl.apply();
    Cabana::deep_copy( data_host, data );
    checkScatter<BuildType>( tag, data_host, my_size, my_rank, num_local );
    checkSizeAndCapacity( scatter_int, num_recv, num_send, overalloc );
    checkSizeAndCapacity( scatter_dbl, num_recv, num_send, overalloc );

    // Gather again, this time with slices.
    slice_int = Cabana::slice<0>( data );
    slice_dbl = Cabana::slice<1>( data );
    auto gather_int = createGather( *halo, slice_int, overalloc );
    auto gather_dbl = createGather( *halo, slice_dbl, overalloc );
    gather_int.apply();
    gather_dbl.apply();
    Cabana::deep_copy( data_host, data );
    checkGatherSlice<BuildType>( tag, data_host, my_size, my_rank, num_local );
    checkSizeAndCapacity( gather_int, num_send, num_recv, overalloc );
    checkSizeAndCapacity( gather_dbl, num_send, num_recv, overalloc );

    // Now check the reserve/shrink functionality with AoSoA.
    // This call should do nothing since the overallocation is still taken into
    // account.
    gather.shrinkToFit( true );
    scatter_int.shrinkToFit( true );
    scatter_dbl.shrinkToFit( true );
    checkSizeAndCapacity( gather, num_send, num_recv, overalloc );
    checkSizeAndCapacity( scatter_int, num_recv, num_send, overalloc );
    checkSizeAndCapacity( scatter_dbl, num_recv, num_send, overalloc );

    //  After another shrink (now without any overallocation) sizes should have
    //  changed.
    gather.shrinkToFit();
    scatter_int.shrinkToFit();
    scatter_dbl.shrinkToFit();
    checkSizeAndCapacity( gather, num_send, num_recv, 1.0 );
    checkSizeAndCapacity( scatter_int, num_recv, num_send, 1.0 );
    checkSizeAndCapacity( scatter_dbl, num_recv, num_send, 1.0 );

    // Last, increase the overallocation factor.
    overalloc = 5.0;
    gather_int.reserve( *halo, slice_int, overalloc );
    gather_dbl.reserve( *halo, slice_dbl, overalloc );
    scatter_int.reserve( *halo, slice_int, overalloc );
    scatter_dbl.reserve( *halo, slice_dbl, overalloc );
    checkSizeAndCapacity( gather_int, num_send, num_recv, overalloc );
    checkSizeAndCapacity( gather_dbl, num_send, num_recv, overalloc );
    checkSizeAndCapacity( scatter_int, num_send, num_recv, overalloc );
    checkSizeAndCapacity( scatter_dbl, num_send, num_recv, overalloc );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
using HaloTestTypes = ::testing::Types<std::tuple<Cabana::Mpi, Cabana::Export>,
                                       std::tuple<Cabana::Mpi, Cabana::Import>>;

template <typename T>
class HaloTypedTest : public ::testing::Test
{
  protected:
    using CommType = typename std::tuple_element<0, T>::type;
    using BuildType = typename std::tuple_element<1, T>::type;
};

TYPED_TEST_SUITE_P( HaloTypedTest );

// 'Unique' tests:
// Export version: test without collisions (each ghost is unique)
// Import version: test with no collision in first gather
// Behavior, and consequently tests, differ between export/import build type
TYPED_TEST_P( HaloTypedTest, Unique )
{
    testHalo( UniqueTestTag{}, typename TestFixture::CommType{},
              typename TestFixture::BuildType{}, true );
    testHaloBuffers( UniqueTestTag{}, typename TestFixture::CommType{},
                     typename TestFixture::BuildType{}, true );
}

TYPED_TEST_P( HaloTypedTest, UniqueNoTopo )
{
    testHalo( UniqueTestTag{}, typename TestFixture::CommType{},
              typename TestFixture::BuildType{}, false );
    testHaloBuffers( UniqueTestTag{}, typename TestFixture::CommType{},
                     typename TestFixture::BuildType{}, false );
}

// 'All' tests:
// Export version: test with collisions (each ghost is duplicated on all ranks)
// Import version: test with multiple collisions in first gather
// Behavior is identical between export/import build types because the
// communication is symmetrical. Test logic unchanged between the two build
// types.
TYPED_TEST_P( HaloTypedTest, All )
{
    testHalo( AllTestTag{}, typename TestFixture::CommType{},
              typename TestFixture::BuildType{}, true );
    testHaloBuffers( AllTestTag{}, typename TestFixture::CommType{},
                     typename TestFixture::BuildType{}, false );
}

TYPED_TEST_P( HaloTypedTest, AllNoTopo )
{
    testHalo( AllTestTag{}, typename TestFixture::CommType{},
              typename TestFixture::BuildType{}, false );
    testHaloBuffers( AllTestTag{}, typename TestFixture::CommType{},
                     typename TestFixture::BuildType{}, false );
}

REGISTER_TYPED_TEST_SUITE_P( HaloTypedTest, Unique, UniqueNoTopo, All,
                             AllNoTopo );

// Instantiate the test suite with the type list. Need a trailing comma
// to avoid an error when compiling with clang++
INSTANTIATE_TYPED_TEST_SUITE_P( HaloTests, HaloTypedTest, HaloTestTypes, );
//---------------------------------------------------------------------------//

} // end namespace Test