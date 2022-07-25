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
    UniqueTestTag()
    {
        int my_size = -1;
        MPI_Comm_size( MPI_COMM_WORLD, &my_size );
        num_local = 2 * my_size;
    }
};
struct AllTestTag
{
    int num_local = 1;
};

struct HaloData
{
    // Create an AoSoA of local data with space allocated for local data.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    using AoSoA_Host_t = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    AoSoA_t aosoa;

    HaloData( Cabana::Halo<TEST_MEMSPACE> halo )
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

auto createHalo( UniqueTestTag, const int use_topology, const int my_size,
                 const int num_local )
{
    std::shared_ptr<Cabana::Halo<TEST_MEMSPACE>> halo;

    // Every rank will send ghosts to all other ranks. Send one element to
    // each rank including yourself. Interleave the sends. The resulting
    // communication plan has ghosts that have one unique destination.
    Kokkos::View<int*, Kokkos::HostSpace> export_ranks_host( "export_ranks",
                                                             my_size );
    Kokkos::View<std::size_t*, Kokkos::HostSpace> export_ids_host( "export_ids",
                                                                   my_size );
    std::vector<int> neighbors( my_size );
    for ( int n = 0; n < my_size; ++n )
    {
        neighbors[n] = n;
        export_ranks_host( n ) = n;
        export_ids_host( n ) = 2 * n + 1;
    }
    auto export_ranks = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), export_ranks_host );
    auto export_ids =
        Kokkos::create_mirror_view_and_copy( TEST_MEMSPACE(), export_ids_host );

    // Create the plan.
    if ( use_topology )
        halo = std::make_shared<Cabana::Halo<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, num_local, export_ids, export_ranks, neighbors );
    else
        halo = std::make_shared<Cabana::Halo<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, num_local, export_ids, export_ranks );

    return halo;
}

auto createHalo( AllTestTag, const int use_topology, const int my_size,
                 const int num_local )
{
    std::shared_ptr<Cabana::Halo<TEST_MEMSPACE>> halo;

    // Every rank will send its single data point as ghosts to all other
    // ranks. This will create collisions in the scatter as every rank will
    // have data for this rank in the summation.
    Kokkos::View<int*, Kokkos::HostSpace> export_ranks_host( "export_ranks",
                                                             my_size );
    Kokkos::View<std::size_t*, TEST_MEMSPACE> export_ids( "export_ids",
                                                          my_size );
    Kokkos::deep_copy( export_ids, 0 );
    std::vector<int> neighbors( my_size );
    for ( int n = 0; n < my_size; ++n )
    {
        neighbors[n] = n;
        export_ranks_host( n ) = n;
    }
    auto export_ranks = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), export_ranks_host );

    // Create the plan.
    if ( use_topology )
        halo = std::make_shared<Cabana::Halo<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, num_local, export_ids, export_ranks, neighbors );
    else
        halo = std::make_shared<Cabana::Halo<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, num_local, export_ids, export_ranks );

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

template <class AoSoAType>
void checkScatter( UniqueTestTag, AoSoAType data_host, const int my_size,
                   const int my_rank, const int num_local )
{
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );

    // Check that the local data was updated. Every ghost had a unique
    // destination so the result should be doubled for those elements that
    // were ghosted.
    for ( int i = 0; i < my_size; ++i )
    {
        EXPECT_EQ( slice_int_host( 2 * i ), my_rank + 1 );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i, 0 ), my_rank + 1 );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i, 1 ), my_rank + 1.5 );

        EXPECT_EQ( slice_int_host( 2 * i + 1 ), 2 * ( my_rank + 1 ) );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i + 1, 0 ), 2 * ( my_rank + 1 ) );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i + 1, 1 ),
                          2 * ( my_rank + 1.5 ) );
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

template <class AoSoAType>
void checkGatherSlice( UniqueTestTag, AoSoAType data_host, const int my_size,
                       const int my_rank, const int num_local )
{
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );

    // Check that the local data remained unchanged.
    for ( int i = 0; i < my_size; ++i )
    {
        EXPECT_EQ( slice_int_host( 2 * i ), my_rank + 1 );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i, 0 ), my_rank + 1 );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i, 1 ), my_rank + 1.5 );

        EXPECT_EQ( slice_int_host( 2 * i + 1 ), 2 * ( my_rank + 1 ) );
        EXPECT_DOUBLE_EQ( slice_dbl_host( 2 * i + 1, 0 ), 2 * ( my_rank + 1 ) );
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
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), 2 * ( my_rank + 1.5 ) );
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
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), 2 * ( send_rank + 1 ) );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), 2 * ( send_rank + 1.5 ) );
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

template <class AoSoAType>
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

template <class AoSoAType>
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

//---------------------------------------------------------------------------//
// Gather/scatter test.
template <class TestTag>
void testHalo( TestTag tag, const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Make a communication plan.
    int num_local = tag.num_local;
    auto halo = createHalo( tag, use_topology, my_size, num_local );

    // Check the plan.
    EXPECT_EQ( halo->numLocal(), num_local );
    EXPECT_EQ( halo->numGhost(), my_size );

    // Create particle data.
    HaloData halo_data( *halo );
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
    checkScatter( tag, data_host, my_size, my_rank, num_local );

    // Gather again, this time with slices.
    Cabana::gather( *halo, slice_int );
    Cabana::gather( *halo, slice_dbl );
    Cabana::deep_copy( data_host, data );
    checkGatherSlice( tag, data_host, my_size, my_rank, num_local );
}

//---------------------------------------------------------------------------//
// Gather/scatter test with persistent buffers.
template <class TestTag>
void testHaloBuffers( TestTag tag, const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Make a communication plan.
    int num_local = tag.num_local;
    auto halo = createHalo( tag, use_topology, my_size, num_local );

    // Check the plan.
    EXPECT_EQ( halo->numLocal(), num_local );
    EXPECT_EQ( halo->numGhost(), my_size );

    // Create particle data.
    HaloData halo_data( *halo );
    auto data = halo_data.createData( my_rank, num_local );

    // Create send and receive buffers with an overallocation.
    double overalloc = 1.5;
    auto gather = createGather( *halo, data, overalloc );

    // Gather by AoSoA using preallocated buffers.
    gather.apply( data );

    // Compare against original host data.
    auto data_host = halo_data.copyToHost();
    checkGatherAoSoA( tag, data_host, my_size, my_rank, num_local );

    // Scatter back the results, now with preallocated slice buffers.
    auto slice_int = Cabana::slice<0>( data );
    auto slice_dbl = Cabana::slice<1>( data );
    auto scatter_int = createScatter( *halo, slice_int, overalloc );
    auto scatter_dbl = createScatter( *halo, slice_dbl, overalloc );
    scatter_int.apply( slice_int );
    scatter_dbl.apply( slice_dbl );
    Cabana::deep_copy( data_host, data );
    checkScatter( tag, data_host, my_size, my_rank, num_local );

    // Gather again, this time with slices.
    auto gather_int = createGather( *halo, slice_int, overalloc );
    auto gather_dbl = createGather( *halo, slice_dbl, overalloc );
    gather_int.apply( slice_int );
    gather_dbl.apply( slice_dbl );
    Cabana::deep_copy( data_host, data );
    checkGatherSlice( tag, data_host, my_size, my_rank, num_local );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
// test without collisions (each ghost is unique)
TEST( TEST_CATEGORY, halo_test_unique )
{
    testHalo( UniqueTestTag{}, true );
    testHaloBuffers( UniqueTestTag{}, true );
}

TEST( TEST_CATEGORY, halo_test_unique_no_topo )
{
    testHalo( UniqueTestTag{}, false );
    testHaloBuffers( UniqueTestTag{}, false );
}

// tests with collisions (each ghost is duplicated on all ranks)
TEST( TEST_CATEGORY, halo_test_all )
{
    testHalo( AllTestTag{}, true );
    testHaloBuffers( AllTestTag{}, false );
}

TEST( TEST_CATEGORY, halo_test_all_no_topo )
{
    testHalo( AllTestTag{}, false );
    testHaloBuffers( AllTestTag{}, false );
}

//---------------------------------------------------------------------------//

} // end namespace Test
