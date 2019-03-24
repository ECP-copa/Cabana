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

#include <Cabana_Halo.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Parallel.hpp>
#include <Cabana_DeepCopy.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <vector>
#include <memory>

namespace Test
{

//---------------------------------------------------------------------------//
// test without collisions
void test1( const bool use_topology )
{
    // Make a communication plan.
    std::shared_ptr<Cabana::Halo<TEST_MEMSPACE> > halo;

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Every rank will send ghosts to all other ranks. Send one element to
    // each rank including yourself. Interleave the sends. The resulting
    // communication plan has ghosts that have one unique destination.
    int num_local = 2 * my_size;
    Kokkos::View<int*,Kokkos::HostSpace> export_ranks_host(
        "export_ranks", my_size );
    Kokkos::View<std::size_t*,Kokkos::HostSpace> export_ids_host(
        "export_ids", my_size );
    std::vector<int> neighbors( my_size );
    for ( int n = 0; n < my_size; ++n )
    {
        neighbors[n] = n;
        export_ranks_host(n) = n;
        export_ids_host(n) = 2*n + 1;
    }
    auto export_ranks = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), export_ranks_host );
    auto export_ids = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), export_ids_host );

    // Create the plan.
    if ( use_topology )
        halo = std::make_shared<Cabana::Halo<TEST_MEMSPACE> >(
            MPI_COMM_WORLD, num_local, export_ids, export_ranks, neighbors );
    else
        halo = std::make_shared<Cabana::Halo<TEST_MEMSPACE> >(
            MPI_COMM_WORLD, num_local, export_ids, export_ranks );

    // Check the plan.
    EXPECT_EQ( halo->numLocal(), num_local );
    EXPECT_EQ( halo->numGhost(), my_size );

    // Create an AoSoA of local data with space allocated for local data.
    using DataTypes = Cabana::MemberTypes<int,double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    AoSoA_t data( halo->numLocal() + halo->numGhost() );
    auto slice_int = data.slice<0>();
    auto slice_dbl = data.slice<1>();

    // Fill the local data.
    auto fill_func =
        KOKKOS_LAMBDA( const int i )
        {
            slice_int(i) = my_rank + 1;
            slice_dbl(i,0) = my_rank + 1;
            slice_dbl(i,1) = my_rank + 1.5;
        };
    Kokkos::RangePolicy<TEST_EXECSPACE>
        range_policy( 0, num_local );
    Kokkos::parallel_for( range_policy, fill_func );
    Kokkos::fence();

    // Gather by AoSoA.
    Cabana::gather( *halo, data );

    // Check the results of the gather.
    Cabana::AoSoA<DataTypes,Cabana::HostSpace> data_host(
        halo->numLocal() + halo->numGhost() );
    auto slice_int_host = data_host.slice<0>();
    auto slice_dbl_host = data_host.slice<1>();
    Cabana::deep_copy( data_host, data );

    // check that the local data didn't change.
    for ( int i = 0; i < my_size; ++i )
    {
        EXPECT_EQ( slice_int_host(2*i), my_rank + 1 );
        EXPECT_EQ( slice_dbl_host(2*i,0), my_rank + 1 );
        EXPECT_EQ( slice_dbl_host(2*i,1), my_rank + 1.5 );

        EXPECT_EQ( slice_int_host(2*i + 1), my_rank + 1 );
        EXPECT_EQ( slice_dbl_host(2*i + 1,0), my_rank + 1 );
        EXPECT_EQ( slice_dbl_host(2*i + 1,1), my_rank + 1.5 );
    }

    // Check that we got one element from everyone.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host(i), my_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,0), my_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,1), my_rank + 1.5 );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host(i), 1 );
            EXPECT_EQ( slice_dbl_host(i,0), 1 );
            EXPECT_EQ( slice_dbl_host(i,1), 1.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host(i), send_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,0), send_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,1), send_rank + 1.5 );
        }
    }

    // Scatter back the results,
    Cabana::scatter( *halo, slice_int );
    Cabana::scatter( *halo, slice_dbl );
    Cabana::deep_copy( data_host, data );

    // Check that the local data was updated. Every ghost had a unique
    // destination so the result should be doubled for those elements that
    // were ghosted.
    for ( int i = 0; i < my_size; ++i )
    {
        EXPECT_EQ( slice_int_host(2*i), my_rank + 1 );
        EXPECT_EQ( slice_dbl_host(2*i,0), my_rank + 1 );
        EXPECT_EQ( slice_dbl_host(2*i,1), my_rank + 1.5 );

        EXPECT_EQ( slice_int_host(2*i + 1), 2 * (my_rank + 1) );
        EXPECT_EQ( slice_dbl_host(2*i + 1,0), 2 * (my_rank + 1 ) );
        EXPECT_EQ( slice_dbl_host(2*i + 1,1), 2 * (my_rank + 1.5) );
    }

    // Check that the ghost data didn't change.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host(i), my_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,0), my_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,1), my_rank + 1.5 );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host(i), 1 );
            EXPECT_EQ( slice_dbl_host(i,0), 1 );
            EXPECT_EQ( slice_dbl_host(i,1), 1.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host(i), send_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,0), send_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,1), send_rank + 1.5 );
        }
    }

    // Gather again, this time with slices.
    Cabana::gather( *halo, slice_int );
    Cabana::gather( *halo, slice_dbl );
    Cabana::deep_copy( data_host, data );

    // Check that the local data remained unchanged.
    for ( int i = 0; i < my_size; ++i )
    {
        EXPECT_EQ( slice_int_host(2*i), my_rank + 1 );
        EXPECT_EQ( slice_dbl_host(2*i,0), my_rank + 1 );
        EXPECT_EQ( slice_dbl_host(2*i,1), my_rank + 1.5 );

        EXPECT_EQ( slice_int_host(2*i + 1), 2 * (my_rank + 1) );
        EXPECT_EQ( slice_dbl_host(2*i + 1,0), 2 * (my_rank + 1 ) );
        EXPECT_EQ( slice_dbl_host(2*i + 1,1), 2 * (my_rank + 1.5) );
    }

    // Check that the ghost data was updated.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host(i), 2 * (my_rank + 1) );
            EXPECT_EQ( slice_dbl_host(i,0), 2 * (my_rank + 1) );
            EXPECT_EQ( slice_dbl_host(i,1), 2 * (my_rank + 1.5) );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host(i), 2 );
            EXPECT_EQ( slice_dbl_host(i,0), 2 );
            EXPECT_EQ( slice_dbl_host(i,1), 3 );
        }
        else
        {
            EXPECT_EQ( slice_int_host(i), 2 * (send_rank + 1) );
            EXPECT_EQ( slice_dbl_host(i,0), 2 * (send_rank + 1) );
            EXPECT_EQ( slice_dbl_host(i,1), 2 * (send_rank + 1.5) );
        }
    }
}

//---------------------------------------------------------------------------//
// test with collisions
void test2( const bool use_topology )
{
    // Make a communication plan.
    std::shared_ptr<Cabana::Halo<TEST_MEMSPACE> > halo;

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Every rank will send its single data point as ghosts to all other
    // ranks. This will create collisions in the scatter as every rank will
    // have data for this rank in the summation.
    int num_local = 1;
    Kokkos::View<int*,Kokkos::HostSpace> export_ranks_host(
        "export_ranks", my_size );
    Kokkos::View<std::size_t*,TEST_MEMSPACE>
        export_ids( "export_ids", my_size );
    Kokkos::deep_copy( export_ids, 0 );
    std::vector<int> neighbors( my_size );
    for ( int n = 0; n < my_size; ++n )
    {
        neighbors[n] = n;
        export_ranks_host(n) = n;
    }
    auto export_ranks = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), export_ranks_host );

    // Create the plan.
    if ( use_topology )
        halo = std::make_shared<Cabana::Halo<TEST_MEMSPACE> >(
            MPI_COMM_WORLD, num_local, export_ids, export_ranks, neighbors );
    else
        halo = std::make_shared<Cabana::Halo<TEST_MEMSPACE> >(
            MPI_COMM_WORLD, num_local, export_ids, export_ranks );

    // Check the plan.
    EXPECT_EQ( halo->numLocal(), num_local );
    EXPECT_EQ( halo->numGhost(), my_size );

    // Create an AoSoA of local data with space allocated for local data.
    using DataTypes = Cabana::MemberTypes<int,double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes,TEST_MEMSPACE>;
    AoSoA_t data( halo->numLocal() + halo->numGhost() );
    auto slice_int = data.slice<0>();
    auto slice_dbl = data.slice<1>();

    // Fill the local data.
    auto fill_func =
        KOKKOS_LAMBDA( const int i )
        {
            slice_int(i) = my_rank + 1;
            slice_dbl(i,0) = my_rank + 1;
            slice_dbl(i,1) = my_rank + 1.5;
        };
    Kokkos::RangePolicy<TEST_EXECSPACE>
        range_policy( 0, num_local );
    Kokkos::parallel_for( range_policy, fill_func );
    Kokkos::fence();

    // Gather by AoSoA.
    Cabana::gather( *halo, data );

    // Check the results of the gather.
    Cabana::AoSoA<DataTypes,Cabana::HostSpace> data_host(
        halo->numLocal() + halo->numGhost() );
    auto slice_int_host = data_host.slice<0>();
    auto slice_dbl_host = data_host.slice<1>();
    Cabana::deep_copy( data_host, data );

    // check that the local data didn't change.
    EXPECT_EQ( slice_int_host(0), my_rank + 1 );
    EXPECT_EQ( slice_dbl_host(0,0), my_rank + 1 );
    EXPECT_EQ( slice_dbl_host(0,1), my_rank + 1.5 );

    // Check that we got one element from everyone.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host(i), my_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,0), my_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,1), my_rank + 1.5 );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host(i), 1 );
            EXPECT_EQ( slice_dbl_host(i,0), 1 );
            EXPECT_EQ( slice_dbl_host(i,1), 1.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host(i), send_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,0), send_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,1), send_rank + 1.5 );
        }
    }

    // Scatter back the results,
    Cabana::scatter( *halo, slice_int );
    Cabana::scatter( *halo, slice_dbl );
    Cabana::deep_copy( data_host, data );

    // Check that the local data was updated. Every ghost was sent to all of
    // the ranks so the result should be multiplied by the number of ranks.
    EXPECT_EQ( slice_int_host(0), (my_size + 1) * (my_rank + 1) );
    EXPECT_EQ( slice_dbl_host(0,0), (my_size + 1) * (my_rank + 1 ) );
    EXPECT_EQ( slice_dbl_host(0,1), (my_size + 1) * (my_rank + 1.5) );

    // Check that the ghost data didn't change.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host(i), my_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,0), my_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,1), my_rank + 1.5 );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host(i), 1 );
            EXPECT_EQ( slice_dbl_host(i,0), 1 );
            EXPECT_EQ( slice_dbl_host(i,1), 1.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host(i), send_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,0), send_rank + 1 );
            EXPECT_EQ( slice_dbl_host(i,1), send_rank + 1.5 );
        }
    }

    // Gather again, this time with slices.
    Cabana::gather( *halo, slice_int );
    Cabana::gather( *halo, slice_dbl );
    Cabana::deep_copy( data_host, data );

    // Check that the local data remained unchanged.
    EXPECT_EQ( slice_int_host(0), (my_size + 1) * (my_rank + 1) );
    EXPECT_EQ( slice_dbl_host(0,0), (my_size + 1) * (my_rank + 1 ) );
    EXPECT_EQ( slice_dbl_host(0,1), (my_size + 1) * (my_rank + 1.5) );

    // Check that the ghost data was updated.
    for ( int i = num_local; i < num_local + my_size; ++i )
    {
        // Self sends are first.
        int send_rank = i - num_local;
        if ( send_rank == 0 )
        {
            EXPECT_EQ( slice_int_host(i), (my_size + 1) * (my_rank + 1) );
            EXPECT_EQ( slice_dbl_host(i,0), (my_size + 1) * (my_rank + 1) );
            EXPECT_EQ( slice_dbl_host(i,1), (my_size + 1) * (my_rank + 1.5) );
        }
        else if ( send_rank == my_rank )
        {
            EXPECT_EQ( slice_int_host(i), (my_size + 1) );
            EXPECT_EQ( slice_dbl_host(i,0), (my_size + 1) );
            EXPECT_EQ( slice_dbl_host(i,1), (my_size + 1) * 1.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host(i), (my_size + 1) * (send_rank + 1) );
            EXPECT_EQ( slice_dbl_host(i,0), (my_size + 1) * (send_rank + 1) );
            EXPECT_EQ( slice_dbl_host(i,1), (my_size + 1) * (send_rank + 1.5) );
        }
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, halo_test_1 )
{ test1(true); }

TEST( TEST_CATEGORY, halo_test_1_no_topo )
{ test1(false); }

TEST( TEST_CATEGORY, halo_test_2 )
{ test2(true); }

TEST( TEST_CATEGORY, halo_test_2_no_topo )
{ test2(false); }

//---------------------------------------------------------------------------//

} // end namespace Test
