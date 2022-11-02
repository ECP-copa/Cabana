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
#include <Cabana_Distributor.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace Test
{

struct DistributorData
{
    // Create an AoSoA of local data with space allocated for local data.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    using AoSoA_Host_t = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    AoSoA_t aosoa;

    DistributorData( const int num_data )
    {
        aosoa = AoSoA_t( "src", num_data );
    }

    AoSoA_t createSrcData1( const int num_data, const int my_rank )
    {
        auto slice_int_src = Cabana::slice<0>( aosoa );
        auto slice_dbl_src = Cabana::slice<1>( aosoa );

        // Fill the data.
        auto fill_func = KOKKOS_LAMBDA( const int i )
        {
            slice_int_src( i ) = my_rank + i;
            slice_dbl_src( i, 0 ) = my_rank + i;
            slice_dbl_src( i, 1 ) = my_rank + i + 0.5;
        };
        Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
        Kokkos::parallel_for( range_policy, fill_func );
        Kokkos::fence();
        return aosoa;
    }

    AoSoA_t createSrcData2( const int num_data, const int my_rank )
    {
        auto slice_int_src = Cabana::slice<0>( aosoa );
        auto slice_dbl_src = Cabana::slice<1>( aosoa );

        // Fill the data.
        auto fill_func = KOKKOS_LAMBDA( const int i )
        {
            slice_int_src( i ) = my_rank;
            slice_dbl_src( i, 0 ) = my_rank;
            slice_dbl_src( i, 1 ) = my_rank + 0.5;
        };
        Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
        Kokkos::parallel_for( range_policy, fill_func );
        Kokkos::fence();
        return aosoa;
    }

    AoSoA_t createDstData( const int num_data )
    {
        return AoSoA_t( "dst", num_data );
    }

    AoSoA_Host_t createHostData( const int num_data )
    {
        // Create empty host copy.
        AoSoA_Host_t aosoa_host( "data_host", num_data );
        return aosoa_host;
    }
};

template <class ExportRankView>
auto createDistributor( ExportRankView export_ranks,
                        const std::vector<int> neighbor_ranks,
                        const int use_topology )
{
    std::shared_ptr<Cabana::Distributor<TEST_MEMSPACE>> distributor;

    // Create the plan.
    if ( use_topology )
        distributor = std::make_shared<Cabana::Distributor<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, export_ranks, neighbor_ranks );
    else
        distributor = std::make_shared<Cabana::Distributor<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, export_ranks );

    return distributor;
}

//---------------------------------------------------------------------------//
void test1( const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Every rank will communicate with itself and send all of its data.
    int num_data = 10;
    Kokkos::View<int*, TEST_MEMSPACE> export_ranks( "export_ranks", num_data );
    Kokkos::deep_copy( export_ranks, my_rank );
    std::vector<int> neighbor_ranks( 1, my_rank );

    auto distributor =
        createDistributor( export_ranks, neighbor_ranks, use_topology );

    // Make some data to migrate.
    DistributorData dist_data( num_data );
    auto data_src = dist_data.createSrcData1( num_data, my_rank );

    // Create a second set of data to which we will migrate.
    auto data_dst = dist_data.createDstData( num_data );
    auto slice_int_dst = Cabana::slice<0>( data_dst );
    auto slice_dbl_dst = Cabana::slice<1>( data_dst );

    // Do the migration
    Cabana::migrate( *distributor, data_src, data_dst );

    // Check the migration.
    auto data_dst_host = dist_data.createHostData( num_data );
    auto slice_int_dst_host = Cabana::slice<0>( data_dst_host );
    auto slice_dbl_dst_host = Cabana::slice<1>( data_dst_host );
    Cabana::deep_copy( data_dst_host, data_dst );
    auto steering = distributor->getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    for ( int i = 0; i < num_data; ++i )
    {
        EXPECT_EQ( slice_int_dst_host( i ), my_rank + host_steering( i ) );
        EXPECT_DOUBLE_EQ( slice_dbl_dst_host( i, 0 ),
                          my_rank + host_steering( i ) );
        EXPECT_DOUBLE_EQ( slice_dbl_dst_host( i, 1 ),
                          my_rank + host_steering( i ) + 0.5 );
    }
}

//---------------------------------------------------------------------------//
void test2( const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Every rank will communicate with itself and send every other piece of
    // data.
    int num_data = 10;
    Kokkos::View<int*, Kokkos::HostSpace> export_ranks_host( "export_ranks",
                                                             num_data );
    for ( int n = 0; n < num_data; ++n )
        export_ranks_host( n ) = ( 0 == n % 2 ) ? my_rank : -1;
    auto export_ranks = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), export_ranks_host );
    std::vector<int> neighbor_ranks( 1, my_rank );

    auto distributor =
        createDistributor( export_ranks, neighbor_ranks, use_topology );

    // Make some data to migrate.
    DistributorData dist_data( num_data );
    auto data = dist_data.createSrcData1( num_data, my_rank );

    // Do the migration in-place
    Cabana::migrate( *distributor, data );

    // Get host copies of the migrated data.
    auto data_host = dist_data.createHostData( num_data / 2 );
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );
    Cabana::deep_copy( data_host, data );

    // Check the migration. We received less than we sent so this should have
    // resized the aosoa.
    auto steering = distributor->getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    EXPECT_EQ( data.size(), num_data / 2 );
    for ( int i = 0; i < num_data / 2; ++i )
    {
        EXPECT_EQ( slice_int_host( i ), my_rank + host_steering( i ) );
        EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ),
                          my_rank + host_steering( i ) );
        EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                          my_rank + host_steering( i ) + 0.5 );
    }
}

//---------------------------------------------------------------------------//
void test3( const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Compute the inverse rank.
    int inverse_rank = my_size - my_rank - 1;

    // Every rank will communicate with the rank that is its inverse.
    int num_data = 10;
    Kokkos::View<int*, TEST_MEMSPACE> export_ranks( "export_ranks", num_data );
    Kokkos::deep_copy( export_ranks, inverse_rank );
    std::vector<int> neighbor_ranks( 1, inverse_rank );

    auto distributor =
        createDistributor( export_ranks, neighbor_ranks, use_topology );

    // Make some data to migrate.
    DistributorData dist_data( num_data );
    auto data_src = dist_data.createSrcData1( num_data, my_rank );
    auto slice_int_src = Cabana::slice<0>( data_src );
    auto slice_dbl_src = Cabana::slice<1>( data_src );

    // Create a second set of data to which we will migrate.
    auto data_dst = dist_data.createDstData( num_data );
    auto slice_int_dst = Cabana::slice<0>( data_dst );
    auto slice_dbl_dst = Cabana::slice<1>( data_dst );

    // Do the migration with slices
    Cabana::migrate( *distributor, slice_int_src, slice_int_dst );
    Cabana::migrate( *distributor, slice_dbl_src, slice_dbl_dst );

    // Exchange steering vectors with your inverse rank so we know what order
    // they sent us stuff in. We thread the creation of the steering vector so
    // its order is not deterministic.
    auto my_steering = distributor->getExportSteering();
    Kokkos::View<std::size_t*, TEST_MEMSPACE> inverse_steering(
        "inv_steering", distributor->totalNumImport() );
    int mpi_tag = 1030;
    MPI_Request request;
    MPI_Irecv( inverse_steering.data(), inverse_steering.size(),
               MPI_UNSIGNED_LONG_LONG, inverse_rank, mpi_tag, MPI_COMM_WORLD,
               &request );
    MPI_Send( my_steering.data(), my_steering.size(), MPI_UNSIGNED_LONG_LONG,
              inverse_rank, mpi_tag, MPI_COMM_WORLD );
    MPI_Status status;
    MPI_Wait( &request, &status );
    auto host_steering = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), inverse_steering );

    // Check the migration.
    auto data_dst_host = dist_data.createHostData( num_data );
    Cabana::deep_copy( data_dst_host, data_dst );
    auto slice_int_dst_host = Cabana::slice<0>( data_dst_host );
    auto slice_dbl_dst_host = Cabana::slice<1>( data_dst_host );
    for ( int i = 0; i < num_data; ++i )
    {
        EXPECT_EQ( slice_int_dst_host( i ), inverse_rank + host_steering( i ) );
        EXPECT_DOUBLE_EQ( slice_dbl_dst_host( i, 0 ),
                          inverse_rank + host_steering( i ) );
        EXPECT_DOUBLE_EQ( slice_dbl_dst_host( i, 1 ),
                          inverse_rank + host_steering( i ) + 0.5 );
    }
}

//---------------------------------------------------------------------------//
void test4( const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Every rank will communicate with all other ranks. Interleave the sends.
    int num_data = 2 * my_size;
    Kokkos::View<int*, Kokkos::HostSpace> export_ranks_host( "export_ranks",
                                                             num_data );
    std::vector<int> neighbor_ranks( my_size );
    for ( int n = 0; n < my_size; ++n )
    {
        export_ranks_host[n] = n;
        export_ranks_host[n + my_size] = n;
        neighbor_ranks[n] = n;
    }
    auto export_ranks = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), export_ranks_host );

    auto distributor =
        createDistributor( export_ranks, neighbor_ranks, use_topology );

    // Make some data to migrate.
    DistributorData dist_data( num_data );
    auto data_src = dist_data.createSrcData2( num_data, my_rank );
    auto slice_int_src = Cabana::slice<0>( data_src );
    auto slice_dbl_src = Cabana::slice<1>( data_src );

    // Create a second set of data to which we will migrate.
    auto data_dst = dist_data.createDstData( num_data );
    auto slice_int_dst = Cabana::slice<0>( data_dst );
    auto slice_dbl_dst = Cabana::slice<1>( data_dst );

    // Do the migration
    Cabana::migrate( *distributor, data_src, data_dst );

    // Check the migration.
    auto data_dst_host = dist_data.createHostData( num_data );
    auto slice_int_dst_host = Cabana::slice<0>( data_dst_host );
    auto slice_dbl_dst_host = Cabana::slice<1>( data_dst_host );
    Cabana::deep_copy( data_dst_host, data_dst );

    // self sends
    EXPECT_EQ( slice_int_dst_host( 0 ), my_rank );
    EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 0, 0 ), my_rank );
    EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 0, 1 ), my_rank + 0.5 );

    EXPECT_EQ( slice_int_dst_host( 1 ), my_rank );
    EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 1, 0 ), my_rank );
    EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 1, 1 ), my_rank + 0.5 );

    // others
    for ( int i = 1; i < my_size; ++i )
    {
        if ( i == my_rank )
        {
            EXPECT_EQ( slice_int_dst_host( 2 * i ), 0 );
            EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 2 * i, 0 ), 0 );
            EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 2 * i, 1 ), 0.5 );

            EXPECT_EQ( slice_int_dst_host( 2 * i + 1 ), 0 );
            EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 2 * i + 1, 0 ), 0 );
            EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 2 * i + 1, 1 ), 0.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_dst_host( 2 * i ), i );
            EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 2 * i, 0 ), i );
            EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 2 * i, 1 ), i + 0.5 );

            EXPECT_EQ( slice_int_dst_host( 2 * i + 1 ), i );
            EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 2 * i + 1, 0 ), i );
            EXPECT_DOUBLE_EQ( slice_dbl_dst_host( 2 * i + 1, 1 ), i + 0.5 );
        }
    }
}

//---------------------------------------------------------------------------//
void test5( const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Every rank will communicate with all other ranks. Interleave the sends
    // and only send every other value.
    int num_data = 2 * my_size;
    Kokkos::View<int*, Kokkos::HostSpace> export_ranks_host( "export_ranks",
                                                             num_data );
    std::vector<int> neighbor_ranks( my_size );
    for ( int n = 0; n < my_size; ++n )
    {
        export_ranks_host[n] = -1;
        export_ranks_host[n + my_size] = n;
        neighbor_ranks[n] = n;
    }
    auto export_ranks = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), export_ranks_host );

    auto distributor =
        createDistributor( export_ranks, neighbor_ranks, use_topology );

    // Make some data to migrate.
    DistributorData dist_data( num_data );
    auto data_src = dist_data.createSrcData2( num_data, my_rank );
    auto slice_int_src = Cabana::slice<0>( data_src );
    auto slice_dbl_src = Cabana::slice<1>( data_src );

    // Create a second set of data to which we will migrate.
    auto data_dst = dist_data.createDstData( my_size );
    auto slice_int_dst = Cabana::slice<0>( data_dst );
    auto slice_dbl_dst = Cabana::slice<1>( data_dst );

    // Do the migration with slices
    Cabana::migrate( *distributor, slice_int_src, slice_int_dst );
    Cabana::migrate( *distributor, slice_dbl_src, slice_dbl_dst );

    // Check the migration.
    auto data_host = dist_data.createHostData( my_size );
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );
    Cabana::deep_copy( data_host, data_dst );

    // self sends
    EXPECT_EQ( slice_int_host( 0 ), my_rank );
    EXPECT_DOUBLE_EQ( slice_dbl_host( 0, 0 ), my_rank );
    EXPECT_DOUBLE_EQ( slice_dbl_host( 0, 1 ), my_rank + 0.5 );

    // others
    for ( int i = 1; i < my_size; ++i )
    {
        if ( i == my_rank )
        {
            EXPECT_EQ( slice_int_host( i ), 0 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), 0 );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), 0.5 );
        }
        else
        {
            EXPECT_EQ( slice_int_host( i ), i );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), i );
            EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), i + 0.5 );
        }
    }
}

//---------------------------------------------------------------------------//
void test6( const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get the comm size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Every has one element and will send that element to rank 0.
    int num_data = 1;
    Kokkos::View<int*, TEST_MEMSPACE> export_ranks( "export_ranks", num_data );
    Kokkos::deep_copy( export_ranks, 0 );
    std::vector<int> neighbor_ranks;
    if ( 0 == my_rank )
    {
        neighbor_ranks.resize( my_size );
        std::iota( neighbor_ranks.begin(), neighbor_ranks.end(), 0 );
    }
    else
    {
        neighbor_ranks.assign( 1, 0 );
    }

    auto distributor =
        createDistributor( export_ranks, neighbor_ranks, use_topology );

    // Make some data to migrate.
    DistributorData dist_data( num_data );
    auto data = dist_data.createSrcData2( num_data, my_rank );
    auto slice_int = Cabana::slice<0>( data );
    auto slice_dbl = Cabana::slice<1>( data );

    // Do the migration
    Cabana::migrate( *distributor, data );

    // Check the change in size.
    if ( 0 == my_rank )
        EXPECT_EQ( data.size(), my_size );
    else
        EXPECT_EQ( data.size(), 0 );

    // Check the migration.
    auto data_host = dist_data.createHostData( distributor->totalNumImport() );
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );
    Cabana::deep_copy( data_host, data );
    auto steering = distributor->getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    for ( std::size_t i = 0; i < distributor->totalNumImport(); ++i )
    {
        EXPECT_EQ( slice_int_host( i ), distributor->neighborRank( i ) );
        EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ),
                          distributor->neighborRank( i ) );
        EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                          distributor->neighborRank( i ) + 0.5 );
    }
}

//---------------------------------------------------------------------------//
void test7( const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get the comm size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Rank 0 starts with all the data and sends one element to every rank.
    int num_data = ( 0 == my_rank ) ? my_size : 0;
    Kokkos::View<int*, TEST_MEMSPACE> export_ranks( "export_ranks", num_data );
    auto fill_ranks = KOKKOS_LAMBDA( const int i ) { export_ranks( i ) = i; };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
    Kokkos::parallel_for( range_policy, fill_ranks );
    Kokkos::fence();
    std::vector<int> neighbor_ranks;
    if ( 0 == my_rank )
    {
        neighbor_ranks.resize( my_size );
        std::iota( neighbor_ranks.begin(), neighbor_ranks.end(), 0 );
    }
    else
    {
        neighbor_ranks.assign( 1, 0 );
    }

    auto distributor =
        createDistributor( export_ranks, neighbor_ranks, use_topology );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t data( "data", num_data );
    auto slice_int = Cabana::slice<0>( data );
    auto slice_dbl = Cabana::slice<1>( data );

    // Fill the data.
    auto fill_func = KOKKOS_LAMBDA( const int i )
    {
        slice_int( i ) = i;
        slice_dbl( i, 0 ) = i;
        slice_dbl( i, 1 ) = i + 0.5;
    };
    Kokkos::parallel_for( range_policy, fill_func );
    Kokkos::fence();

    // Do the migration
    Cabana::migrate( *distributor, data );

    // Check the change in size.
    EXPECT_EQ( data.size(), 1 );

    // Check the migration.
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_host(
        "data_host", distributor->totalNumImport() );
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );
    Cabana::deep_copy( data_host, data );
    EXPECT_EQ( slice_int_host( 0 ), my_rank );
    EXPECT_DOUBLE_EQ( slice_dbl_host( 0, 0 ), my_rank );
    EXPECT_DOUBLE_EQ( slice_dbl_host( 0, 1 ), my_rank + 0.5 );
}

//---------------------------------------------------------------------------//
void test8( const bool use_topology )
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get the comm size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Each rank sends two items. Rank zero sends 1 item to itself and 1 item
    // to the rank with the id 1 larger. The rest of the ranks send 1 item to
    // the rank with id 1 smaller and 1 item to the rank with id 1
    // larger. For problems with 3 or more MPI ranks this creates a situation
    // where rank 0 receives from rank with id (my_size-1) but does not send
    // data to that rank.
    int num_data = 2;
    Kokkos::View<int*, TEST_MEMSPACE> export_ranks( "export_ranks", num_data );
    auto fill_ranks = KOKKOS_LAMBDA( const int )
    {
        export_ranks( 0 ) = ( my_rank == 0 ) ? 0 : my_rank - 1;
        export_ranks( 1 ) = ( my_rank == my_size - 1 ) ? 0 : my_rank + 1;
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, 1 );
    Kokkos::parallel_for( range_policy, fill_ranks );
    Kokkos::fence();

    // Neighbors made unique internally.
    std::vector<int> neighbor_ranks( 3 );
    neighbor_ranks[0] = ( my_rank == 0 ) ? my_size - 1 : my_rank - 1;
    neighbor_ranks[1] = my_rank;
    neighbor_ranks[2] = ( my_rank == my_size - 1 ) ? 0 : my_rank + 1;

    auto distributor =
        createDistributor( export_ranks, neighbor_ranks, use_topology );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t data( "data", num_data );
    auto slice_int = Cabana::slice<0>( data );
    auto slice_dbl = Cabana::slice<1>( data );

    // Fill the data.
    auto fill_func = KOKKOS_LAMBDA( const int )
    {
        slice_int( 0 ) = export_ranks( 0 );
        slice_int( 1 ) = export_ranks( 1 );

        slice_dbl( 0, 0 ) = export_ranks( 0 );
        slice_dbl( 1, 0 ) = export_ranks( 1 );

        slice_dbl( 0, 1 ) = export_ranks( 0 ) + 1;
        slice_dbl( 1, 1 ) = export_ranks( 1 ) + 1;
    };
    Kokkos::parallel_for( range_policy, fill_func );
    Kokkos::fence();

    // Do the migration
    Cabana::migrate( *distributor, data );

    // Check the results.
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_host( "data_host",
                                                           data.size() );
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );
    Cabana::deep_copy( data_host, data );
    for ( unsigned i = 0; i < data.size(); ++i )
    {
        EXPECT_EQ( slice_int_host( i ), my_rank );
        EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), my_rank );
        EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), my_rank + 1 );
    }
}

//---------------------------------------------------------------------------//
void test9( const bool use_topology )
{
    // Edge case where all particles will be removed - nothing is kept, sent, or
    // received.
    int num_data = 2;
    Kokkos::View<int*, TEST_MEMSPACE> export_ranks( "export_ranks", num_data );
    auto fill_ranks = KOKKOS_LAMBDA( const int i ) { export_ranks( i ) = -1; };

    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
    Kokkos::parallel_for( range_policy, fill_ranks );
    Kokkos::fence();

    std::vector<int> neighbor_ranks;

    auto distributor =
        createDistributor( export_ranks, neighbor_ranks, use_topology );

    // Make empty data to migrate.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t data( "data", num_data );
    auto slice_int = Cabana::slice<0>( data );

    // Create empty slice to "copy" into.
    AoSoA_t data_copy( "copy", 0 );
    auto slice_int_copy = Cabana::slice<0>( data_copy );

    // Do empty slice migration.
    Cabana::migrate( *distributor, slice_int, slice_int_copy );

    // Check entries were removed.
    slice_int_copy = Cabana::slice<0>( data_copy );
    EXPECT_EQ( slice_int_copy.size(), 0 );

    // Do empty in-place migration.
    Cabana::migrate( *distributor, data );

    // Check entries were removed.
    EXPECT_EQ( data.size(), 0 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, distributor_test_1 ) { test1( true ); }

TEST( TEST_CATEGORY, distributor_test_2 ) { test2( true ); }

TEST( TEST_CATEGORY, distributor_test_3 ) { test3( true ); }

TEST( TEST_CATEGORY, distributor_test_4 ) { test4( true ); }

TEST( TEST_CATEGORY, distributor_test_5 ) { test5( true ); }

TEST( TEST_CATEGORY, distributor_test_6 ) { test6( true ); }

TEST( TEST_CATEGORY, distributor_test_7 ) { test7( true ); }

TEST( TEST_CATEGORY, distributor_test_8 ) { test8( true ); }

TEST( TEST_CATEGORY, distributor_test_9 ) { test9( true ); }

TEST( TEST_CATEGORY, distributor_test_1_no_topo ) { test1( false ); }

TEST( TEST_CATEGORY, distributor_test_2_no_topo ) { test2( false ); }

TEST( TEST_CATEGORY, distributor_test_3_no_topo ) { test3( false ); }

TEST( TEST_CATEGORY, distributor_test_4_no_topo ) { test4( false ); }

TEST( TEST_CATEGORY, distributor_test_5_no_topo ) { test5( false ); }

TEST( TEST_CATEGORY, distributor_test_6_no_topo ) { test6( false ); }

TEST( TEST_CATEGORY, distributor_test_7_no_topo ) { test7( false ); }

TEST( TEST_CATEGORY, distributor_test_8_no_topo ) { test8( false ); }

TEST( TEST_CATEGORY, distributor_test_9_no_topo ) { test9( false ); }

//---------------------------------------------------------------------------//

} // end namespace Test
