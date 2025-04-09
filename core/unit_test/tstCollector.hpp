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
#include <Cabana_Collector.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace Test
{

//---------------------------------------------------------------------------//
// void test0( const bool use_topology )
// {
//     // Make a communication plan.
//     std::shared_ptr<Cabana::Collector<Kokkos::HostSpace>> collector;

//     // Get my rank.
//     int my_rank = -1;
//     MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

//     // Every rank will communicate with itself and send all of its data.
//     int num_import_requests = 0;
//     if (my_rank == 0) num_import_requests = 6;
//     else if (my_rank == 1) num_import_requests = 5;
//     else if (my_rank == 2) num_import_requests = 5;
//     else if (my_rank == 3) num_import_requests = 5;
//     Kokkos::View<int*, Kokkos::HostSpace> import_ranks( "import_ranks", num_import_requests );
//     Kokkos::View<int*, Kokkos::HostSpace> import_ids( "import_ids", num_import_requests );
//     if (my_rank == 0)
//     {
//         import_ranks(0) = 2;
//         import_ranks(1) = 3;
//         import_ranks(2) = 2;
//         import_ranks(3) = 3;
//         import_ranks(4) = 2;
//         import_ranks(5) = 3;
//         import_ids(0) = 0;
//         import_ids(1) = 1;
//         import_ids(2) = 2;
//         import_ids(3) = 3;
//         import_ids(4) = 4;
//         import_ids(5) = 5;
//     }
//     else if (my_rank == 1)
//     {
//         import_ranks(0) = 0;
//         import_ranks(1) = 2;
//         import_ranks(2) = 0;
//         import_ranks(3) = 2;
//         import_ranks(4) = 2;
//         import_ids(0) = 0;
//         import_ids(1) = 1;
//         import_ids(2) = 2;
//         import_ids(3) = 3;
//         import_ids(4) = 4;
//     }
//     else if (my_rank == 2)
//     {
//         import_ranks(0) = 1;
//         import_ranks(1) = 3;
//         import_ranks(2) = 1;
//         import_ranks(3) = 3;
//         import_ranks(4) = 1;
//         import_ids(0) = 0;
//         import_ids(1) = 1;
//         import_ids(2) = 2;
//         import_ids(3) = 3;
//         import_ids(4) = 4;
//     }
//     else if (my_rank == 3)
//     {
//         import_ranks(0) = 0;
//         import_ranks(1) = 2;
//         import_ranks(2) = 0;
//         import_ranks(3) = 2;
//         import_ranks(4) = 0;
//         import_ids(0) = 0;
//         import_ids(1) = 1;
//         import_ids(2) = 2;
//         import_ids(3) = 3;
//         import_ids(4) = 4;
//     }

//     std::vector<int> neighbor_ranks( 1, my_rank );

//     // Create the plan.
//     if ( false )
//         // collector = std::make_shared<Cabana::Collector<Kokkos::HostSpace>>(
//         //     MPI_COMM_WORLD, import_ranks, import_ids, neighbor_ranks );
//     else
//         collector = std::make_shared<Cabana::Collector<Kokkos::HostSpace>>(
//             MPI_COMM_WORLD, import_ranks, import_ids );

//     // Make some data to migrate.
//     int num_data = 10;
//     using DataTypes = Cabana::MemberTypes<int, double[2]>;
//     using AoSoA_t = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
//     AoSoA_t data_src( "src", num_data );
//     auto slice_int_src = Cabana::slice<0>( data_src );
//     auto slice_dbl_src = Cabana::slice<1>( data_src );

//     // Fill the data.
//     auto fill_func = KOKKOS_LAMBDA( const int i )
//     {
//         slice_int_src( i ) = my_rank + i;
//         slice_dbl_src( i, 0 ) = my_rank + i;
//         slice_dbl_src( i, 1 ) = my_rank + i + 0.5;
//     };
//     Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
//     Kokkos::parallel_for( range_policy, fill_func );
//     Kokkos::fence();

//     // Create a second set of data to which we will migrate.
//     AoSoA_t data_dst( "dst", num_data );
//     auto slice_int_dst = Cabana::slice<0>( data_dst );
//     auto slice_dbl_dst = Cabana::slice<1>( data_dst );

//     // Do the migration
//     Cabana::migrate( *collector, data_src, data_dst );

//     // Check the migration.
//     Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_dst_host( "data_dst_host",
//                                                                 num_data );
//     auto slice_int_dst_host = Cabana::slice<0>( data_dst_host );
//     auto slice_dbl_dst_host = Cabana::slice<1>( data_dst_host );
//     Cabana::deep_copy( data_dst_host, data_dst );
//     auto steering = collector->getExportSteering();
//     auto host_steering =
//         Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
//     for ( int i = 0; i < num_data; ++i )
//     {
//         EXPECT_EQ( slice_int_dst_host( i ), my_rank + host_steering( i ) );
//         EXPECT_DOUBLE_EQ( slice_dbl_dst_host( i, 0 ),
//                             my_rank + host_steering( i ) );
//         EXPECT_DOUBLE_EQ( slice_dbl_dst_host( i, 1 ),
//                             my_rank + host_steering( i ) + 0.5 );
//     }
// }

//---------------------------------------------------------------------------//
void test1( const bool use_topology )
{
    // Make a communication plan.
    std::shared_ptr<Cabana::Collector<TEST_MEMSPACE>> collector;

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Every rank will communicate with itself and send all of its data.
    int num_data = 10;
    Kokkos::View<int*, TEST_MEMSPACE> import_ranks( "export_ranks", num_data );
    Kokkos::View<int*, TEST_MEMSPACE> import_ids( "import_ids", num_data );
    Kokkos::deep_copy( import_ranks, my_rank );
    std::vector<int> neighbor_ranks( 1, my_rank );

    // Fill the import_ids.
    auto fill_func0 = KOKKOS_LAMBDA( const int i )
    {
        import_ids( i ) = i;
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy0( 0, num_data );
    Kokkos::parallel_for( range_policy0, fill_func0 );
    Kokkos::fence();

    // Create the plan.
    if ( use_topology )
    {
        // collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
        //     MPI_COMM_WORLD, export_ranks, neighbor_ranks );
    }
    else
        collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, import_ranks, import_ids );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t data_src( "src", num_data );
    auto slice_int_src = Cabana::slice<0>( data_src );
    auto slice_dbl_src = Cabana::slice<1>( data_src );

    // Fill the data.
    auto fill_func1 = KOKKOS_LAMBDA( const int i )
    {
        slice_int_src( i ) = my_rank + i;
        slice_dbl_src( i, 0 ) = my_rank + i;
        slice_dbl_src( i, 1 ) = my_rank + i + 0.5;
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy1( 0, num_data );
    Kokkos::parallel_for( range_policy1, fill_func1 );
    Kokkos::fence();

    // Create a second set of data to which we will migrate.
    AoSoA_t data_dst( "dst", num_data );
    auto slice_int_dst = Cabana::slice<0>( data_dst );
    auto slice_dbl_dst = Cabana::slice<1>( data_dst );

    // Do the migration
    Cabana::migrate( *collector, data_src, data_dst );

    // Check the migration.
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_dst_host( "data_dst_host",
                                                               num_data );
    auto slice_int_dst_host = Cabana::slice<0>( data_dst_host );
    auto slice_dbl_dst_host = Cabana::slice<1>( data_dst_host );
    Cabana::deep_copy( data_dst_host, data_dst );
    auto steering = collector->getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    for ( int i = 0; i < num_data; ++i )
    {
        ASSERT_EQ( slice_int_dst_host( i ), my_rank + host_steering( i ) ) << "Rank " << my_rank << "\n";
        ASSERT_DOUBLE_EQ( slice_dbl_dst_host( i, 0 ),
                          my_rank + host_steering( i ) ) << "Rank " << my_rank << "\n";
        ASSERT_DOUBLE_EQ( slice_dbl_dst_host( i, 1 ),
                          my_rank + host_steering( i ) + 0.5 ) << "Rank " << my_rank << "\n";
    }
}

// //---------------------------------------------------------------------------//
void test2( const bool use_topology )
{
    // Make a communication plan.
    std::shared_ptr<Cabana::Collector<TEST_MEMSPACE>> collector;

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Every rank will communicate with itself and import every other piece of
    // data.
    int num_data = 10;
    Kokkos::View<int*, Kokkos::HostSpace> import_ids_host( "import_ranks",
                                                             num_data/2 );
    Kokkos::View<int*, Kokkos::HostSpace> import_ranks_host( "import_ranks", num_data/2 );

    for ( int n = 0; n < num_data/2; ++n )
    {
        import_ranks_host( n ) = my_rank;
        import_ids_host( n ) = num_data*2;
    }
    auto import_ranks = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), import_ranks_host );
    auto import_ids = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), import_ids_host );
    // std::vector<int> neighbor_ranks( 1, my_rank );

    // Create the plan
    if ( use_topology )
    {
        // collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
        //     MPI_COMM_WORLD, export_ranks, neighbor_ranks );
    }
    else
        collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, import_ranks, import_ids );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t data( "data", num_data );
    auto slice_int = Cabana::slice<0>( data );
    auto slice_dbl = Cabana::slice<1>( data );

    // Fill the data.
    auto fill_func = KOKKOS_LAMBDA( const int i )
    {
        slice_int( i ) = my_rank + i;
        slice_dbl( i, 0 ) = my_rank + i;
        slice_dbl( i, 1 ) = my_rank + i + 0.5;
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
    Kokkos::parallel_for( range_policy, fill_func );
    Kokkos::fence();

    // Do the migration in-place
    Cabana::migrate( *collector, data );

    // Get host copies of the migrated data.
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_host( "data_host",
                                                           num_data / 2 );
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );
    Cabana::deep_copy( data_host, data );

    // Check the migration. We received less than we sent so this should have
    // resized the aososa.
    auto steering = collector->getExportSteering();
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
    // Make a communication plan.
    std::shared_ptr<Cabana::Collector<TEST_MEMSPACE>> collector;

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Compute the inverse rank.
    int inverse_rank = my_size - my_rank - 1;

    // Every rank will import all the data from its inverse rank.
    int num_data = 10;
    Kokkos::View<int*, TEST_MEMSPACE> import_ranks( "import_ranks", num_data );
    Kokkos::View<int*, TEST_MEMSPACE> import_ids( "import_ids", num_data );
    Kokkos::deep_copy( import_ranks, inverse_rank );

    // Fill the import_ids.
    auto fill_func0 = KOKKOS_LAMBDA( const int i )
    {
        import_ids( i ) = i;
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy0( 0, num_data );
    Kokkos::parallel_for( range_policy0, fill_func0 );
    Kokkos::fence();

    std::vector<int> neighbor_ranks( 1, inverse_rank );

    // Create the plan with both export ranks and the topology.
    if ( use_topology )
        printf("TODO\n");
        // collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
        //     MPI_COMM_WORLD, export_ranks, neighbor_ranks );
    else
        collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, import_ranks, import_ids );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t data_src( "data_src", num_data );
    auto slice_int_src = Cabana::slice<0>( data_src );
    auto slice_dbl_src = Cabana::slice<1>( data_src );

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

    // Create a second set of data to which we will migrate.
    AoSoA_t data_dst( "data_dst", num_data );
    auto slice_int_dst = Cabana::slice<0>( data_dst );
    auto slice_dbl_dst = Cabana::slice<1>( data_dst );

    // Do the migration with slices
    Cabana::migrate( *collector, slice_int_src, slice_int_dst );
    Cabana::migrate( *collector, slice_dbl_src, slice_dbl_dst );

    // Exchange steering vectors with your inverse rank so we know what order
    // they sent us stuff in. We thread the creation of the steering vector so
    // its order is not deterministic.
    auto my_steering = collector->getExportSteering();
    Kokkos::View<std::size_t*, TEST_MEMSPACE> inverse_steering(
        "inv_steering", collector->totalNumImport() );
    int mpi_tag = 1030;
    MPI_Request request;
    MPI_Irecv( inverse_steering.data(), inverse_steering.size(),
               MPI_UNSIGNED_LONG, inverse_rank, mpi_tag, MPI_COMM_WORLD,
               &request );
    MPI_Send( my_steering.data(), my_steering.size(), MPI_UNSIGNED_LONG,
              inverse_rank, mpi_tag, MPI_COMM_WORLD );
    MPI_Status status;
    MPI_Wait( &request, &status );
    auto host_steering = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), inverse_steering );

    // Check the migration.
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_dst_host( "data_dst_host",
                                                               num_data );
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
    // Make a communication plan.
    std::shared_ptr<Cabana::Collector<TEST_MEMSPACE>> collector;

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Every rank will import all data from all other ranks. Interleave the imports.
    int num_data = 2 * my_size;
    Kokkos::View<int*, Kokkos::HostSpace> import_ranks_host( "import_ranks",
                                                             num_data );
    Kokkos::View<int*, Kokkos::HostSpace> import_ids_host( "import_ids", num_data );

    for ( int n = 0; n < num_data; ++n )
    {
        import_ids_host( n ) = num_data % my_size;
    }
    
    std::vector<int> neighbor_ranks( my_size );
    for ( int n = 0; n < my_size; ++n )
    {
        import_ranks_host[n] = n;
        import_ranks_host[n + my_size] = n;
        neighbor_ranks[n] = n;
    }
    auto import_ranks = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), import_ranks_host );
    auto import_ids = Kokkos::create_mirror_view_and_copy(
        TEST_MEMSPACE(), import_ids_host );

    // Create the plan
    if ( use_topology )
        printf("TODO\n");
        // collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
        //     MPI_COMM_WORLD, export_ranks, neighbor_ranks );
    else
        collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, import_ranks, import_ids );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t data_src( "data_src", num_data );
    auto slice_int_src = Cabana::slice<0>( data_src );
    auto slice_dbl_src = Cabana::slice<1>( data_src );

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

    // Create a second set of data to which we will migrate.
    AoSoA_t data_dst( "data_dst", num_data );
    auto slice_int_dst = Cabana::slice<0>( data_dst );
    auto slice_dbl_dst = Cabana::slice<1>( data_dst );

    // Do the migration
    Cabana::migrate( *collector, data_src, data_dst );

    // Check the migration.
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_dst_host( "data_dst_host",
                                                               num_data );
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
    // Make a communication plan.
    std::shared_ptr<Cabana::Collector<TEST_MEMSPACE>> collector;

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get the comm size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Every rank has one element.
    // Rank 0 will import one element from each rank.
    int num_data = 1;
    Kokkos::View<int*, TEST_MEMSPACE> import_ranks( "import_ranks", my_size );
    Kokkos::View<int*, TEST_MEMSPACE> import_ids( "import_ids", my_size );
    Kokkos::deep_copy( import_ranks, 0 );
    Kokkos::deep_copy( import_ids, 0 );
    std::vector<int> neighbor_ranks;
    if ( 0 == my_rank )
    {
        neighbor_ranks.resize( my_size );
        std::iota( neighbor_ranks.begin(), neighbor_ranks.end(), 0 );

        // Fill the import_ids and ranks.
        auto fill_func0 = KOKKOS_LAMBDA( const int i )
        {
            import_ids( i ) = 0;
            import_ranks( i ) = i;
        };
        Kokkos::RangePolicy<TEST_EXECSPACE> range_policy0( 0, my_size );
        Kokkos::parallel_for( range_policy0, fill_func0 );
        Kokkos::fence();
    }
    else
    {
        neighbor_ranks.assign( 1, 0 );

        // No other rank is collecting
        Kokkos::resize(import_ranks, 0);
        Kokkos::resize(import_ids, 0);
    }

    // Create the plan.
    if ( use_topology )
        printf("TODO\n");
        // collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
        //     MPI_COMM_WORLD, import_ranks, neighbor_ranks );
    else
        collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, import_ranks, import_ids );

    // Make some data to migrate.
    using DataTypes = Cabana::MemberTypes<int, double[2]>;
    using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
    AoSoA_t data( "data", num_data );
    auto slice_int = Cabana::slice<0>( data );
    auto slice_dbl = Cabana::slice<1>( data );

    // Fill the data.
    auto fill_func = KOKKOS_LAMBDA( const int i )
    {
        slice_int( i ) = my_rank;
        slice_dbl( i, 0 ) = my_rank;
        slice_dbl( i, 1 ) = my_rank + 0.5;
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
    Kokkos::parallel_for( range_policy, fill_func );
    Kokkos::fence();

    // Do the migration
    Cabana::migrate( *collector, data );

    // Check the change in size.
    if ( 0 == my_rank )
        EXPECT_EQ( data.size(), my_size ) << "Rank " << my_rank << "\n";
    else
        EXPECT_EQ( data.size(), 0 ) << "Rank " << my_rank << "\n";

    // Check the migration.
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_host(
        "data_host", collector->totalNumImport() );
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );
    Cabana::deep_copy( data_host, data );
    auto steering = collector->getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    for ( std::size_t i = 0; i < collector->totalNumImport(); ++i )
    {
        EXPECT_EQ( slice_int_host( i ), collector->neighborRank( i ) ) << "Rank " << my_rank << "\n";
        EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ),
                          collector->neighborRank( i ) ) << "Rank " << my_rank << "\n";
        EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ),
                          collector->neighborRank( i ) + 0.5 ) << "Rank " << my_rank << "\n";
    }
}

//---------------------------------------------------------------------------//
void test6( const bool use_topology )
{
    // Make a communication plan.
    std::shared_ptr<Cabana::Collector<TEST_MEMSPACE>> collector;

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get the comm size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Rank 0 starts with all the data 
    // Each rank imports one element from Rank 0
    int num_data = ( 0 == my_rank ) ? my_size : 0;
    std::vector<int> neighbor_ranks;

    int import_size = ( 0 == my_rank ) ? 0 : 1;
    Kokkos::View<int*, TEST_MEMSPACE> import_ranks( "import_ranks", import_size );
    Kokkos::View<int*, TEST_MEMSPACE> import_ids( "import_ids", import_size );
    Kokkos::deep_copy( import_ranks, 0 );
    Kokkos::deep_copy( import_ids, 0 );

    if ( 0 == my_rank )
    {
        neighbor_ranks.resize( my_size );
        std::iota( neighbor_ranks.begin(), neighbor_ranks.end(), 0 );
    }
    else
    {
        neighbor_ranks.assign( 1, 0 );

        // Fill the import_ids and ranks.
        auto fill_func0 = KOKKOS_LAMBDA( const int i )
        {
            import_ids( i ) = my_rank;
            import_ranks( i ) = 0;
            printf("R%d: import_id(%d): %d\n", my_rank, i, import_ids(i));
        };
        Kokkos::RangePolicy<TEST_EXECSPACE> range_policy0( 0, import_size );
        Kokkos::parallel_for( range_policy0, fill_func0 );
        Kokkos::fence();
    }

    // Create the plan.
    if ( use_topology )
        printf("TODO\n");
        // collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
        //     MPI_COMM_WORLD, import_ranks, neighbor_ranks );
    else
    {
        collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
            MPI_COMM_WORLD, import_ranks, import_ids );
    }

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
        // printf("R%d: i%d: int, d1, d2: %d, %0.1lf, %0.1lf\n", my_rank, i, slice_int(i), slice_dbl(i, 0), slice_dbl(i, 1));
    };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy1( 0, num_data );
    Kokkos::parallel_for( range_policy1, fill_func );
    Kokkos::fence();

    // Do the migration
    Cabana::migrate( *collector, data );

    // Check the change in size and import counts.
    if (my_rank == 0)
    {
        ASSERT_EQ( data.size(), 4 ) << "Rank " << my_rank << "\n";
        ASSERT_EQ( collector->totalNumImport(), 0 ) << "Rank " << my_rank << "\n";
    }
    else
    {
        ASSERT_EQ( data.size(), 1 ) << "Rank " << my_rank << "\n";
        ASSERT_EQ( collector->totalNumImport(), 1 ) << "Rank " << my_rank << "\n";
    }

    // Check the migration.
    Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_host(
        "data_host", data.size() );
    auto slice_int_host = Cabana::slice<0>( data_host );
    auto slice_dbl_host = Cabana::slice<1>( data_host );
    Cabana::deep_copy( data_host, data );
    for (int i = 0; i < data_host.size(); i++)
    {
        printf("Recv: R%d: i%d: int, d1, d2: %d, %0.1lf, %0.1lf\n", my_rank,
            i, slice_int_host(i), slice_dbl_host(i, 0), slice_dbl_host(i, 1));
    }
    if (my_rank == 0) return; // Rank 0 has no data after migration
    ASSERT_EQ( slice_int_host( 0 ), my_rank ) << "Rank " << my_rank << "\n";
    // ASSERT_DOUBLE_EQ( slice_dbl_host( 0, 0 ), my_rank ) << "Rank " << my_rank << "\n";
    // ASSERT_DOUBLE_EQ( slice_dbl_host( 0, 1 ), my_rank + 0.5 ) << "Rank " << my_rank << "\n";
}

// //---------------------------------------------------------------------------//
// void test8( const bool use_topology )
// {
//     // Make a communication plan.
//     std::shared_ptr<Cabana::Collector<TEST_MEMSPACE>> collector;

//     // Get my rank.
//     int my_rank = -1;
//     MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

//     // Get the comm size.
//     int my_size = -1;
//     MPI_Comm_size( MPI_COMM_WORLD, &my_size );

//     // Each rank sends two items. Rank zero sends 1 item to itself and 1 item
//     // to the rank with the id 1 larger. The rest of the ranks send 1 item to
//     // the rank with id 1 smaller and 1 item to the rank with id 1
//     // larger. For problems with 3 or more MPI ranks this creates a situation
//     // where rank 0 receives from rank with id (my_size-1) but does not send
//     // data to that rank.
//     int num_data = 2;
//     Kokkos::View<int*, TEST_MEMSPACE> export_ranks( "export_ranks", num_data );
//     auto fill_ranks = KOKKOS_LAMBDA( const int )
//     {
//         export_ranks( 0 ) = ( my_rank == 0 ) ? 0 : my_rank - 1;
//         export_ranks( 1 ) = ( my_rank == my_size - 1 ) ? 0 : my_rank + 1;
//     };
//     Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, 1 );
//     Kokkos::parallel_for( range_policy, fill_ranks );
//     Kokkos::fence();

//     // Neighbors made unique internally.
//     std::vector<int> neighbor_ranks( 3 );
//     neighbor_ranks[0] = ( my_rank == 0 ) ? my_size - 1 : my_rank - 1;
//     neighbor_ranks[1] = my_rank;
//     neighbor_ranks[2] = ( my_rank == my_size - 1 ) ? 0 : my_rank + 1;

//     // Create the plan.
//     if ( use_topology )
//         collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
//             MPI_COMM_WORLD, export_ranks, neighbor_ranks );
//     else
//         collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
//             MPI_COMM_WORLD, export_ranks );

//     // Make some data to migrate.
//     using DataTypes = Cabana::MemberTypes<int, double[2]>;
//     using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
//     AoSoA_t data( "data", num_data );
//     auto slice_int = Cabana::slice<0>( data );
//     auto slice_dbl = Cabana::slice<1>( data );

//     // Fill the data.
//     auto fill_func = KOKKOS_LAMBDA( const int )
//     {
//         slice_int( 0 ) = export_ranks( 0 );
//         slice_int( 1 ) = export_ranks( 1 );

//         slice_dbl( 0, 0 ) = export_ranks( 0 );
//         slice_dbl( 1, 0 ) = export_ranks( 1 );

//         slice_dbl( 0, 1 ) = export_ranks( 0 ) + 1;
//         slice_dbl( 1, 1 ) = export_ranks( 1 ) + 1;
//     };
//     Kokkos::parallel_for( range_policy, fill_func );
//     Kokkos::fence();

//     // Do the migration
//     Cabana::migrate( *collector, data );

//     // Check the results.
//     Cabana::AoSoA<DataTypes, Kokkos::HostSpace> data_host( "data_host",
//                                                            data.size() );
//     auto slice_int_host = Cabana::slice<0>( data_host );
//     auto slice_dbl_host = Cabana::slice<1>( data_host );
//     Cabana::deep_copy( data_host, data );
//     for ( unsigned i = 0; i < data.size(); ++i )
//     {
//         EXPECT_EQ( slice_int_host( i ), my_rank );
//         EXPECT_DOUBLE_EQ( slice_dbl_host( i, 0 ), my_rank );
//         EXPECT_DOUBLE_EQ( slice_dbl_host( i, 1 ), my_rank + 1 );
//     }
// }

// //---------------------------------------------------------------------------//
// void test9( const bool use_topology )
// {
//     // Make a communication plan.
//     std::shared_ptr<Cabana::Collector<TEST_MEMSPACE>> collector;

//     // Edge case where all particles will be removed - nothing is kept, sent, or
//     // received.
//     int num_data = 2;
//     Kokkos::View<int*, TEST_MEMSPACE> export_ranks( "export_ranks", num_data );
//     auto fill_ranks = KOKKOS_LAMBDA( const int i ) { export_ranks( i ) = -1; };

//     Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
//     Kokkos::parallel_for( range_policy, fill_ranks );
//     Kokkos::fence();

//     // Create the plan.
//     if ( use_topology )
//     {
//         std::vector<int> neighbor_ranks;
//         collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
//             MPI_COMM_WORLD, export_ranks, neighbor_ranks );
//     }
//     else
//     {
//         collector = std::make_shared<Cabana::Collector<TEST_MEMSPACE>>(
//             MPI_COMM_WORLD, export_ranks );
//     }

//     // Make empty data to migrate.
//     using DataTypes = Cabana::MemberTypes<int, double[2]>;
//     using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
//     AoSoA_t data( "data", num_data );
//     auto slice_int = Cabana::slice<0>( data );

//     // Create empty slice to "copy" into.
//     AoSoA_t data_copy( "copy", 0 );
//     auto slice_int_copy = Cabana::slice<0>( data_copy );

//     // Do empty slice migration.
//     Cabana::migrate( *collector, slice_int, slice_int_copy );

//     // Check entries were removed.
//     slice_int_copy = Cabana::slice<0>( data_copy );
//     EXPECT_EQ( slice_int_copy.size(), 0 );

//     // Do empty in-place migration.
//     Cabana::migrate( *collector, data );

//     // Check entries were removed.
//     EXPECT_EQ( data.size(), 0 );
// }

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
// TEST( Collector, Test0 ) { test0( true ); }

// TEST( Collector, Test1 ) { test1( true ); }

// TEST( Collector, Test2 ) { test2( true ); }

// TEST( Collector, Test3 ) { test3( true ); }

// TEST( Collector, Test4 ) { test4( true ); }

// TEST( Collector, Test5 ) { test5( true ); }

// TEST( Collector, Test6 ) { test6( true ); }

// TEST( Collector, Test7 ) { test7( true ); }

// TEST( Collector, Test8 ) { test8( true ); }

// TEST( Collector, Test9 ) { test9( true ); }

// TEST( Collector, Test1NoTopo ) { test1( false ); }

// TEST( Collector, Test2NoTopo ) { test2( false ); }

// TEST( Collector, Test3NoTopo ) { test3( false ); }

// TEST( Collector, Test4NoTopo ) { test4( false ); }

// TEST( Collector, Test5NoTopo ) { test5( false ); }

TEST( Collector, Test6NoTopo ) { test6( false ); }

// TEST( Collector, Test7NoTopo ) { test7( false ); }

// TEST( Collector, Test8NoTopo ) { test8( false ); }

// TEST( Collector, Test9NoTopo ) { test9( false ); }

//---------------------------------------------------------------------------//

} // end namespace Test
