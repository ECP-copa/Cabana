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
#include <Cabana_CommunicationPlan.hpp>
#include <Cabana_DeepCopy.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <set>
#include <vector>

namespace Test
{
//---------------------------------------------------------------------------//
class CommPlanTester : public Cabana::CommunicationPlan<TEST_MEMSPACE>
{
  public:
    using memory_space = TEST_MEMSPACE;
    using size_type = typename TEST_MEMSPACE::size_type;

    CommPlanTester( MPI_Comm comm )
        : Cabana::CommunicationPlan<memory_space>( comm )
    {
    }

    template <class ViewType>
    Kokkos::View<size_type*, memory_space>
    createFromExportsAndNeighbors( const ViewType& element_export_ranks,
                                   const std::vector<int>& neighbor_ranks )
    {
        return this->createFromTopology( Cabana::Export(), element_export_ranks,
                                                   neighbor_ranks );
    }

    template <class ViewType>
    Kokkos::View<size_type*, memory_space>
    createFromExports( const ViewType& element_export_ranks )
    {
        return this->createFromNoTopology( Cabana::Export(), element_export_ranks );
    }

    template <class ViewType>
    std::tuple<Kokkos::View<typename ViewType::size_type*,
                            typename ViewType::memory_space>,
               Kokkos::View<int*, typename ViewType::memory_space>,
               Kokkos::View<int*, typename ViewType::memory_space>>
    createFromImportsAndNeighbors( const ViewType& element_import_ranks,
                                   const ViewType& element_import_ids,
                                   const std::vector<int>& neighbor_ranks )
    {
        return this->createFromTopology( Cabana::CommDriver::Import(),
            element_import_ranks, element_import_ids, neighbor_ranks );
    }

    template <class ViewType>
    std::tuple<Kokkos::View<typename ViewType::size_type*,
                            typename ViewType::memory_space>,
               Kokkos::View<int*, typename ViewType::memory_space>,
               Kokkos::View<int*, typename ViewType::memory_space>>
    createFromImports( const ViewType& element_import_ranks,
                       const ViewType& element_import_ids )
    {
        return this->createFromNoTopology( Cabana::CommDriver::Import(), element_import_ranks,
                                            element_import_ids );
    }

    template <class ViewType>
    void createSteering( Kokkos::View<size_type*, memory_space> neighbor_ids,
                         const ViewType& element_export_ranks )
    {
        this->createExportSteering( neighbor_ids, element_export_ranks );
    }

    template <class RankViewType, class IdViewType>
    void createSteering( Kokkos::View<size_type*, memory_space> neighbor_ids,
                         const RankViewType& element_export_ranks,
                         const IdViewType& element_export_ids )
    {
        this->createExportSteering( neighbor_ids, element_export_ranks,
                                    element_export_ids );
    }
};

//---------------------------------------------------------------------------//
void test1( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_tmp( MPI_COMM_WORLD );
    auto comm_plan = comm_tmp;

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Every rank will communicate with itself and send all of its data.
    int num_data = 10;
    Kokkos::View<int*, TEST_MEMSPACE> export_ranks( "export_ranks", num_data );
    Kokkos::deep_copy( export_ranks, my_rank );
    std::vector<int> neighbor_ranks( 1, my_rank );

    // Create the plan.
    using size_type = typename TEST_MEMSPACE::size_type;
    Kokkos::View<size_type*, TEST_MEMSPACE> neighbor_ids;
    if ( use_topology )
        neighbor_ids = comm_plan.createFromExportsAndNeighbors(
            export_ranks, neighbor_ranks );
    else
        neighbor_ids = comm_plan.createFromExports( export_ranks );

    // Check the plan.
    EXPECT_EQ( comm_plan.numNeighbor(), 1 );
    EXPECT_EQ( comm_plan.neighborRank( 0 ), my_rank );
    EXPECT_EQ( comm_plan.numExport( 0 ), num_data );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data );
    EXPECT_EQ( comm_plan.numImport( 0 ), num_data );
    EXPECT_EQ( comm_plan.totalNumImport(), num_data );

    // Create the export steering vector.
    comm_plan.createSteering( neighbor_ids, export_ranks );

    // Check the steering vector. We thread the creation of the steering
    // vector so we don't really know what order it is in - only that it is
    // grouped by the ranks to which we are exporting. In this case just sort
    // the steering vector and make sure all of the ids are there. We can do
    // this because we are only sending to one rank.
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    std::sort( host_steering.data(),
               host_steering.data() + host_steering.size() );
    EXPECT_EQ( host_steering.size(), num_data );
    for ( int n = 0; n < num_data; ++n )
        EXPECT_EQ( n, host_steering( n ) );
}

//---------------------------------------------------------------------------//
void test2( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_plan( MPI_COMM_WORLD );

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

    // Create the plan
    using size_type = typename TEST_MEMSPACE::size_type;
    Kokkos::View<size_type*, TEST_MEMSPACE> neighbor_ids;
    if ( use_topology )
        neighbor_ids = comm_plan.createFromExportsAndNeighbors(
            export_ranks, neighbor_ranks );
    else
        neighbor_ids = comm_plan.createFromExports( export_ranks );

    // Check the plan.
    EXPECT_EQ( comm_plan.numNeighbor(), 1 );
    EXPECT_EQ( comm_plan.neighborRank( 0 ), my_rank );
    EXPECT_EQ( comm_plan.numExport( 0 ), num_data / 2 );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data / 2 );
    EXPECT_EQ( comm_plan.numImport( 0 ), num_data / 2 );
    EXPECT_EQ( comm_plan.totalNumImport(), num_data / 2 );

    // Create the export steering vector.
    Kokkos::View<std::size_t*, Kokkos::HostSpace> export_ids_host(
        "export_ids", export_ranks.size() );
    std::iota( export_ids_host.data(),
               export_ids_host.data() + export_ranks.size(), 0 );
    auto element_export_ids =
        Kokkos::create_mirror_view_and_copy( TEST_MEMSPACE(), export_ids_host );
    comm_plan.createSteering( neighbor_ids, export_ranks, element_export_ids );

    // Check the steering vector.  We thread the creation of the steering
    // vector so we don't really know what order it is in - only that it is
    // grouped by the ranks to which we are exporting. In this case just sort
    // the steering vector and make sure all of the ids are there. We can do
    // this because we are only sending to one rank.
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    std::sort( host_steering.data(),
               host_steering.data() + host_steering.size() );
    EXPECT_EQ( host_steering.size(), num_data / 2 );
    for ( int n = 0; n < num_data / 2; ++n )
        EXPECT_EQ( n * 2, host_steering( n ) );
}

//---------------------------------------------------------------------------//
void test3( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_plan( MPI_COMM_WORLD );

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

    // Create the plan with both export ranks and the topology.
    using size_type = typename TEST_MEMSPACE::size_type;
    Kokkos::View<size_type*, TEST_MEMSPACE> neighbor_ids;
    if ( use_topology )
        neighbor_ids = comm_plan.createFromExportsAndNeighbors(
            export_ranks, neighbor_ranks );
    else
        neighbor_ids = comm_plan.createFromExports( export_ranks );

    // Check the plan.
    EXPECT_EQ( comm_plan.numNeighbor(), 1 );
    EXPECT_EQ( comm_plan.neighborRank( 0 ), inverse_rank );
    EXPECT_EQ( comm_plan.numExport( 0 ), num_data );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data );
    EXPECT_EQ( comm_plan.numImport( 0 ), num_data );
    EXPECT_EQ( comm_plan.totalNumImport(), num_data );

    // Create the export steering vector.`
    comm_plan.createSteering( neighbor_ids, export_ranks );

    // Check the steering vector. We thread the creation of the steering
    // vector so we don't really know what order it is in - only that it is
    // grouped by the ranks to which we are exporting. In this case just sort
    // the steering vector and make sure all of the ids are there. We can do
    // this because we are only sending to one rank.
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    std::sort( host_steering.data(),
               host_steering.data() + host_steering.size() );
    for ( int n = 0; n < num_data; ++n )
        EXPECT_EQ( n, host_steering( n ) );
}

//---------------------------------------------------------------------------//
void test4( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_plan( MPI_COMM_WORLD );

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

    // Create the plan
    using size_type = typename TEST_MEMSPACE::size_type;
    Kokkos::View<size_type*, TEST_MEMSPACE> neighbor_ids;
    if ( use_topology )
        neighbor_ids = comm_plan.createFromExportsAndNeighbors(
            export_ranks, neighbor_ranks );
    else
        neighbor_ids = comm_plan.createFromExports( export_ranks );

    // Check the plan. Note that if we are sending to ourselves (which we are)
    // that then that data is listed as the first neighbor.
    EXPECT_EQ( comm_plan.numNeighbor(), my_size );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data );
    EXPECT_EQ( comm_plan.totalNumImport(), num_data );

    // self send
    EXPECT_EQ( comm_plan.neighborRank( 0 ), my_rank );
    EXPECT_EQ( comm_plan.numExport( 0 ), 2 );
    EXPECT_EQ( comm_plan.numImport( 0 ), 2 );

    // others
    for ( int n = 1; n < my_size; ++n )
    {
        // the algorithm will swap this rank and the first one.
        if ( n == my_rank )
            EXPECT_EQ( comm_plan.neighborRank( n ), 0 );
        else
            EXPECT_EQ( comm_plan.neighborRank( n ), n );

        EXPECT_EQ( comm_plan.numExport( n ), 2 );
        EXPECT_EQ( comm_plan.numImport( n ), 2 );
    }

    // Create the export steering vector.
    comm_plan.createSteering( neighbor_ids, export_ranks );

    // Check the steering vector. The algorithm will pack the ids according to
    // send rank and self sends will appear first.
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    EXPECT_EQ( host_steering.size(), num_data );

    // self sends - we don't know which order the self sends are in but we
    // know they are the first 2.
    if ( my_rank == (int)host_steering( 0 ) )
    {
        EXPECT_EQ( host_steering( 0 ), my_rank );
        EXPECT_EQ( host_steering( 1 ), my_rank + my_size );
    }
    else
    {
        EXPECT_EQ( host_steering( 1 ), my_rank );
        EXPECT_EQ( host_steering( 0 ), my_rank + my_size );
    }

    // others. again, we don't know which order the vector was made in but we
    // do know they are grouped by the rank to which we are sending and we
    // know how those ranks are ordered.
    for ( int n = 1; n < my_size; ++n )
    {
        if ( n == my_rank )
        {
            if ( 0 == host_steering( 2 * n ) )
            {
                EXPECT_EQ( host_steering( 2 * n ), 0 );
                EXPECT_EQ( host_steering( 2 * n + 1 ), my_size );
            }
            else
            {
                EXPECT_EQ( host_steering( 2 * n + 1 ), 0 );
                EXPECT_EQ( host_steering( 2 * n ), my_size );
            }
        }
        else
        {
            if ( n == (int)host_steering( 2 * n ) )
            {
                EXPECT_EQ( host_steering( 2 * n ), n );
                EXPECT_EQ( host_steering( 2 * n + 1 ), n + my_size );
            }
            else
            {
                EXPECT_EQ( host_steering( 2 * n + 1 ), n );
                EXPECT_EQ( host_steering( 2 * n ), n + my_size );
            }
        }
    }
}

//---------------------------------------------------------------------------//
void test5( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_plan( MPI_COMM_WORLD );

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

    // Create the plan
    using size_type = typename TEST_MEMSPACE::size_type;
    Kokkos::View<size_type*, TEST_MEMSPACE> neighbor_ids;
    if ( use_topology )
        neighbor_ids = comm_plan.createFromExportsAndNeighbors(
            export_ranks, neighbor_ranks );
    else
        neighbor_ids = comm_plan.createFromExports( export_ranks );

    // Check the plan. Note that if we are sending to ourselves (which we are)
    // that then that data is listed as the first neighbor.
    EXPECT_EQ( comm_plan.numNeighbor(), my_size );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data / 2 );
    EXPECT_EQ( comm_plan.totalNumImport(), num_data / 2 );

    // self send
    EXPECT_EQ( comm_plan.neighborRank( 0 ), my_rank );
    EXPECT_EQ( comm_plan.numExport( 0 ), 1 );
    EXPECT_EQ( comm_plan.numImport( 0 ), 1 );

    // others
    for ( int n = 1; n < my_size; ++n )
    {
        // the algorithm will swap this rank and the first one.
        if ( n == my_rank )
            EXPECT_EQ( comm_plan.neighborRank( n ), 0 );
        else
            EXPECT_EQ( comm_plan.neighborRank( n ), n );

        EXPECT_EQ( comm_plan.numExport( n ), 1 );
        EXPECT_EQ( comm_plan.numImport( n ), 1 );
    }

    // Create the export steering vector.
    comm_plan.createSteering( neighbor_ids, export_ranks );

    // Check the steering vector. The algorithm will pack the ids according to
    // send rank and self sends will appear first.
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    EXPECT_EQ( host_steering.size(), my_size );

    // self sends
    EXPECT_EQ( host_steering( 0 ), my_rank + my_size );

    // others
    for ( int n = 1; n < my_size; ++n )
    {
        if ( n == my_rank )
            EXPECT_EQ( host_steering( n ), my_size );
        else
            EXPECT_EQ( host_steering( n ), n + my_size );
    }
}

//---------------------------------------------------------------------------//
void test6( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_plan( MPI_COMM_WORLD );

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

    // Create the plan.
    using size_type = typename TEST_MEMSPACE::size_type;
    Kokkos::View<size_type*, TEST_MEMSPACE> neighbor_ids;
    if ( use_topology )
        neighbor_ids = comm_plan.createFromExportsAndNeighbors(
            export_ranks, neighbor_ranks );
    else
        neighbor_ids = comm_plan.createFromExports( export_ranks );

    // Check the plan.
    if ( 0 == my_rank )
    {
        EXPECT_EQ( comm_plan.numNeighbor(), my_size );
        EXPECT_EQ( comm_plan.numImport( 0 ), 1 );
        EXPECT_EQ( comm_plan.totalNumImport(), my_size );
    }
    else
    {
        EXPECT_EQ( comm_plan.numNeighbor(), 1 );
        EXPECT_EQ( comm_plan.neighborRank( 0 ), 0 );
        EXPECT_EQ( comm_plan.numImport( 0 ), 0 );
        EXPECT_EQ( comm_plan.totalNumImport(), 0 );
    }
    EXPECT_EQ( comm_plan.numExport( 0 ), num_data );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data );

    // Create the export steering vector.
    comm_plan.createSteering( neighbor_ids, export_ranks );

    // Check the steering vector. We thread the creation of the steering
    // vector so we don't really know what order it is in - only that it is
    // grouped by the ranks to which we are exporting. In this case just sort
    // the steering vector and make sure all of the ids are there. We can do
    // this because we are only sending to one rank.
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    EXPECT_EQ( host_steering.size(), num_data );
    EXPECT_EQ( 0, host_steering( 0 ) );
}

//---------------------------------------------------------------------------//
void test7( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_plan( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Every rank will communicate with itself and send all of its data. Use a
    // Cabana slice to make sure that it's syntax usage in the communication
    // plan is equivalent to a kokkos view.
    int num_data = 10;
    using member_types = Cabana::MemberTypes<int>;
    Cabana::AoSoA<member_types, TEST_MEMSPACE> aosoa( "aosoa", num_data );
    auto export_ranks = Cabana::slice<0>( aosoa, "export_ranks" );
    Cabana::deep_copy( export_ranks, my_rank );
    std::vector<int> neighbor_ranks( 1, my_rank );

    // Create the plan.
    using size_type = typename TEST_MEMSPACE::size_type;
    Kokkos::View<size_type*, TEST_MEMSPACE> neighbor_ids;
    if ( use_topology )
        neighbor_ids = comm_plan.createFromExportsAndNeighbors(
            export_ranks, neighbor_ranks );
    else
        neighbor_ids = comm_plan.createFromExports( export_ranks );

    // Check the plan.
    EXPECT_EQ( comm_plan.numNeighbor(), 1 );
    EXPECT_EQ( comm_plan.neighborRank( 0 ), my_rank );
    EXPECT_EQ( comm_plan.numExport( 0 ), num_data );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data );
    EXPECT_EQ( comm_plan.numImport( 0 ), num_data );
    EXPECT_EQ( comm_plan.totalNumImport(), num_data );

    // Create the export steering vector.
    comm_plan.createSteering( neighbor_ids, export_ranks );

    // Check the steering vector. We thread the creation of the steering
    // vector so we don't really know what order it is in - only that it is
    // grouped by the ranks to which we are exporting. In this case just sort
    // the steering vector and make sure all of the ids are there. We can do
    // this because we are only sending to one rank.
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    std::sort( host_steering.data(),
               host_steering.data() + host_steering.size() );
    EXPECT_EQ( host_steering.size(), num_data );
    for ( int n = 0; n < num_data; ++n )
        EXPECT_EQ( n, host_steering( n ) );
}

//---------------------------------------------------------------------------//
void testTopology()
{
    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    std::vector<int> neighbor_ranks = { 0, 2, 2,  1,  1,      1,
                                        2, 3, 10, 27, my_rank };
    auto unique_ranks =
        Cabana::getUniqueTopology( MPI_COMM_WORLD, neighbor_ranks );

    // Check this rank is first.
    EXPECT_EQ( my_rank, unique_ranks[0] );

    // Check each rank is unique.
    for ( std::size_t n = 0; n < unique_ranks.size(); ++n )
        for ( std::size_t m = 0; m < unique_ranks.size(); ++m )
            if ( n != m )
            {
                EXPECT_NE( unique_ranks[n], unique_ranks[m] );
            }
}

//---------------------------------------------------------------------------//
void test8( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_plan( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Every rank will import from itself and import all of its data.
    int num_data = 10;
    Kokkos::View<int*, TEST_MEMSPACE> import_ranks( "import_ranks", num_data );
    Kokkos::View<int*, TEST_MEMSPACE> import_ids( "import_ids", num_data );
    Kokkos::deep_copy( import_ranks, my_rank );

    // Fill the import_ids.
    auto fill_func = KOKKOS_LAMBDA( const int i ) { import_ids( i ) = i; };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
    Kokkos::parallel_for( range_policy, fill_func );
    Kokkos::fence();

    std::vector<int> neighbor_ranks( 1, my_rank );

    // Create the plan.
    using size_type = typename TEST_MEMSPACE::size_type;
    std::tuple<Kokkos::View<size_type*, TEST_MEMSPACE>,
               Kokkos::View<int*, TEST_MEMSPACE>,
               Kokkos::View<int*, TEST_MEMSPACE>>
        neighbor_ids_ranks_indices;
    if ( use_topology )
        neighbor_ids_ranks_indices = comm_plan.createFromImportsAndNeighbors(
            import_ranks, import_ids, neighbor_ranks );
    else
        neighbor_ids_ranks_indices =
            comm_plan.createFromImports( import_ranks, import_ids );

    // Check the plan.
    EXPECT_EQ( comm_plan.numNeighbor(), 1 );
    EXPECT_EQ( comm_plan.neighborRank( 0 ), my_rank );
    EXPECT_EQ( comm_plan.numExport( 0 ), num_data );
    EXPECT_EQ( comm_plan.totalNumExport(), num_data );
    EXPECT_EQ( comm_plan.numImport( 0 ), num_data );
    EXPECT_EQ( comm_plan.totalNumImport(), num_data );
    EXPECT_EQ( comm_plan.exportSize(), num_data );

    // Create the export steering vector.
    comm_plan.createSteering( std::get<0>( neighbor_ids_ranks_indices ),
                              std::get<1>( neighbor_ids_ranks_indices ),
                              std::get<2>( neighbor_ids_ranks_indices ) );

    // Check the steering vector. We thread the creation of the steering
    // vector so we don't really know what order it is in - only that it is
    // grouped by the ranks to which we are exporting. In this case just sort
    // the steering vector and make sure all of the ids are there. We can do
    // this because we are only sending to one rank.
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    std::sort( host_steering.data(),
               host_steering.data() + host_steering.size() );
    EXPECT_EQ( host_steering.size(), num_data );
    for ( int n = 0; n < num_data; ++n )
        EXPECT_EQ( n, host_steering( n ) );
}

//---------------------------------------------------------------------------//
void test9( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_plan( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Compute the inverse rank.
    int inverse_rank = my_size - my_rank - 1;

    // Every rank will import all the data from its inverse rank
    int num_data = 10;
    Kokkos::View<int*, TEST_MEMSPACE> import_ranks( "import_ranks", num_data );
    Kokkos::View<int*, TEST_MEMSPACE> import_ids( "import_ids", num_data );
    Kokkos::deep_copy( import_ranks, inverse_rank );

    // Fill the import_ids.
    auto fill_func = KOKKOS_LAMBDA( const int i ) { import_ids( i ) = i; };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
    Kokkos::parallel_for( range_policy, fill_func );
    Kokkos::fence();

    std::vector<int> neighbor_ranks( 1, inverse_rank );

    // Create the plan.
    using size_type = typename TEST_MEMSPACE::size_type;
    std::tuple<Kokkos::View<size_type*, TEST_MEMSPACE>,
               Kokkos::View<int*, TEST_MEMSPACE>,
               Kokkos::View<int*, TEST_MEMSPACE>>
        neighbor_ids_ranks_indices;
    if ( use_topology )
        neighbor_ids_ranks_indices = comm_plan.createFromImportsAndNeighbors(
            import_ranks, import_ids, neighbor_ranks );
    else
        neighbor_ids_ranks_indices =
            comm_plan.createFromImports( import_ranks, import_ids );

    // Check the plan.
    EXPECT_EQ( comm_plan.numNeighbor(), 1 ) << "Rank " << my_rank << "\n";
    EXPECT_EQ( comm_plan.neighborRank( 0 ), inverse_rank )
        << "Rank " << my_rank << "\n";
    EXPECT_EQ( comm_plan.numExport( 0 ), num_data )
        << "Rank " << my_rank << "\n";
    EXPECT_EQ( comm_plan.totalNumExport(), num_data )
        << "Rank " << my_rank << "\n";
    EXPECT_EQ( comm_plan.numImport( 0 ), num_data )
        << "Rank " << my_rank << "\n";
    EXPECT_EQ( comm_plan.totalNumImport(), num_data )
        << "Rank " << my_rank << "\n";

    // Create the export steering vector.
    comm_plan.createSteering( std::get<0>( neighbor_ids_ranks_indices ),
                              std::get<1>( neighbor_ids_ranks_indices ),
                              std::get<2>( neighbor_ids_ranks_indices ) );

    // Check the steering vector. We thread the creation of the steering
    // vector so we don't really know what order it is in - only that it is
    // grouped by the ranks to which we are exporting. In this case just sort
    // the steering vector and make sure all of the ids are there. We can do
    // this because we are only sending to one rank.
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    std::sort( host_steering.data(),
               host_steering.data() + host_steering.size() );
    EXPECT_EQ( host_steering.size(), num_data );
    for ( int n = 0; n < num_data; ++n )
        EXPECT_EQ( n, host_steering( n ) );
}

//---------------------------------------------------------------------------//
void test10( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_plan( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get my size.
    int comm_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    // Every rank will import data from its previous and next ranks
    int num_data = 10;
    Kokkos::View<int*, TEST_MEMSPACE> import_ranks( "import_ranks", num_data );
    Kokkos::View<int*, TEST_MEMSPACE> import_ids( "import_ids", num_data );
    int previous_rank = ( my_rank == 0 ) ? ( comm_size - 1 ) : ( my_rank - 1 );
    int next_rank = ( my_rank == comm_size - 1 ) ? 0 : ( my_rank + 1 );

    // Create subviews
    auto subview_0_4 = Kokkos::subview( import_ranks, std::make_pair( 0, 5 ) );
    auto subview_5_9 = Kokkos::subview( import_ranks, std::make_pair( 5, 10 ) );

    Kokkos::deep_copy( subview_0_4, previous_rank );
    Kokkos::deep_copy( subview_5_9, next_rank );

    // Fill the import_ids.
    auto fill_func = KOKKOS_LAMBDA( const int i ) { import_ids( i ) = i; };
    Kokkos::RangePolicy<TEST_EXECSPACE> range_policy( 0, num_data );
    Kokkos::parallel_for( range_policy, fill_func );
    Kokkos::fence();

    std::vector<int> neighbor_ranks = { previous_rank, next_rank };

    // Create the plan.
    using size_type = typename TEST_MEMSPACE::size_type;
    std::tuple<Kokkos::View<size_type*, TEST_MEMSPACE>,
               Kokkos::View<int*, TEST_MEMSPACE>,
               Kokkos::View<int*, TEST_MEMSPACE>>
        neighbor_ids_ranks_indices;
    if ( use_topology )
        neighbor_ids_ranks_indices = comm_plan.createFromImportsAndNeighbors(
            import_ranks, import_ids, neighbor_ranks );
    else
        neighbor_ids_ranks_indices =
            comm_plan.createFromImports( import_ranks, import_ids );

    // Check the plan.
    if ( comm_size < 3 )
        EXPECT_EQ( comm_plan.numNeighbor(), 1 ) << "Rank " << my_rank << "\n";
    else
        EXPECT_EQ( comm_plan.numNeighbor(), 2 ) << "Rank " << my_rank << "\n";

    // neighborRank built differently depending on rank. Check that all ranks
    // are present
    std::set<int> expected_neighbors = { previous_rank, next_rank };
    std::set<int> actual_neighbors;
    for ( int i = 0; i < comm_plan.numNeighbor(); i++ )
        actual_neighbors.insert( comm_plan.neighborRank( i ) );

    EXPECT_EQ( expected_neighbors, actual_neighbors )
        << "Rank " << my_rank << "\n";

    // If comm_size is < 3, we export num_data either ourselves
    // or one other rank instead of two other ranks.
    if ( comm_size < 3 )
    {
        EXPECT_EQ( comm_plan.numExport( 0 ), num_data )
            << "Rank " << my_rank << "\n";
        EXPECT_EQ( comm_plan.numImport( 0 ), num_data )
            << "Rank " << my_rank << "\n";
    }
    else
    {
        EXPECT_EQ( comm_plan.numExport( 0 ), num_data / 2 )
            << "Rank " << my_rank << "\n";
        EXPECT_EQ( comm_plan.numExport( 1 ), num_data / 2 )
            << "Rank " << my_rank << "\n";
        EXPECT_EQ( comm_plan.numImport( 0 ), num_data / 2 )
            << "Rank " << my_rank << "\n";
        EXPECT_EQ( comm_plan.numImport( 1 ), num_data / 2 )
            << "Rank " << my_rank << "\n";
    }

    EXPECT_EQ( comm_plan.totalNumExport(), num_data )
        << "Rank " << my_rank << "\n";
    EXPECT_EQ( comm_plan.totalNumImport(), num_data )
        << "Rank " << my_rank << "\n";

    // Create the export steering vector.
    comm_plan.createSteering( std::get<0>( neighbor_ids_ranks_indices ),
                              std::get<1>( neighbor_ids_ranks_indices ),
                              std::get<2>( neighbor_ids_ranks_indices ) );

    // Check the steering vector. We thread the creation of the steering
    // vector so we don't really know what order it is in - only that it is
    // grouped by the ranks to which we are exporting. In this case just sort
    // the steering vector and make sure all of the ids are there. We can do
    // this because we are sending all ids between the previous/next rank
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    std::sort( host_steering.data(),
               host_steering.data() + host_steering.size() );
    EXPECT_EQ( host_steering.size(), num_data );
    for ( int n = 0; n < num_data; ++n )
        EXPECT_EQ( n, host_steering( n ) );
}

//---------------------------------------------------------------------------//
void test11( const bool use_topology )
{
    // Make a communication plan.
    CommPlanTester comm_plan( MPI_COMM_WORLD );

    // Get my rank.
    int my_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );

    // Get the comm size.
    int my_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &my_size );

    // Rank 0 imports one element from each rank
    // Every has one element
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
        Kokkos::resize( import_ranks, 0 );
        Kokkos::resize( import_ids, 0 );
    }

    // Create the plan.
    using size_type = typename TEST_MEMSPACE::size_type;
    std::tuple<Kokkos::View<size_type*, TEST_MEMSPACE>,
               Kokkos::View<int*, TEST_MEMSPACE>,
               Kokkos::View<int*, TEST_MEMSPACE>>
        neighbor_ids_ranks_indices;
    if ( use_topology )
        neighbor_ids_ranks_indices = comm_plan.createFromImportsAndNeighbors(
            import_ranks, import_ids, neighbor_ranks );
    else
        neighbor_ids_ranks_indices =
            comm_plan.createFromImports( import_ranks, import_ids );

    // Check the plan.
    if ( 0 == my_rank )
    {
        EXPECT_EQ( comm_plan.numNeighbor(), my_size )
            << "Rank " << my_rank << "\n";
        EXPECT_EQ( comm_plan.numImport( 0 ), 1 ) << "Rank " << my_rank << "\n";
        EXPECT_EQ( comm_plan.totalNumImport(), my_size )
            << "Rank " << my_rank << "\n";
    }
    else
    {
        EXPECT_EQ( comm_plan.numNeighbor(), 1 ) << "Rank " << my_rank << "\n";
        EXPECT_EQ( comm_plan.neighborRank( 0 ), 0 )
            << "Rank " << my_rank << "\n";
        EXPECT_EQ( comm_plan.numImport( 0 ), 0 ) << "Rank " << my_rank << "\n";
        EXPECT_EQ( comm_plan.totalNumImport(), 0 )
            << "Rank " << my_rank << "\n";
    }
    EXPECT_EQ( comm_plan.numExport( 0 ), num_data )
        << "Rank " << my_rank << "\n";
    EXPECT_EQ( comm_plan.totalNumExport(), num_data )
        << "Rank " << my_rank << "\n";

    // Create the export steering vector.
    comm_plan.createSteering( std::get<0>( neighbor_ids_ranks_indices ),
                              std::get<1>( neighbor_ids_ranks_indices ),
                              std::get<2>( neighbor_ids_ranks_indices ) );

    // Check the steering vector. We thread the creation of the steering
    // vector so we don't really know what order it is in - only that it is
    // grouped by the ranks to which we are exporting. In this case just sort
    // the steering vector and make sure all of the ids are there. We can do
    // this because we are only sending to one rank.
    auto steering = comm_plan.getExportSteering();
    auto host_steering =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), steering );
    EXPECT_EQ( host_steering.size(), num_data ) << "Rank " << my_rank << "\n";
    EXPECT_EQ( 0, host_steering( 0 ) ) << "Rank " << my_rank << "\n";
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

// Export tests
TEST( CommPlan, Test1 ) { test1( true ); }

TEST( CommPlan, Test2 ) { test2( true ); }

TEST( CommPlan, Test3 ) { test3( true ); }

TEST( CommPlan, Test4 ) { test4( true ); }

TEST( CommPlan, Test5 ) { test5( true ); }

TEST( CommPlan, Test6 ) { test6( true ); }

TEST( CommPlan, Test7 ) { test7( true ); }

TEST( CommPlan, Test1NoTopo ) { test1( false ); }

TEST( CommPlan, Test2NoTopo ) { test2( false ); }

TEST( CommPlan, Test3NoTopo ) { test3( false ); }

TEST( CommPlan, Test4NoTopo ) { test4( false ); }

TEST( CommPlan, Test5NoTopo ) { test5( false ); }

TEST( CommPlan, Test6NoTopo ) { test6( false ); }

TEST( CommPlan, Test7NoTopo ) { test7( false ); }

TEST( CommPlan, TestTopology ) { testTopology(); }

// Import tests
TEST( CommPlan, Test8 ) { test8( true ); }

TEST( CommPlan, Test9 ) { test9( true ); }

TEST( CommPlan, Test10 ) { test10( true ); }

TEST( CommPlan, Test11 ) { test11( true ); }

TEST( CommPlan, Test8NoTopo ) { test8( false ); }

TEST( CommPlan, Test9NoTopo ) { test9( false ); }

TEST( CommPlan, Test10NoTopo ) { test10( false ); }

TEST( CommPlan, Test11NoTopo ) { test11( false ); }

//---------------------------------------------------------------------------//

} // end namespace Test
