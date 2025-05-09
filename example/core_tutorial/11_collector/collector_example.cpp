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

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

#include <mpi.h>

//---------------------------------------------------------------------------//
// Collector example.
//---------------------------------------------------------------------------//
void collectorExample()
{
    /*
      The collector is a communication plan allowing for the collection of
      data from one uniquely-owned distribution to another uniquely-owned
      distribution. Data collection may be applied to entire AoSoA data
      structures as well as slices.

      In this example we will demonstrate building a collector communication
      plan and collecting data.

      Note: The collector uses MPI for data migration. MPI is initialized
      and finalized in the main function below.

      Note: The collector uses GPU-aware MPI communication. If AoSoA data is
      allocated in GPU memory, this feature will be used automatically.
    */

    std::cout << "Cabana Collector Example\n" << std::endl;

    /*
       Get parameters from the communicator. We will use MPI_COMM_WORLD for
       this example but any MPI communicator may be used.
    */
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    int comm_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    /*
      Declare the AoSoA parameters.
    */
    using DataTypes = Cabana::MemberTypes<int, int>;
    const int VectorLength = 8;
    using MemorySpace = Kokkos::HostSpace;

    /*
       Create the AoSoA.
    */
    int num_tuple = 100;
    Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> aosoa( "A", num_tuple );

    /*
      Create slices with the MPI rank and a local ID so we can follow where the
      data goes. One might consider using a parallel for loop in this case -
      especially when the code being written is for an arbitrary memory space.
     */
    auto slice_ranks = Cabana::slice<0>( aosoa );
    auto slice_ids = Cabana::slice<1>( aosoa );
    for ( int i = 0; i < num_tuple; ++i )
    {
        slice_ranks( i ) = comm_rank;
        slice_ids( i ) = i;
    }

    /*
      Before collecting the data, let's print out the data in the slices
      on one rank.
    */
    if ( comm_rank == 0 )
    {
        std::cout << "BEFORE collection" << std::endl
                  << "(Rank " << comm_rank << ") ";
        for ( std::size_t i = 0; i < slice_ranks.size(); ++i )
            std::cout << slice_ranks( i ) << " ";
        std::cout << std::endl
                  << "(" << slice_ranks.size() << " ranks before collection)"
                  << std::endl
                  << "(Rank " << comm_rank << ") ";
        for ( std::size_t i = 0; i < slice_ids.size(); ++i )
            std::cout << slice_ids( i ) << " ";
        std::cout << std::endl
                  << "(" << slice_ids.size() << " IDs before collection)"
                  << std::endl
                  << std::endl;
    }

    /*
      Build a communication plan where we collect the first 10 elements of
      data from the next highest rank in the communication and collect the
      second 10 elements from the next lowest rank in the communication.
    */
    Kokkos::View<int*, MemorySpace> import_ranks( "import_ranks", 20 );
    Kokkos::View<int*, MemorySpace> import_ids( "import_ids", 20 );

    // First 10 are collected from the next rank. Note that this view will
    // most often be filled within a parallel_for but we do so in serial
    // here for demonstration purposes.
    int previous_rank = ( comm_rank == 0 ) ? comm_size - 1 : comm_rank - 1;
    int next_rank = ( comm_rank == comm_size - 1 ) ? 0 : comm_rank + 1;
    for ( int i = 0; i < 10; ++i )
    {
        import_ranks( i ) = next_rank;
        import_ids( i ) = i;
    }

    // Next 10 elements will be collected from the previous rank.
    for ( int i = 10; i < 20; ++i )
    {
        import_ranks( i ) = previous_rank;
        import_ids( i ) = i;
    }

    /*
      We have two ways to make a collector. In the first case we know which
      ranks we are importing the data from but not the ranks we are sending
      data to. In the second we know the topology of the communication plan
      (i.e. the ranks we send and receive from).

      We know that we will only send/receive from this rank and the
      next/previous rank so use that information in this case because this
      substantially reduces the amount of communication needed to compose the
      communication plan. If this neighbor data were not supplied, extra
      global communication would be needed to generate a list of neighbors.
     */
    std::vector<int> neighbors = { previous_rank, comm_rank, next_rank };
    std::sort( neighbors.begin(), neighbors.end() );
    auto unique_end = std::unique( neighbors.begin(), neighbors.end() );
    neighbors.resize( std::distance( neighbors.begin(), unique_end ) );
    Cabana::Collector<MemorySpace> collector(
        MPI_COMM_WORLD, num_tuple, import_ranks, import_ids, neighbors );

    /*
      There are three choices for applying the collector: 1) Migrating the
      data into a new AoSoA, 2) Migrating the data into an existing AoSoA, 3)
      Migrating using slices. We will go through each next.
     */

    /*
      1) MIGRATING TO A NEW AOSOA

      The following creates a new AoSoA and migrates into the AoSoA.
     */

    // Make a new AoSoA. Note that this has the same data types, vector
    // length, and memory space as the original aosoa.
    //
    // We are importing 20 elements, so the AoSoA should be size 20.
    // Since we know how much data we are importing, we can size this
    // manually, or call collector.totalNumImport().
    Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> imports(
        "imports", collector.totalNumImport() );

    // Do the migration.
    Cabana::migrate( collector, aosoa, imports );

    /*
      2) MIGRATING SLICES

      We can migrate each slice individually as well. This is useful when not
      all data in an AoSoA needs to be moved to a new decomposition.
     */
    auto slice_ranks_dst = Cabana::slice<0>( imports );
    auto slice_ids_dst = Cabana::slice<1>( imports );
    Cabana::migrate( collector, slice_ranks, slice_ranks_dst );
    Cabana::migrate( collector, slice_ids, slice_ids_dst );

    /*
      3) IN-PLACE MIGRATION.

      In many cases a user may want to use the same AoSoA and not manage a
      second temporary copy for the data collection.

      To use the same AoSoA, the size of the AoSoA must be
      num_owned + num_imported. Imported data is placed at the end of
      the AoSoA.

      Note: Any existing slices created from this AoSoA will be invalidated if
      the data structure must be resized to meet the size requirement.
     */
    aosoa.resize( num_tuple + collector.totalNumImport() );
    Cabana::migrate( collector, aosoa );

    /*
      Having migrated the data, let's print out the in-place case on one rank.
      We re-slice because the previous slices are no longer valid.
    */
    slice_ranks = Cabana::slice<0>( aosoa );
    slice_ids = Cabana::slice<1>( aosoa );

    if ( comm_rank == 0 )
    {
        std::cout << "AFTER migration" << std::endl
                  << "(Rank " << comm_rank << ") ";
        for ( std::size_t i = 0; i < slice_ranks.size(); ++i )
            std::cout << slice_ranks( i ) << " ";
        std::cout << std::endl
                  << "(" << slice_ranks.size() << " ranks after migrate)"
                  << std::endl
                  << "(Rank " << comm_rank << ") ";
        for ( std::size_t i = 0; i < slice_ids.size(); ++i )
            std::cout << slice_ids( i ) << " ";
        std::cout << std::endl
                  << "(" << slice_ids.size() << " IDs after migrate)"
                  << std::endl;
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        collectorExample();
    }
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
