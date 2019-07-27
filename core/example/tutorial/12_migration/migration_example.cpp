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

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>

#include <mpi.h>

//---------------------------------------------------------------------------//
// Migration example.
//---------------------------------------------------------------------------//
void migrationExample() {
    /*
      The distributor is a communication plan allowing for the migration of
      data from one uniquely-owned distribution to another uniquely-owned
      distribution. Data migration may be applied to entire AoSoA data
      structures as well as slices.

      In this example we will demonstrate building a distributor communication
      plan and migrating data.

      Note: The distributor uses MPI for data migration. MPI is initialized
      and finalized in the main function below.

      Note: The distributor uses GPU-aware MPI communication. If AoSoA data is
      allocated in GPU memory, this feature will be used automatically.
    */

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
    using ExecutionSpace = Kokkos::Serial;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Create the AoSoA.
    */
    int num_tuple = 100;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "A", num_tuple );

    /*
      Create slices and assign data. The data values are equal to id of this
      rank so we can track where the data goes. One might consider using a
      parallel for loop in this case - especially when the code being written
      is for an arbitrary memory space.
     */
    auto slice_0 = Cabana::slice<0>( aosoa );
    auto slice_1 = Cabana::slice<1>( aosoa );
    for ( int i = 0; i < num_tuple; ++i ) {
        slice_0( i ) = comm_rank;
        slice_1( i ) = comm_rank;
    }

    /*
      Build a communication plan where the first 10 elements of our data is
      passed to the next highest rank in the communication, the next 10
      elements are to be discarded, and the last 80 elements stay on this
      rank.
    */
    Kokkos::View<int *, DeviceType> export_ranks( "export_ranks", num_tuple );

    // First 10 go to the next rank. Note that this view will most often be
    // filled within a parallel_for but we do so in serial here for
    // demonstration purposes.
    int previous_rank = ( comm_rank == 0 ) ? comm_size - 1 : comm_rank - 1;
    int next_rank = ( comm_rank == comm_size - 1 ) ? 0 : comm_rank + 1;
    for ( int i = 0; i < 10; ++i )
        export_ranks( i ) = next_rank;

    // Next 10 elements will be discarded. Use an export rank of -1 to
    // indicate this.
    for ( int i = 10; i < 20; ++i )
        export_ranks( i ) = -1;

    // The last 80 elements stay on this process.
    for ( int i = 20; i < num_tuple; ++i )
        export_ranks( i ) = comm_rank;

    /*
      We have two ways to make a distributor. In the first case we know which
      ranks we are sending the data to but not the ranks we are receiving data
      from. In the second we know the topology of the communication plan
      (i.e. the ranks we send and receive from).

      We know that we will only send/receive from this rank and the
      next/previous rank so use that information in this case because this
      substantially reduces the amount of communication needed to compose the
      communication plan. If this neighbor data were not supplied, extra
      global communication would be needed to generate a list of neighbors.
     */
    std::vector<int> neighbors = {previous_rank, comm_rank, next_rank};
    std::sort( neighbors.begin(), neighbors.end() );
    auto unique_end = std::unique( neighbors.begin(), neighbors.end() );
    neighbors.resize( std::distance( neighbors.begin(), unique_end ) );
    Cabana::Distributor<DeviceType> distributor( MPI_COMM_WORLD, export_ranks,
                                                 neighbors );

    /*
      There are three choices for applying the distributor: 1) Migrating the
      entire AoSoA to a new AoSoA, 2) Migrating the AoSoA in-place, 3)
      Migrating using slices. We will go through each next.
     */

    /*
      1) MIGRATING TO A NEW AOSOA

      The following creates a new AoSoA and migrates the entire AoSoA.
     */

    // Make a new AoSoA. Note that this has the same data types, vector
    // length, and memory space as the original aosoa.
    //
    // Also note how this AoSoA is sized. The distrubutor computes how many
    // imported elements each rank will recieve. We discard 10 elements, get
    // 10 from our neighbor, and keep 80 of our own so this number should be 90.
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> destination(
        distributor.totalNumImport() );

    // Do the migration.
    Cabana::migrate( distributor, aosoa, destination );

    /*
      2) MIGRATING SLICES

      We can migrate each slice individually as well. This is useful when not
      all data in an AoSoA needs to be moved to a new decomposition.
     */
    auto slice_0_dst = Cabana::slice<0>( destination );
    auto slice_1_dst = Cabana::slice<1>( destination );
    Cabana::migrate( distributor, slice_0, slice_0_dst );
    Cabana::migrate( distributor, slice_1, slice_1_dst );

    /*
      3) IN-PLACE MIGRATION.

      In many cases a user may want to use the same AoSoA and not manage a
      second temporary copy for the data migration. In-place migration does
      this automatically by moving the data to the new decomposition and
      resizing the AoSoA automatically.

      In the code below, the AoSoA should be size 100 on input and size 90 on
      output and contain the new data.

      Note: Any existing slices created from this AoSoA may be invalidated as
      the data structure will be resized during migration.
     */
    Cabana::migrate( distributor, aosoa );
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char *argv[] ) {
    MPI_Init( &argc, &argv );

    Kokkos::ScopeGuard scope_guard( argc, argv );

    migrationExample();

    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
