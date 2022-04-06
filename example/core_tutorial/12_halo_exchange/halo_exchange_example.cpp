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

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

#include <mpi.h>

//---------------------------------------------------------------------------//
// Halo exchange example.
//---------------------------------------------------------------------------//
void haloExchangeExample()
{
    /*
      The halo is a communication plan designed from halo exchange where some
      locally-owned elements on each rank are used as ghost data on other
      ranks. The halo supplies both forward and reverse communication
      operations. In the forward operation (the gather), data is sent from the
      uniquely-owned decomposition to the ghosted decomposition. In the
      reverse operation (the scatter), data is sent from the ghosted
      decomposition back to the uniquely-owned decomposition and collisions
      are resolved.

      In this example we will demonstrate building a halo communication
      plan and performing both scatter and gather operations.

      Note: The halo uses MPI for data movement. MPI is initialized
      and finalized in the main function below.

      Note: The halo uses GPU-aware MPI communication. If AoSoA data is
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
    using DataTypes = Cabana::MemberTypes<double, double>;
    const int VectorLength = 8;
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Create the AoSoA.
    */
    int num_tuple = 100;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "my_aosoa",
                                                              num_tuple );

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
      Before migrating the data, let's print out the data in the slices
      on one rank.
    */
    if ( comm_rank == 0 )
    {
        std::cout << "BEFORE exchange" << std::endl
                  << "(Rank " << comm_rank << ") ";
        for ( std::size_t i = 0; i < slice_ranks.size(); ++i )
            std::cout << slice_ranks( i ) << " ";
        std::cout << std::endl
                  << "(" << slice_ranks.size() << " ranks before exchange)"
                  << std::endl
                  << "(Rank " << comm_rank << ") ";
        for ( std::size_t i = 0; i < slice_ids.size(); ++i )
            std::cout << slice_ids( i ) << " ";
        std::cout << std::endl
                  << "(" << slice_ids.size() << " IDs before exchange)"
                  << std::endl
                  << std::endl;
    }

    /*
      Build a halo where the last 10 elements are sent to the next rank.
    */
    int local_num_send = 10;
    Kokkos::View<int*, DeviceType> export_ranks( "export_ranks",
                                                 local_num_send );
    Kokkos::View<int*, DeviceType> export_ids( "export_ids", local_num_send );

    // Last 10 elements (elements 90-99) go to the next rank. Note that this
    // view will most often be filled within a parallel_for but we do so in
    // serial here for demonstration purposes.
    int previous_rank = ( comm_rank == 0 ) ? comm_size - 1 : comm_rank - 1;
    int next_rank = ( comm_rank == comm_size - 1 ) ? 0 : comm_rank + 1;
    for ( int i = 0; i < local_num_send; ++i )
    {
        export_ranks( i ) = next_rank;
        export_ids( i ) = i + num_tuple - 10;
    }

    /*
      We have two ways to make a halo. In the first case we know what ranks we
      are sending the data to but not the ranks we are receiving data from. In
      the second we know the topology of the communication plan (i.e. the
      ranks we send and receive from).

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
    Cabana::Halo<DeviceType> halo( MPI_COMM_WORLD, num_tuple, export_ids,
                                   export_ranks, neighbors );

    /*
      Resize the AoSoA to allow for additional ghost data. We can get the
      number of ghosts directly from the halo as well as the number of local
      elements (which is equal to num_tuple). We should be getting 10 ghosts
      from our neighbor so the new size should be 110 in this case.

      The halo always puts the ghost data at the end of the aosoa. In this
      case the first 100 elements are those that we started with and the next
      10 are the ghosts we got from our neighbor. If there are multiple ranks
      that are sending us ghosts, the ghost data will blocked by their owning
      rank such that all ghosts from a single rank appear in consecutive
      order.
     */
    aosoa.resize( halo.numLocal() + halo.numGhost() );

    /*
      Get new slices after resizing.
     */
    slice_ranks = Cabana::slice<0>( aosoa );
    slice_ids = Cabana::slice<1>( aosoa );

    /*
      Gather data for the ghosts on this rank from our neighbors that own
      them. We can do this with slices or the entire AoSoA. The last ten
      elements in the AoSoA should now have data from our neighbor (with
      their previous local ID), with a total size of 110.
     */
    Cabana::gather( halo, aosoa );

    /*
      Having exchanged the data, let's print out the data on one rank.
    */
    if ( comm_rank == 0 )
    {
        std::cout << "AFTER gather" << std::endl
                  << "(Rank " << comm_rank << ") ";
        for ( std::size_t i = 0; i < slice_ranks.size(); ++i )
            std::cout << slice_ranks( i ) << " ";
        std::cout << std::endl
                  << "(" << slice_ranks.size() << " ranks after gather)"
                  << std::endl
                  << "(Rank " << comm_rank << ") ";
        for ( std::size_t i = 0; i < slice_ids.size(); ++i )
            std::cout << slice_ids( i ) << " ";
        std::cout << std::endl
                  << "(" << slice_ids.size() << " IDs after gather)"
                  << std::endl
                  << std::endl;
    }

    /*
      Scatter the ghost data we have back to its owning rank. Any collisions
      will be resolved on the owning rank (i.e. the owning rank has an element
      that is a ghost on several ranks). The scatter sums all ghosted values
      into the locally-owned value. Because of the nature of the scatter, it
      can only be performed with slices and not the entire AoSoA.

      Elements 90-99 will now have their original values plus the values they
      received from their neighbors. We only have one neighbor in this case so
      these elements should now be doubled, with an unchanged total slice size.
     */
    Cabana::scatter( halo, slice_ranks );
    Cabana::scatter( halo, slice_ids );

    /*
      Having exchanged the data, let's print out the data on one rank.
    */
    if ( comm_rank == 0 )
    {
        std::cout << "AFTER scatter" << std::endl
                  << "(Rank " << comm_rank << ") ";
        for ( std::size_t i = 0; i < slice_ranks.size(); ++i )
            std::cout << slice_ranks( i ) << " ";
        std::cout << std::endl
                  << "(" << slice_ranks.size() << " ranks after scatter)"
                  << std::endl
                  << "(Rank " << comm_rank << ") ";
        for ( std::size_t i = 0; i < slice_ids.size(); ++i )
            std::cout << slice_ids( i ) << " ";
        std::cout << std::endl
                  << "(" << slice_ids.size() << " IDs after scatter)"
                  << std::endl;
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    Kokkos::ScopeGuard scope_guard( argc, argv );

    haloExchangeExample();

    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
