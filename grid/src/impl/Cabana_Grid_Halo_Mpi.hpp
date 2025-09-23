/****************************************************************************
 * Copyright (c) 2018-2025 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Grid_Halo_Mpi.hpp
  \brief Multi-node grid scatter/gather implemented with vanilla MPI
*/
#ifndef CABANA_GRID_HALO_MPI_HPP
#define CABANA_GRID_HALO_MPI_HPP

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_IndexSpace.hpp>

#include <Cabana_ParameterPack.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace Grid
{

//---------------------------------------------------------------------------//
/*!
  \brief Grid Halo class. Variant that uses vanilla MPI as the communication
  backend.
*/
template <class MemorySpace>
class Halo<MemorySpace, Mpi> : public HaloBase<MemorySpace>
{
  public:
    using typename HaloBase<MemorySpace>::memory_space;

    /*!
      \brief Constructor.
      \tparam The arrays types to construct the halo for.
      \param pattern The halo pattern to use for halo communication.
      \param width Halo cell width. Must be less than or equal to the halo
      width of the block.
      \param arrays The arrays to build the halo for. These arrays must be
      provided in the same order
    */
    template <class Pattern, class... ArrayTypes>
    Halo( const Pattern& pattern, const int width, const ArrayTypes&... arrays )
        : HaloBase<MemorySpace>( pattern, width, arrays... )
    {
    }

    /*!
      \brief Gather data into our ghosts from their owners.

      \param exec_space The execution space to use for pack/unpack.

      \param arrays The arrays to gather. NOTE: These arrays must be given in
      the same order as in the constructor. These could technically be
      different arrays, they just need to have the same layouts and data types
      as the input arrays.
    */
    template <class ExecutionSpace, class... ArrayTypes>
    void gather( const ExecutionSpace& exec_space,
                 const ArrayTypes&... arrays ) const
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::gather" );

        // Get the number of neighbors. Return if we have none.
        int num_n = this->_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // Get the MPI communicator.
        auto comm = this->getComm( arrays... );

        // Allocate requests.
        std::vector<MPI_Request> requests( 2 * num_n, MPI_REQUEST_NULL );

        // Pick a tag to use for communication. This object has its own
        // communication space so any tag will do.
        const int mpi_tag = 1234;

        // Post receives.
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < this->_ghosted_buffers[n].size() )
            {
                MPI_Irecv( this->_ghosted_buffers[n].data(),
                           this->_ghosted_buffers[n].size(), MPI_BYTE,
                           this->_neighbor_ranks[n],
                           mpi_tag + this->_receive_tags[n], comm,
                           &requests[n] );
            }
        }

        // Pack send buffers and post sends.
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < this->_owned_buffers[n].size() )
            {
                // Pack the send buffer.
                this->packBuffer( exec_space, this->_owned_buffers[n],
                                  this->_owned_steering[n], arrays.view()... );

                // Post a send.
                MPI_Isend( this->_owned_buffers[n].data(),
                           this->_owned_buffers[n].size(), MPI_BYTE,
                           this->_neighbor_ranks[n],
                           mpi_tag + this->_send_tags[n], comm,
                           &requests[num_n + n] );
            }
        }

        // Unpack receive buffers.
        bool unpack_complete = false;
        while ( !unpack_complete )
        {
            // Get the next buffer to unpack.
            int unpack_index = MPI_UNDEFINED;
            MPI_Waitany( num_n, requests.data(), &unpack_index,
                         MPI_STATUS_IGNORE );

            // If there are no more buffers to unpack we are done.
            if ( MPI_UNDEFINED == unpack_index )
            {
                unpack_complete = true;
            }

            // Otherwise unpack the next buffer.
            else
            {
                this->unpackBuffer( ScatterReduce::Replace(), exec_space,
                                    this->_ghosted_buffers[unpack_index],
                                    this->_ghosted_steering[unpack_index],
                                    arrays.view()... );
            }
        }

        // Wait on send requests.
        MPI_Waitall( num_n, requests.data() + num_n, MPI_STATUSES_IGNORE );
    }

    /*!
      \brief Scatter data from our ghosts to their owners using the given type
      of reduce operation.
      \param reduce_op The functor used to reduce the results.
      \param exec_space The execution space to use for pack/unpack.
      \param arrays The arrays to scatter.
    */
    template <class ExecutionSpace, class ReduceOp, class... ArrayTypes>
    void scatter( const ExecutionSpace& exec_space, const ReduceOp& reduce_op,
                  const ArrayTypes&... arrays ) const
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::Grid::scatter" );

        // Get the number of neighbors. Return if we have none.
        int num_n = this->_neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // Get the MPI communicator.
        auto comm = this->getComm( arrays... );

        // Requests.
        std::vector<MPI_Request> requests( 2 * num_n, MPI_REQUEST_NULL );

        // Pick a tag to use for communication. This object has its own
        // communication space so any tag will do.
        const int mpi_tag = 2345;

        // Post receives for all neighbors that are not self sends.
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < this->_owned_buffers[n].size() )
            {
                MPI_Irecv( this->_owned_buffers[n].data(),
                           this->_owned_buffers[n].size(), MPI_BYTE,
                           this->_neighbor_ranks[n],
                           mpi_tag + this->_receive_tags[n], comm,
                           &requests[n] );
            }
        }

        // Pack send buffers and post sends.
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < this->_ghosted_buffers[n].size() )
            {
                // Pack the send buffer.
                this->packBuffer( exec_space, this->_ghosted_buffers[n],
                                  this->_ghosted_steering[n],
                                  arrays.view()... );

                // Post a send.
                MPI_Isend( this->_ghosted_buffers[n].data(),
                           this->_ghosted_buffers[n].size(), MPI_BYTE,
                           this->_neighbor_ranks[n],
                           mpi_tag + this->_send_tags[n], comm,
                           &requests[num_n + n] );
            }
        }

        // Unpack receive buffers.
        bool unpack_complete = false;
        while ( !unpack_complete )
        {
            // Get the next buffer to unpack.
            int unpack_index = MPI_UNDEFINED;
            MPI_Waitany( num_n, requests.data(), &unpack_index,
                         MPI_STATUS_IGNORE );

            // If there are no more buffers to unpack we are done.
            if ( MPI_UNDEFINED == unpack_index )
            {
                unpack_complete = true;
            }

            // Otherwise unpack the next buffer and apply the reduce operation.
            else
            {
                this->unpackBuffer(
                    reduce_op, exec_space, this->_owned_buffers[unpack_index],
                    this->_owned_steering[unpack_index], arrays.view()... );
            }

            // Wait on send requests.
            MPI_Waitall( num_n, requests.data() + num_n, MPI_STATUSES_IGNORE );
        }
    }
};

} // end namespace Grid
} // end namespace Cabana

#endif // end CABANA_GRID_HALO_MPI_HPP