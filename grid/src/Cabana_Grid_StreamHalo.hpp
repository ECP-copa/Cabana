/****************************************************************************
 * Copyright (c) 2018-2024 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Grid_StreamHalo.hpp
  \brief Stream-triggered multi-node grid scatter/gather
*/
#ifndef CABANA_GRID_STREAMHALO_HPP
#define CABANA_GRID_STREAMHALO_HPP

#include <Cabana_Grid_Halo.hpp>

namespace Cabana
{
namespace Grid
{

//---------------------------------------------------------------------------//
// StreamHalo
// ---------------------------------------------------------------------------//
/*!
  General multiple array halo communication plan for migrating shared data
  between blocks. Arrays may be defined on different entity types and have
  different data types.

  The halo operates on an arbitrary set of arrays. Each of these arrays must
  be defined on the same local grid meaning they that share the same
  communicator and halo size. The arrays must also reside in the same memory
  space. These requirements are checked at construction.
*/
template <class MemorySpace, class CommSpace>
class StreamHalo : public Halo<MemorySpace>;
{

  public:
    //! Memory space.
    using memory_space = MemorySpace;

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
    StreamHalo( const Pattern& pattern, const int width, const ArrayTypes&... arrays )
    {
        // Spatial dimension.
        const std::size_t num_space_dim = Pattern::num_space_dim;

        // Get the local grid.
        auto local_grid = getLocalGrid( arrays... );

        // Function to get the local id of the neighbor.
        auto neighbor_id = []( const std::array<int, num_space_dim>& ijk )
        {
            int id = ijk[0];
            for ( std::size_t d = 1; d < num_space_dim; ++d )
                id += num_space_dim * id + ijk[d];
            return id;
        };

        // Neighbor id flip function. This lets us compute what neighbor we
        // are relative to a given neighbor.
        auto flip_id = [=]( const std::array<int, num_space_dim>& ijk )
        {
            std::array<int, num_space_dim> flip_ijk;
            for ( std::size_t d = 0; d < num_space_dim; ++d )
                flip_ijk[d] = -ijk[d];
            return flip_ijk;
        };

        // Get the neighbor ranks we will exchange with in the halo and
        // allocate buffers. If any of the exchanges are self sends mark these
        // so we know which send buffers correspond to which receive buffers.
        auto neighbors = pattern.getNeighbors();
        for ( const auto& n : neighbors )
        {
            // Get the rank of the neighbor.
            int rank = local_grid->neighborRank( n );

            // If this is a valid rank add it as a neighbor.
            if ( rank >= 0 )
            {
                // Add the rank.
                _neighbor_ranks.push_back( rank );

                // Set the tag we will use to send data to this neighbor. The
                // receiving rank should have a matching tag.
                _send_tags.push_back( neighbor_id( n ) );

                // Set the tag we will use to receive data from this
                // neighbor. The sending rank should have a matching tag.
                _receive_tags.push_back( neighbor_id( flip_id( n ) ) );

                // Create communication data for owned entities.
                buildCommData( Own(), width, n, _owned_buffers, _owned_steering,
                               arrays... );

                // Create communication data for ghosted entities.
                buildCommData( Ghost(), width, n, _ghosted_buffers,
                               _ghosted_steering, arrays... );
            }
        }
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
        int num_n = _neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // Get the MPI communicator.
        auto comm = getComm( arrays... );

        // Allocate requests.
        std::vector<MPI_Request> requests( 2 * num_n, MPI_REQUEST_NULL );

        // Pick a tag to use for communication. This object has its own
        // communication space so any tag will do.
        const int mpi_tag = 1234;

        // Post receives.
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < _ghosted_buffers[n].size() )
            {
                MPI_Irecv( _ghosted_buffers[n].data(),
                           _ghosted_buffers[n].size(), MPI_BYTE,
                           _neighbor_ranks[n], mpi_tag + _receive_tags[n], comm,
                           &requests[n] );
            }
        }

        // Pack send buffers and post sends.
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < _owned_buffers[n].size() )
            {
                // Pack the send buffer.
                packBuffer( exec_space, _owned_buffers[n], _owned_steering[n],
                            arrays.view()... );

                // Post a send.
                MPI_Isend( _owned_buffers[n].data(), _owned_buffers[n].size(),
                           MPI_BYTE, _neighbor_ranks[n],
                           mpi_tag + _send_tags[n], comm,
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
                unpackBuffer( ScatterReduce::Replace(), exec_space,
                              _ghosted_buffers[unpack_index],
                              _ghosted_steering[unpack_index],
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
        int num_n = _neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // Get the MPI communicator.
        auto comm = getComm( arrays... );

        // Requests.
        std::vector<MPI_Request> requests( 2 * num_n, MPI_REQUEST_NULL );

        // Pick a tag to use for communication. This object has its own
        // communication space so any tag will do.
        const int mpi_tag = 2345;

        // Post receives for all neighbors that are not self sends.
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < _owned_buffers[n].size() )
            {
                MPI_Irecv( _owned_buffers[n].data(), _owned_buffers[n].size(),
                           MPI_BYTE, _neighbor_ranks[n],
                           mpi_tag + _receive_tags[n], comm, &requests[n] );
            }
        }

        // Pack send buffers and post sends.
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < _ghosted_buffers[n].size() )
            {
                // Pack the send buffer.
                packBuffer( exec_space, _ghosted_buffers[n],
                            _ghosted_steering[n], arrays.view()... );

                // Post a send.
                MPI_Isend( _ghosted_buffers[n].data(),
                           _ghosted_buffers[n].size(), MPI_BYTE,
                           _neighbor_ranks[n], mpi_tag + _send_tags[n], comm,
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
                unpackBuffer( reduce_op, exec_space,
                              _owned_buffers[unpack_index],
                              _owned_steering[unpack_index], arrays.view()... );
            }

            // Wait on send requests.
            MPI_Waitall( num_n, requests.data() + num_n, MPI_STATUSES_IGNORE );
        }
    }
};

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_STREAMHALO_HPP
