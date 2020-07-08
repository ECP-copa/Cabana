/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_HALO_HPP
#define CAJITA_HALO_HPP

#include <Cajita_Array.hpp>
#include <Cajita_Hilbert.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_MpiTraits.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Halo exchange patterns.
//---------------------------------------------------------------------------//
// Base class.
class HaloPattern
{
  public:
    // Default constructor.
    HaloPattern() {}

    // Destructor
    virtual ~HaloPattern() = default;

    // Assign the neighbors that are in the halo pattern.
    void setNeighbors( const std::vector<std::array<int, 3>> &neighbors )
    {
        _neighbors = neighbors;
    }

    // Get the neighbors that are in the halo pattern.
    std::vector<std::array<int, 3>> getNeighbors() const { return _neighbors; }

  private:
    std::vector<std::array<int, 3>> _neighbors;
};

// Full halo with all 26 adjacent blocks.
class FullHaloPattern : public HaloPattern
{
  public:
    FullHaloPattern()
        : HaloPattern()
    {
        std::vector<std::array<int, 3>> neighbors;
        neighbors.reserve( 26 );
        for ( int i = -1; i < 2; ++i )
            for ( int j = -1; j < 2; ++j )
                for ( int k = -1; k < 2; ++k )
                    if ( !( i == 0 && j == 0 && k == 0 ) )
                        neighbors.push_back( {i, j, k} );
        this->setNeighbors( neighbors );
    }
};

//---------------------------------------------------------------------------//
// Scatter reduction.
//---------------------------------------------------------------------------//
namespace ScatterReduce
{

// Sum values from neighboring ranks into this rank's data.
struct Sum
{
    template <class ViewType, class BufferType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const ViewType &view, const BufferType &buffer, const int i,
                const int j, const int k, const int l ) const
    {
        view( i, j, k, l ) += buffer( i, j, k, l );
    }
};

// Assign this rank's data to be the minimum of it and its neighbor ranks'
// values.
struct Min
{
    template <class ViewType, class BufferType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const ViewType &view, const BufferType &buffer, const int i,
                const int j, const int k, const int l ) const
    {
        if ( view( i, j, k, l ) > buffer( i, j, k, l ) )
            view( i, j, k, l ) = buffer( i, j, k, l );
    }
};

// Assign this rank's data to be the maximum of it and its neighbor ranks'
// values.
struct Max
{
    template <class ViewType, class BufferType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const ViewType &view, const BufferType &buffer, const int i,
                const int j, const int k, const int l ) const
    {
        if ( view( i, j, k, l ) < buffer( i, j, k, l ) )
            view( i, j, k, l ) = buffer( i, j, k, l );
    }
};

// Replace this rank's data with its neighbor ranks' values. Note that if
// multiple ranks scatter back to the same grid locations then the value
// assigned will be from one of the neighbors but it is undetermined from
// which neighbor that value will come.
struct Replace
{
    template <class ViewType, class BufferType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const ViewType &view, const BufferType &buffer, const int i,
                const int j, const int k, const int l ) const
    {
        view( i, j, k, l ) = buffer( i, j, k, l );
    }
};

} // end namespace ScatterReduce

//---------------------------------------------------------------------------//
// Halo exchange communication plan for migrating shared data between blocks.
//---------------------------------------------------------------------------//
template <class Scalar, class DeviceType>
class Halo
{
  public:
    // Scalar type.
    using value_type = Scalar;

    // Device type.
    using device_type = DeviceType;

    // Execution space.
    using execution_space = typename device_type::execution_space;

    // Memory space.
    using memory_space = typename device_type::memory_space;

    // View type.
    using view_type = Kokkos::View<value_type ****, device_type>;

    /*!
      \brief Constructor.
      \param layout The array layout to build the halo for.
      \param pattern The halo pattern to use for halo communication
      \param width An option halo cell width. The default is to use the entire
      halo width of the block.
    */
    template <class ArrayLayout_t>
    Halo( const ArrayLayout_t &layout, const HaloPattern &pattern,
          const int width = -1 )
    {
        // Duplicate the communicator so we have our own communication space.
        MPI_Comm_dup( layout.localGrid()->globalGrid().comm(), &_comm );

        // Function to get the local id of the neighbor.
        auto neighbor_id = []( const int i, const int j, const int k ) {
            int nk = k + 1;
            int nj = j + 1;
            int ni = i + 1;
            return nk + 3 * ( nj + 3 * ni );
        };

        // Neighbor id flip function. This lets us compute what neighbor we
        // are relative to a given neighbor.
        auto flip = []( const int i ) {
            if ( i == -1 )
                return 1;
            else if ( i == 0 )
                return 0;
            else
                return -1;
        };

        // Get the neighbor ranks we will exchange with in the halo and
        // allocate buffers. If any of the exchanges are self sends mark these
        // so we know which send buffers correspond to which receive buffers.
        auto neighbors = pattern.getNeighbors();
        for ( const auto &n : neighbors )
        {
            // Get the neighbor ids.
            auto i = n[Dim::I];
            auto j = n[Dim::J];
            auto k = n[Dim::K];

            // Get the rank of the neighbor.
            int rank = layout.localGrid()->neighborRank( i, j, k );

            // If this is a valid rank add it as a neighbor.
            if ( rank >= 0 )
            {
                // Add the rank.
                _neighbor_ranks.push_back( rank );

                // Set the tag we will use to send data to this
                // neighbor. The receiving rank should have a
                // matching tag.
                _send_tags.push_back( neighbor_id( i, j, k ) );

                // Set the tag we will use to receive data from
                // this neighbor. The sending rank should have a
                // matching tag.
                _receive_tags.push_back(
                    neighbor_id( flip( i ), flip( j ), flip( k ) ) );

                // Get the owned index space we share with this
                // neighbor.
                _owned_spaces.push_back(
                    layout.sharedIndexSpace( Own(), i, j, k, width ) );

                // Create the buffer of data we own that we share
                // with this neighbor.
                _owned_buffers.push_back( createView<value_type, device_type>(
                    "halo_owned_buffer", _owned_spaces.back() ) );

                // Get the ghosted index space we share with this
                // neighbor.
                _ghosted_spaces.push_back(
                    layout.sharedIndexSpace( Ghost(), i, j, k, width ) );

                // Create the buffer of ghost data that is owned
                // by our neighbor
                _ghosted_buffers.push_back( createView<value_type, device_type>(
                    "halo_ghosted_buffer", _ghosted_spaces.back() ) );
            }
        }
    }

    // Destructor.
    ~Halo() { MPI_Comm_free( &_comm ); }

    /*!
      \brief Gather data into our ghosts from their owners.
      \param array The array to gather.
    */
    template <class Array_t>
    void gather( const Array_t &array ) const
    {
        // Check that the array type is valid.
        static_assert(
            std::is_same<value_type, typename Array_t::value_type>::value,
            "Array value type does not match halo value type" );
        static_assert(
            std::is_same<memory_space, typename Array_t::memory_space>::value,
            "Array memory space does not match halo memory space" );

        // Get the number of neighbors. Return if we have none.
        int num_n = _neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // Get the array data.
        auto view = array.view();

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
                           _ghosted_buffers[n].size(),
                           MpiTraits<value_type>::type(), _neighbor_ranks[n],
                           mpi_tag + _receive_tags[n], _comm, &requests[n] );
            }
        }

        // Pack send buffers and post sends.
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < _owned_buffers[n].size() )
            {
                // Pack the send buffer.
                auto subview = createSubview( view, _owned_spaces[n] );
                Kokkos::deep_copy( _owned_buffers[n], subview );

                // Post a send.
                MPI_Isend( _owned_buffers[n].data(), _owned_buffers[n].size(),
                        MpiTraits<value_type>::type(), _neighbor_ranks[n],
                        mpi_tag + _send_tags[n], _comm,
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
                if ( std::string( typeid(view.layout()).name() ).find("LayoutHilbert") != std::string::npos ) {
                    view_type ghostedBuffer = _ghosted_buffers[unpack_index];
                    Cajita::hilbertCopy( view, _ghosted_buffers[unpack_index], _ghosted_spaces[unpack_index] );
                }
                else {
                    auto subview =
                        createSubview( view, _ghosted_spaces[unpack_index] );
                    Kokkos::deep_copy( subview, _ghosted_buffers[unpack_index] );
                }
            }
        }

        // Wait on send requests.
        MPI_Waitall( num_n, requests.data() + num_n, MPI_STATUSES_IGNORE );
    }

    /*!
      \brief Scatter data from our ghosts to their owners. Default sum
      implementation.
      \param array The array to scatter.
    */
    template <class Array_t>
    void scatter( const Array_t &array ) const
    {
        scatter( array, ScatterReduce::Sum() );
    }

    /*!
      \brief Scatter data from our ghosts to their owners using the given type
      of reduce operation.
      \param array The array to scatter.
      \param reduce The functor used to reduce the results.
    */
    template <class Array_t, class ReduceFunc>
    void scatter( const Array_t &array, const ReduceFunc &reduce ) const
    {
        // Check that the array type is valid.
        static_assert(
            std::is_same<value_type, typename Array_t::value_type>::value,
            "Array value type does not match halo value type" );
        static_assert(
            std::is_same<memory_space, typename Array_t::memory_space>::value,
            "Array memory space does not match halo memory space" );

        // Get the number of neighbors. Return if we have none.
        int num_n = _neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // Get the array data.
        auto view = array.view();

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
                           MpiTraits<value_type>::type(), _neighbor_ranks[n],
                           mpi_tag + _receive_tags[n], _comm, &requests[n] );
            }
        }

        // Pack send buffers and post sends.
        for ( int n = 0; n < num_n; ++n )
        {
            // Only process this neighbor if there is work to do.
            if ( 0 < _ghosted_buffers[n].size() )
            {
                // Pack the send buffer.
                auto subview = createSubview( view, _ghosted_spaces[n] );
                Kokkos::deep_copy( _ghosted_buffers[n], subview );

                // Post a send.
                MPI_Isend(
                    _ghosted_buffers[n].data(), _ghosted_buffers[n].size(),
                    MpiTraits<value_type>::type(), _neighbor_ranks[n],
                    mpi_tag + _send_tags[n], _comm, &requests[num_n + n] );
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
                auto subview =
                    createSubview( view, _owned_spaces[unpack_index] );
                auto owned_buffer = _owned_buffers[unpack_index];
                IndexSpace<4> scatter_space(
                    {static_cast<long>( subview.extent( 0 ) ),
                     static_cast<long>( subview.extent( 1 ) ),
                     static_cast<long>( subview.extent( 2 ) ),
                     static_cast<long>( subview.extent( 3 ) )} );
                Kokkos::parallel_for(
                    "Cajita::Halo::scatterOwned",
                    createExecutionPolicy( scatter_space, execution_space() ),
                    KOKKOS_LAMBDA( const int i, const int j, const int k,
                                   const int l ) {
                        reduce( subview, owned_buffer, i, j, k, l );
                    } );
            }

            // Wait on send requests.
            MPI_Waitall( num_n, requests.data() + num_n, MPI_STATUSES_IGNORE );
        }
    }

  private:
    MPI_Comm _comm;
    std::vector<int> _neighbor_ranks;
    std::vector<int> _send_tags;
    std::vector<int> _receive_tags;
    std::vector<IndexSpace<4>> _owned_spaces;
    std::vector<view_type> _owned_buffers;
    std::vector<IndexSpace<4>> _ghosted_spaces;
    std::vector<view_type> _ghosted_buffers;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a halo with a layout.
  \param layout The array layout to build the halo for.
  \note The scalar type and device type must be specified so the proper
  buffers may be allocated. This means a halo constructed via this method is
  only compatible with arrays that have the same scalar and device type.
*/
template <class Scalar, class Device, class EntityType, class MeshType>
std::shared_ptr<Halo<Scalar, Device>>
createHalo( const ArrayLayout<EntityType, MeshType> &layout,
            const HaloPattern &pattern, const int width = -1 )
{
    return std::make_shared<Halo<Scalar, Device>>( layout, pattern, width );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a halo.
  \param layout The array to build the halo for.
  \note The scalar type and device type are specified via the input arrays so
  the proper buffers may be allocated. This means a halo constructed via this
  method is only compatible with arrays that have the same scalar and device
  type as the input array.
*/
template <class Scalar, class EntityType, class MeshType, class... Params>
std::shared_ptr<
    Halo<typename Array<Scalar, EntityType, MeshType, Params...>::value_type,
         typename Array<Scalar, EntityType, MeshType, Params...>::device_type>>
createHalo( const Array<Scalar, EntityType, MeshType, Params...> &array,
            const HaloPattern &pattern, const int width = -1 )
{
    return std::make_shared<Halo<
        typename Array<Scalar, EntityType, MeshType, Params...>::value_type,
        typename Array<Scalar, EntityType, MeshType, Params...>::device_type>>(
        *( array.layout() ), pattern, width );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_HALO_HPP
