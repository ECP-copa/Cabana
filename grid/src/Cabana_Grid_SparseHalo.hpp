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

#ifndef CABANA_GRID_SPARSEHALO_HPP
#define CABANA_GRID_SPARSEHALO_HPP

#include <Cabana_MemberTypes.hpp>
#include <Cabana_SoA.hpp>
#include <Cabana_Tuple.hpp>

#include <Cabana_Grid_HaloBase.hpp> // to get the pattern and tags defined here
#include <Cabana_Grid_SparseArray.hpp>
#include <Cabana_Grid_SparseIndexSpace.hpp>
#include <Cabana_Grid_Types.hpp>

#include <Kokkos_Core.hpp>

#include <numeric>
#include <type_traits>

namespace Cabana
{
namespace Grid
{
namespace Experimental
{
//---------------------------------------------------------------------------//
// Halo
// ---------------------------------------------------------------------------//
/*!
  General multiple array halo communication plan for migrating shared data
  between sparse grid blocks.
  Arrays may be defined on different entity types and have different data types.

  The halo operates on an arbitrary set of arrays. Each of these arrays must
  be defined on the same local grid meaning they that share the same
  communicator and halo size. The arrays must also reside in the same memory
  space. These requirements are checked at construction.
*/
template <class MemorySpace, class DataTypes, class EntityType,
          std::size_t NumSpaceDim, unsigned long long cellBitsPerTileDim,
          typename Value = int, typename Key = uint64_t>
class SparseHalo
{
  public:
    //! sparse array dimension number
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    //! memory space to store the sparse grid
    using memory_space = MemorySpace;
    //! entity type on sparse grid
    using entity_type = EntityType;

    //! sparse grid halo pattern (TODO currently reusing Halo's Node pattern)
    using halo_pattern_type = NodeHaloPattern<NumSpaceDim>;

    //! value type of entities on sparse grid
    using value_type = Value;
    //! key type in sparse map
    using key_type = Key;
    //! invalid key in sparse map
    enum KeyValue
    {
        invalid_key = ~static_cast<key_type>( 0 )
    };

    //! data members in AoSoA structure
    using aosoa_member_types = DataTypes;
    //! AoSoA tuple type
    using tuple_type = Cabana::Tuple<aosoa_member_types>;

    //! AoSoA member data type
    template <std::size_t M>
    using member_data_type =
        typename Cabana::MemberTypeAtIndex<M, aosoa_member_types>::type;
    //! AoSoA member #
    static constexpr std::size_t member_num = aosoa_member_types::size;
    //! sparse grid hierarchy: cell id bit# per dimension
    static constexpr unsigned long long cell_bits_per_tile_dim =
        cellBitsPerTileDim;
    //! sparse grid hierarchy: cell # per dimension
    static constexpr unsigned long long cell_num_per_tile =
        1 << ( cell_bits_per_tile_dim * 3 );

    //! communication data buffer view type
    using buffer_view = Kokkos::View<tuple_type*, memory_space>;
    //! communication steering view type, used to check whether there are common
    //! sparse grids in the halo region
    using steering_view = Kokkos::View<key_type*, memory_space>;
    //! tile index space type TODO
    using tile_index_space =
        TileIndexSpace<num_space_dim, cell_bits_per_tile_dim>;
    //! index (own or ghost)
    enum Index
    {
        own = 0,
        ghost = 1,
        total = 2
    };

    //! view type used to count common sparse grid # with neighbors
    //! [0] shared_owned_num [1] shared_ghost_num
    using counting_view = Kokkos::View<int[2], memory_space,
                                       Kokkos::MemoryTraits<Kokkos::Atomic>>;

    //---------------------------------------------------------------------------//
    /*!
        \brief constructor
        \tparam LocalGridType local grid type
        \param pattern The halo pattern to use for halo communication
        \param sparse_array Sparse array to communicate
    */
    template <class SparseArrayType>
    SparseHalo( const halo_pattern_type pattern,
                const std::shared_ptr<SparseArrayType>& sparse_array )
        : _pattern( pattern )
    {
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

        // compute total number of bytes in each SoA structure of the sparse
        // grid (the spasre grid is in AoSoA manner)
        auto soa_byte_array = compute_member_size_list();
        for ( std::size_t i = 0; i < member_num; ++i )
            _soa_member_bytes[i] = soa_byte_array[i];
        _soa_total_bytes = std::max(
            std::accumulate( soa_byte_array.begin(), soa_byte_array.end(), 0 ),
            static_cast<int>( sizeof( tuple_type ) ) );

        // Get the local grid the array uses.
        auto local_grid = sparse_array->layout().localGrid();

        // linear MPI rank ID of the current working rank
        _self_rank =
            local_grid->neighborRank( std::array<int, 3>( { 0, 0, 0 } ) );

        // set the linear neighbor rank ID
        // set up correspondence between sending and receiving buffers
        auto neighbors = _pattern.getNeighbors();
        for ( const auto& n : neighbors )
        {
            // neighbor rank linear ID
            int rank = local_grid->neighborRank( n );

            // if neighbor is valid
            if ( rank >= 0 )
            {
                // add neighbor to the list
                _neighbor_ranks.push_back( rank );
                // set the tag to use for sending data to this neighbor
                // the corresponding neighbor will have the same receiving tag
                _send_tags.push_back( neighbor_id( n ) );
                // set the tag to use for receiving data from this neighbor
                // the corresponding neighbor will have the same sending tag
                _receive_tags.push_back( neighbor_id( flip_id( n ) ) );

                // build communication data for owned entries
                buildCommData( Own(), local_grid, n, _owned_buffers,
                               _owned_tile_steering, _owned_tile_spaces );
                // build communication data for ghosted entries
                buildCommData( Ghost(), local_grid, n, _ghosted_buffers,
                               _ghosted_tile_steering, _ghosted_tile_spaces );

                auto& own_index_space = _owned_tile_spaces.back();
                auto& ghost_index_space = _ghosted_tile_spaces.back();
                int tmp_steering_size =
                    own_index_space.sizeTile() > ghost_index_space.sizeTile()
                        ? own_index_space.sizeTile()
                        : ghost_index_space.sizeTile();
                _tmp_tile_steering.push_back(
                    steering_view( "tmp_tile_steering", tmp_steering_size ) );

                _valid_counting.push_back(
                    counting_view( "halo_valid_counting" ) );
                _neighbor_counting.push_back(
                    counting_view( "halo_neighbor_valid_counting" ) );
                Kokkos::deep_copy( _valid_counting.back(), 0 );
                Kokkos::deep_copy( _neighbor_counting.back(), 0 );

                _valid_neighbor_ids.emplace_back( n );
            }
        }
    }

    //! Get the communicator.
    template <class SparseArrayType>
    MPI_Comm getComm( const SparseArrayType sparse_array ) const
    {
        return sparse_array.layout().localGrid()->globalGrid().comm();
    }

    /*!
        \brief Build communication data.
        \tparam DecompositionTag  decomposition tag type
        \tparam LocalGridType  sparse local grid type
        \param decomposition_tag tag to indicate if it's owned or ghosted halo
        \param local_grid sparse local grid shared pointer
        \param nid neighbor local id (ijk in pattern)
        \param buffers buffer to be used to store communicated data
        \param steering steering to be used to guide communications
        \param spaces sparse tile index spaces
    */
    template <class DecompositionTag, class LocalGridType>
    void buildCommData( DecompositionTag decomposition_tag,
                        const std::shared_ptr<LocalGridType>& local_grid,
                        const std::array<int, num_space_dim>& nid,
                        std::vector<buffer_view>& buffers,
                        std::vector<steering_view>& steering,
                        std::vector<tile_index_space>& spaces )
    {
        // get the halo sparse tile index space sharsed with the neighbor
        spaces.push_back(
            local_grid->template sharedTileIndexSpace<cell_bits_per_tile_dim>(
                decomposition_tag, entity_type(), nid ) );
        auto& index_space = spaces.back();

        // allocate the buffer to store shared data with given neighbor
        buffers.push_back(
            buffer_view( "halo_buffer", index_space.sizeCell() ) );
        // allocate the steering to guide the communication with the given
        // neighbor
        steering.push_back(
            steering_view( "halo_tile_steering", index_space.sizeTile() ) );
        // clear steering (init)
        Kokkos::deep_copy( steering.back(), invalid_key );
    }

    //---------------------------------------------------------------------------//
    /*!
        \brief update tile index space according to current partition
        \tparam LocalGridType sparse local grid type
        \param local_grid sparse local grid pointer
    */
    template <class LocalGridType>
    void updateTileSpace( const std::shared_ptr<LocalGridType>& local_grid )
    {
        // clear index space array first
        _owned_tile_spaces.clear();
        _ghosted_tile_spaces.clear();

        // loop over all neighbors and update the shared tile index space
        auto neighbors = _pattern.getNeighbors();
        for ( std::size_t i = 0; i < _valid_neighbor_ids.size(); ++i )
        {
            // get neighbor relative id
            auto& n = _valid_neighbor_ids[i];
            // get neighbor linear MPI rank ID
            int rank = local_grid->neighborRank( n );
            // check if neighbor rank is valid
            // the neighbor id should always be valid (as all should be
            // well-prepared during construction/initialization)
            if ( rank == _neighbor_ranks[i] )
            {
                // get shared tile index space from local grid
                _owned_tile_spaces.push_back(
                    local_grid
                        ->template sharedTileIndexSpace<cell_bits_per_tile_dim>(
                            Own(), entity_type(), n ) );
                _ghosted_tile_spaces.push_back(
                    local_grid
                        ->template sharedTileIndexSpace<cell_bits_per_tile_dim>(
                            Ghost(), entity_type(), n ) );

                // reference to tile index spaces
                auto& own_index_space = _owned_tile_spaces.back();
                auto& ghost_index_space = _ghosted_tile_spaces.back();

                // number of tiles inside each shared tile space
                int own_tile_size = own_index_space.sizeTile();
                int ghost_tile_size = ghost_index_space.sizeTile();
                int tmp_steering_size = own_tile_size > ghost_tile_size
                                            ? own_tile_size
                                            : ghost_tile_size;

                // check if the data steering and buffers require resize
                // since the partition may changes during simulation process
                // the size of shared spaces may also change accordingly
                if ( own_tile_size > _owned_tile_steering[i].extent( 0 ) )
                {
                    Kokkos::resize( _owned_tile_steering[i], own_tile_size );
                    Kokkos::resize( _owned_buffers[i],
                                    own_index_space.sizeCell() );
                }
                if ( ghost_tile_size > _ghosted_tile_steering[i].extent( 0 ) )
                {
                    Kokkos::resize( _ghosted_tile_steering[i],
                                    ghost_tile_size );
                    Kokkos::resize( _ghosted_buffers[i],
                                    ghost_index_space.sizeCell() );
                }
                if ( tmp_steering_size > _tmp_tile_steering[i].extent( 0 ) )
                    Kokkos::resize( _tmp_tile_steering[i], tmp_steering_size );
            }
            else
                std::runtime_error(
                    "Cabana::Grid::Experimental::SparseHalo::updateTileSpace: "
                    "Neighbor rank doesn't match id" );
        }
    }

    //---------------------------------------------------------------------------//
    /*!
        \brief register valid halos (according to grid activation status in
       sparse map) in the steerings
        \tparam ExecSpace execution space
        \tparam SparseMapType sparse map type
        \tparam scalar_type scalar type (type of dx)
        \param map sparse map
    */
    template <class ExecSpace, class SparseMapType>
    void register_halo( SparseMapType& map )
    {
        // return if there's no valid neighbors
        int num_n = _neighbor_ranks.size();
        if ( 0 == num_n )
            return;

        // loop for all valid neighbors
        for ( int nid = 0; nid < num_n; nid++ )
        {
            auto& owned_space = _owned_tile_spaces[nid];
            auto& owned_steering = _owned_tile_steering[nid];

            auto& ghosted_space = _ghosted_tile_spaces[nid];
            auto& ghosted_steering = _ghosted_tile_steering[nid];

            auto& counting = _valid_counting[nid];

            // check the sparse map, if there are valid tiles in corresponding
            // shared tile index space, add the tile key into the steering view,
            // this steering keys will later be used for halo data collection

            Kokkos::parallel_for(
                Kokkos::RangePolicy<ExecSpace>( 0, map.capacity() ),
                KOKKOS_LAMBDA( const int index ) {
                    if ( map.valid_at( index ) )
                    {
                        auto tile_key = map.key_at( index );
                        int ti, tj, tk;
                        map.key2ijk( tile_key, ti, tj, tk );
                        if ( owned_space.tileInRange( ti, tj, tk ) )
                        {
                            owned_steering( counting( Index::own )++ ) =
                                tile_key;
                        }
                        else if ( ghosted_space.tileInRange( ti, tj, tk ) )
                        {
                            ghosted_steering( counting( Index::ghost )++ ) =
                                tile_key;
                        }
                    }
                } );
            Kokkos::fence();
        }
    }

    //---------------------------------------------------------------------------//
    /*!
        \brief clear guiding information in sparse halo,
        \param comm MPI communicator

       This information needs to be cleared before steering and counting, then
       recollected before halo communication in each step.
    */
    void clear( MPI_Comm comm )
    {
        // clear counting
        for ( std::size_t i = 0; i < _valid_counting.size(); ++i )
            Kokkos::deep_copy( _valid_counting[i], 0 );
        for ( std::size_t i = 0; i < _neighbor_counting.size(); ++i )
            Kokkos::deep_copy( _neighbor_counting[i], 0 );
        // clear steering
        for ( std::size_t i = 0; i < _owned_tile_steering.size(); ++i )
            Kokkos::deep_copy( _owned_tile_steering[i], invalid_key );
        for ( std::size_t i = 0; i < _ghosted_tile_steering.size(); ++i )
            Kokkos::deep_copy( _ghosted_tile_steering[i], invalid_key );
        for ( std::size_t i = 0; i < _tmp_tile_steering.size(); ++i )
            Kokkos::deep_copy( _tmp_tile_steering[i], invalid_key );
        // sync
        MPI_Barrier( comm );
    }

    //---------------------------------------------------------------------------//
    /*!
        \brief neighbor tile counting, communication needed only if the counting
       is non-zero
        \param comm MPI communicator
        \param is_neighbor_counting_collected label if the neighbor has already
       been collected; if true, it means all neighbor counting information is
       up-to-date and there's no need for recollection
    */
    void collectNeighborCounting(
        MPI_Comm comm, const bool is_neighbor_counting_collected = false ) const
    {
        // the valid halo size is already counted, no need to recount
        if ( is_neighbor_counting_collected )
            return;

        // number of neighbors
        int num_n = _neighbor_ranks.size();

        // MPI request to collect counting information in shared owned and
        // shared ghosted space
        std::vector<MPI_Request> counting_requests( 2 * num_n,
                                                    MPI_REQUEST_NULL );
        const int mpi_tag_counting = 1234;

        // receive from all neighbors
        for ( int nid = 0; nid < num_n; ++nid )
        {
            MPI_Irecv( _neighbor_counting[nid].data(),
                       Index::total * sizeof( int ), MPI_BYTE,
                       _neighbor_ranks[nid],
                       mpi_tag_counting + _receive_tags[nid], comm,
                       &counting_requests[nid] );
        }
        // send to all valid neighbors
        for ( int nid = 0; nid < num_n; ++nid )
        {
            MPI_Isend( _valid_counting[nid].data(),
                       Index::total * sizeof( int ), MPI_BYTE,
                       _neighbor_ranks[nid], mpi_tag_counting + _send_tags[nid],
                       comm, &counting_requests[nid + num_n] );
        }

        // wait until all counting data sending finished
        const int ec = MPI_Waitall( num_n, counting_requests.data() + num_n,
                                    MPI_STATUSES_IGNORE );

        // check if the counting communication succeed
        if ( MPI_SUCCESS != ec )
            throw std::logic_error(
                "Cabana::Grid::Experimental::SparseHalo::"
                "collectNeighborCounting: counting sending failed." );
        MPI_Barrier( comm );
    }

    /*!
       \brief collect all valid ranks for sparse grid scatter operations
       \param comm MPI communicator
       \param valid_sends neighbor array id that requires data from current rank
       \param valid_recvs neighbor array id the current ranks requires data from
       \param is_neighbor_counting_collected label if the neighbor has already
       been collected; if true, it means all neighbor counting information is
       up-to-date and there's no need for recollection
    */
    void scatterValidSendAndRecvRanks(
        MPI_Comm comm, std::vector<int>& valid_sends,
        std::vector<int>& valid_recvs,
        const bool is_neighbor_counting_collected = false ) const
    {
        // collect neighbor counting if needed
        collectNeighborCounting( comm, is_neighbor_counting_collected );

        // loop over all valid neighbors to check if there's data communication
        // needed
        for ( std::size_t nid = 0; nid < _neighbor_ranks.size(); ++nid )
        {
            // reference to counting in owned and ghosted share spaces
            auto h_counting = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), _valid_counting[nid] );
            auto h_neighbor_counting = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), _neighbor_counting[nid] );

            // if the current ghosted space and the neighbor's owned share
            // space is non-emty, we need to send data to the neighbor
            if ( !( h_counting( Index::ghost ) == 0 ||
                    h_neighbor_counting( Index::own ) == 0 ) )
            {
                valid_sends.push_back( nid );
            }
            // if the current owned share space and the neighbor's ghosted share
            // space is non-emty, the neighbor will send data to us and we need
            // to receive data accordingly
            if ( !( h_counting( Index::own ) == 0 ||
                    h_neighbor_counting( Index::ghost ) == 0 ) )
            {
                valid_recvs.push_back( nid );
            }
        }
    }

    /*!
       \brief collect all valid ranks for sparse grid gather operations
       \param comm MPI communicator
       \param valid_sends neighbor array id that requires data from current rank
       \param valid_recvs neighbor array id the current ranks requires data from
       \param is_neighbor_counting_collected label if the neighbor has already
       been collected; if true, it means all neighbor counting information is
       up-to-date and there's no need for recollection
    */
    void gatherValidSendAndRecvRanks(
        MPI_Comm comm, std::vector<int>& valid_sends,
        std::vector<int>& valid_recvs,
        const bool is_neighbor_counting_collected = false ) const
    {
        // collect neighbor counting if needed
        collectNeighborCounting( comm, is_neighbor_counting_collected );

        // loop over all valid neighbors to check if there's data communication
        // needed
        for ( std::size_t nid = 0; nid < _neighbor_ranks.size(); ++nid )
        {
            auto h_counting = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), _valid_counting[nid] );
            auto h_neighbor_counting = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), _neighbor_counting[nid] );

            // if the current owned space and the neighbor's ghosted share
            // space is non-emty, we need to send data to the neighbor
            if ( !( h_counting( Index::own ) == 0 ||
                    h_neighbor_counting( Index::ghost ) == 0 ) )
            {
                valid_sends.push_back( nid );
            }

            // if the current ghosted space and the neighbor's owned share
            // space is non-emty, we need to send data to the neighbor
            if ( !( h_counting( Index::ghost ) == 0 ||
                    h_neighbor_counting( Index::own ) == 0 ) )
            {
                valid_recvs.push_back( nid );
            }
        }
    }

    /*!
         \brief Gather data into our ghosted share space from their owners.
         \tparam ExecSpace execution space
         \tparam SparseArrayType sparse array type
         \tparam SparseMapType sparse map type
         \param exec_space execution space
         \param sparse_array sparse AoSoA array used to store grid data
         \param is_neighbor_counting_collected label if the neighbor has already
       been collected; if true, it means all neighbor counting information is
       up-to-date and there's no need for recollection
    */
    template <class ExecSpace, class SparseArrayType>
    void gather( const ExecSpace& exec_space, SparseArrayType& sparse_array,
                 const bool is_neighbor_counting_collected = false ) const
    {
        // return if no valid neighbor
        if ( 0 == _neighbor_ranks.size() )
            return;

        // Get the MPI communicator.
        auto comm = getComm( sparse_array );

        const auto& map = sparse_array.layout().sparseMap();

        // communicate "counting" among neighbors, to decide if the grid data
        // communication is needed
        std::vector<int> valid_sends;
        std::vector<int> valid_recvs;
        gatherValidSendAndRecvRanks( comm, valid_sends, valid_recvs,
                                     is_neighbor_counting_collected );
        MPI_Barrier( comm );

        // ------------------------------------------------------------------
        // communicate steering (array keys) for all valid sends and receives
        std::vector<MPI_Request> steering_requests(
            valid_recvs.size() + valid_sends.size(), MPI_REQUEST_NULL );
        const int mpi_tag_steering = 3214;

        // get the steering keys from valid neighbors to get all grids that
        // we need to receive
        // loop over all neighbors that will send data to the current rank
        for ( std::size_t i = 0; i < valid_recvs.size(); ++i )
        {
            int nid = valid_recvs[i];
            Kokkos::View<int[2], Kokkos::HostSpace> h_neighbor_counting(
                "tmp_host_neighbor_counting" );
            Kokkos::deep_copy( h_neighbor_counting, _neighbor_counting[nid] );

            MPI_Irecv( _tmp_tile_steering[nid].data(),
                       h_neighbor_counting( Index::own ) * sizeof( key_type ),
                       MPI_BYTE, _neighbor_ranks[nid],
                       mpi_tag_steering + _receive_tags[nid], comm,
                       &steering_requests[i] );
        }

        // send the steering keys to valid neighbors
        // loop over all neighbors that requires our owned data
        for ( std::size_t i = 0; i < valid_sends.size(); ++i )
        {
            int nid = valid_sends[i];
            Kokkos::View<int[2], Kokkos::HostSpace> h_counting(
                "tmp_host_counting" );
            Kokkos::deep_copy( h_counting, _valid_counting[nid] );

            MPI_Isend( _owned_tile_steering[nid].data(),
                       h_counting( Index::own ) * sizeof( key_type ), MPI_BYTE,
                       _neighbor_ranks[nid], mpi_tag_steering + _send_tags[nid],
                       comm, &steering_requests[i + valid_recvs.size()] );
        }

        // wait for all sending work finish
        const int ec_ss = MPI_Waitall(
            valid_sends.size(), steering_requests.data() + valid_recvs.size(),
            MPI_STATUSES_IGNORE );
        if ( MPI_SUCCESS != ec_ss )
            throw std::logic_error( "Cabana::Grid::Experimental::SparseHalo::"
                                    "gather: steering sending failed." );
        MPI_Barrier( comm );

        // ------------------------------------------------------------------
        // communicate sparse array data
        // Pick a tag to use for communication. This object has its own
        // communication space so any tag will do.
        std::vector<MPI_Request> requests(
            valid_recvs.size() + valid_sends.size(), MPI_REQUEST_NULL );
        const int mpi_tag = 2345;

        // post receives
        for ( std::size_t i = 0; i < valid_recvs.size(); ++i )
        {
            int nid = valid_recvs[i];
            Kokkos::View<int[2], Kokkos::HostSpace> h_neighbor_counting(
                "tmp_host_neighbor_counting" );
            ;
            Kokkos::deep_copy( h_neighbor_counting, _neighbor_counting[nid] );

            MPI_Irecv( _ghosted_buffers[nid].data(),
                       h_neighbor_counting( Index::own ) * cell_num_per_tile *
                           _soa_total_bytes,
                       MPI_BYTE, _neighbor_ranks[nid],
                       mpi_tag + _receive_tags[nid], comm, &requests[i] );
        }

        // pack send buffers and post sends
        for ( std::size_t i = 0; i < valid_sends.size(); ++i )
        {
            int nid = valid_sends[i];
            Kokkos::View<int[2], Kokkos::HostSpace> h_counting(
                "tmp_host_counting" );
            ;
            Kokkos::deep_copy( h_counting, _valid_counting[nid] );

            packBuffer( exec_space, _owned_buffers[nid],
                        _owned_tile_steering[nid], sparse_array,
                        h_counting( Index::own ) );
            Kokkos::fence();

            MPI_Isend(
                _owned_buffers[nid].data(),
                h_counting( Index::own ) * cell_num_per_tile * _soa_total_bytes,
                MPI_BYTE, _neighbor_ranks[nid], mpi_tag + _send_tags[nid], comm,
                &requests[i + valid_recvs.size()] );
        }

        // unpack receive buffers
        for ( std::size_t i = 0; i < valid_recvs.size(); ++i )
        {
            // get the next buffer to unpack
            int unpack_index = MPI_UNDEFINED;
            MPI_Waitany( valid_recvs.size(), requests.data(), &unpack_index,
                         MPI_STATUS_IGNORE );

            // in theory we should receive enough buffers to unpack
            // if not there could be some problems
            if ( MPI_UNDEFINED == unpack_index )
                std::runtime_error(
                    std::string( "Cabana::Grid::Experimental::SparseHalo::"
                                 "gather: data receiving failed, "
                                 "get only " ) +
                    std::to_string( i ) + ", need " +
                    std::to_string( valid_recvs.size() ) );
            // otherwise unpack the next buffer
            else
            {
                int nid = valid_recvs[unpack_index];
                auto h_neighbor_counting = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), _neighbor_counting[nid] );
                unpackBuffer( ScatterReduce::Replace(), exec_space,
                              _ghosted_buffers[nid], _tmp_tile_steering[nid],
                              sparse_array, map,
                              h_neighbor_counting( Index::own ) );
                Kokkos::fence();
            }
        }

        // wait to finish all send requests
        const int ec_data = MPI_Waitall( valid_sends.size(),
                                         requests.data() + valid_recvs.size(),
                                         MPI_STATUSES_IGNORE );
        if ( MPI_SUCCESS != ec_data )
            throw std::logic_error(
                "sparse_halo_gather: data sending failed." );

        // reinit steerings for next round of communication
        for ( std::size_t i = 0; i < _tmp_tile_steering.size(); ++i )
            Kokkos::deep_copy( _tmp_tile_steering[i], invalid_key );
        MPI_Barrier( comm );
    }

    /*!
        \brief Scatter data from our ghosts to their owners using the given type
     of reduce operation.
        \tparam ExecSpace execution space
        \tparam ReduceOp The type of reduction functor
        \tparam SparseArrayType sparse array type
        \tparam SparseMapType sparse map type
        \param exec_space execution space
        \param reduce_op The functor used to reduce the results
        \param sparse_array sparse AoSoA array used to store grid data
        \param is_neighbor_counting_collected label if the neighbor has already
       been collected; if true, it means all neighbor counting information is
       up-to-date and there's no need for recollection
    */
    template <class ExecSpace, class ReduceOp, class SparseArrayType>
    void scatter( const ExecSpace& exec_space, const ReduceOp& reduce_op,
                  SparseArrayType& sparse_array,
                  const bool is_neighbor_counting_collected = false ) const
    {
        // return if no valid neighbor
        if ( 0 == _neighbor_ranks.size() )
            return;

        // Get the MPI communicator.
        auto comm = getComm( sparse_array );

        const auto& map = sparse_array.layout().sparseMap();

        // communicate "counting" among neighbors, to decide if the grid data
        // transfer is needed
        std::vector<int> valid_sends;
        std::vector<int> valid_recvs;
        scatterValidSendAndRecvRanks( comm, valid_sends, valid_recvs,
                                      is_neighbor_counting_collected );
        MPI_Barrier( comm );

        // ------------------------------------------------------------------
        // communicate steering (array keys) for all valid sends and receives
        std::vector<MPI_Request> steering_requests(
            valid_recvs.size() + valid_sends.size(), MPI_REQUEST_NULL );
        const int mpi_tag_steering = 214;

        // get the steering keys from valid neighbors to know all grids that
        // we need to receive
        // loop over all neighbors that will send data to the current rank
        for ( std::size_t i = 0; i < valid_recvs.size(); ++i )
        {
            int nid = valid_recvs[i];
            Kokkos::View<int[2], Kokkos::HostSpace> h_neighbor_counting(
                "tmp_host_neighbor_counting" );
            ;
            Kokkos::deep_copy( h_neighbor_counting, _neighbor_counting[nid] );

            MPI_Irecv( _tmp_tile_steering[nid].data(),
                       h_neighbor_counting( Index::ghost ) * sizeof( key_type ),
                       MPI_BYTE, _neighbor_ranks[nid],
                       mpi_tag_steering + _receive_tags[nid], comm,
                       &steering_requests[i] );
        }

        // send the steering keys to valid neighbors
        // loop over all neighbors that requires our owned data
        for ( std::size_t i = 0; i < valid_sends.size(); ++i )
        {
            int nid = valid_sends[i];
            Kokkos::View<int[2], Kokkos::HostSpace> h_counting(
                "tmp_host_counting" );
            ;
            Kokkos::deep_copy( h_counting, _valid_counting[nid] );

            MPI_Isend( _ghosted_tile_steering[nid].data(),
                       h_counting( Index::ghost ) * sizeof( key_type ),
                       MPI_BYTE, _neighbor_ranks[nid],
                       mpi_tag_steering + _send_tags[nid], comm,
                       &steering_requests[i + valid_recvs.size()] );
        }

        // wait for all sending work finish
        const int ec_ss = MPI_Waitall(
            valid_sends.size(), steering_requests.data() + valid_recvs.size(),
            MPI_STATUSES_IGNORE );
        if ( MPI_SUCCESS != ec_ss )
            throw std::logic_error( "Cabana::Grid::Experimental::SparseHalo::"
                                    "scatter: steering sending failed." );
        MPI_Barrier( comm );

        // ------------------------------------------------------------------
        // communicate sparse array data
        // Pick a tag to use for communication. This object has its own
        // communication space so any tag will do.
        std::vector<MPI_Request> requests(
            valid_recvs.size() + valid_sends.size(), MPI_REQUEST_NULL );
        const int mpi_tag = 345;

        // post receives
        for ( std::size_t i = 0; i < valid_recvs.size(); ++i )
        {
            int nid = valid_recvs[i];
            Kokkos::View<int[2], Kokkos::HostSpace> h_neighbor_counting(
                "tmp_host_neighbor_counting" );
            ;
            Kokkos::deep_copy( h_neighbor_counting, _neighbor_counting[nid] );

            MPI_Irecv( _owned_buffers[nid].data(),
                       h_neighbor_counting( Index::ghost ) * cell_num_per_tile *
                           _soa_total_bytes,
                       MPI_BYTE, _neighbor_ranks[nid],
                       mpi_tag + _receive_tags[nid], comm, &requests[i] );
        }

        // pack send buffers and post sends
        for ( std::size_t i = 0; i < valid_sends.size(); ++i )
        {
            int nid = valid_sends[i];
            Kokkos::View<int[2], Kokkos::HostSpace> h_counting(
                "tmp_host_counting" );
            ;
            Kokkos::deep_copy( h_counting, _valid_counting[nid] );
            packBuffer( exec_space, _ghosted_buffers[nid],
                        _ghosted_tile_steering[nid], sparse_array,
                        h_counting( Index::ghost ) );
            Kokkos::fence();

            MPI_Isend( _ghosted_buffers[nid].data(),
                       h_counting( Index::ghost ) * cell_num_per_tile *
                           _soa_total_bytes,
                       MPI_BYTE, _neighbor_ranks[nid],
                       mpi_tag + _send_tags[nid], comm,
                       &requests[i + valid_recvs.size()] );
        }

        // unpack receive buffers
        for ( std::size_t i = 0; i < valid_recvs.size(); ++i )
        {
            // get the next buffer to unpack
            int unpack_index = MPI_UNDEFINED;
            MPI_Waitany( valid_recvs.size(), requests.data(), &unpack_index,
                         MPI_STATUS_IGNORE );

            // in theory we should receive enough buffers to unpack
            // if not there could be some problems
            if ( MPI_UNDEFINED == unpack_index )
                std::runtime_error(
                    std::string( "sparse_halo_scatter: data receiving failed, "
                                 "get only " ) +
                    std::to_string( i ) + ", need " +
                    std::to_string( valid_recvs.size() ) );
            // otherwise unpack the next buffer with the given reduce operator
            else
            {
                int nid = valid_recvs[unpack_index];
                auto h_neighbor_counting = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), _neighbor_counting[nid] );
                unpackBuffer( reduce_op, exec_space, _owned_buffers[nid],
                              _tmp_tile_steering[nid], sparse_array, map,
                              h_neighbor_counting( Index::ghost ) );
                Kokkos::fence();
            }
        }

        // wait to finish all send requests
        const int ec_data = MPI_Waitall( valid_sends.size(),
                                         requests.data() + valid_recvs.size(),
                                         MPI_STATUSES_IGNORE );
        if ( MPI_SUCCESS != ec_data )
            throw std::logic_error( "Cabana::Grid::Experimental::SparseHalo::"
                                    "scatter: data sending failed." );

        // reinit steerings for next round of communication
        for ( std::size_t i = 0; i < _tmp_tile_steering.size(); ++i )
            Kokkos::deep_copy( _tmp_tile_steering[i], invalid_key );
        MPI_Barrier( comm );
    }

    //---------------------------------------------------------------------------//
    /*!
        \brief Pack sparse arrays at halo regions into a buffer
        \tparam ExecSpace execution space type
        \tparam SparseArrayType sparse array type
        \param exec_space execution space
        \param buffer buffer to store sparse array data and to communicate
        \param tile_steering Kokkos view to store halo tile keys
        \param sparse_array sparse array (all sparse grids on current rank)
        \param count number of halo grids to pack
    */
    template <class ExecSpace, class SparseArrayType>
    void packBuffer( const ExecSpace& exec_space, const buffer_view& buffer,
                     const steering_view& tile_steering,
                     SparseArrayType& sparse_array, const int count ) const
    {
        Kokkos::parallel_for(
            "Cabana::Grid::Experimental::SparseHalo::packBuffer",
            Kokkos::RangePolicy<ExecSpace>( exec_space, 0, count ),
            KOKKOS_LAMBDA( const int i ) {
                if ( tile_steering( i ) != invalid_key )
                {
                    const int buffer_idx = i * cell_num_per_tile;
                    auto tile_key = tile_steering( i );
                    for ( int lcid = 0; lcid < (int)cell_num_per_tile; ++lcid )
                    {
                        buffer( buffer_idx + lcid ) =
                            sparse_array.getTuple( tile_key, lcid );
                    }
                }
            } );
    }

    //---------------------------------------------------------------------------//
    //! Reduce an element into the buffer. Sum reduction.
    template <class T>
    KOKKOS_INLINE_FUNCTION static void
    unpackOp( ScatterReduce::Sum, const T& buffer_val, T& array_val )
    {
        array_val += buffer_val;
    }

    //! Reduce an element into the buffer. Min reduction.
    template <class T>
    KOKKOS_INLINE_FUNCTION static void
    unpackOp( ScatterReduce::Min, const T& buffer_val, T& array_val )
    {
        if ( buffer_val < array_val )
            array_val = buffer_val;
    }

    //! Reduce an element into the buffer. Max reduction.
    template <class T>
    KOKKOS_INLINE_FUNCTION static void
    unpackOp( ScatterReduce::Max, const T& buffer_val, T& array_val )
    {
        if ( buffer_val > array_val )
            array_val = buffer_val;
    }

    //! Reduce an element into the buffer. Replace reduction.
    template <class T>
    KOKKOS_INLINE_FUNCTION static void
    unpackOp( ScatterReduce::Replace, const T& buffer_val, T& array_val )
    {
        array_val = buffer_val;
    }

    //---------------------------------------------------------------------------//
    /*!
        \brief Unpack a sparse arrays element (a tuple) in a buffer
        (for case tuple members with rank == 3)
        \tparam ReduceOp reduce functor type
        \tparam N element ID inside a SoA tuple (N-th data member)
        \tparam M rank number of the current element (N-th data member)
        \tparam SoAType SoA type in sparse array (which is an AoSoA)
        \param reduce_op reduce operation
        \param src_tuple source tuple
        \param dst_soa destination SoA to store copied data
        \param soa_idx tuple index inside the destination SoA
        \param extents element member extents in all ranks
    */
    template <class ReduceOp, std::size_t N, std::size_t M, class SoAType>
    KOKKOS_FORCEINLINE_FUNCTION static std::enable_if_t<3 == M, void>
    unpackTupleMember( const ReduceOp& reduce_op, const tuple_type& src_tuple,
                       SoAType& dst_soa, const int soa_idx,
                       const Kokkos::Array<std::size_t, M>& extents,
                       const std::integral_constant<std::size_t, N>,
                       const std::integral_constant<std::size_t, M> )
    {
        for ( std::size_t d0 = 0; d0 < extents[0]; ++d0 )
            for ( int d1 = 0; d1 < extents[1]; ++d1 )
                for ( int d2 = 0; d2 < extents[2]; ++d2 )
                {
                    unpackOp( reduce_op,
                              Cabana::get<N>( src_tuple, d0, d1, d2 ),
                              Cabana::get<N>( dst_soa, soa_idx, d0, d1, d2 ) );
                }
    }

    /*!
        \brief Unpack a sparse arrays element (a tuple) in a buffer
        (for case tuple members with rank == 2)
        \tparam ReduceOp reduce functor type
        \tparam N element ID inside a SoA tuple (N-th data member)
        \tparam M rank number of the current element (N-th data member)
        \tparam SoAType SoA type in sparse array (which is an AoSoA)
        \param reduce_op reduce operation
        \param src_tuple source tuple
        \param dst_soa destination SoA to store copied data
        \param soa_idx tuple index inside the destination SoA
        \param extents element member extents in all ranks
    */
    template <class ReduceOp, std::size_t N, std::size_t M, class SoAType>
    KOKKOS_FORCEINLINE_FUNCTION static std::enable_if_t<2 == M, void>
    unpackTupleMember( const ReduceOp& reduce_op, const tuple_type& src_tuple,
                       SoAType& dst_soa, const int soa_idx,
                       const Kokkos::Array<std::size_t, M>& extents,
                       const std::integral_constant<std::size_t, N>,
                       const std::integral_constant<std::size_t, M> )
    {
        for ( std::size_t d0 = 0; d0 < extents[0]; ++d0 )
            for ( int d1 = 0; d1 < extents[1]; ++d1 )
            {
                unpackOp( reduce_op, Cabana::get<N>( src_tuple, d0, d1 ),
                          Cabana::get<N>( dst_soa, soa_idx, d0, d1 ) );
            }
    }

    /*!
        \brief Unpack a sparse arrays element (a tuple) in a buffer
        (for case tuple members with rank == 1)
        \tparam ReduceOp reduce functor type
        \tparam N element ID inside a SoA tuple (N-th data member)
        \tparam M rank number of the current element (N-th data member)
        \tparam SoAType SoA type in sparse array (which is an AoSoA)
        \param reduce_op reduce operation
        \param src_tuple source tuple
        \param dst_soa destination SoA to store copied data
        \param soa_idx tuple index inside the destination SoA
        \param extents element member extents in all ranks
    */
    template <class ReduceOp, std::size_t N, std::size_t M, class SoAType>
    KOKKOS_FORCEINLINE_FUNCTION static std::enable_if_t<1 == M, void>
    unpackTupleMember( const ReduceOp& reduce_op, const tuple_type& src_tuple,
                       SoAType& dst_soa, const int soa_idx,
                       const Kokkos::Array<std::size_t, M>& extents,
                       const std::integral_constant<std::size_t, N>,
                       const std::integral_constant<std::size_t, M> )
    {
        for ( std::size_t d0 = 0; d0 < extents[0]; ++d0 )
        {
            unpackOp( reduce_op, Cabana::get<N>( src_tuple, d0 ),
                      Cabana::get<N>( dst_soa, soa_idx, d0 ) );
        }
    }

    /*!
        \brief Unpack a sparse arrays element (a tuple) in a buffer
        (for case tuple members with rank == 0)
        \tparam ReduceOp reduce functor type
        \tparam N element ID inside a SoA tuple (N-th data member)
        \tparam M rank number of the current element (N-th data member)
        \tparam SoAType SoA type in sparse array (which is an AoSoA)
        \param reduce_op reduce operation
        \param src_tuple source tuple
        \param dst_soa destination SoA to store copied data
        \param soa_idx tuple index inside the destination SoA
    */
    template <class ReduceOp, std::size_t N, std::size_t M, class SoAType>
    KOKKOS_FORCEINLINE_FUNCTION static std::enable_if_t<0 == M, void>
    unpackTupleMember( const ReduceOp& reduce_op, const tuple_type& src_tuple,
                       SoAType& dst_soa, const int soa_idx,
                       const Kokkos::Array<std::size_t, M>&,
                       const std::integral_constant<std::size_t, N>,
                       const std::integral_constant<std::size_t, M> )
    {
        unpackOp( reduce_op, Cabana::get<N>( src_tuple ),
                  Cabana::get<N>( dst_soa, soa_idx ) );
    }

    /*!
        \brief Unpack a sparse arrays tuple for it's member with index 0
        \tparam ReduceOp reduce functor type
        \tparam SoAType SoA type in sparse array (which is an AoSoA)
        \param reduce_op reduce operation
        \param src_tuple source tuple
        \param dst_soa destination SoA to store copied data
        \param soa_idx tuple index inside the destination SoA
    */
    template <class ReduceOp, class SoAType>
    KOKKOS_FORCEINLINE_FUNCTION static void
    unpackTuple( const ReduceOp& reduce_op, const tuple_type& src_tuple,
                 SoAType& dst_soa, const int soa_idx,
                 const std::integral_constant<std::size_t, 0> )
    {
        using current_type = member_data_type<0>;
        auto extents = compute_member_extents<current_type>();
        unpackTupleMember(
            reduce_op, src_tuple, dst_soa, soa_idx, extents,
            std::integral_constant<std::size_t, 0>(),
            std::integral_constant<std::size_t,
                                   std::rank<current_type>::value>() );
    }

    /*!
        \brief Unpack a sparse arrays tuple for all members when element ID!=0
        \tparam ReduceOp reduce functor type
        \tparam SoAType SoA type in sparse array (which is an AoSoA)
        \tparam N Unpack N-th data member in this call
        \param reduce_op reduce operation
        \param src_tuple source tuple
        \param dst_soa destination SoA to store copied data
        \param soa_idx tuple index inside the destination SoA
    */
    template <class ReduceOp, std::size_t N, class SoAType>
    KOKKOS_FORCEINLINE_FUNCTION static void
    unpackTuple( const ReduceOp& reduce_op, const tuple_type& src_tuple,
                 SoAType& dst_soa, const int soa_idx,
                 const std::integral_constant<std::size_t, N> )
    {
        using current_type = member_data_type<N>;
        auto extents = compute_member_extents<current_type>();
        unpackTupleMember(
            reduce_op, src_tuple, dst_soa, soa_idx, extents,
            std::integral_constant<std::size_t, N>(),
            std::integral_constant<std::size_t,
                                   std::rank<current_type>::value>() );

        if ( N > 1 )
        {
            // recurcively unpack the next tuple element
            unpackTuple( reduce_op, src_tuple, dst_soa, soa_idx,
                         std::integral_constant<std::size_t, N - 1>() );
        }
        else
        {
            unpackTuple( reduce_op, src_tuple, dst_soa, soa_idx,
                         std::integral_constant<std::size_t, 0>() );
        }
    }

    /*!
        \brief Unpack a sparse array communication buffer
        \tparam ReduceOp reduce functor type
        \tparam ExecSpace execution space type
        \tparam SparseArrayType sparse array type
        \tparam SparseMapType sparse map type
        \param reduce_op reduce operation
        \param exec_space execution space
        \param buffer buffer to store sparse array data and to communicate
        \param tile_steering Kokkos view to store halo tile keys
        \param sparse_array sparse array (all sparse grids on current rank)
        \param map sparse map that has valid grids registered
        \param count number of halo grids to unpack
    */
    template <class ReduceOp, class ExecSpace, class SparseArrayType,
              class SparseMapType>
    void unpackBuffer( const ReduceOp& reduce_op, const ExecSpace& exec_space,
                       const buffer_view& buffer,
                       const steering_view& tile_steering,
                       const SparseArrayType& sparse_array, SparseMapType& map,
                       const int count ) const
    {
        Kokkos::parallel_for(
            "Cabana::Grid::Experimental::SparseHalo::unpackBuffer",
            Kokkos::RangePolicy<ExecSpace>( exec_space, 0, count ),
            KOKKOS_LAMBDA( const int i ) {
                if ( tile_steering( i ) != invalid_key )
                {
                    auto tile_key = tile_steering( i );
                    if ( map.isValidKey( tile_key ) )
                    {
                        int ti, tj, tk;
                        map.key2ijk( tile_key, ti, tj, tk );

                        auto tile_id = map.queryTileFromTileKey( tile_key );
                        const int buffer_idx = i * cell_num_per_tile;
                        for ( int lcid = 0; lcid < (int)cell_num_per_tile;
                              ++lcid )
                        {
                            auto& tuple = buffer( buffer_idx + lcid );
                            auto& data_access =
                                sparse_array.accessTile( tile_id );
                            unpackTuple(
                                reduce_op, tuple, data_access, lcid,
                                std::integral_constant<std::size_t,
                                                       member_num - 1>() );
                        }
                    }
                }
            } );
    }

  private:
    // These functions may be useful if added to Cabana_MemberType.hpp
    // compute member size
    template <std::size_t M>
    static constexpr std::size_t compute_member_size()
    {
        return sizeof( member_data_type<M> );
    }

    template <typename Sequence>
    struct compute_member_size_list_impl;

    template <std::size_t... Is>
    struct compute_member_size_list_impl<std::index_sequence<Is...>>
    {
        std::array<std::size_t, member_num> operator()()
        {
            return { compute_member_size<Is>()... };
        }
    };

    template <std::size_t N = member_num,
              typename Indices = std::make_index_sequence<N>>
    std::array<std::size_t, member_num> compute_member_size_list()
    {
        compute_member_size_list_impl<Indices> op;
        return op();
    }

    // member extent
    template <typename Type, std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION static constexpr std::size_t
    compute_one_member_extent()
    {
        return std::extent<Type, M>::value;
    }

    template <class Type, std::size_t M, typename Sequence>
    struct compute_member_extents_impl;

    template <class Type, std::size_t M, std::size_t... Is>
    struct compute_member_extents_impl<Type, M, std::index_sequence<Is...>>
    {
        KOKKOS_FORCEINLINE_FUNCTION
        Kokkos::Array<std::size_t, M> operator()()
        {
            return { compute_one_member_extent<Type, Is>()... };
        }
    };

    template <class Type, std::size_t M = std::rank<Type>::value,
              typename Indices = std::make_index_sequence<M>>
    KOKKOS_FORCEINLINE_FUNCTION static Kokkos::Array<std::size_t, M>
    compute_member_extents()
    {
        compute_member_extents_impl<Type, M, Indices> op;
        return op();
    }

  private:
    // Current MPI linear rank ID
    int _self_rank;
    // Hallo pattern
    halo_pattern_type _pattern;

    // neighbor rank linear MPI rank IDs
    std::vector<int> _neighbor_ranks;
    // valid neigber rank indices; valid means require data communication
    std::vector<std::array<int, num_space_dim>> _valid_neighbor_ids;
    // sending tags
    std::vector<int> _send_tags;
    // receiving tags
    std::vector<int> _receive_tags;

    // owned view buffers
    std::vector<buffer_view> _owned_buffers;
    // ghosted view buffers
    std::vector<buffer_view> _ghosted_buffers;

    // owned tile key steerings
    std::vector<steering_view> _owned_tile_steering;
    // key steering buffers (used to store valid keys get from neighbors)
    std::vector<steering_view> _tmp_tile_steering;
    // ghosted tile key steerings
    std::vector<steering_view> _ghosted_tile_steering;

    // valid halo grid counting on current rank (each element map to a neighbor)
    std::vector<counting_view> _valid_counting;
    // valid halo grid counting on corresponding neighbor ranks
    std::vector<counting_view> _neighbor_counting;

    // owned tile space
    std::vector<tile_index_space> _owned_tile_spaces;
    // ghosted tile space
    std::vector<tile_index_space> _ghosted_tile_spaces;

    // SoA member bytes num
    Kokkos::Array<std::size_t, member_num> _soa_member_bytes;
    // SoA total bytes count
    std::size_t _soa_total_bytes;
};

//---------------------------------------------------------------------------//
// Sparse halo creation.
//---------------------------------------------------------------------------//
/*!
  \brief SparseHalo creation function.
  \param pattern The pattern to build the sparse halo from.
  \param array The sparse array over which to build the halo.
*/
template <class MemorySpace, unsigned long long cellBitsPerTileDim,
          class DataTypes, class EntityType, class MeshType,
          class SparseMapType, class Pattern, typename Value = int,
          typename Key = uint64_t>
auto createSparseHalo(
    const Pattern& pattern,
    const std::shared_ptr<SparseArray<DataTypes, MemorySpace, EntityType,
                                      MeshType, SparseMapType>>
        array )
{
    using array_type = SparseArray<DataTypes, MemorySpace, EntityType, MeshType,
                                   SparseMapType>;
    using memory_space = typename array_type::memory_space;
    static constexpr std::size_t num_space_dim = array_type::num_space_dim;
    return std::make_shared<
        SparseHalo<memory_space, DataTypes, EntityType, num_space_dim,
                   cellBitsPerTileDim, Value, Key>>( pattern, array );
}

} // namespace Experimental
} // namespace Grid
} // namespace Cabana

#endif // CABANA_GRID_SPARSEHALO_HPP
