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

/*!
  \file Cajita_DynamicPartitioner.hpp
  \brief Multi-node dynamic grid partitioner
*/
#ifndef CAJITA_DYNAMICPARTITIONER_HPP
#define CAJITA_DYNAMICPARTITIONER_HPP

#include <Cajita_Partitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <vector>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
/*!
  Dynamic mesh block partitioner. (Current Version: Support 3D only)
  \tparam Device Kokkos device type.
  \tparam CellPerTileDim Cells per tile per dimension.
  \tparam NumSpaceDim Dimemsion (The current version support 3D only)
*/
template <typename Device, unsigned long long CellPerTileDim = 4,
          std::size_t NumSpaceDim = 3>
class DynamicPartitioner : public BlockPartitioner<NumSpaceDim>
{
  public:
    //! dimension
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    //! Kokkos device type.
    using device_type = Device;
    //! Kokkos memory space.
    using memory_space = typename Device::memory_space;
    //! Kokkos execution space.
    using execution_space = typename Device::execution_space;

    //! Workload device view.
    using workload_view = Kokkos::View<int***, memory_space>;
    //! Partition device view.
    using partition_view = Kokkos::View<int* [num_space_dim], memory_space>;
    //! Workload host view.
    using workload_view_host =
        Kokkos::View<int***, typename execution_space::array_layout,
                     Kokkos::HostSpace>;
    //! Partition host view.
    using partition_view_host =
        Kokkos::View<int* [num_space_dim],
                     typename execution_space::array_layout, Kokkos::HostSpace>;

    //! Number of bits (per dimension) needed to index the cells inside a tile
    static constexpr unsigned long long cell_bits_per_tile_dim =
        bitCount( CellPerTileDim );
    //! Number of cells inside each tile (per dimension)
    //! Tile size reset to power of 2
    static constexpr unsigned long long cell_num_per_tile_dim =
        1 << cell_bits_per_tile_dim;

    /*!
      \brief Constructor - automatically compute ranks_per_dim from MPI
      communicator
      \param comm MPI communicator to decide the rank nums in each dimension
      \param max_workload_coeff threshold factor for re-partition
      \param workload_num total workload(particle/tile) number, used to compute
      workload_threshold
      \param num_step_rebalance the simulation step number after which one
      should check if repartition is needed
      \param global_cells_per_dim 3D array, global cells in each dimension
      \param max_optimize_iteration max iteration number to run the optimization
    */
    DynamicPartitioner(
        MPI_Comm comm, float max_workload_coeff, int workload_num,
        int num_step_rebalance,
        const std::array<int, num_space_dim>& global_cells_per_dim,
        int max_optimize_iteration = 10 )
        : _workload_threshold(
              static_cast<int>( max_workload_coeff * workload_num ) )
        , _num_step_rebalance( num_step_rebalance )
        , _max_optimize_iteration( max_optimize_iteration )
    {
        // compute the ranks_per_dim from MPI communicator
        allocate( global_cells_per_dim );
        ranksPerDimension( comm );
    }

    /*!
      \brief Constructor - user-defined ranks_per_dim
      communicator
      \param comm MPI communicator to decide the rank nums in each dimension
      \param max_workload_coeff threshold factor for re-partition
      \param workload_num total workload(particle/tile) number, used to compute
      workload_threshold
      \param num_step_rebalance the simulation step number after which one
      should check if repartition is needed
      \param ranks_per_dim 3D array, user-defined MPI rank constrains in per
      dimension
      \param global_cells_per_dim 3D array, global cells in each dimension
      \param max_optimize_iteration max iteration number to run the optimization
    */
    DynamicPartitioner(
        MPI_Comm comm, float max_workload_coeff, int workload_num,
        int num_step_rebalance,
        const std::array<int, num_space_dim>& ranks_per_dim,
        const std::array<int, num_space_dim>& global_cells_per_dim,
        int max_optimize_iteration = 10 )
        : _workload_threshold(
              static_cast<int>( max_workload_coeff * workload_num ) )
        , _num_step_rebalance( num_step_rebalance )
        , _max_optimize_iteration( max_optimize_iteration )
    {
        allocate( global_cells_per_dim );
        std::copy( ranks_per_dim.begin(), ranks_per_dim.end(),
                   _ranks_per_dim.data() );

        // init MPI topology
        int comm_size;
        MPI_Comm_size( comm, &comm_size );
        MPI_Dims_create( comm_size, num_space_dim, _ranks_per_dim.data() );
    }

    /*!
      \brief Compute the number of MPI ranks in each dimension of the grid
      from the given MPI communicator
      \param comm The communicator to use for the partitioning
    */
    std::array<int, num_space_dim> ranksPerDimension( MPI_Comm comm )
    {
        int comm_size;
        MPI_Comm_size( comm, &comm_size );

        std::array<int, num_space_dim> ranks_per_dim;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            ranks_per_dim[d] = 0;
        MPI_Dims_create( comm_size, num_space_dim, ranks_per_dim.data() );

        std::copy( ranks_per_dim.begin(), ranks_per_dim.end(),
                   _ranks_per_dim.data() );

        return ranks_per_dim;
    }

    /*!
      \brief Get the number of MPI ranks in each dimension of the grid
      from the given MPI communicator
      \param comm The communicator to use for the partitioning
    */
    std::array<int, num_space_dim>
    ranksPerDimension( MPI_Comm comm,
                       const std::array<int, num_space_dim>& ) const override
    {
        std::array<int, num_space_dim> ranks_per_dim = {
            _ranks_per_dim[0], _ranks_per_dim[1], _ranks_per_dim[2] };
        int comm_size;
        MPI_Comm_size( comm, &comm_size );
        int nrank = 1;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            nrank *= _ranks_per_dim[d];
        if ( comm_size != nrank )
            throw std::runtime_error(
                "DynamicPartitioner ranks do not match comm size" );
        return ranks_per_dim;
    }

    /*!
      \brief Get the tile number in each dimension owned by the current MPI rank
      \param cart_comm MPI cartesian communicator
    */
    std::array<int, num_space_dim>
    ownedTilesPerDimension( MPI_Comm cart_comm ) const
    {
        // Get the Cartesian topology index of this rank.
        std::array<int, num_space_dim> cart_rank;
        int linear_rank;
        MPI_Comm_rank( cart_comm, &linear_rank );
        MPI_Cart_coords( cart_comm, linear_rank, num_space_dim,
                         cart_rank.data() );

        // Get the tiles per dimension and the remainder.
        std::array<int, num_space_dim> tiles_per_dim;
        auto rec_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _rectangle_partition_dev );
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            tiles_per_dim[d] = rec_mirror( cart_rank[d] + 1, d ) -
                               rec_mirror( cart_rank[d], d );
        return tiles_per_dim;
    }

    /*!
      \brief Get the cell number in each dimension owned by the current MPI rank
      \param cart_comm MPI cartesian communicator
    */
    std::array<int, num_space_dim>
    ownedCellsPerDimension( MPI_Comm cart_comm ) const
    {
        auto tiles_per_dim = ownedTilesPerDimension( cart_comm );
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            // compute cells_per_dim from tiles_per_dim
            tiles_per_dim[d] <<= cell_bits_per_tile_dim;
        }
        return tiles_per_dim;
    }

    /*!
       \brief Get the owned number of tiles and the global tile offset of the
       current MPI rank.
       \param cart_comm The MPI Cartesian communicator for the partitioning.
       \param owned_num_tile (Return) The owned number of tiles of the current
       MPI rank in each dimension.
       \param global_tile_offset (Return) The global tile offset of the current
       MPI rank in each dimension
     */
    void
    ownedTileInfo( MPI_Comm cart_comm,
                   std::array<int, num_space_dim>& owned_num_tile,
                   std::array<int, num_space_dim>& global_tile_offset ) const
    {
        // Get the Cartesian topology index of this rank.
        std::array<int, num_space_dim> cart_rank;
        int linear_rank;
        MPI_Comm_rank( cart_comm, &linear_rank );
        MPI_Cart_coords( cart_comm, linear_rank, num_space_dim,
                         cart_rank.data() );

        // Get the tiles per dimension and the remainder.
        auto rec_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _rectangle_partition_dev );
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            owned_num_tile[d] = rec_mirror( cart_rank[d] + 1, d ) -
                                rec_mirror( cart_rank[d], d );
            global_tile_offset[d] = rec_mirror( cart_rank[d], d );
        }
    }

    /*!
      \brief Get the owned number of cells and the global cell offset of the
      current MPI rank.
      \param cart_comm The MPI Cartesian communicator for the partitioning.
      \param owned_num_cell (Return) The owned number of cells of the current
      MPI rank in each dimension.
      \param global_cell_offset (Return) The global cell offset of the current
      MPI rank in each dimension
    */
    void ownedCellInfo(
        MPI_Comm cart_comm, const std::array<int, num_space_dim>&,
        std::array<int, num_space_dim>& owned_num_cell,
        std::array<int, num_space_dim>& global_cell_offset ) const override
    {
        ownedTileInfo( cart_comm, owned_num_cell, global_cell_offset );
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            // compute cells_per_dim from tiles_per_dim
            owned_num_cell[d] <<= cell_bits_per_tile_dim;
            global_cell_offset[d] <<= cell_bits_per_tile_dim;
        }
    }

    /*!
      \brief Initialize the tile partition; partition in each dimension
      has the form [0, p_1, ..., p_n, total_tile_num], so the partition
      would be [0, p_1), [p_1, p_2) ... [p_n, total_tile_num]
      \param rec_partition_i partition array in dimension i
      \param rec_partition_j partition array in dimension j
      \param rec_partition_k partition array in dimension k
    */
    void initializeRecPartition( std::vector<int>& rec_partition_i,
                                 std::vector<int>& rec_partition_j,
                                 std::vector<int>& rec_partition_k )
    {

        int max_size = 0;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            max_size =
                max_size < _ranks_per_dim[d] ? _ranks_per_dim[d] : max_size;

        typedef typename execution_space::array_layout layout;
        Kokkos::View<int* [num_space_dim], layout, Kokkos::HostSpace>
            rectangle_partition( "rectangle_partition_host", max_size + 1 );

        for ( int id = 0; id < _ranks_per_dim[0] + 1; ++id )
            rectangle_partition( id, 0 ) = rec_partition_i[id];

        for ( int id = 0; id < _ranks_per_dim[1] + 1; ++id )
            rectangle_partition( id, 1 ) = rec_partition_j[id];

        for ( int id = 0; id < _ranks_per_dim[2] + 1; ++id )
            rectangle_partition( id, 2 ) = rec_partition_k[id];

        _rectangle_partition_dev = Kokkos::create_mirror_view_and_copy(
            memory_space(), rectangle_partition );
    }

    /*!
      \brief Get the current partition.
      Copy partition from the device view to host std::array<vector>
    */
    std::array<std::vector<int>, num_space_dim> getCurrentPartition()
    {
        std::array<std::vector<int>, num_space_dim> rec_part;
        auto rec_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _rectangle_partition_dev );
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            rec_part[d].resize( _ranks_per_dim[d] + 1 );
            for ( int id = 0; id < _ranks_per_dim[d] + 1; ++id )
            {
                rec_part[d][id] = rec_mirror( id, d );
            }
        }
        return rec_part;
    }

    /*!
         \brief set all elements in _workload_per_tile and _workload_prefix_sum
         matrix to 0
       */
    void resetWorkload()
    {
        Kokkos::deep_copy( _workload_per_tile, 0 );
        Kokkos::deep_copy( _workload_prefix_sum, 0 );
    }

    /*!
      \brief compute the workload in the current MPI rank from particle
      positions (each particle count for 1 workload value)
      \param view particle positions view
      \param particle_num total particle number
      \param global_lower_corner the coordinate of the domain global lower
      corner
      \param dx cell dx size
      \param comm MPI communicator used for workload reduction
    */
    template <class ParticlePosViewType, typename ArrayType, typename CellUnit>
    void setLocalWorkloadByParticles( const ParticlePosViewType& view,
                           int particle_num,
                           const ArrayType& global_lower_corner,
                           const CellUnit dx, MPI_Comm comm )
    {
        resetWorkload();
        // make a local copy
        auto workload = _workload_per_tile;
        Kokkos::Array<CellUnit, num_space_dim> lower_corner;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            lower_corner[d] = global_lower_corner[d];
        }

        Kokkos::parallel_for(
            "compute_local_workload_parpos",
            Kokkos::RangePolicy<execution_space>( 0, particle_num ),
            KOKKOS_LAMBDA( const int i ) {
                int ti = static_cast<int>(
                             ( view( i, 0 ) - lower_corner[0] ) / dx - 0.5 ) >>
                         cell_bits_per_tile_dim;
                int tj = static_cast<int>(
                             ( view( i, 1 ) - lower_corner[1] ) / dx - 0.5 ) >>
                         cell_bits_per_tile_dim;
                int tz = static_cast<int>(
                             ( view( i, 2 ) - lower_corner[2] ) / dx - 0.5 ) >>
                         cell_bits_per_tile_dim;
                Kokkos::atomic_increment( &workload( ti + 1, tj + 1, tz + 1 ) );
            } );
        Kokkos::fence();
        // Wait for other ranks' workload to be ready
        MPI_Barrier( comm );
    }

    /*!
      \brief compute the workload in the current MPI rank from sparseMap
      (the workload of a tile is 1 if the tile is occupied, 0 otherwise)
      \param sparseMap sparseMap in the current rank
      \param comm MPI communicator used for workload reduction
    */
    template <class SparseMapType>
    void setLocalWorkloadBySparseMap( const SparseMapType& sparseMap, MPI_Comm comm )
    {
        resetWorkload();
        // make a local copy
        auto workload = _workload_per_tile;
        Kokkos::parallel_for(
            "compute_local_workload_sparsmap",
            Kokkos::RangePolicy<execution_space>( 0, sparseMap.capacity() ),
            KOKKOS_LAMBDA( uint32_t i ) {
                if ( sparseMap.valid_at( i ) )
                {
                    auto key = sparseMap.key_at( i );
                    int ti, tj, tk;
                    sparseMap.key2ijk( key, ti, tj, tk );
                    Kokkos::atomic_increment(
                        &workload( ti + 1, tj + 1, tk + 1 ) );
                }
            } );
        Kokkos::fence();
        // Wait for other ranks' workload to be ready
        MPI_Barrier( comm );
    }

    /*!
      \brief 1. reduce the total workload in all MPI ranks; 2. compute the
      workload prefix sum matrix for all MPI ranks
      \param comm MPI communicator used for workload reduction
    */
    void computeFullPrefixSum( MPI_Comm comm )
    {
        // local copy
        auto workload = _workload_per_tile;
        auto prefix_sum = _workload_prefix_sum;
        int total_size = _workload_per_tile.size();

        // MPI all reduce: compute workload in all MPI ranks from the local
        // workload matrix, save the results in _workload_prefix_sum
        MPI_Allreduce( workload.data(), prefix_sum.data(), total_size, MPI_INT,
                       MPI_SUM, comm );
        MPI_Barrier( comm );

        // compute the prefix sum (in three dimensions)
        // prefix sum in the dimension 0
        for ( int j = 0;
              j < static_cast<int>( _workload_prefix_sum.extent( 1 ) ); ++j )
            for ( int k = 0;
                  k < static_cast<int>( _workload_prefix_sum.extent( 2 ) );
                  ++k )
                Kokkos::parallel_scan(
                    "scan_prefix_sum_dim0",
                    Kokkos::RangePolicy<execution_space>(
                        0, _workload_prefix_sum.extent( 0 ) ),
                    KOKKOS_LAMBDA( const int i, int& update,
                                   const bool final ) {
                        const float val_i = prefix_sum( i, j, k );
                        update += val_i;
                        if ( final )
                        {
                            prefix_sum( i, j, k ) = update;
                        }
                    } );
        Kokkos::fence();

        // prefix sum in the dimension 1
        for ( int i = 0;
              i < static_cast<int>( _workload_prefix_sum.extent( 0 ) ); ++i )
            for ( int k = 0;
                  k < static_cast<int>( _workload_prefix_sum.extent( 2 ) );
                  ++k )
                Kokkos::parallel_scan(
                    "scan_prefix_sum_dim1",
                    Kokkos::RangePolicy<execution_space>(
                        0, _workload_prefix_sum.extent( 1 ) ),
                    KOKKOS_LAMBDA( const int j, int& update,
                                   const bool final ) {
                        const float val_i = prefix_sum( i, j, k );
                        update += val_i;
                        if ( final )
                        {
                            prefix_sum( i, j, k ) = update;
                        }
                    } );
        Kokkos::fence();

        // prefix sum in the dimension 2
        for ( int i = 0;
              i < static_cast<int>( _workload_prefix_sum.extent( 0 ) ); ++i )
            for ( int j = 0;
                  j < static_cast<int>( _workload_prefix_sum.extent( 1 ) );
                  ++j )
                Kokkos::parallel_scan(
                    "scan_prefix_sum_dim2",
                    Kokkos::RangePolicy<execution_space>(
                        0, _workload_prefix_sum.extent( 2 ) ),
                    KOKKOS_LAMBDA( const int k, int& update,
                                   const bool final ) {
                        const float val_i = prefix_sum( i, j, k );
                        update += val_i;
                        if ( final )
                        {
                            prefix_sum( i, j, k ) = update;
                        }
                    } );
        Kokkos::fence();
    }

    /*!
      \brief iteratively optimize the partition
      \param comm MPI communicator used for workload reduction
      \return iteration number
    */
    int optimizePartition( MPI_Comm comm )
    {
        computeFullPrefixSum( comm );
        MPI_Barrier( comm );

        // each iteration covers partitioner optization in all three dimensions
        // (with a random dim sequence)
        for ( int i = 0; i < _max_optimize_iteration; ++i )
        {
            bool is_changed = false; // record changes in current iteration
            bool dim_covered[3] = { false, false, false };
            for ( int d = 0; d < 3; ++d )
            {
                int random_dim_id = std::rand() % num_space_dim;
                while ( dim_covered[random_dim_id] )
                    random_dim_id = std::rand() % num_space_dim;

                bool is_dim_changed = false; // record changes in current dim
                optimizePartitionAlongDim( random_dim_id, is_dim_changed );

                // update control info
                is_changed = is_changed || is_dim_changed;
                dim_covered[random_dim_id] = true;
            }
            // return if the current partition is optimal
            if ( !is_changed )
                return i;
        }
        return _max_optimize_iteration;
    }

    /*!
      \brief optimize the partition in three dimensions seperately
      \param iter_seed seed number to choose the starting dimension of the
      optimization
      \param is_changed label if the partition is changed after the optimization
    */
    void optimizePartitionAlongDim( int iter_seed, bool& is_changed )
    {
        is_changed = false;
        // loop over three dimensions, optimize the partition in dimension di
        for ( int iter_id = iter_seed;
              iter_id < iter_seed + static_cast<int>( num_space_dim );
              ++iter_id )
        {
            int di = iter_id % num_space_dim;
            // compute the dimensions that should be fixed (dj and dk)
            int dj = ( di + 1 ) % num_space_dim;
            int dk = ( di + 2 ) % num_space_dim;
            auto rank_k = _ranks_per_dim[dk];

            auto rank = _ranks_per_dim[di];
            auto rec_mirror = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), _rectangle_partition_dev );
            auto rec_partition = _rectangle_partition_dev;

            SubWorkloadFunctor<partition_view, workload_view>
                compute_sub_workload( _rectangle_partition_dev,
                                      _workload_prefix_sum );

            // compute average workload in the dimension di
            Kokkos::View<int*, memory_space> ave_workload(
                "ave_workload", _ranks_per_dim[dj] * _ranks_per_dim[dk] );
            Kokkos::parallel_for(
                "compute_average_workload",
                Kokkos::RangePolicy<execution_space>(
                    0, _ranks_per_dim[dj] * _ranks_per_dim[dk] ),
                KOKKOS_LAMBDA( uint32_t jnk ) {
                    // compute rank_id in the fixed dimensions
                    int j = static_cast<int>( jnk / rank_k );
                    int k = static_cast<int>( jnk % rank_k );
                    // compute the average workload with the partition of the
                    // two fixed dimensions
                    ave_workload( jnk ) =
                        compute_sub_workload( di, 0, rec_partition( rank, di ),
                                              dj, j, dk, k ) /
                        rank;
                } );
            Kokkos::fence();

            // point_i: current partition position
            int point_i = 1;
            // equal_start_point: register the beginning pos of potentially
            // equivalent partitions
            int equal_start_point = 1;
            // last_point: the opimized position for the lask partition
            int last_point = 0;
            // current_workload: the workload between [last_point, point_i)
            for ( int current_rank = 1; current_rank < rank; current_rank++ )
            {
                int last_diff = __INT_MAX__;
                while ( true )
                {
                    int diff;
                    Kokkos::parallel_reduce(
                        "diff_reduce",
                        Kokkos::RangePolicy<execution_space>(
                            0, _ranks_per_dim[dj] * _ranks_per_dim[dk] ),
                        KOKKOS_LAMBDA( const int jnk, int& update ) {
                            int j = static_cast<int>( jnk / rank_k );
                            int k = static_cast<int>( jnk % rank_k );
                            int current_workload = compute_sub_workload(
                                di, last_point, point_i, dj, j, dk, k );
                            auto wl =
                                current_workload - ave_workload( jnk );
                            // compute absolute diff (rather than squares to
                            // avoid potential overflow)
                            // TODO: update when Kokkos::abs() available
                            wl = wl > 0 ? wl : -wl;
                            update += wl;
                        },
                        diff );
                    Kokkos::fence();

                    // record the new optimal position
                    if ( diff <= last_diff )
                    {
                        // register starting points of potentially equivalent
                        // partitions
                        if ( diff != last_diff )
                            equal_start_point = point_i;

                        // check if point_i reach the total_tile_num
                        if ( point_i == rec_mirror( rank, di ) )
                        {
                            rec_mirror( current_rank, di ) = point_i;
                            break;
                        }

                        last_diff = diff;
                        point_i++;
                    }
                    else
                    {
                        // final optimal position - middle position of all
                        // potentially equivalent partitions
                        if ( rec_mirror( current_rank, di ) !=
                             ( point_i - 1 + equal_start_point ) / 2 )
                        {
                            rec_mirror( current_rank, di ) =
                                ( point_i - 1 + equal_start_point ) / 2;
                            is_changed = true;
                        }
                        last_point = point_i - 1;
                        break;
                    }
                } // end while (optimization for the current rank)
            }     // end for (all partition/rank in the optimized dimension)
            Kokkos::deep_copy( _rectangle_partition_dev, rec_mirror );
        } // end for (3 dimensions)
    }

    /*!
      \brief compute the total workload on the current MPI rank
      \param cart_comm MPI cartesian communicator
      \return total workload on current rank
    */
    int currentRankWorkload( MPI_Comm cart_comm )
    {
        auto rec_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _rectangle_partition_dev );
        auto prefix_sum_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _workload_prefix_sum );

        return currentRankWorkload( cart_comm, rec_mirror, prefix_sum_mirror );
    }

    /*!
       \brief compute the total workload on the current MPI rank
       \param cart_comm MPI cartesian communicator
       \param rec_view Host mirror of _rec_partition_dev
       \param prefix_sum_view Host mirror of _workload_prefix_sum
       \return total workload on current rank
     */
    template <typename PartitionViewHost, typename WorkloadViewHost>
    int currentRankWorkload( MPI_Comm cart_comm, PartitionViewHost& rec_view,
                             WorkloadViewHost& prefix_sum_view )
    {
        SubWorkloadFunctor<PartitionViewHost, WorkloadViewHost>
            compute_sub_workload_host( rec_view, prefix_sum_view );

        // Get the Cartesian topology index of this rank.
        Kokkos::Array<int, num_space_dim> cart_rank;
        int linear_rank;
        MPI_Comm_rank( cart_comm, &linear_rank );
        MPI_Cart_coords( cart_comm, linear_rank, num_space_dim,
                         cart_rank.data() );

        // compute total workload of the current rank
        int workload_current_rank = compute_sub_workload_host(
            0, rec_view( cart_rank[0], 0 ), rec_view( cart_rank[0] + 1, 0 ), 1,
            cart_rank[1], 2, cart_rank[2] );

        return workload_current_rank;
    }

    /*!
    \brief compute the average workload on each MPI rank
    \return average workload on each rank
    */
    int averageRankWorkload()
    {
        auto prefix_sum_view = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _workload_prefix_sum );
        // compute total workload of the current rank
        return averageRankWorkload( prefix_sum_view );
    }

    /*!
    \brief compute the average workload on each MPI rank
    \param prefix_sum_view Host mirror of _workload_prefix_sum
    \return average workload on each rank
    */
    template <typename WorkloadViewHost>
    int averageRankWorkload( WorkloadViewHost& prefix_sum_view )
    {
        // compute total workload of the current rank
        return prefix_sum_view( prefix_sum_view.extent( 0 ) - 1,
                                prefix_sum_view.extent( 1 ) - 1,
                                prefix_sum_view.extent( 2 ) - 1 ) /
               ( _ranks_per_dim[0] * _ranks_per_dim[1] * _ranks_per_dim[2] );
    }

    /*!
      \brief compute the imbalance factor for the current partition
      \param cart_comm MPI cartesian communicator
      \return the imbalance factor = workload on current rank / ave_workload
    */
    float computeImbalanceFactor( MPI_Comm cart_comm )
    {
        auto rec_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _rectangle_partition_dev );
        auto prefix_sum_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _workload_prefix_sum );

        int workload_current_rank =
            currentRankWorkload( cart_comm, rec_mirror, prefix_sum_mirror );
        int workload_ave_rank = averageRankWorkload( prefix_sum_mirror );

        return static_cast<float>( workload_current_rank ) /
               static_cast<float>( workload_ave_rank );
    }

    /*!
      \brief functor to compute the sub workload in a given region (from the
      prefix sum)
    */
    template <typename PartitionView, typename WorkloadView>
    struct SubWorkloadFunctor
    {
        //! Rectilinear partition
        PartitionView rec_partition;
        //! Workload prefix sum matrix
        WorkloadView workload_prefix_sum;

        //! Constructor.
        SubWorkloadFunctor( PartitionView rec_par, WorkloadView pre_sum )
            : rec_partition( rec_par )
            , workload_prefix_sum( pre_sum )
        {
        }

        //! compute the workload in region rounded by:
        //! [i_start, i_end) in dim_i
        //! [partition[j], partition[j+1]) in dim_j
        //! [partition[k], partition[k+1]) in dim_k
        KOKKOS_INLINE_FUNCTION int operator()( int dim_i, int i_start,
                                               int i_end, int dim_j, int j,
                                               int dim_k, int k ) const
        {
            int end[num_space_dim], start[num_space_dim];
            end[dim_i] = i_end;
            end[dim_j] = rec_partition( j + 1, dim_j );
            end[dim_k] = rec_partition( k + 1, dim_k );

            start[dim_i] = i_start;
            start[dim_j] = rec_partition( j, dim_j );
            start[dim_k] = rec_partition( k, dim_k );

            // S[i][j][k] = S[i-1][j][k] + S[i][j-1][k] + S[i][j][k-1] -
            // S[i-1][j-1][k]
            // - S[i][j-1][k-1] - S[i-1][j][k-1] + S[i-1][j-1][k-1] + a[i][j][k]
            return workload_prefix_sum( end[0], end[1], end[2] ) // S[i][j][k]
                   - workload_prefix_sum( start[0], end[1],
                                          end[2] ) // S[i-1][j][k]
                   - workload_prefix_sum( end[0], start[1],
                                          end[2] ) // S[i][j-1][k]
                   - workload_prefix_sum( end[0], end[1],
                                          start[2] ) // S[i][j][k-1]
                   + workload_prefix_sum( start[0], start[1],
                                          end[2] ) // S[i-1][j-1][k]
                   + workload_prefix_sum( end[0], start[1],
                                          start[2] ) // S[i][j-1][k-1]
                   + workload_prefix_sum( start[0], end[1],
                                          start[2] ) // S[i-1][j][k-1]
                   - workload_prefix_sum( start[0], start[1],
                                          start[2] ); // S[i-1][j-1][k-1]
        }
    };

  private:
    // workload_threshold
    int _workload_threshold;
    // default check point for re-balance
    int _num_step_rebalance;
    // max_optimize iterations
    int _max_optimize_iteration;

    // represent the rectangle partition in each dimension
    // with form [0, p_1, ..., p_n, cell_num], n = rank num in current
    // dimension, partition in this dimension would be [0, p_1), [p_1, p_2) ...
    // [p_n, total_tile_num] (unit: tile)
    partition_view _rectangle_partition_dev;
    // the workload of each tile on current
    workload_view _workload_per_tile;
    // 3d prefix sum of the workload of each tile on current
    workload_view _workload_prefix_sum;
    // ranks per dimension
    Kokkos::Array<int, num_space_dim> _ranks_per_dim;

    void allocate( const std::array<int, num_space_dim>& global_cells_per_dim )
    {

        _workload_per_tile = workload_view(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "workload_per_tile" ),
            ( global_cells_per_dim[0] >> cell_bits_per_tile_dim ) + 1,
            ( global_cells_per_dim[1] >> cell_bits_per_tile_dim ) + 1,
            ( global_cells_per_dim[2] >> cell_bits_per_tile_dim ) + 1 );

        _workload_prefix_sum = workload_view(
            Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                "workload_prefix_sum" ),
            ( global_cells_per_dim[0] >> cell_bits_per_tile_dim ) + 1,
            ( global_cells_per_dim[1] >> cell_bits_per_tile_dim ) + 1,
            ( global_cells_per_dim[2] >> cell_bits_per_tile_dim ) + 1 );
    }
};
} // end namespace Cajita

#endif // end CAJITA_DYNAMICPARTITIONER_HPP
