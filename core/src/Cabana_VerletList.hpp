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

/*!
  \file Cabana_VerletList.hpp
  \brief Verlet grid-accelerated neighbor list
*/
#ifndef CABANA_VERLETLIST_HPP
#define CABANA_VERLETLIST_HPP

#include <Cabana_LinkedCellList.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <cassert>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Verlet List Memory Layout Tag.
//---------------------------------------------------------------------------//
//! CSR (compressed sparse row) neighbor list layout.
struct VerletLayoutCSR
{
};

//! 2D array neighbor list layout.
struct VerletLayout2D
{
};

//---------------------------------------------------------------------------//
// Verlet List Data.
//---------------------------------------------------------------------------//
template <class MemorySpace, class LayoutTag>
struct VerletListData;

//! Store the VerletList compressed sparse row (CSR) neighbor data.
template <class MemorySpace>
struct VerletListData<MemorySpace, VerletLayoutCSR>
{
    //! Kokkos memory space.
    using memory_space = MemorySpace;

    //! Number of neighbors per particle.
    Kokkos::View<int*, memory_space> counts;

    //! Offsets into the neighbor list.
    Kokkos::View<int*, memory_space> offsets;

    //! Neighbor list.
    Kokkos::View<int*, memory_space> neighbors;

    //! Add a neighbor to the list.
    KOKKOS_INLINE_FUNCTION
    void addNeighbor( const int pid, const int nid ) const
    {
        neighbors( offsets( pid ) +
                   Kokkos::atomic_fetch_add( &counts( pid ), 1 ) ) = nid;
    }

    //! Modify a neighbor in the list.
    KOKKOS_INLINE_FUNCTION
    void setNeighbor( const int pid, const int nid, const int new_id ) const
    {
        neighbors( offsets( pid ) + nid ) = new_id;
    }
};

//! Store the VerletList 2D neighbor data.
template <class MemorySpace>
struct VerletListData<MemorySpace, VerletLayout2D>
{
    //! Kokkos memory space.
    using memory_space = MemorySpace;

    //! Number of neighbors per particle.
    Kokkos::View<int*, memory_space> counts;

    //! Neighbor list.
    Kokkos::View<int**, memory_space> neighbors;

    //! Actual maximum neighbors per particle (potentially less than allocated
    //! space).
    std::size_t max_n;

    //! Add a neighbor to the list.
    KOKKOS_INLINE_FUNCTION
    void addNeighbor( const int pid, const int nid ) const
    {
        std::size_t count = Kokkos::atomic_fetch_add( &counts( pid ), 1 );
        if ( count < neighbors.extent( 1 ) )
            neighbors( pid, count ) = nid;
    }

    //! Modify a neighbor in the list.
    KOKKOS_INLINE_FUNCTION
    void setNeighbor( const int pid, const int nid, const int new_id ) const
    {
        neighbors( pid, nid ) = new_id;
    }
};

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//

namespace Impl
{
//! \cond Impl

//---------------------------------------------------------------------------//
// Verlet List Builder
//---------------------------------------------------------------------------//
template <class DeviceType, class PositionType, class RandomAccessPositionType,
          class AlgorithmTag, class LayoutTag, class BuildOpTag,
          std::size_t NumSpaceDim = 3>
struct VerletListBuilder
{
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    // Types.
    using device = DeviceType;
    using PositionValueType = typename PositionType::value_type;
    using memory_space = typename device::memory_space;
    using execution_space = typename device::execution_space;

    // List data.
    VerletListData<memory_space, LayoutTag> _data;

    // Neighbor cutoff.
    PositionValueType rsqr;

    // Positions.
    RandomAccessPositionType _position;
    std::size_t pid_begin, pid_end;

    // Binning Data.
    BinningData<memory_space> bin_data_1d;
    LinkedCellList<memory_space, PositionValueType, num_space_dim>
        linked_cell_list;

    // Check to count or refill.
    bool refill;
    bool count;

    // Maximum allocated neighbors per particle
    std::size_t alloc_n;

    // Constructor.
    VerletListBuilder( PositionType positions, const std::size_t begin,
                       const std::size_t end,
                       const PositionValueType neighborhood_radius,
                       const PositionValueType cell_size_ratio,
                       const PositionValueType grid_min[num_space_dim],
                       const PositionValueType grid_max[num_space_dim],
                       const std::size_t max_neigh )
        : pid_begin( begin )
        , pid_end( end )
        , alloc_n( max_neigh )
    {
        count = true;
        refill = false;

        // Create the count view.
        _data.counts = Kokkos::View<int*, memory_space>( "num_neighbors",
                                                         size( positions ) );

        // Make a guess for the number of neighbors per particle for 2D lists.
        initCounts( LayoutTag() );

        // Shallow copy for random access read-only memory.
        _position = positions;

        // Bin the particles in the grid. Don't actually sort them but make a
        // permutation vector. Note that we are binning all particles here and
        // not just the requested range. This is because all particles are
        // treated as candidates for neighbors.
        double grid_size = cell_size_ratio * neighborhood_radius;
        PositionValueType grid_delta[num_space_dim];
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            grid_delta[d] = grid_size;

        linked_cell_list = createLinkedCellList<num_space_dim>(
            _position, grid_delta, grid_min, grid_max, neighborhood_radius,
            cell_size_ratio );
        bin_data_1d = linked_cell_list.binningData();

        // We will use the square of the distance for neighbor determination.
        rsqr = neighborhood_radius * neighborhood_radius;
    }

    // Neighbor count team operator (only used for CSR lists).
    struct CountNeighborsTag
    {
    };
    using CountNeighborsPolicy =
        Kokkos::TeamPolicy<execution_space, CountNeighborsTag,
                           Kokkos::IndexType<int>,
                           Kokkos::Schedule<Kokkos::Dynamic>>;

    KOKKOS_INLINE_FUNCTION
    void
    operator()( const CountNeighborsTag&,
                const typename CountNeighborsPolicy::member_type& team ) const
    {
        // The league rank of the team is the cardinal cell index we are
        // working on.
        int cell = team.league_rank();

        // Get the stencil for this cell.
        Kokkos::Array<int, num_space_dim> min;
        Kokkos::Array<int, num_space_dim> max;
        linked_cell_list.getStencilCells( cell, min, max );

        // Operate on the particles in the bin.
        std::size_t b_offset = bin_data_1d.binOffset( cell );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team, 0, bin_data_1d.binSize( cell ) ),
            [&]( const int bi )
            {
                // Get the true particle id. The binned particle index is the
                // league rank of the team.
                std::size_t pid = linked_cell_list.permutation( bi + b_offset );

                if ( ( pid >= pid_begin ) && ( pid < pid_end ) )
                {
                    // Cache the particle coordinates.
                    Kokkos::Array<PositionValueType, num_space_dim> xp;
                    for ( std::size_t d = 0; d < num_space_dim; ++d )
                        xp[d] = _position( pid, d );

                    // Loop over the cell stencil.
                    int stencil_count =
                        linkedcell_stencil_count( team, min, max, xp, pid );

                    Kokkos::single( Kokkos::PerThread( team ), [&]()
                                    { _data.counts( pid ) = stencil_count; } );
                }
            } );
    }

    template <class TeamType, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, int>
    linkedcell_stencil_count( TeamType team, const Kokkos::Array<int, 3> min,
                              const Kokkos::Array<int, 3> max,
                              const Kokkos::Array<PositionValueType, 3> xp,
                              const std::size_t pid ) const
    {
        int stencil_count = 0;
        Kokkos::Array<int, 3> ijk;
        for ( int i = min[0]; i < max[0]; ++i )
            for ( int j = min[1]; j < max[1]; ++j )
                for ( int k = min[2]; k < max[2]; ++k )
                {
                    ijk = { i, j, k };
                    linkedcell_bin( team, ijk, xp, pid, stencil_count );
                }
        return stencil_count;
    }

    template <class TeamType, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, int>
    linkedcell_stencil_count( TeamType team, const Kokkos::Array<int, 2> min,
                              const Kokkos::Array<int, 2> max,
                              const Kokkos::Array<PositionValueType, 2> xp,
                              const std::size_t pid ) const
    {
        int stencil_count = 0;
        Kokkos::Array<int, 2> ij;
        for ( int i = min[0]; i < max[0]; ++i )
            for ( int j = min[1]; j < max[1]; ++j )
            {
                ij = { i, j };
                linkedcell_bin( team, ij, xp, pid, stencil_count );
            }
        return stencil_count;
    }

    template <class TeamType>
    KOKKOS_INLINE_FUNCTION void
    linkedcell_bin( TeamType team, const Kokkos::Array<int, num_space_dim> ijk,
                    const Kokkos::Array<PositionValueType, num_space_dim> xp,
                    const std::size_t pid, int& stencil_count ) const
    {
        // See if we should actually check this box for neighbors.
        if ( linked_cell_list.cellStencil().grid.minDistanceToPoint(
                 xp, ijk ) <= rsqr )
        {
            std::size_t n_offset = linked_cell_list.binOffset( ijk );
            std::size_t num_n = linked_cell_list.binSize( ijk );

            // Check the particles in this bin to see if they are neighbors. If
            // they are add to the count for this bin.
            int cell_count = 0;
            neighbor_reduce( team, pid, xp, n_offset, num_n, cell_count,
                             BuildOpTag() );
            stencil_count += cell_count;
        }
    }

    // Neighbor count team vector loop (only used for CSR lists).
    KOKKOS_INLINE_FUNCTION void
    neighbor_reduce( const typename CountNeighborsPolicy::member_type& team,
                     const std::size_t pid,
                     const Kokkos::Array<double, num_space_dim> xp,
                     const int n_offset, const int num_n, int& cell_count,
                     TeamVectorOpTag ) const
    {
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange( team, num_n ),
            [&]( const int n, int& local_count )
            { neighbor_kernel( pid, xp, n_offset, n, local_count ); },
            cell_count );
    }

    // Neighbor count serial loop (only used for CSR lists).
    KOKKOS_INLINE_FUNCTION
    void neighbor_reduce( const typename CountNeighborsPolicy::member_type,
                          const std::size_t pid,
                          const Kokkos::Array<double, num_space_dim> xp,
                          const int n_offset, const int num_n, int& cell_count,
                          TeamOpTag ) const
    {
        for ( int n = 0; n < num_n; n++ )
            neighbor_kernel( pid, xp, n_offset, n, cell_count );
    }

    // Neighbor count kernel
    KOKKOS_INLINE_FUNCTION
    void neighbor_kernel( const int pid,
                          const Kokkos::Array<double, num_space_dim> xp,
                          const int n_offset, const int n,
                          int& local_count ) const
    {
        //  Get the true id of the candidate  neighbor.
        std::size_t nid = linked_cell_list.permutation( n_offset + n );

        // Cache the candidate neighbor particle coordinates.
        Kokkos::Array<double, num_space_dim> xn;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            xn[d] = _position( nid, d );

        // If this could be a valid neighbor, continue.
        if ( NeighborDiscriminator<AlgorithmTag>::isValid( pid, xp, nid, xn ) )
        {
            // Calculate the distance between the particle and its candidate
            // neighbor.
            PositionValueType dist_sqr = 0.0;
            for ( std::size_t d = 0; d < num_space_dim; ++d )
            {
                PositionValueType dx = xp[d] - xn[d];
                dist_sqr += dx * dx;
            }

            // If within the cutoff add to the count.
            if ( dist_sqr <= rsqr )
                local_count += 1;
        }
    }

    // Process the CSR counts by computing offsets and allocating the neighbor
    // list.
    template <class KokkosMemorySpace>
    struct OffsetScanOp
    {
        using kokkos_mem_space = KokkosMemorySpace;
        Kokkos::View<int*, kokkos_mem_space> counts;
        Kokkos::View<int*, kokkos_mem_space> offsets;
        KOKKOS_INLINE_FUNCTION
        void operator()( const int i, int& update, const bool final_pass ) const
        {
            if ( final_pass )
                offsets( i ) = update;
            update += counts( i );
        }
    };

    void initCounts( VerletLayoutCSR ) {}

    void initCounts( VerletLayout2D )
    {
        if ( alloc_n > 0 )
        {
            count = false;

            _data.neighbors = Kokkos::View<int**, memory_space>(
                Kokkos::ViewAllocateWithoutInitializing( "neighbors" ),
                _data.counts.size(), alloc_n );
        }
    }

    void processCounts( VerletLayoutCSR )
    {
        // Allocate offsets.
        _data.offsets = Kokkos::View<int*, memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "neighbor_offsets" ),
            _data.counts.size() );

        // Calculate offsets from counts and the total number of counts.
        OffsetScanOp<memory_space> offset_op;
        offset_op.counts = _data.counts;
        offset_op.offsets = _data.offsets;
        int total_num_neighbor;
        Kokkos::RangePolicy<execution_space> range_policy(
            0, _data.counts.extent( 0 ) );
        Kokkos::parallel_scan( "Cabana::VerletListBuilder::offset_scan",
                               range_policy, offset_op, total_num_neighbor );
        Kokkos::fence();

        // Allocate the neighbor list.
        _data.neighbors = Kokkos::View<int*, memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "neighbors" ),
            total_num_neighbor );

        // Reset the counts. We count again when we fill.
        Kokkos::deep_copy( _data.counts, 0 );
    }

    // Process 2D counts by computing the maximum number of neighbors and
    // reallocating the 2D data structure if needed.
    void processCounts( VerletLayout2D )
    {
        // Calculate the maximum number of neighbors.
        auto counts = _data.counts;
        int max;
        Kokkos::Max<int> max_reduce( max );
        Kokkos::parallel_reduce(
            "Cabana::VerletListBuilder::reduce_max",
            Kokkos::RangePolicy<execution_space>( 0, _data.counts.size() ),
            KOKKOS_LAMBDA( const int i, int& value ) {
                if ( counts( i ) > value )
                    value = counts( i );
            },
            max_reduce );
        Kokkos::fence();
        _data.max_n = static_cast<std::size_t>( max );

        // Reallocate the neighbor list if previous size is exceeded.
        if ( count or _data.max_n > _data.neighbors.extent( 1 ) )
        {
            refill = true;
            Kokkos::deep_copy( _data.counts, 0 );
            _data.neighbors = Kokkos::View<int**, memory_space>(
                Kokkos::ViewAllocateWithoutInitializing( "neighbors" ),
                _data.counts.size(), _data.max_n );
        }
    }

    // Neighbor count team operator.
    struct FillNeighborsTag
    {
    };
    using FillNeighborsPolicy =
        Kokkos::TeamPolicy<execution_space, FillNeighborsTag,
                           Kokkos::IndexType<int>,
                           Kokkos::Schedule<Kokkos::Dynamic>>;
    KOKKOS_INLINE_FUNCTION
    void
    operator()( const FillNeighborsTag&,
                const typename FillNeighborsPolicy::member_type& team ) const
    {
        // The league rank of the team is the cardinal cell index we are
        // working on.
        int cell = team.league_rank();

        // Get the stencil for this cell.
        Kokkos::Array<int, num_space_dim> min;
        Kokkos::Array<int, num_space_dim> max;
        linked_cell_list.getStencilCells( cell, min, max );

        // Operate on the particles in the bin.
        std::size_t b_offset = bin_data_1d.binOffset( cell );
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team, 0, bin_data_1d.binSize( cell ) ),
            [&]( const int bi )
            {
                // Get the true particle id. The binned particle index is the
                // league rank of the team.
                std::size_t pid = linked_cell_list.permutation( bi + b_offset );

                if ( ( pid >= pid_begin ) && ( pid < pid_end ) )
                {
                    // Cache the particle coordinates.
                    Kokkos::Array<PositionValueType, num_space_dim> xp;
                    for ( std::size_t d = 0; d < num_space_dim; ++d )
                        xp[d] = _position( pid, d );

                    linkedcell_stencil_fill( team, min, max, xp, pid );
                }
            } );
    }

    template <class TeamType, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    linkedcell_stencil_fill( TeamType team, const Kokkos::Array<int, 3> min,
                             const Kokkos::Array<int, 3> max,
                             const Kokkos::Array<PositionValueType, 3> xp,
                             const std::size_t pid ) const
    {
        Kokkos::Array<int, 3> ijk;
        for ( int i = min[0]; i < max[0]; ++i )
            for ( int j = min[1]; j < max[1]; ++j )
                for ( int k = min[2]; k < max[2]; ++k )
                {
                    ijk = { i, j, k };
                    linkedcell_bin( team, ijk, xp, pid );
                }
    }

    template <class TeamType, std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    linkedcell_stencil_fill( TeamType team, const Kokkos::Array<int, 2> min,
                             const Kokkos::Array<int, 2> max,
                             const Kokkos::Array<PositionValueType, 2> xp,
                             const std::size_t pid ) const
    {
        Kokkos::Array<int, 2> ij;
        for ( int i = min[0]; i < max[0]; ++i )
            for ( int j = min[1]; j < max[1]; ++j )
            {
                ij = { i, j };
                linkedcell_bin( team, ij, xp, pid );
            }
    }

    template <class TeamType>
    KOKKOS_INLINE_FUNCTION void
    linkedcell_bin( TeamType team, const Kokkos::Array<int, num_space_dim> ijk,
                    const Kokkos::Array<PositionValueType, num_space_dim> xp,
                    const std::size_t pid ) const
    {
        // See if we should actually check this box for neighbors.
        if ( linked_cell_list.cellStencil().grid.minDistanceToPoint(
                 xp, ijk ) <= rsqr )
        {
            // Check the particles in this bin to see if they are neighbors.
            std::size_t n_offset = linked_cell_list.binOffset( ijk );
            int num_n = linked_cell_list.binSize( ijk );
            neighbor_for( team, pid, xp, n_offset, num_n, BuildOpTag() );
        }
    }

    // Neighbor fill team vector loop.
    KOKKOS_INLINE_FUNCTION void
    neighbor_for( const typename FillNeighborsPolicy::member_type& team,
                  const std::size_t pid,
                  const Kokkos::Array<double, num_space_dim> xp,
                  const int n_offset, const int num_n, TeamVectorOpTag ) const
    {
        Kokkos::parallel_for( Kokkos::ThreadVectorRange( team, num_n ),
                              [&]( const int n )
                              { neighbor_kernel( pid, xp, n_offset, n ); } );
    }

    // Neighbor fill serial loop.
    KOKKOS_INLINE_FUNCTION
    void neighbor_for( const typename FillNeighborsPolicy::member_type team,
                       const std::size_t pid,
                       const Kokkos::Array<double, num_space_dim> xp,
                       const int n_offset, const int num_n, TeamOpTag ) const
    {
        for ( int n = 0; n < num_n; n++ )
            Kokkos::single( Kokkos::PerThread( team ), [&]()
                            { neighbor_kernel( pid, xp, n_offset, n ); } );
    }

    // Neighbor fill kernel.
    KOKKOS_INLINE_FUNCTION
    void neighbor_kernel( const int pid,
                          const Kokkos::Array<double, num_space_dim> xp,
                          const int n_offset, const int n ) const
    {
        //  Get the true id of the candidate neighbor.
        std::size_t nid = linked_cell_list.permutation( n_offset + n );

        // Cache the candidate neighbor particle coordinates.
        Kokkos::Array<double, num_space_dim> xn;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            xn[d] = _position( nid, d );

        // If this could be a valid neighbor, continue.
        if ( NeighborDiscriminator<AlgorithmTag>::isValid( pid, xp, nid, xn ) )
        {
            // Calculate the distance between the particle and its candidate
            // neighbor.
            PositionValueType dist_sqr = 0.0;
            for ( std::size_t d = 0; d < num_space_dim; ++d )
            {
                PositionValueType dx = xp[d] - xn[d];
                dist_sqr += dx * dx;
            }

            // If within the cutoff increment the neighbor count and add as a
            // neighbor at that index.
            if ( dist_sqr <= rsqr )
            {
                _data.addNeighbor( pid, nid );
            }
        }
    }
};

// Builder creation functions. This is only necessary to define the different
// random access types.
template <class DeviceType, class PositionType, class AlgorithmTag,
          class LayoutTag, class BuildOpTag, std::size_t NumSpaceDim>
auto createVerletListBuilder(
    PositionType x, const std::size_t begin, const std::size_t end,
    const typename PositionType::value_type radius,
    const typename PositionType::value_type cell_size_ratio,
    const typename PositionType::value_type grid_min[NumSpaceDim],
    const typename PositionType::value_type grid_max[NumSpaceDim],
    const std::size_t max_neigh,
    typename std::enable_if<( is_slice<PositionType>::value ), int>::type* = 0 )
{
    using RandomAccessPositionType = typename PositionType::random_access_slice;
    return VerletListBuilder<DeviceType, PositionType, RandomAccessPositionType,
                             AlgorithmTag, LayoutTag, BuildOpTag, NumSpaceDim>(
        x, begin, end, radius, cell_size_ratio, grid_min, grid_max, max_neigh );
}

template <class DeviceType, class PositionType, class AlgorithmTag,
          class LayoutTag, class BuildOpTag, std::size_t NumSpaceDim>
auto createVerletListBuilder(
    PositionType x, const std::size_t begin, const std::size_t end,
    const typename PositionType::value_type radius,
    const typename PositionType::value_type cell_size_ratio,
    const typename PositionType::value_type grid_min[NumSpaceDim],
    const typename PositionType::value_type grid_max[NumSpaceDim],
    const std::size_t max_neigh,
    typename std::enable_if<( Kokkos::is_view<PositionType>::value ),
                            int>::type* = 0 )
{
    using RandomAccessPositionType =
        Kokkos::View<typename PositionType::value_type**, DeviceType,
                     Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    return VerletListBuilder<DeviceType, PositionType, RandomAccessPositionType,
                             AlgorithmTag, LayoutTag, BuildOpTag, NumSpaceDim>(
        x, begin, end, radius, cell_size_ratio, grid_min, grid_max, max_neigh );
}

//---------------------------------------------------------------------------//

//! \endcond
} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Neighbor list implementation based on binning particles on a Cartesian
  grid with cells of the same size as the interaction distance.

  \tparam MemorySpace The Kokkos memory space for storing the neighbor list.

  \tparam AlgorithmTag Tag indicating whether to build a full or half neighbor
  list.

  \tparam LayoutTag Tag indicating whether to use a CSR or 2D data layout.

  \tparam BuildTag Tag indicating whether to use hierarchical team or team
  vector parallelism when building neighbor lists.

  Neighbor list implementation most appropriate for somewhat regularly
  distributed particles due to the use of a Cartesian grid.
*/
template <class MemorySpace, class AlgorithmTag, class LayoutTag,
          class BuildTag = TeamVectorOpTag, std::size_t NumSpaceDim = 3>
class VerletList
{
  public:
    static_assert( Kokkos::is_memory_space<MemorySpace>::value, "" );

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    //! Kokkos memory space in which the neighbor list data resides.
    using memory_space = MemorySpace;

    //! Kokkos default execution space for this memory space.
    using execution_space = typename memory_space::execution_space;

    //! Verlet list data.
    VerletListData<memory_space, LayoutTag> _data;

    /*!
      \brief Default constructor.
    */
    VerletList() {}

    /*!
      \brief VerletList constructor. Given a list of particle positions and
      a neighborhood radius calculate the neighbor list.

      \param x The particle positions

      \param begin The beginning particle index to compute neighbors for.

      \param end The end particle index to compute neighbors for.

      \param neighborhood_radius The radius of the neighborhood. Particles
      within this radius are considered neighbors. This is effectively the
      grid cell size in each dimension.

      \param cell_size_ratio The ratio of the cell size in the Cartesian grid
      to the neighborhood radius. For example, if the cell size ratio is 0.5
      then the cells will be half the size of the neighborhood radius in each
      dimension.

      \param grid_min The minimum value of the grid containing the particles
      in each dimension.

      \param grid_max The maximum value of the grid containing the particles
      in each dimension.

      \param max_neigh Optional maximum number of neighbors per particle to
      pre-allocate the neighbor list. Potentially avoids recounting with 2D
      layout only.

      Particles outside of the neighborhood radius will not be considered
      neighbors. Only compute the neighbors of those that are within the given
      range. All particles are candidates for being a neighbor, regardless of
      whether or not they are in the range.
    */
    template <class PositionType>
    VerletList(
        PositionType x, const std::size_t begin, const std::size_t end,
        const typename PositionType::value_type neighborhood_radius,
        const typename PositionType::value_type cell_size_ratio,
        const typename PositionType::value_type grid_min[num_space_dim],
        const typename PositionType::value_type grid_max[num_space_dim],
        const std::size_t max_neigh = 0,
        typename std::enable_if<( is_slice<PositionType>::value ||
                                  Kokkos::is_view<PositionType>::value ),
                                int>::type* = 0 )
    {
        build( x, begin, end, neighborhood_radius, cell_size_ratio, grid_min,
               grid_max, max_neigh );
    }

    /*!
      \brief Given a list of particle positions and a neighborhood radius
      calculate the neighbor list.
    */
    template <class PositionType>
    void
    build( PositionType x, const std::size_t begin, const std::size_t end,
           const typename PositionType::value_type neighborhood_radius,
           const typename PositionType::value_type cell_size_ratio,
           const typename PositionType::value_type grid_min[num_space_dim],
           const typename PositionType::value_type grid_max[num_space_dim],
           const std::size_t max_neigh = 0,
           typename std::enable_if<( is_slice<PositionType>::value ||
                                     Kokkos::is_view<PositionType>::value ),
                                   int>::type* = 0 )
    {
        // Use the default execution space.
        build( execution_space{}, x, begin, end, neighborhood_radius,
               cell_size_ratio, grid_min, grid_max, max_neigh );
    }
    /*!
      \brief Given a list of particle positions and a neighborhood radius
      calculate the neighbor list.
    */
    template <class PositionType, class ExecutionSpace>
    void
    build( ExecutionSpace, PositionType x, const std::size_t begin,
           const std::size_t end,
           const typename PositionType::value_type neighborhood_radius,
           const typename PositionType::value_type cell_size_ratio,
           const typename PositionType::value_type grid_min[num_space_dim],
           const typename PositionType::value_type grid_max[num_space_dim],
           const std::size_t max_neigh = 0,
           typename std::enable_if<( is_slice<PositionType>::value ||
                                     Kokkos::is_view<PositionType>::value ),
                                   int>::type* = 0 )
    {
        Kokkos::Profiling::ScopedRegion region( "Cabana::VerletList::build" );

        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        assert( end >= begin );
        assert( end <= size( x ) );

        using device_type = Kokkos::Device<ExecutionSpace, memory_space>;

        // Create a builder functor.
        auto builder = Impl::createVerletListBuilder<device_type, PositionType,
                                                     AlgorithmTag, LayoutTag,
                                                     BuildTag, num_space_dim>(
            x, begin, end, neighborhood_radius, cell_size_ratio, grid_min,
            grid_max, max_neigh );

        // For each particle in the range check each neighboring bin for
        // neighbor particles. Bins are at least the size of the neighborhood
        // radius so the bin in which the particle resides and any surrounding
        // bins are guaranteed to contain the neighboring particles.
        // For CSR lists, we count, then fill neighbors. For 2D lists, we
        // count and fill at the same time, unless the array size is exceeded,
        // at which point only counting is continued to reallocate and refill.
        typename decltype( builder )::FillNeighborsPolicy fill_policy(
            builder.bin_data_1d.numBin(), Kokkos::AUTO, 4 );
        if ( builder.count )
        {
            typename decltype( builder )::CountNeighborsPolicy count_policy(
                builder.bin_data_1d.numBin(), Kokkos::AUTO, 4 );
            Kokkos::parallel_for( "Cabana::VerletList::count_neighbors",
                                  count_policy, builder );
        }
        else
        {
            builder.processCounts( LayoutTag() );
            Kokkos::parallel_for( "Cabana::VerletList::fill_neighbors",
                                  fill_policy, builder );
        }
        Kokkos::fence();

        // Process the counts by computing offsets and allocating the neighbor
        // list, if needed.
        builder.processCounts( LayoutTag() );

        // For each particle in the range fill (or refill) its part of the
        // neighbor list.
        if ( builder.count or builder.refill )
        {
            Kokkos::parallel_for( "Cabana::VerletList::fill_neighbors",
                                  fill_policy, builder );
            Kokkos::fence();
        }

        // Get the data from the builder.
        _data = builder._data;
    }

    //! Modify a neighbor in the list; for example, mark it as a broken bond.
    KOKKOS_INLINE_FUNCTION
    void setNeighbor( const std::size_t particle_index,
                      const std::size_t neighbor_index,
                      const int new_index ) const
    {
        _data.setNeighbor( particle_index, neighbor_index, new_index );
    }
};

//---------------------------------------------------------------------------//
// Neighbor list interface implementation.
//---------------------------------------------------------------------------//
//! CSR VerletList NeighborList interface.
template <class MemorySpace, class AlgorithmTag, class BuildTag,
          std::size_t Dim>
class NeighborList<
    VerletList<MemorySpace, AlgorithmTag, VerletLayoutCSR, BuildTag, Dim>>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Neighbor list type.
    using list_type =
        VerletList<MemorySpace, AlgorithmTag, VerletLayoutCSR, BuildTag, Dim>;

    //! Get the total number of neighbors across all particles.
    KOKKOS_INLINE_FUNCTION
    static std::size_t totalNeighbor( const list_type& list )
    {
        // Size of the allocated memory gives total neighbors.
        return list._data.neighbors.extent( 0 );
    }

    //! Get the maximum number of neighbors across all particles.
    KOKKOS_INLINE_FUNCTION
    static std::size_t maxNeighbor( const list_type& list )
    {
        std::size_t num_p = list._data.counts.size();
        return Impl::maxNeighbor( list, num_p );
    }

    //! Get the number of neighbors for a given particle index.
    KOKKOS_INLINE_FUNCTION
    static std::size_t numNeighbor( const list_type& list,
                                    const std::size_t particle_index )
    {
        return list._data.counts( particle_index );
    }

    //! Get the id for a neighbor for a given particle index and the index of
    //! the neighbor relative to the particle.
    KOKKOS_INLINE_FUNCTION
    static std::size_t getNeighbor( const list_type& list,
                                    const std::size_t particle_index,
                                    const std::size_t neighbor_index )
    {
        return list._data.neighbors( list._data.offsets( particle_index ) +
                                     neighbor_index );
    }
};

//---------------------------------------------------------------------------//
//! 2D VerletList NeighborList interface.
template <class MemorySpace, class AlgorithmTag, class BuildTag,
          std::size_t Dim>
class NeighborList<
    VerletList<MemorySpace, AlgorithmTag, VerletLayout2D, BuildTag, Dim>>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Neighbor list type.
    using list_type =
        VerletList<MemorySpace, AlgorithmTag, VerletLayout2D, BuildTag, Dim>;

    //! Get the total number of neighbors across all particles.
    KOKKOS_INLINE_FUNCTION
    static std::size_t totalNeighbor( const list_type& list )
    {
        std::size_t num_p = list._data.counts.size();
        return Impl::totalNeighbor( list, num_p );
    }

    //! Get the maximum number of neighbors per particle.
    KOKKOS_INLINE_FUNCTION
    static std::size_t maxNeighbor( const list_type& list )
    {
        // Stored during neighbor search.
        return list._data.max_n;
    }

    //! Get the number of neighbors for a given particle index.
    KOKKOS_INLINE_FUNCTION
    static std::size_t numNeighbor( const list_type& list,
                                    const std::size_t particle_index )
    {
        return list._data.counts( particle_index );
    }

    //! Get the id for a neighbor for a given particle index and the index of
    //! the neighbor relative to the particle.
    KOKKOS_INLINE_FUNCTION
    static std::size_t getNeighbor( const list_type& list,
                                    const std::size_t particle_index,
                                    const std::size_t neighbor_index )
    {
        return list._data.neighbors( particle_index, neighbor_index );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end  CABANA_VERLETLIST_HPP
