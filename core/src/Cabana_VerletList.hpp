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
  \file Cabana_VerletList.hpp
  \brief Verlet grid-accelerated neighbor list
*/
#ifndef CABANA_VERLETLIST_HPP
#define CABANA_VERLETLIST_HPP

#include <Cabana_LinkedCellList.hpp>
#include <Cabana_N2NeighborList.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>
#include <impl/Cabana_CartesianGrid.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace Cabana
{

namespace Impl
{
//! \cond Impl

//---------------------------------------------------------------------------//
// Cell stencil.
template <class Scalar>
struct LinkedCellStencil
{
    Scalar rsqr;
    CartesianGrid<double> grid;
    int max_cells_dir;
    int max_cells;
    int cell_range;

    LinkedCellStencil( const Scalar neighborhood_radius,
                       const Scalar cell_size_ratio, const Scalar grid_min[3],
                       const Scalar grid_max[3] )
        : rsqr( neighborhood_radius * neighborhood_radius )
    {
        Scalar dx = neighborhood_radius * cell_size_ratio;
        grid = CartesianGrid<double>( grid_min[0], grid_min[1], grid_min[2],
                                      grid_max[0], grid_max[1], grid_max[2], dx,
                                      dx, dx );
        cell_range = std::ceil( 1 / cell_size_ratio );
        max_cells_dir = 2 * cell_range + 1;
        max_cells = max_cells_dir * max_cells_dir * max_cells_dir;
    }

    // Given a cell, get the index bounds of the cell stencil.
    KOKKOS_INLINE_FUNCTION
    void getCells( const int cell, int& imin, int& imax, int& jmin, int& jmax,
                   int& kmin, int& kmax ) const
    {
        int i, j, k;
        grid.ijkBinIndex( cell, i, j, k );

        kmin = ( k - cell_range > 0 ) ? k - cell_range : 0;
        kmax =
            ( k + cell_range + 1 < grid._nz ) ? k + cell_range + 1 : grid._nz;

        jmin = ( j - cell_range > 0 ) ? j - cell_range : 0;
        jmax =
            ( j + cell_range + 1 < grid._ny ) ? j + cell_range + 1 : grid._ny;

        imin = ( i - cell_range > 0 ) ? i - cell_range : 0;
        imax =
            ( i + cell_range + 1 < grid._nx ) ? i + cell_range + 1 : grid._nx;
    }
};

//---------------------------------------------------------------------------//
// Verlet List Builder
//---------------------------------------------------------------------------//
template <class DeviceType, class PositionSlice, class AlgorithmTag,
          class LayoutTag, class BuildOpTag>
struct VerletListBuilder
    : public N2NeighborListBuilder<DeviceType, PositionSlice, AlgorithmTag,
                                   LayoutTag, BuildOpTag>
{
    // Types.
    using n2_list_type =
        N2NeighborListBuilder<DeviceType, PositionSlice, AlgorithmTag,
                              LayoutTag, BuildOpTag>;
    using PositionValueType = typename n2_list_type::PositionValueType;
    using device = DeviceType;
    using memory_space = typename device::memory_space;
    using execution_space = typename device::execution_space;

    // List data.
    using n2_list_type::_data;

    // Neighbor cutoff.
    using n2_list_type::rsqr;

    // Positions.
    using n2_list_type::pid_begin;
    using n2_list_type::pid_end;
    using n2_list_type::position;

    // Check to count or refill.
    using n2_list_type::count;
    using n2_list_type::refill;

    // Maximum neighbors per particle
    using n2_list_type::max_n;

    using CountNeighborsPolicy = typename n2_list_type::CountNeighborsPolicy;
    using CountNeighborsTag = typename n2_list_type::CountNeighborsTag;
    using FillNeighborsPolicy = typename n2_list_type::FillNeighborsPolicy;
    using FillNeighborsTag = typename n2_list_type::FillNeighborsTag;

    // Binning Data.
    BinningData<device> bin_data_1d;
    LinkedCellList<device> linked_cell_list;

    // Cell stencil.
    LinkedCellStencil<PositionValueType> cell_stencil;

    // Constructor.
    VerletListBuilder( PositionSlice slice, const std::size_t begin,
                       const std::size_t end,
                       const PositionValueType neighborhood_radius,
                       const PositionValueType cell_size_ratio,
                       const PositionValueType grid_min[3],
                       const PositionValueType grid_max[3],
                       const std::size_t max_neigh )
        : n2_list_type( slice, begin, end, neighborhood_radius, max_neigh )
        , cell_stencil( neighborhood_radius, cell_size_ratio, grid_min,
                        grid_max )
    {
        // Bin the particles in the grid. Don't actually sort them but make a
        // permutation vector. Note that we are binning all particles here and
        // not just the requested range. This is because all particles are
        // treated as candidates for neighbors.
        double grid_size = cell_size_ratio * neighborhood_radius;
        PositionValueType grid_delta[3] = { grid_size, grid_size, grid_size };
        linked_cell_list =
            LinkedCellList<device>( position, grid_delta, grid_min, grid_max );
        bin_data_1d = linked_cell_list.binningData();
    }

    // Neighbor count team operator (only used for CSR lists).
    KOKKOS_INLINE_FUNCTION
    void
    operator()( const CountNeighborsTag&,
                const typename CountNeighborsPolicy::member_type& team ) const
    {
        // The league rank of the team is the cardinal cell index we are
        // working on.
        int cell = team.league_rank();

        // Get the stencil for this cell.
        int imin, imax, jmin, jmax, kmin, kmax;
        cell_stencil.getCells( cell, imin, imax, jmin, jmax, kmin, kmax );

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
                    double x_p = position( pid, 0 );
                    double y_p = position( pid, 1 );
                    double z_p = position( pid, 2 );

                    // Loop over the cell stencil.
                    int stencil_count = 0;
                    for ( int i = imin; i < imax; ++i )
                        for ( int j = jmin; j < jmax; ++j )
                            for ( int k = kmin; k < kmax; ++k )
                            {
                                // See if we should actually check this box for
                                // neighbors.
                                if ( cell_stencil.grid.minDistanceToPoint(
                                         x_p, y_p, z_p, i, j, k ) <= rsqr )
                                {
                                    std::size_t n_offset =
                                        linked_cell_list.binOffset( i, j, k );
                                    std::size_t num_n =
                                        linked_cell_list.binSize( i, j, k );

                                    // Check the particles in this bin to see if
                                    // they are neighbors. If they are add to
                                    // the count for this bin.
                                    int cell_count = 0;
                                    neighbor_reduce( team, pid, x_p, y_p, z_p,
                                                     n_offset, num_n,
                                                     cell_count, BuildOpTag() );
                                    stencil_count += cell_count;
                                }
                            }
                    Kokkos::single( Kokkos::PerThread( team ), [&]()
                                    { _data.counts( pid ) = stencil_count; } );
                }
            } );
    }

    // Neighbor count team vector loop (only used for CSR lists).
    KOKKOS_INLINE_FUNCTION void
    neighbor_reduce( const typename CountNeighborsPolicy::member_type& team,
                     const std::size_t pid, const double x_p, const double y_p,
                     const double z_p, const int n_offset, const int num_n,
                     int& cell_count, TeamVectorOpTag ) const
    {
        Kokkos::parallel_reduce(
            Kokkos::ThreadVectorRange( team, num_n ),
            [&]( const int n, int& local_count )
            {
                //  Get the true id of the candidate  neighbor.
                std::size_t nid = linked_cell_list.permutation( n_offset + n );

                this->neighbor_kernel( pid, x_p, y_p, z_p, nid, local_count );
            },
            cell_count );
    }

    // Neighbor count serial loop (only used for CSR lists).
    KOKKOS_INLINE_FUNCTION
    void neighbor_reduce( const typename CountNeighborsPolicy::member_type,
                          const std::size_t pid, const double x_p,
                          const double y_p, const double z_p,
                          const int n_offset, const int num_n, int& cell_count,
                          TeamOpTag ) const
    {
        for ( int n = 0; n < num_n; n++ )
        {
            //  Get the true id of the candidate  neighbor.
            std::size_t nid = linked_cell_list.permutation( n_offset + n );

            this->neighbor_kernel( pid, x_p, y_p, z_p, nid, cell_count );
        }
    }

    KOKKOS_INLINE_FUNCTION
    void
    operator()( const FillNeighborsTag&,
                const typename FillNeighborsPolicy::member_type& team ) const
    {
        // The league rank of the team is the cardinal cell index we are
        // working on.
        int cell = team.league_rank();

        // Get the stencil for this cell.
        int imin, imax, jmin, jmax, kmin, kmax;
        cell_stencil.getCells( cell, imin, imax, jmin, jmax, kmin, kmax );

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
                    double x_p = position( pid, 0 );
                    double y_p = position( pid, 1 );
                    double z_p = position( pid, 2 );

                    // Loop over the cell stencil.
                    for ( int i = imin; i < imax; ++i )
                        for ( int j = jmin; j < jmax; ++j )
                            for ( int k = kmin; k < kmax; ++k )
                            {
                                // See if we should actually check this box for
                                // neighbors.
                                if ( cell_stencil.grid.minDistanceToPoint(
                                         x_p, y_p, z_p, i, j, k ) <= rsqr )
                                {
                                    // Check the particles in this bin to see if
                                    // they are neighbors.
                                    std::size_t n_offset =
                                        linked_cell_list.binOffset( i, j, k );
                                    int num_n =
                                        linked_cell_list.binSize( i, j, k );
                                    neighbor_for( team, pid, x_p, y_p, z_p,
                                                  n_offset, num_n,
                                                  BuildOpTag() );
                                }
                            }
                }
            } );
    }

    // Neighbor fill team vector loop.
    KOKKOS_INLINE_FUNCTION void
    neighbor_for( const typename FillNeighborsPolicy::member_type& team,
                  const std::size_t pid, const double x_p, const double y_p,
                  const double z_p, const int n_offset, const int num_n,
                  TeamVectorOpTag ) const
    {
        Kokkos::parallel_for(
            Kokkos::ThreadVectorRange( team, num_n ),
            [&]( const int n )
            {
                // Get the true id of the candidate neighbor.
                std::size_t nid = linked_cell_list.permutation( n_offset + n );

                this->neighbor_kernel( pid, x_p, y_p, z_p, nid );
            } );
    }

    // Neighbor fill serial loop.
    KOKKOS_INLINE_FUNCTION
    void neighbor_for( const typename FillNeighborsPolicy::member_type team,
                       const std::size_t pid, const double x_p,
                       const double y_p, const double z_p, const int n_offset,
                       const int num_n, TeamOpTag ) const
    {
        for ( int n = 0; n < num_n; n++ )
            Kokkos::single(
                Kokkos::PerThread( team ),
                [&]()
                {
                    // Get the true id of the candidate neighbor.
                    std::size_t nid =
                        linked_cell_list.permutation( n_offset + n );

                    this->neighbor_kernel( pid, x_p, y_p, z_p, nid );
                } );
    }
};

//---------------------------------------------------------------------------//

//! \endcond
} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Neighbor list implementation based on binning particles on a 3d
  Cartesian grid with cells of the same size as the interaction distance.

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
          class BuildTag = TeamVectorOpTag>
class VerletList
{
  public:
    static_assert( Kokkos::is_memory_space<MemorySpace>::value, "" );

    //! Kokkos memory space in which the neighbor list data resides.
    using memory_space = MemorySpace;

    //! Kokkos default execution space for this memory space.
    using execution_space = typename memory_space::execution_space;

    //! Verlet list data.
    NeighborListData<memory_space, LayoutTag> _data;

    /*!
      \brief Default constructor.
    */
    VerletList() {}

    /*!
      \brief VerletList constructor. Given a list of particle positions and
      a neighborhood radius calculate the neighbor list.

      \param x The slice containing the particle positions

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
    template <class PositionSlice>
    VerletList( PositionSlice x, const std::size_t begin, const std::size_t end,
                const typename PositionSlice::value_type neighborhood_radius,
                const typename PositionSlice::value_type cell_size_ratio,
                const typename PositionSlice::value_type grid_min[3],
                const typename PositionSlice::value_type grid_max[3],
                const std::size_t max_neigh = 0,
                typename std::enable_if<( is_slice<PositionSlice>::value ),
                                        int>::type* = 0 )
    {
        build( x, begin, end, neighborhood_radius, cell_size_ratio, grid_min,
               grid_max, max_neigh );
    }

    /*!
      \brief Given a list of particle positions and a neighborhood radius
      calculate the neighbor list.
    */
    template <class PositionSlice>
    void build( PositionSlice x, const std::size_t begin, const std::size_t end,
                const typename PositionSlice::value_type neighborhood_radius,
                const typename PositionSlice::value_type cell_size_ratio,
                const typename PositionSlice::value_type grid_min[3],
                const typename PositionSlice::value_type grid_max[3],
                const std::size_t max_neigh = 0 )
    {
        // Use the default execution space.
        build( execution_space{}, x, begin, end, neighborhood_radius,
               cell_size_ratio, grid_min, grid_max, max_neigh );
    }
    /*!
      \brief Given a list of particle positions and a neighborhood radius
      calculate the neighbor list.
    */
    template <class PositionSlice, class ExecutionSpace>
    void build( ExecutionSpace, PositionSlice x, const std::size_t begin,
                const std::size_t end,
                const typename PositionSlice::value_type neighborhood_radius,
                const typename PositionSlice::value_type cell_size_ratio,
                const typename PositionSlice::value_type grid_min[3],
                const typename PositionSlice::value_type grid_max[3],
                const std::size_t max_neigh = 0 )
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        assert( end >= begin );
        assert( end <= x.size() );

        using device_type = Kokkos::Device<ExecutionSpace, memory_space>;

        // Create a builder functor.
        using builder_type =
            Impl::VerletListBuilder<device_type, PositionSlice, AlgorithmTag,
                                    LayoutTag, BuildTag>;
        builder_type builder( x, begin, end, neighborhood_radius,
                              cell_size_ratio, grid_min, grid_max, max_neigh );

        // For each particle in the range check each neighboring bin for
        // neighbor particles. Bins are at least the size of the neighborhood
        // radius so the bin in which the particle resides and any surrounding
        // bins are guaranteed to contain the neighboring particles.
        // For CSR lists, we count, then fill neighbors. For 2D lists, we
        // count and fill at the same time, unless the array size is exceeded,
        // at which point only counting is continued to reallocate and refill.
        typename builder_type::FillNeighborsPolicy fill_policy(
            builder.bin_data_1d.numBin(), Kokkos::AUTO, 4 );
        if ( builder.count )
        {
            typename builder_type::CountNeighborsPolicy count_policy(
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
};

//---------------------------------------------------------------------------//
// Neighbor list interface implementation.
//---------------------------------------------------------------------------//
//! CSR VerletList NeighborList interface.
template <class MemorySpace, class AlgorithmTag, class BuildTag>
class NeighborList<
    VerletList<MemorySpace, AlgorithmTag, NeighborLayoutCSR, BuildTag>>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Neighbor list type.
    using list_type =
        VerletList<MemorySpace, AlgorithmTag, NeighborLayoutCSR, BuildTag>;

    //! Get the total number of neighbors (maximum size of CSR list).
    KOKKOS_INLINE_FUNCTION
    static std::size_t maxNeighbor( const list_type& list )
    {
        return list._data.neighbors.extent( 0 );
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
template <class MemorySpace, class AlgorithmTag, class BuildTag>
class NeighborList<
    VerletList<MemorySpace, AlgorithmTag, NeighborLayout2D, BuildTag>>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Neighbor list type.
    using list_type =
        VerletList<MemorySpace, AlgorithmTag, NeighborLayout2D, BuildTag>;

    //! Get the maximum number of neighbors per particle.
    KOKKOS_INLINE_FUNCTION
    static std::size_t maxNeighbor( const list_type& list )
    {
        return list._data.neighbors.extent( 1 );
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

//! CSR VerletList NeighborList interface (backwards compatability copy).
template <class MemorySpace, class AlgorithmTag, class BuildTag>
class NeighborList<
    VerletList<MemorySpace, AlgorithmTag, VerletLayoutCSR, BuildTag>>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Neighbor list type.
    using list_type =
        VerletList<MemorySpace, AlgorithmTag, VerletLayoutCSR, BuildTag>;

    //! Get the total number of neighbors (maximum size of CSR list).
    KOKKOS_INLINE_FUNCTION
    static std::size_t maxNeighbor( const list_type& list )
    {
        return list._data.neighbors.extent( 0 );
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

//! 2D VerletList NeighborList interface (backwards compatability copy).
template <class MemorySpace, class AlgorithmTag, class BuildTag>
class NeighborList<
    VerletList<MemorySpace, AlgorithmTag, VerletLayout2D, BuildTag>>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Neighbor list type.
    using list_type =
        VerletList<MemorySpace, AlgorithmTag, VerletLayout2D, BuildTag>;

    //! Get the maximum number of neighbors per particle.
    KOKKOS_INLINE_FUNCTION
    static std::size_t maxNeighbor( const list_type& list )
    {
        return list._data.neighbors.extent( 1 );
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
