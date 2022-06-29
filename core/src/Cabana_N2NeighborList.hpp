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
  \file Cabana_N2NeighborList.hpp
  \brief Neighbor list without grid acceleration
*/
#ifndef CABANA_N2LIST_HPP
#define CABANA_N2LIST_HPP

#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace Cabana
{

namespace Impl
{
//! \cond Impl

struct N2ListTag
{
};

//---------------------------------------------------------------------------//
// Neighbor List Builder
//---------------------------------------------------------------------------//
template <class DeviceType, class PositionSlice, class AlgorithmTag,
          class LayoutTag, class BuildOpTag>
struct N2NeighborListBuilder
{
    // Types.
    using device = DeviceType;
    using PositionValueType = typename PositionSlice::value_type;
    using RandomAccessPositionSlice =
        typename PositionSlice::random_access_slice;
    using memory_space = typename device::memory_space;
    using execution_space = typename device::execution_space;

    // List data.
    NeighborListData<memory_space, LayoutTag> _data;

    // Neighbor cutoff.
    PositionValueType rsqr;

    // Positions.
    RandomAccessPositionSlice position;
    std::size_t pid_begin, pid_end;

    // Check to count or refill.
    bool refill;
    bool count;

    // Maximum neighbors per particle
    std::size_t max_n;

    // Constructor.
    N2NeighborListBuilder( PositionSlice slice, const std::size_t begin,
                           const std::size_t end,
                           const PositionValueType neighborhood_radius,
                           const std::size_t max_neigh )
        : pid_begin( begin )
        , pid_end( end )
        , max_n( max_neigh )
    {
        count = true;
        refill = false;

        // Create the count view.
        _data.counts =
            Kokkos::View<int*, memory_space>( "num_neighbors", slice.size() );

        // Make a guess for the number of neighbors per particle for 2D
        // lists.
        initCounts( LayoutTag() );

        // Get the positions with random access read-only memory.
        position = slice;

        // We will use the square of the distance for neighbor
        // determination.
        rsqr = neighborhood_radius * neighborhood_radius;
    }

    // Neighbor count team operator (only used for CSR lists).
    struct CountNeighborsTag
    {
    };
    KOKKOS_INLINE_FUNCTION
    void operator()( const CountNeighborsTag&, const int pid ) const
    {
        if ( ( pid >= pid_begin ) && ( pid < pid_end ) )
        {
            // Cache the particle coordinates.
            double x_p = position( pid, 0 );
            double y_p = position( pid, 1 );
            double z_p = position( pid, 2 );

            // Check to see if the particles are neighbors.
            int count = 0;
            // Note that we loop over all particles as potential neighbors
            // of particles in the given range.
            for ( std::size_t nid = 0; nid < position.size(); nid++ )
            // Check to see if the particles are neighbors.
            {
                neighbor_kernel( pid, x_p, y_p, z_p, nid, count );
            }
            _data.counts( pid ) = count;
        }
    }

    using CountNeighborsPolicy =
        Kokkos::TeamPolicy<execution_space, CountNeighborsTag,
                           Kokkos::IndexType<int>,
                           Kokkos::Schedule<Kokkos::Dynamic>>;
    KOKKOS_INLINE_FUNCTION
    void
    operator()( const CountNeighborsTag&,
                const typename CountNeighborsPolicy::member_type& team ) const
    {
        // The league rank of the team is the current particle.
        std::size_t pid = team.league_rank() + pid_begin;
        if ( ( pid >= pid_begin ) && ( pid < pid_end ) )
        {
            // Cache the particle coordinates.
            double x_p = position( pid, 0 );
            double y_p = position( pid, 1 );
            double z_p = position( pid, 2 );

            // Check to see if the particles are neighbors.
            int count = 0;
            // Note that we loop over all particles as potential neighbors
            // of particles in the given range.
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange( team, position.size() ),
                [&]( const int nid, int& local_count )
                {
                    // Check to see if the particles are neighbors.
                    neighbor_kernel( pid, x_p, y_p, z_p, nid, local_count );
                },
                count );
            _data.counts( pid ) = count;
        }
    }

    // Neighbor count kernel
    KOKKOS_INLINE_FUNCTION
    void neighbor_kernel( const int pid, const double x_p, const double y_p,
                          const double z_p, const int nid,
                          int& local_count ) const
    {
        // Cache the candidate neighbor particle coordinates.
        double x_n = position( nid, 0 );
        double y_n = position( nid, 1 );
        double z_n = position( nid, 2 );

        // If this could be a valid neighbor, continue.
        if ( NeighborDiscriminator<AlgorithmTag>::isValid(
                 pid, x_p, y_p, z_p, nid, x_n, y_n, z_n ) )
        {
            // Calculate the distance between the particle and its candidate
            // neighbor.
            PositionValueType dx = x_p - x_n;
            PositionValueType dy = y_p - y_n;
            PositionValueType dz = z_p - z_n;
            PositionValueType dist_sqr = dx * dx + dy * dy + dz * dz;

            // If within the cutoff add to the count.
            if ( dist_sqr <= rsqr )
                local_count += 1;
        }
    }

    // Process the CSR counts by computing offsets and allocating the
    // neighbor list.
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

    void initCounts( NeighborLayoutCSR ) {}

    void initCounts( NeighborLayout2D )
    {
        if ( max_n > 0 )
        {
            count = false;

            _data.neighbors = Kokkos::View<int**, memory_space>(
                Kokkos::ViewAllocateWithoutInitializing( "neighbors" ),
                _data.counts.size(), max_n );
        }
    }

    void processCounts( NeighborLayoutCSR )
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
        Kokkos::parallel_scan( "Cabana::NeighborListBuilder::offset_scan",
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
    void processCounts( NeighborLayout2D )
    {
        // Calculate the maximum number of neighbors.
        auto counts = _data.counts;
        int max_num_neighbor = 0;
        Kokkos::Max<int> max_reduce( max_num_neighbor );
        Kokkos::parallel_reduce(
            "Cabana::NeighborListBuilder::reduce_max",
            Kokkos::RangePolicy<execution_space>( 0, _data.counts.size() ),
            KOKKOS_LAMBDA( const int i, int& value ) {
                if ( counts( i ) > value )
                    value = counts( i );
            },
            max_reduce );
        Kokkos::fence();

        // Reallocate the neighbor list if previous size is exceeded.
        if ( count or ( std::size_t )
                              max_num_neighbor > _data.neighbors.extent( 1 ) )
        {
            refill = true;
            Kokkos::deep_copy( _data.counts, 0 );
            _data.neighbors = Kokkos::View<int**, memory_space>(
                Kokkos::ViewAllocateWithoutInitializing( "neighbors" ),
                _data.counts.size(), max_num_neighbor );
        }
    }

    // Compatibility wrapper for old tags.
    void processCounts( VerletLayoutCSR )
    {
        processCounts( NeighborLayoutCSR{} );
    }
    void processCounts( VerletLayout2D )
    {
        processCounts( NeighborLayout2D{} );
    }
    void initCounts( VerletLayoutCSR ) { initCounts( NeighborLayoutCSR{} ); }
    void initCounts( VerletLayout2D ) { initCounts( NeighborLayout2D{} ); }

    // Neighbor count team operator.
    struct FillNeighborsTag
    {
    };
    KOKKOS_INLINE_FUNCTION
    void operator()( const FillNeighborsTag&, const int pid ) const
    {
        if ( ( pid >= pid_begin ) && ( pid < pid_end ) )
        {
            // Cache the particle coordinates.
            double x_p = position( pid, 0 );
            double y_p = position( pid, 1 );
            double z_p = position( pid, 2 );

            // Note that we loop over all particles as potential neighbors
            // of particles in the given range.
            for ( std::size_t nid = 0; nid < position.size(); nid++ )
                // Check to see if the particles are neighbors.
                neighbor_kernel( pid, x_p, y_p, z_p, nid );
        }
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
        // The league rank of the team is the current particle.
        std::size_t pid = team.league_rank() + pid_begin;
        if ( ( pid >= pid_begin ) && ( pid < pid_end ) )
        {
            // Cache the particle coordinates.
            double x_p = position( pid, 0 );
            double y_p = position( pid, 1 );
            double z_p = position( pid, 2 );

            // Note that we loop over all particles as potential neighbors
            // of particles in the given range.
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange( team, position.size() ),
                [&]( const int nid )
                {
                    // Check to see if the particles are neighbors.
                    neighbor_kernel( pid, x_p, y_p, z_p, nid );
                } );
        }
    };

    // Neighbor fill kernel.
    KOKKOS_INLINE_FUNCTION
    void neighbor_kernel( const int pid, const double x_p, const double y_p,
                          const double z_p, const int nid ) const
    {
        // Cache the candidate neighbor particle coordinates.
        double x_n = position( nid, 0 );
        double y_n = position( nid, 1 );
        double z_n = position( nid, 2 );

        // If this could be a valid neighbor, continue.
        if ( NeighborDiscriminator<AlgorithmTag>::isValid(
                 pid, x_p, y_p, z_p, nid, x_n, y_n, z_n ) )
        {
            // Calculate the distance between the particle and its candidate
            // neighbor.
            PositionValueType dx = x_p - x_n;
            PositionValueType dy = y_p - y_n;
            PositionValueType dz = z_p - z_n;
            PositionValueType dist_sqr = dx * dx + dy * dy + dz * dz;

            // If within the cutoff increment the neighbor count and add as
            // a neighbor at that index.
            if ( dist_sqr <= rsqr )
            {
                _data.addNeighbor( pid, nid );
            }
        }
    }
};

//---------------------------------------------------------------------------//

//! \endcond
} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Neighbor list implementation based on interaction distance.

  \tparam MemorySpace The Kokkos memory space for storing the neighbor list.

  \tparam AlgorithmTag Tag indicating whether to build a full or half
  neighbor list.

  \tparam LayoutTag Tag indicating whether to use a CSR or 2D data layout.

  \tparam BuildTag Tag indicating whether to use serial or team parallelism when
  building neighbor lists.
*/
template <class MemorySpace, class AlgorithmTag, class LayoutTag,
          class BuildTag = TeamOpTag>
class N2NeighborList
{
  public:
    static_assert( Kokkos::is_memory_space<MemorySpace>::value, "" );

    //! Kokkos memory space in which the neighbor list data resides.
    using memory_space = MemorySpace;

    //! Kokkos default execution space for this memory space.
    using execution_space = typename memory_space::execution_space;

    //! Neighbor list data.
    NeighborListData<memory_space, LayoutTag> _data;

    /*!
      \brief Default constructor.
    */
    N2NeighborList() {}

    /*!
      \brief N2NeighborList constructor. Given a list of particle positions
      and a neighborhood radius calculate the neighbor list.

      \param x The slice containing the particle positions

      \param begin The beginning particle index to compute neighbors for.

      \param end The end particle index to compute neighbors for.

      \param neighborhood_radius The radius of the neighborhood. Particles
      within this radius are considered neighbors.

      \param max_neigh Optional maximum number of neighbors per particle to
      pre-allocate the neighbor list. Potentially avoids recounting with 2D
      layout only.

      Particles outside of the neighborhood radius will not be considered
      neighbors. Only compute the neighbors of those that are within the
      given range. All particles are candidates for being a neighbor,
      regardless of whether or not they are in the range.
    */
    template <class PositionSlice>
    N2NeighborList(
        PositionSlice x, const std::size_t begin, const std::size_t end,
        const typename PositionSlice::value_type neighborhood_radius,
        const std::size_t max_neigh = 0,
        typename std::enable_if<( is_slice<PositionSlice>::value ),
                                int>::type* = 0 )
    {
        build( x, begin, end, neighborhood_radius, max_neigh );
    }

    /*!
      \brief Given a list of particle positions and a neighborhood radius
      calculate the neighbor list.
    */
    template <class PositionSlice>
    void build( PositionSlice x, const std::size_t begin, const std::size_t end,
                const typename PositionSlice::value_type neighborhood_radius,
                const std::size_t max_neigh = 0 )
    {
        // Use the default execution space.
        build( execution_space{}, x, begin, end, neighborhood_radius,
               max_neigh );
    }
    /*!
      \brief Given a list of particle positions and a neighborhood radius
      calculate the neighbor list.
    */
    template <class PositionSlice, class ExecutionSpace>
    void build( ExecutionSpace, PositionSlice x, const std::size_t begin,
                const std::size_t end,
                const typename PositionSlice::value_type neighborhood_radius,
                const std::size_t max_neigh = 0 )
    {
        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );

        assert( end >= begin );
        assert( end <= x.size() );

        using device_type = Kokkos::Device<ExecutionSpace, memory_space>;

        // Create a builder functor.
        using builder_type =
            Impl::N2NeighborListBuilder<device_type, PositionSlice,
                                        AlgorithmTag, LayoutTag, BuildTag>;
        builder_type builder( x, begin, end, neighborhood_radius, max_neigh );

        // For each particle in the range check for neighbor particles. For CSR
        // lists, we count, then fill neighbors. For 2D lists, we count and fill
        // at the same time, unless the array size is exceeded, at which point
        // only counting is continued to reallocate and refill.
        typename builder_type::FillNeighborsPolicy fill_policy(
            end - begin, Kokkos::AUTO, 4 );
        if ( builder.count )
        {
            typename builder_type::CountNeighborsPolicy count_policy(
                end - begin, Kokkos::AUTO, 4 );
            Kokkos::parallel_for( "Cabana::N2NeighborList::count_neighbors",
                                  count_policy, builder );
        }
        else
        {
            builder.processCounts( LayoutTag() );
            Kokkos::parallel_for( "Cabana::N2NeighborList::fill_neighbors",
                                  fill_policy, builder );
        }
        Kokkos::fence();

        // Process the counts by computing offsets and allocating the
        // neighbor list, if needed.
        builder.processCounts( LayoutTag() );

        // For each particle in the range fill (or refill) its part of the
        // neighbor list.
        if ( builder.count or builder.refill )
        {
            Kokkos::parallel_for( "Cabana::N2NeighborList::fill_neighbors",
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
//! CSR N2 NeighborList interface.
template <class MemorySpace, class AlgorithmTag, class BuildTag>
class NeighborList<
    N2NeighborList<MemorySpace, AlgorithmTag, NeighborLayoutCSR, BuildTag>>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Neighbor list type.
    using list_type =
        N2NeighborList<MemorySpace, AlgorithmTag, NeighborLayoutCSR, BuildTag>;

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

    //! Get the id for a neighbor for a given particle index and the index
    //! of the neighbor relative to the particle.
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
//! 2D N2 NeighborList interface.
template <class MemorySpace, class AlgorithmTag, class BuildTag>
class NeighborList<
    N2NeighborList<MemorySpace, AlgorithmTag, NeighborLayout2D, BuildTag>>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Neighbor list type.
    using list_type =
        N2NeighborList<MemorySpace, AlgorithmTag, NeighborLayout2D, BuildTag>;

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

    //! Get the id for a neighbor for a given particle index and the index
    //! of the neighbor relative to the particle.
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

#endif // end  CABANA_N2LIST_HPP
