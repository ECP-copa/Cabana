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

#ifndef CABANA_VERLETLIST_HPP
#define CABANA_VERLETLIST_HPP

#include <Cabana_NeighborList.hpp>
#include <Cabana_LinkedCellList.hpp>
#include <impl/Cabana_CartesianGrid.hpp>

#include <Kokkos_Core.hpp>

namespace Cabana
{
namespace Impl
{
//---------------------------------------------------------------------------//
// Neighborhood discriminator.
template<class Tag>
class NeighborDiscriminator;

// Full list specialization.
template<>
class NeighborDiscriminator<FullNeighborTag>
{
  public:
    // Full neighbor lists count and store the neighbors of all
    // particles. The only criteria for a potentially valid neighbor is
    // that the particle does not neighbor itself (i.e. the particle index
    // "p" is not the same as the neighbor index "n").
    KOKKOS_INLINE_FUNCTION
    static bool isValid( const std::size_t p,
                         const double, const double, const double,
                         const std::size_t n,
                         const double, const double, const double )
    {
        return ( p != n );
    }
};

// Half list specialization.
template<>
class NeighborDiscriminator<HalfNeighborTag>
{
  public:
    // Half neighbor lists only store half of the neighbors be eliminating
    // duplicate pairs such that the fact that particle "p" neighbors
    // particle "n" is stored in the list but "n" neighboring "p" is not
    // stored but rather implied. We discriminate by only storing neighbors
    // who's coordinates are greater in the x direction. If they are the same
    // then the y direction is checked next and finally the z direction if the
    // y coordinates are the same.
    KOKKOS_INLINE_FUNCTION
    static bool isValid( const std::size_t p,
                         const double xp, const double yp, const double zp,
                         const std::size_t n,
                         const double xn, const double yn, const double zn )
    {
        return ( (p != n) &&
                 ( (xn>xp)  ||
                   ( (xn==xp) && ( (yn>yp) || ((yn==yp) && (zn>zp) )))) );
    }
};

//---------------------------------------------------------------------------//
// Cell stencil.
template<class Scalar>
struct LinkedCellStencil
{
    Scalar rsqr;
    CartesianGrid<double> grid;
    int max_cells_dir;
    int max_cells;
    int cell_range;

    LinkedCellStencil( const Scalar neighborhood_radius,
                       const Scalar cell_size_ratio,
                       const Scalar grid_min[3],
                       const Scalar grid_max[3] )
        : rsqr( neighborhood_radius * neighborhood_radius )
    {
        Scalar dx = neighborhood_radius * cell_size_ratio;
        grid = CartesianGrid<double>( grid_min[0], grid_min[1], grid_min[2],
                                      grid_max[0], grid_max[1], grid_max[2],
                                      dx, dx, dx );
        cell_range = std::ceil( 1 / cell_size_ratio );
        max_cells_dir = 2 * cell_range + 1;
        max_cells = max_cells_dir * max_cells_dir * max_cells_dir;
    }

    // Given a cell, get the index bounds of the cell stencil.
    KOKKOS_INLINE_FUNCTION
    void getCells( const int cell,
                   int& imin,
                   int& imax,
                   int& jmin,
                   int& jmax,
                   int& kmin,
                   int& kmax ) const
    {
        int i, j, k;
        grid.ijkBinIndex( cell, i, j, k );

        kmin = (k - cell_range > 0) ? k - cell_range : 0;
        kmax = (k + cell_range + 1 < grid._nz)
               ? k + cell_range + 1 : grid._nz;

        jmin = (j - cell_range > 0) ? j - cell_range : 0;
        jmax = (j + cell_range + 1 < grid._ny)
               ? j + cell_range + 1 : grid._ny;

        imin = (i - cell_range > 0) ? i - cell_range : 0;
        imax = (i + cell_range + 1 < grid._nx)
               ? i + cell_range + 1 : grid._nx;
    }
};

//---------------------------------------------------------------------------//
template<class PositionSlice, class AlgorithmTag>
struct VerletListBuilder
{
    // Types.
    using PositionValueType = typename PositionSlice::value_type;
    using RandomAccessPositionSlice =
        typename PositionSlice::random_access_slice;
    using memory_space = typename PositionSlice::memory_space;
    using execution_space = typename PositionSlice::execution_space;

    // Number of neighbors per particle.
    Kokkos::View<int*,memory_space> counts;

    // Offsets into the neighbor list.
    Kokkos::View<int*,memory_space> offsets;

    // Neighbor list.
    Kokkos::View<int*,memory_space> neighbors;

    // Neighbor cutoff.
    PositionValueType rsqr;

    // Positions.
    RandomAccessPositionSlice position;
    std::size_t pid_begin, pid_end;

    // Binning Data.
    BinningData<memory_space> bin_data_1d;
    LinkedCellList<memory_space> linked_cell_list;

    // Cell stencil.
    LinkedCellStencil<PositionValueType> cell_stencil;

    // Constructor.
    VerletListBuilder(
        PositionSlice slice,
        const std::size_t begin,
        const std::size_t end,
        const PositionValueType neighborhood_radius,
        const PositionValueType cell_size_ratio,
        const PositionValueType grid_min[3],
        const PositionValueType grid_max[3])
        : counts( "num_neighbors", slice.size() )
        , offsets( "neighbor_offsets", slice.size() )
        , pid_begin(begin), pid_end(end)
        , cell_stencil( neighborhood_radius, cell_size_ratio, grid_min, grid_max )
    {
        // Get the positions with random access read-only memory.
        position = slice;

        // Bin the particles in the grid. Don't actually sort them but make a
        // permutation vector. Note that we are binning all particles here and
        // not just the requested range. This is because all particles are
        // treated as candidates for neighbors.
        double grid_size = cell_size_ratio * neighborhood_radius;
        PositionValueType grid_delta[3] = { grid_size, grid_size, grid_size };
        linked_cell_list = LinkedCellList<memory_space>(
            position, grid_delta, grid_min, grid_max );
        bin_data_1d = linked_cell_list.binningData();

        // We will use the square of the distance for neighbor determination.
        rsqr = neighborhood_radius * neighborhood_radius;
    }

    // Neighbor count team operator.
    struct CountNeighborsTag {};
    using CountNeighborsPolicy =
        Kokkos::TeamPolicy<execution_space,
                           CountNeighborsTag,
                           Kokkos::IndexType<int>,
                           Kokkos::Schedule<Kokkos::Dynamic> >;
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const CountNeighborsTag&,
        const typename CountNeighborsPolicy::member_type& team ) const
    {
        // The league rank of the team is the cardinal cell index we are
        // working on.
        int cell = team.league_rank();

        // Get the stencil for this cell.
        int imin, imax, jmin, jmax, kmin, kmax;
        cell_stencil.getCells( cell, imin, imax, jmin, jmax, kmin, kmax );

        // Operate on the particles in the bin.
        std::size_t b_offset = bin_data_1d.binOffset(cell);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team,0,bin_data_1d.binSize(cell)),
            [&] ( const int bi )
            {
                // Get the true particle id. The binned particle index is the
                // league rank of the team.
                std::size_t pid = linked_cell_list.permutation( bi + b_offset );

                if (( pid >= pid_begin ) && ( pid  < pid_end ))
                {
                    // Cache the particle coordinates.
                    double x_p = position(pid,0);
                    double y_p = position(pid,1);
                    double z_p = position(pid,2);

                    // Loop over the cell stencil.
                    int stencil_count = 0;
                    for ( int i = imin; i < imax; ++i )
                        for ( int j = jmin; j < jmax; ++j )
                            for ( int k = kmin; k < kmax; ++k )
                            {
                                // See if we should actually check this box for
                                // neighbors.
                                if ( cell_stencil.grid.minDistanceToPoint(x_p,y_p,z_p,i,j,k)
                                   <= rsqr )
                                {
                                    // Check the particles in this bin to see if they are
                                    // neighbors. If they are add to the count for this bin.
                                    int cell_count = 0;
                                    std::size_t a_offset = linked_cell_list.binOffset(i,j,k);
                                    Kokkos::parallel_reduce(
                                        Kokkos::ThreadVectorRange(
                                            team,linked_cell_list.binSize(i,j,k)),
                                        [&] ( const int n, int& local_count ) {

                                            //  Get the true id of the candidate neighbor.
                                            std::size_t nid =
                                                linked_cell_list.permutation( a_offset + n );

                                            // Cache the candidate neighbor particle
                                            // coordinates.
                                            double x_n = position(nid,0);
                                            double y_n = position(nid,1);
                                            double z_n = position(nid,2);

                                            // If this could be a valid neighbor, continue.
                                            if ( NeighborDiscriminator<AlgorithmTag>::isValid(
                                                     pid,x_p,y_p,z_p,nid,x_n,y_n,z_n) )
                                            {
                                                // Calculate the distance between the particle
                                                // and its candidate neighbor.
                                                PositionValueType dx = x_p - x_n;
                                                PositionValueType dy = y_p - y_n;
                                                PositionValueType dz = z_p - z_n;
                                                PositionValueType dist_sqr = dx*dx + dy*dy + dz*dz;

                                                // If within the cutoff add to the count.
                                                if ( dist_sqr <= rsqr )
                                                    local_count += 1;
                                            }
                                         },
                                        cell_count );
                                    stencil_count += cell_count;
                                }
                            }
                    Kokkos::single(Kokkos::PerThread(team), [&] () {
                        counts(pid) = stencil_count;
                    });
                }
            });
    }

    // Process the counts by computing offsets and allocating the neighbor
    // list.
    template<class KokkosMemorySpace>
    struct OffsetScanOp
    {
        using kokkos_mem_space = KokkosMemorySpace;
        Kokkos::View<int*,kokkos_mem_space> counts;
        Kokkos::View<int*,kokkos_mem_space> offsets;
        KOKKOS_INLINE_FUNCTION
        void operator()( const int i, int& update, const bool final_pass ) const
        {
            if ( final_pass ) offsets(i) = update;
            update += counts(i);
        }
    };
    void processCounts()
    {
        // Calculate offsets from counts and the total number of counts.
        OffsetScanOp<memory_space> offset_op;
        offset_op.counts = counts;
        offset_op.offsets = offsets;
        int total_num_neighbor;
        Kokkos::RangePolicy<execution_space> range_policy(
            0, counts.extent(0) );
        Kokkos::parallel_scan(
            "Cabana::VerletListBuilder::offset_scan",
            range_policy, offset_op, total_num_neighbor );
        Kokkos::fence();

        // Allocate the neighbor list.
        neighbors = Kokkos::View<int*,memory_space>(
            "neighbors", total_num_neighbor );

        // Reset the counts. We count again when we fill.
        Kokkos::deep_copy( counts, 0 );
    }

    // Neighbor count team operator.
    struct FillNeighborsTag {};
    using FillNeighborsPolicy =
        Kokkos::TeamPolicy<execution_space,
                           FillNeighborsTag,
                           Kokkos::IndexType<int>,
                           Kokkos::Schedule<Kokkos::Dynamic> >;
    KOKKOS_INLINE_FUNCTION
    void operator()(
        const FillNeighborsTag&,
        const typename FillNeighborsPolicy::member_type& team ) const
    {
        // The league rank of the team is the cardinal cell index we are
        // working on.
        int cell = team.league_rank();

        // Get the stencil for this cell.
        int imin, imax, jmin, jmax, kmin, kmax;
        cell_stencil.getCells( cell, imin, imax, jmin, jmax, kmin, kmax );

        // Operate on the particles in the bin.
        std::size_t b_offset = bin_data_1d.binOffset(cell);
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team,0,bin_data_1d.binSize(cell)),
            [&] ( const int bi )
            {
                // Get the true particle id. The binned particle index is the
                // league rank of the team.
                std::size_t pid = linked_cell_list.permutation( bi + b_offset );

                if (( pid >= pid_begin ) && ( pid  < pid_end ))
                {
                    // Cache the particle coordinates.
                    double x_p = position(pid,0);
                    double y_p = position(pid,1);
                    double z_p = position(pid,2);

                    // Loop over the cell stencil.
                    for ( int i = imin; i < imax; ++i )
                        for ( int j = jmin; j < jmax; ++j )
                            for ( int k = kmin; k < kmax; ++k )
                            {
                                // See if we should actually check this box for
                                // neighbors.
                                if ( cell_stencil.grid.minDistanceToPoint(x_p,y_p,z_p,i,j,k)
                                     <= rsqr )
                                {
                                    // Check the particles in this bin to see if they are
                                    // neighbors.
                                    std::size_t a_offset = linked_cell_list.binOffset(i,j,k);
                                    Kokkos::parallel_for(
                                        Kokkos::ThreadVectorRange(
                                            team,linked_cell_list.binSize(i,j,k)),
                                        [&] ( const int n ) {

                                            //  Get the true id of the candidate neighbor.
                                            std::size_t nid =
                                                linked_cell_list.permutation( a_offset + n );

                                            // Cache the candidate neighbor particle coordinates.
                                            double x_n = position(nid,0);
                                            double y_n = position(nid,1);
                                            double z_n = position(nid,2);

                                            // If this could be a valid neighbor, continue.
                                            if ( NeighborDiscriminator<AlgorithmTag>::isValid(
                                                     pid,x_p,y_p,z_p,nid,x_n,y_n,z_n) )
                                            {
                                                // Calculate the distance between the particle
                                                // and its candidate neighbor.
                                                PositionValueType dx = x_p - x_n;
                                                PositionValueType dy = y_p - y_n;
                                                PositionValueType dz = z_p - z_n;
                                                PositionValueType dist_sqr = dx*dx + dy*dy + dz*dz;

                                                // If within the cutoff increment the neighbor
                                                // count and add as a neighbor at that index.
                                                if ( dist_sqr <= rsqr )
                                                {
                                                    neighbors(
                                                        offsets(pid) +
                                                        Kokkos::atomic_fetch_add(&counts(pid),1) )
                                                        = nid;
                                                }
                                            }
                                        });
                                }
                            }
                }
            });
    }
};

//---------------------------------------------------------------------------//

} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \class VerletList
  \brief Neighbor list implementation based on binning particles on a 3d
  Cartesian grid with cells of the same size as the interaction distance.

  Neighbor list implementation most appropriate for somewhat regularly
  distributed particles due to the use of a Cartesian grid.
*/
template<class MemorySpace, class AlgorithmTag>
class VerletList
{
  public:

    // The memory space in which the neighbor list data resides.
    using memory_space = MemorySpace;

    // Number of neighbors per particle.
    Kokkos::View<int*,memory_space> _counts;

    // Offsets into the neighbor list.
    Kokkos::View<int*,memory_space> _offsets;

    // Neighbor list.
    Kokkos::View<int*,memory_space> _neighbors;

    /*!
      \brief Given a list of particle positions and a neighborhood radius calculate
      the neighbor list.

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

      Particles outside of the neighborhood radius will not be considered
      neighbors. Only compute the neighbors of those that are within the given
      range. All particles are candidates for being a neighbor, regardless of
      whether or not they are in the range.
    */
    template<class PositionSlice>
    VerletList(
        PositionSlice x,
        const std::size_t begin,
        const std::size_t end,
        const typename PositionSlice::value_type neighborhood_radius,
        const typename PositionSlice::value_type cell_size_ratio,
        const typename PositionSlice::value_type grid_min[3],
        const typename PositionSlice::value_type grid_max[3],
        typename std::enable_if<(is_slice<PositionSlice>::value),int>::type * = 0 )
    {
        // Create a builder functor.
        using builder_type =
            Impl::VerletListBuilder<PositionSlice,AlgorithmTag>;
        builder_type builder( x, begin, end,
                              neighborhood_radius, cell_size_ratio,
                              grid_min, grid_max );

        // For each particle in the range check each neighboring bin for
        // neighbor particles. Bins are at least the size of the neighborhood
        // radius so the bin in which the particle resides and any surrounding
        // bins are guaranteed to contain the neighboring particles.
        typename builder_type::CountNeighborsPolicy
            count_policy( builder.bin_data_1d.numBin(), Kokkos::AUTO, 4 );
        Kokkos::parallel_for(
            "Cabana::VerletList::count_neighbors",
            count_policy, builder );
        Kokkos::fence();

        // Process the counts by computing offsets and allocating the neighbor
        // list.
        builder.processCounts();

        // For each particle in the range fill its part of the neighbor list.
        typename builder_type::FillNeighborsPolicy
            fill_policy( builder.bin_data_1d.numBin(), Kokkos::AUTO, 4 );
        Kokkos::parallel_for(
            "Cabana::VerletList::fill_neighbors",
            fill_policy, builder );
        Kokkos::fence();

        // Get the data from the builder.
        _counts = builder.counts;
        _offsets = builder.offsets;
        _neighbors = builder.neighbors;
    }
};

//---------------------------------------------------------------------------//
// Neighbor list interface implementation.
//---------------------------------------------------------------------------//
template<class MemorySpace, class AlgorithmTag>
class NeighborList<VerletList<MemorySpace,AlgorithmTag> >
{
  public:

    using list_type = VerletList<MemorySpace,AlgorithmTag>;

    using TypeTag = AlgorithmTag;

    // Get the number of neighbors for a given particle index.
    KOKKOS_INLINE_FUNCTION
    static int numNeighbor( const list_type& list,
                            const std::size_t particle_index )
    {
        return list._counts( particle_index );
    }

    // Get the id for a neighbor for a given particle index and the index of
    // the neighbor relative to the particle.
    KOKKOS_INLINE_FUNCTION
    static int getNeighbor( const list_type& list,
                            const std::size_t particle_index,
                            const int neighbor_index )
    {
        return list._neighbors(
            list._offsets(particle_index) + neighbor_index );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end  CABANA_VERLETLIST_HPP
