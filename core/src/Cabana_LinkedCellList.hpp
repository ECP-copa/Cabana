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
  \file Cabana_LinkedCellList.hpp
  \brief Linked cell list binning and sorting
*/
#ifndef CABANA_LINKEDCELLLIST_HPP
#define CABANA_LINKEDCELLLIST_HPP

#include <Cabana_NeighborList.hpp>
#include <Cabana_Slice.hpp>
#include <Cabana_Sort.hpp>
#include <impl/Cabana_CartesianGrid.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cassert>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \brief Data describing the bin sizes and offsets resulting from a binning
  operation on a 3d regular Cartesian grid.
*/
template <class DeviceType>
class LinkedCellList
{
  public:
    //! Kokkos device_type.
    using device_type = DeviceType;
    //! Kokkos memory space.
    using memory_space = typename device_type::memory_space;
    //! Kokkos execution space.
    using execution_space = typename device_type::execution_space;
    //! Memory space size type.
    using size_type = typename memory_space::size_type;
    //! Binning view type.
    using CountView = Kokkos::View<int*, device_type>;
    //! Offset view type.
    using OffsetView = Kokkos::View<size_type*, device_type>;

    /*!
      \brief Default constructor.
    */
    LinkedCellList() {}

    /*!
      \brief Slice constructor

      \tparam SliceType Slice type for positions.

      \param positions Slice of positions.

      \param grid_delta Grid sizes in each cardinal direction.

      \param grid_min Grid minimum value in each direction.

      \param grid_max Grid maximum value in each direction.
    */
    template <class SliceType>
    LinkedCellList(
        SliceType positions, const typename SliceType::value_type grid_delta[3],
        const typename SliceType::value_type grid_min[3],
        const typename SliceType::value_type grid_max[3],
        typename std::enable_if<( is_slice<SliceType>::value ), int>::type* =
            0 )
        : _grid( grid_min[0], grid_min[1], grid_min[2], grid_max[0],
                 grid_max[1], grid_max[2], grid_delta[0], grid_delta[1],
                 grid_delta[2] )
    {
        std::size_t np = positions.size();
        allocate( totalBins(), np );
        build( positions, 0, np );
    }

    /*!
      \brief Slice range constructor

      \tparam SliceType Slice type for positions.

      \param positions Slice of positions.

      \param begin The beginning index of the AoSoA range to sort.

      \param end The end index of the AoSoA range to sort.

      \param grid_delta Grid sizes in each cardinal direction.

      \param grid_min Grid minimum value in each direction.

      \param grid_max Grid maximum value in each direction.
    */
    template <class SliceType>
    LinkedCellList(
        SliceType positions, const std::size_t begin, const std::size_t end,
        const typename SliceType::value_type grid_delta[3],
        const typename SliceType::value_type grid_min[3],
        const typename SliceType::value_type grid_max[3],
        typename std::enable_if<( is_slice<SliceType>::value ), int>::type* =
            0 )
        : _grid( grid_min[0], grid_min[1], grid_min[2], grid_max[0],
                 grid_max[1], grid_max[2], grid_delta[0], grid_delta[1],
                 grid_delta[2] )
    {
        allocate( totalBins(), end - begin );
        build( positions, begin, end );
    }

    //! Number of binned particles.
    KOKKOS_INLINE_FUNCTION
    int numParticles() const { return _permutes.extent( 0 ); }

    /*!
      \brief Get the total number of bins.
      \return the total number of bins.
    */
    KOKKOS_INLINE_FUNCTION
    int totalBins() const { return _grid.totalNumCells(); }

    /*!
      \brief Get the number of bins in a given dimension.
      \param dim The dimension to get the number of bins for.
      \return The number of bins.
    */
    KOKKOS_INLINE_FUNCTION
    int numBin( const int dim ) const { return _grid.numBin( dim ); }

    /*!
      \brief Given the ijk index of a bin get its cardinal index.
      \param i The i bin index (x).
      \param j The j bin index (y).
      \param k The k bin index (z).
      \return The cardinal bin index.

      Note that the Kokkos sort orders the bins such that the i index moves
      the slowest and the k index mvoes the fastest.
    */
    KOKKOS_INLINE_FUNCTION
    size_type cardinalBinIndex( const int i, const int j, const int k ) const
    {
        return _grid.cardinalCellIndex( i, j, k );
    }

    /*!
      \brief Given the cardinal index of a bin get its ijk indices.
      \param cardinal The cardinal bin index.
      \param i The i bin index (x).
      \param j The j bin index (y).
      \param k The k bin index (z).

      Note that the Kokkos sort orders the bins such that the i index moves
      the slowest and the k index mvoes the fastest.
    */
    KOKKOS_INLINE_FUNCTION
    void ijkBinIndex( const int cardinal, int& i, int& j, int& k ) const
    {
        _grid.ijkBinIndex( cardinal, i, j, k );
    }

    /*!
      \brief Given a bin get the number of particles it contains.
      \param i The i bin index (x).
      \param j The j bin index (y).
      \param k The k bin index (z).
      \return The number of particles in the bin.
    */
    KOKKOS_INLINE_FUNCTION
    int binSize( const int i, const int j, const int k ) const
    {
        return _bin_data.binSize( cardinalBinIndex( i, j, k ) );
    }

    /*!
      \brief Given a bin get the particle index at which it sorts.
      \param i The i bin index (x).
      \param j The j bin index (y).
      \param k The k bin index (z).
      \return The starting particle index of the bin.
    */
    KOKKOS_INLINE_FUNCTION
    size_type binOffset( const int i, const int j, const int k ) const
    {
        return _bin_data.binOffset( cardinalBinIndex( i, j, k ) );
    }

    /*!
      \brief Given a local particle id in the binned layout, get the id of the
      particle in the old (unbinned) layout.
      \param particle_id The id of the particle in the binned layout.
      \return The particle id in the old (unbinned) layout.
    */
    KOKKOS_INLINE_FUNCTION
    size_type permutation( const int particle_id ) const
    {
        return _bin_data.permutation( particle_id );
    }

    /*!
      \brief The beginning particle index binned by the linked cell list.
    */
    KOKKOS_INLINE_FUNCTION
    std::size_t rangeBegin() const { return _bin_data.rangeBegin(); }

    /*!
      \brief The ending particle index binned by the linked cell list.
    */
    KOKKOS_INLINE_FUNCTION
    std::size_t rangeEnd() const { return _bin_data.rangeEnd(); }

    /*!
      \brief Get the 1d bin data.
      \return The 1d bin data.
    */
    BinningData<DeviceType> binningData() const { return _bin_data; }

    /*!
      \brief Build the linked cell list with a subset of particles.

      \tparam SliceType Slice type for positions.

      \param positions Slice of positions.

      \param begin The beginning index of the slice range to sort.

      \param end The end index of the slice range to sort.
    */
    template <class SliceType>
    void build( SliceType positions, const std::size_t begin,
                const std::size_t end )
    {
        assert( end >= begin );
        assert( end <= positions.size() );

        // Resize the binning data. Note that the permutation vector spans
        // only the length of begin-end;
        std::size_t ncell = totalBins();
        if ( _counts.extent( 0 ) != ncell )
        {
            Kokkos::resize( _counts, ncell );
            Kokkos::resize( _offsets, ncell );
        }
        std::size_t nparticles = end - begin;
        if ( _permutes.extent( 0 ) != nparticles )
        {
            Kokkos::resize( _permutes, nparticles );
        }

        // Get local copies of class data for lambda function capture.
        auto grid = _grid;
        auto counts = _counts;
        auto offsets = _offsets;
        auto permutes = _permutes;

        // Count.
        Kokkos::RangePolicy<execution_space> particle_range( begin, end );
        Kokkos::deep_copy( _counts, 0 );
        auto counts_sv = Kokkos::Experimental::create_scatter_view( _counts );
        auto cell_count = KOKKOS_LAMBDA( const std::size_t p )
        {
            int i, j, k;
            grid.locatePoint( positions( p, 0 ), positions( p, 1 ),
                              positions( p, 2 ), i, j, k );
            auto counts_data = counts_sv.access();
            counts_data( grid.cardinalCellIndex( i, j, k ) ) += 1;
        };
        Kokkos::parallel_for( "Cabana::LinkedCellList::build::cell_count",
                              particle_range, cell_count );
        Kokkos::fence();
        Kokkos::Experimental::contribute( _counts, counts_sv );

        // Compute offsets.
        Kokkos::RangePolicy<execution_space> cell_range( 0, ncell );
        auto offset_scan = KOKKOS_LAMBDA( const std::size_t c, int& update,
                                          const bool final_pass )
        {
            if ( final_pass )
                offsets( c ) = update;
            update += counts( c );
        };
        Kokkos::parallel_scan( "Cabana::LinkedCellList::build::offset_scan",
                               cell_range, offset_scan );
        Kokkos::fence();

        // Reset counts.
        Kokkos::deep_copy( _counts, 0 );

        // Compute the permutation vector.
        auto create_permute = KOKKOS_LAMBDA( const std::size_t p )
        {
            int i, j, k;
            grid.locatePoint( positions( p, 0 ), positions( p, 1 ),
                              positions( p, 2 ), i, j, k );
            auto cell_id = grid.cardinalCellIndex( i, j, k );
            int c = Kokkos::atomic_fetch_add( &counts( cell_id ), 1 );
            permutes( offsets( cell_id ) + c ) = p;
        };
        Kokkos::parallel_for( "Cabana::LinkedCellList::build::create_permute",
                              particle_range, create_permute );
        Kokkos::fence();

        // Create the binning data.
        _bin_data =
            BinningData<DeviceType>( begin, end, _counts, _offsets, _permutes );
    }

    /*!
      \brief Build the linked cell list with all particles.

      \tparam SliceType Slice type for positions.

      \param positions Slice of positions.
    */
    template <class SliceType>
    void build( SliceType positions )
    {
        build( positions, 0, positions.size() );
    }

    /*!
      \brief Get the bin cell index for each binned particle.

      \param begin The beginning index of the sorted range.

      \param end The end index of the sorted range.
    */
    auto cells( const std::size_t begin, const std::size_t end ) const
    {
        CountView particle_cells( "particle_cells", end - begin );
        Kokkos::parallel_for(
            "Cabana::LinkedCellList::find_bin_index",
            Kokkos::RangePolicy<execution_space>( 0, totalBins() ),
            KOKKOS_LAMBDA( const int i ) {
                int bin_ijk[3];
                ijkBinIndex( i, bin_ijk[0], bin_ijk[1], bin_ijk[2] );
                auto offset = binOffset( bin_ijk[0], bin_ijk[1], bin_ijk[2] );
                auto size = binSize( bin_ijk[0], bin_ijk[1], bin_ijk[2] );
                for ( size_t p = offset; p < offset + size; ++p )
                    particle_cells( p ) = i;
            } );
        return particle_cells;
    }

  private:
    BinningData<DeviceType> _bin_data;
    Impl::CartesianGrid<double> _grid;

    CountView _counts;
    OffsetView _offsets;
    OffsetView _permutes;

    void allocate( const int ncell, const int nparticles )
    {
        _counts = CountView(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "counts" ),
            ncell );
        _offsets = OffsetView(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "offsets" ),
            ncell );
        _permutes = OffsetView(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "permutes" ),
            nparticles );
    }
};

//---------------------------------------------------------------------------//
//! \cond Impl
template <typename>
struct is_linked_cell_list_impl : public std::false_type
{
};

template <typename DeviceType>
struct is_linked_cell_list_impl<LinkedCellList<DeviceType>>
    : public std::true_type
{
};
//! \endcond

//! LinkedCellList static type checker.
template <class T>
struct is_linked_cell_list
    : public is_linked_cell_list_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
/*!
  \brief Given a linked cell list permute an AoSoA.

  \tparam LinkedCellListType The linked cell list type.

  \tparam AoSoA_t The AoSoA type.

  \param linked_cell_list The linked cell list to permute the AoSoA with.

  \param aosoa The AoSoA to permute.
 */
template <class LinkedCellListType, class AoSoA_t>
void permute(
    const LinkedCellListType& linked_cell_list, AoSoA_t& aosoa,
    typename std::enable_if<( is_linked_cell_list<LinkedCellListType>::value &&
                              is_aosoa<AoSoA_t>::value ),
                            int>::type* = 0 )
{
    permute( linked_cell_list.binningData(), aosoa );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a linked cell list permute a slice.

  \tparam LinkedCellListType The linked cell list type.

  \tparam SliceType The slice type.

  \param linked_cell_list The linked cell list to permute the slice with.

  \param slice The slice to permute.
 */
template <class LinkedCellListType, class SliceType>
void permute(
    const LinkedCellListType& linked_cell_list, SliceType& slice,
    typename std::enable_if<( is_linked_cell_list<LinkedCellListType>::value &&
                              is_slice<SliceType>::value ),
                            int>::type* = 0 )
{
    permute( linked_cell_list.binningData(), slice );
}

//---------------------------------------------------------------------------//
//! Stencil of cells surrounding each cell.
template <class Scalar>
struct LinkedCellStencil
{
    //! Cutoff squared.
    Scalar rsqr;
    //! Background grid.
    Impl::CartesianGrid<double> grid;
    //! Maximum cells per dimension.
    int max_cells_dir;
    //! Maximum total cells.
    int max_cells;
    //! Range of cells to search based on cutoff.
    int cell_range;

    //! Constructor
    LinkedCellStencil( const Scalar neighborhood_radius,
                       const Scalar cell_size_ratio, const Scalar grid_min[3],
                       const Scalar grid_max[3] )
        : rsqr( neighborhood_radius * neighborhood_radius )
    {
        Scalar dx = neighborhood_radius * cell_size_ratio;
        grid = Impl::CartesianGrid<double>(
            grid_min[0], grid_min[1], grid_min[2], grid_max[0], grid_max[1],
            grid_max[2], dx, dx, dx );
        cell_range = std::ceil( 1 / cell_size_ratio );
        max_cells_dir = 2 * cell_range + 1;
        max_cells = max_cells_dir * max_cells_dir * max_cells_dir;
    }

    //! Given a cell, get the index bounds of the cell stencil.
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
//! LinkedCellList NeighborList interface.
template <class DeviceType>
class NeighborList<LinkedCellList<DeviceType>>
{
  public:
    //! Kokkos memory space.
    using device_type = DeviceType;
    //! Neighbor list type.
    using list_type = LinkedCellList<DeviceType>;

    //! Get the maximum number of neighbors per particle.
    template <class CellIndexType, class StencilType>
    KOKKOS_INLINE_FUNCTION static std::size_t
    maxNeighbor( const list_type& list, const CellIndexType cell,
                 const StencilType stencil )
    {
        int total_count = 0;
        for ( int p = 0; p < list.numParticles(); ++p )
            total_count += numNeighbor( list, cell, stencil, p );
        return total_count;
    }

    //! Get the number of neighbors for a given particle index.
    template <class CellIndexType, class StencilType>
    KOKKOS_INLINE_FUNCTION static std::size_t
    numNeighbor( const list_type& list, const CellIndexType cell,
                 const StencilType stencil, const std::size_t particle_index )
    {
        int total_count = 0;
        int imin, imax, jmin, jmax, kmin, kmax;
        stencil.getCells( cell( particle_index ), imin, imax, jmin, jmax, kmin,
                          kmax );

        // Loop over the cell stencil.
        for ( int i = imin; i < imax; ++i )
            for ( int j = jmin; j < jmax; ++j )
                for ( int k = kmin; k < kmax; ++k )
                {
                    total_count += list.binSize( i, j, k );
                }
        return total_count;
    }

    //! Get the id for a neighbor for a given particle index and the index of
    //! the neighbor relative to the particle.
    KOKKOS_INLINE_FUNCTION
    static std::size_t getNeighbor( const list_type& list, const std::size_t,
                                    const std::size_t neighbor_index,
                                    const bool sorted = true )
    {
        if ( sorted )
            return neighbor_index;
        else
            return list.permutation( neighbor_index );
    }
};

} // end namespace Cabana

#endif // end CABANA_SORT_HPP
