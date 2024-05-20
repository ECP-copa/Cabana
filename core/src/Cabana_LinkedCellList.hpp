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
  \file Cabana_LinkedCellList.hpp
  \brief Linked cell list binning (spatial sorting) and neighbor iteration.
*/
#ifndef CABANA_LINKEDCELLLIST_HPP
#define CABANA_LINKEDCELLLIST_HPP

#include <Cabana_NeighborList.hpp>
#include <Cabana_Slice.hpp>
#include <Cabana_Sort.hpp>
#include <Cabana_Utils.hpp>
#include <impl/Cabana_CartesianGrid.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cassert>

namespace Cabana
{
//---------------------------------------------------------------------------//
//! Stencil of cells surrounding each cell.

template <class Scalar, std::size_t NumSpaceDim = 3>
struct LinkedCellStencil
{
    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    //! Background grid.
    Impl::CartesianGrid<Scalar, num_space_dim> grid;
    //! Maximum cells per dimension.
    int max_cells_dir;
    //! Maximum total cells.
    int max_cells;
    //! Range of cells to search based on cutoff.
    int cell_range;

    //! Default Constructor
    LinkedCellStencil() {}

    //! Constructor
    LinkedCellStencil( const Scalar neighborhood_radius,
                       const Scalar cell_size_ratio,
                       const Scalar grid_min[num_space_dim],
                       const Scalar grid_max[num_space_dim] )
    {
        Scalar dx = neighborhood_radius * cell_size_ratio;
        Scalar grid_dx[num_space_dim];
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            grid_dx[d] = dx;
        grid = Impl::CartesianGrid<Scalar>( grid_min, grid_max, grid_dx );
        cell_range = std::ceil( 1 / cell_size_ratio );
        max_cells_dir = 2 * cell_range + 1;
        max_cells = max_cells_dir * max_cells_dir * max_cells_dir;
    }

    //! Given a cell, get the index bounds of the cell stencil.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    getCells( const int cell, int& imin, int& imax, int& jmin, int& jmax,
              int& kmin, int& kmax ) const
    {
        int i, j, k;
        grid.ijkBinIndex( cell, i, j, k );

        kmin = ( k - cell_range > 0 ) ? k - cell_range : 0;
        kmax = ( k + cell_range + 1 < grid._nx[2] ) ? k + cell_range + 1
                                                    : grid._nx[2];

        jmin = ( j - cell_range > 0 ) ? j - cell_range : 0;
        jmax = ( j + cell_range + 1 < grid._nx[1] ) ? j + cell_range + 1
                                                    : grid._nx[1];

        imin = ( i - cell_range > 0 ) ? i - cell_range : 0;
        imax = ( i + cell_range + 1 < grid._nx[0] ) ? i + cell_range + 1
                                                    : grid._nx[0];
    }

    //! Given a cell, get the index bounds of the cell stencil.
    KOKKOS_INLINE_FUNCTION void
    getCells( const int cell, Kokkos::Array<int, num_space_dim>& min,
              Kokkos::Array<int, num_space_dim>& max ) const
    {
        Kokkos::Array<int, num_space_dim> ijk;
        grid.ijkBinIndex( cell, ijk );

        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            min[d] = ( ijk[d] - cell_range > 0 ) ? ijk[d] - cell_range : 0;
            max[d] = ( ijk[d] + cell_range + 1 < grid._nx[d] )
                         ? ijk[d] + cell_range + 1
                         : grid._nx[d];
        }
    }
};

//---------------------------------------------------------------------------//
/*!
  \brief Data describing the bin sizes and offsets resulting from a binning
  operation on a 3d regular Cartesian grid.
  \note Defaults to double precision for backwards compatibility.
*/
template <class MemorySpace, class Scalar = double, std::size_t NumSpaceDim = 3>
class LinkedCellList
{
  public:
    // FIXME: add static_assert that this is a valid MemorySpace.

    // FIXME: extracting the self type for backwards compatibility with previous
    // template on DeviceType. Should simply be MemorySpace after next release.
    //! Memory space.
    using memory_space = typename MemorySpace::memory_space;
    // FIXME: replace warning with memory space assert after next release.
    static_assert( Impl::deprecated( Kokkos::is_device<MemorySpace>() ) );

    //! Default device type.
    using device_type [[deprecated]] = typename memory_space::device_type;
    //! Default execution space.
    using execution_space = typename memory_space::execution_space;
    //! Memory space size type.
    using size_type = typename memory_space::size_type;

    //! Binning view type.
    using CountView = Kokkos::View<int*, memory_space>;
    //! Offset view type.
    using OffsetView = Kokkos::View<size_type*, memory_space>;
    //! Stencil type.
    using stencil_type = Cabana::LinkedCellStencil<Scalar>;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

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
    LinkedCellList( SliceType positions, const Scalar grid_delta[3],
                    const Scalar grid_min[num_space_dim],
                    const Scalar grid_max[num_space_dim],
                    typename std::enable_if<( is_slice<SliceType>::value ),
                                            int>::type* = 0 )
        : _begin( 0 )
        , _end( positions.size() )
        , _grid( grid_min, grid_max, grid_delta )
        , _cell_stencil( grid_delta[0], 1.0, grid_min, grid_max )
        , _sorted( false )
    {
        std::size_t np = positions.size();
        allocate( totalBins(), np );
        build( positions, 0, np );
    }

    /*!
      \brief Slice constructor

      \tparam SliceType Slice type for positions.

      \param positions Slice of positions.
      \param begin The beginning index of particles to bin or find neighbors
      for. Particles outside this range will NOT be considered as candidate
      neighbors.
      \param end The end index of particles to bin or find neighbors
      for. Particles outside this range will NOT be considered as candidate
      neighbors.
      \param grid_delta Grid sizes in each cardinal direction.
      \param grid_min Grid minimum value in each direction.
      \param grid_max Grid maximum value in each direction.
    */
    template <class SliceType>
    LinkedCellList(
        SliceType positions, const std::size_t begin, const std::size_t end,
        const Scalar grid_delta[num_space_dim],
        const Scalar grid_min[num_space_dim],
        const Scalar grid_max[num_space_dim],
        typename std::enable_if<( is_slice<SliceType>::value ), int>::type* =
            0 )
        : _begin( begin )
        , _end( end )
        , _grid( grid_min, grid_max, grid_delta )
        , _cell_stencil( grid_delta[0], 1.0, grid_min, grid_max )
        , _sorted( false )
    {
        allocate( totalBins(), end - begin );
        build( positions, begin, end );
    }

    /*!
      \brief Slice constructor

      \tparam SliceType Slice type for positions.

      \param positions Slice of positions.
      \param grid_delta Grid sizes in each cardinal direction.
      \param grid_min Grid minimum value in each direction.
      \param grid_max Grid maximum value in each direction.
      \param neighborhood_radius Radius for neighbors.
      \param cell_size_ratio Ratio of the cell size to the neighborhood size.
    */
    template <class SliceType>
    LinkedCellList(
        SliceType positions, const Scalar grid_delta[num_space_dim],
        const Scalar grid_min[num_space_dim],
        const Scalar grid_max[num_space_dim], const Scalar neighborhood_radius,
        const Scalar cell_size_ratio = 1,
        typename std::enable_if<( is_slice<SliceType>::value ), int>::type* =
            0 )
        : _begin( 0 )
        , _end( positions.size() )
        , _grid( grid_min, grid_max, grid_delta )
        , _cell_stencil( neighborhood_radius, cell_size_ratio, grid_min,
                         grid_max )
        , _sorted( false )
    {
        std::size_t np = positions.size();
        allocate( totalBins(), np );
        build( positions, 0, np );
    }

    /*!
      \brief Slice range constructor

      \tparam SliceType Slice type for positions.

      \param positions Slice of positions.
      \param begin The beginning index of particles to bin or find neighbors
      for. Particles outside this range will NOT be considered as candidate
      neighbors.
      \param end The end index of particles to bin or find neighbors
      for. Particles outside this range will NOT be considered as candidate
      neighbors.
      \param grid_delta Grid sizes in each cardinal direction.
      \param grid_min Grid minimum value in each direction.
      \param grid_max Grid maximum value in each direction.
      \param neighborhood_radius Radius for neighbors.
      \param cell_size_ratio Ratio of the cell size to the neighborhood size.
    */
    template <class SliceType>
    LinkedCellList(
        SliceType positions, const std::size_t begin, const std::size_t end,
        const Scalar grid_delta[num_space_dim],
        const Scalar grid_min[num_space_dim],
        const Scalar grid_max[num_space_dim], const Scalar neighborhood_radius,
        const Scalar cell_size_ratio = 1,
        typename std::enable_if<( is_slice<SliceType>::value ), int>::type* =
            0 )
        : _begin( begin )
        , _end( end )
        , _grid( grid_min, grid_max, grid_delta )
        , _cell_stencil( neighborhood_radius, cell_size_ratio, grid_min,
                         grid_max )
        , _sorted( false )
    {
        allocate( totalBins(), end - begin );
        build( positions, begin, end );
    }

    //! Number of binned particles.
    KOKKOS_INLINE_FUNCTION
    int numParticles() const { return _permutes.extent( 0 ); }

    //! Beginning of binned range.
    KOKKOS_INLINE_FUNCTION
    std::size_t getParticleBegin() const { return _begin; }

    //! End of binned range.
    KOKKOS_INLINE_FUNCTION
    std::size_t getParticleEnd() const { return _end; }

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
      \brief Given the ijk index of a bin get its cardinal index.
      \param ijk The bin indices in (x,y,z).
      \return The cardinal bin index.

      Note that the Kokkos sort orders the bins such that the i index moves
      the slowest and the k index mvoes the fastest.
    */
    KOKKOS_INLINE_FUNCTION
    size_type
    cardinalBinIndex( const Kokkos::Array<int, num_space_dim>& ijk ) const
    {
        return _grid.cardinalCellIndex( ijk );
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
      \brief Given the cardinal index of a bin get its ijk indices.
      \param cardinal The cardinal bin index.
      \param ijk The bin indices in (x,y,z).

      Note that the Kokkos sort orders the bins such that the i index moves
      the slowest and the k index mvoes the fastest.
    */
    KOKKOS_INLINE_FUNCTION
    void ijkBinIndex( const int cardinal,
                      Kokkos::Array<int, num_space_dim>& ijk ) const
    {
        _grid.ijkBinIndex( cardinal, ijk );
    }

    /*!
      \brief Given a bin get the number of particles it contains.
      \param ijk The bin indices in (x,y,z).
      \return The number of particles in the bin.
    */
    KOKKOS_INLINE_FUNCTION
    int binSize( const Kokkos::Array<int, num_space_dim> ijk ) const
    {
        return _bin_data.binSize( cardinalBinIndex( ijk ) );
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
      \brief Given a bin get the particle index at which it sorts.
      \param ijk The bin indices in (x,y,z).
      \return The starting particle index of the bin.
    */
    KOKKOS_INLINE_FUNCTION
    size_type binOffset( const Kokkos::Array<int, num_space_dim> ijk ) const
    {
        return _bin_data.binOffset( cardinalBinIndex( ijk ) );
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
      \brief Get the linked cell stencil.
      \return Cell stencil.
    */
    KOKKOS_INLINE_FUNCTION
    stencil_type cellStencil() const { return _cell_stencil; }

    /*!
      \brief Get the 1d bin data.
      \return The 1d bin data.
    */
    BinningData<MemorySpace> binningData() const { return _bin_data; }

    /*!
      \brief Build the linked cell list with a subset of particles.

      \tparam ExecutionSpace Kokkos execution space.
      \tparam SliceType Slice type for positions.

      \param positions Slice of positions.
      \param begin The beginning index of particles to bin or find neighbors
      for. Particles outside this range will NOT be considered as candidate
      neighbors.
      \param end The end index of particles to bin or find neighbors
      for. Particles outside this range will NOT be considered as candidate
      neighbors.
    */
    template <class ExecutionSpace, class SliceType>
    void build( ExecutionSpace, SliceType positions, const std::size_t begin,
                const std::size_t end )
    {
        Kokkos::Profiling::ScopedRegion region(
            "Cabana::LinkedCellList::build" );

        static_assert( is_accessible_from<memory_space, ExecutionSpace>{}, "" );
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
        Kokkos::RangePolicy<ExecutionSpace> particle_range( begin, end );
        Kokkos::deep_copy( _counts, 0 );
        auto counts_sv = Kokkos::Experimental::create_scatter_view( _counts );
        auto cell_count = KOKKOS_LAMBDA( const std::size_t p )
        {
            Kokkos::Array<int, num_space_dim> ijk;
            Kokkos::Array<Scalar, num_space_dim> pos;
            for ( std::size_t d = 0; d < num_space_dim; ++d )
                pos[d] = positions( p, d );
            grid.locatePoint( pos, ijk );
            auto counts_data = counts_sv.access();
            counts_data( grid.cardinalCellIndex( ijk ) ) += 1;
        };
        Kokkos::parallel_for( "Cabana::LinkedCellList::build::cell_count",
                              particle_range, cell_count );
        Kokkos::fence();
        Kokkos::Experimental::contribute( _counts, counts_sv );

        // Compute offsets.
        Kokkos::RangePolicy<ExecutionSpace> cell_range( 0, ncell );
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
            Kokkos::Array<int, num_space_dim> ijk;
            Kokkos::Array<Scalar, num_space_dim> pos;
            for ( std::size_t d = 0; d < num_space_dim; ++d )
                pos[d] = positions( p, d );
            grid.locatePoint( pos, ijk );
            auto cell_id = grid.cardinalCellIndex( ijk );
            int c = Kokkos::atomic_fetch_add( &counts( cell_id ), 1 );
            permutes( offsets( cell_id ) + c ) = p;
        };
        Kokkos::parallel_for( "Cabana::LinkedCellList::build::create_permute",
                              particle_range, create_permute );
        Kokkos::fence();

        // Create the binning data.
        _bin_data = BinningData<MemorySpace>( begin, end, _counts, _offsets,
                                              _permutes );

        // Store the bin per particle (for neighbor iteration).
        storeParticleBins();
    }

    /*!
      \brief Build the linked cell list with a subset of particles.

      \tparam SliceType Slice type for positions.

      \param positions Slice of positions.
      \param begin The beginning index of particles to bin or find neighbors
      for. Particles outside this range will NOT be considered as candidate
      neighbors.
      \param end The end index of particles to bin or find neighbors
      for. Particles outside this range will NOT be considered as candidate
      neighbors.
    */
    template <class SliceType>
    void build( SliceType positions, const std::size_t begin,
                const std::size_t end )
    {
        // Use the default execution space.
        build( execution_space{}, positions, begin, end );
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
      \brief Store the bin cell index for each binned particle.
    */
    void storeParticleBins()
    {
        Kokkos::parallel_for(
            "Cabana::LinkedCellList::storeBinIndices",
            Kokkos::RangePolicy<execution_space>( 0, totalBins() ), *this );
    }

    /*!
      \brief Get the bin cell index for each binned particle.
      \note This View only stores bins for particles which are binned.
    */
    auto getParticleBins() const { return _particle_bins; }

    /*!
      \brief Get the bin cell index of the input particle
    */
    KOKKOS_INLINE_FUNCTION
    auto getParticleBin( const int particle_index ) const
    {
        assert( particle_index >= static_cast<int>( _begin ) );
        assert( particle_index < static_cast<int>( _end ) );
        return _particle_bins( particle_index - _begin );
    }

    /*!
      \brief Determines which particles belong to bin i
      \param i the cardinal bin index of the current bin
    */
    KOKKOS_FUNCTION void operator()( const int i ) const
    {
        Kokkos::Array<int, num_space_dim> bin_ijk;
        ijkBinIndex( i, bin_ijk );
        auto offset = binOffset( bin_ijk );
        auto size = binSize( bin_ijk );
        for ( size_t p = offset; p < offset + size; ++p )
        {
            if ( _sorted )
            {
                _particle_bins( p ) = i;
            }
            else
            {
                _particle_bins( permutation( p ) - _begin ) = i;
            }
        }
    }

    /*!
      \brief Updates the status of the list sorting
      \param sorted Bool that is true if the list has been sorted
    */
    void update( const bool sorted ) { _sorted = sorted; }

    /*!
      \brief Returns whether the list has been sorted or not
    */
    KOKKOS_INLINE_FUNCTION
    auto sorted() const { return _sorted; }

    /*!
      \brief Get the cell indices for the stencil about cell
    */
    KOKKOS_INLINE_FUNCTION
    void getStencilCells( const int cell, int& imin, int& imax, int& jmin,
                          int& jmax, int& kmin, int& kmax ) const
    {
        _cell_stencil.getCells( cell, imin, imax, jmin, jmax, kmin, kmax );
    }

    /*!
      \brief Get the cell indices for the stencil about cell
    */
    KOKKOS_INLINE_FUNCTION
    void getStencilCells( const int cell,
                          Kokkos::Array<int, num_space_dim>& min,
                          Kokkos::Array<int, num_space_dim>& max ) const
    {
        _cell_stencil.getCells( cell, min, max );
    }

    /*!
      \brief Get a candidate neighbor particle at a given binned offset.
      \param offset Particle offset in the binned layout.
    */
    KOKKOS_INLINE_FUNCTION
    auto getParticle( const int offset ) const
    {
        std::size_t j;
        if ( !sorted() )
            j = permutation( offset );
        else
            j = offset + getParticleBegin();
        return j;
    }

  private:
    std::size_t _begin;
    std::size_t _end;

    // Building the linked cell.
    BinningData<MemorySpace> _bin_data;
    Impl::CartesianGrid<Scalar> _grid;

    CountView _counts;
    OffsetView _offsets;
    OffsetView _permutes;

    // Iterating over the linked cell.
    stencil_type _cell_stencil;

    bool _sorted;
    CountView _particle_bins;

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
        // This is only used for iterating over the particles (not building the
        // permutaion vector).
        _particle_bins = CountView(
            Kokkos::view_alloc( Kokkos::WithoutInitializing, "counts" ),
            nparticles );
    }
};

/*!
  \brief Creation function for linked cell list.
  \return LinkedCellList.
*/
template <class SliceType, class Scalar, std::size_t NumSpaceDim = 3>
auto createLinkedCellList( SliceType positions,
                           const Scalar grid_delta[NumSpaceDim],
                           const Scalar grid_min[NumSpaceDim],
                           const Scalar grid_max[NumSpaceDim] )
{
    using memory_space = typename SliceType::memory_space;
    return LinkedCellList<memory_space, Scalar, NumSpaceDim>(
        positions, grid_delta, grid_min, grid_max );
}

/*!
  \brief Creation function for linked cell list with partial range.
  \return LinkedCellList.
*/
template <class SliceType, class Scalar, std::size_t NumSpaceDim = 3>
auto createLinkedCellList( SliceType positions, const std::size_t begin,
                           const std::size_t end,
                           const Scalar grid_delta[NumSpaceDim],
                           const Scalar grid_min[NumSpaceDim],
                           const Scalar grid_max[NumSpaceDim] )
{
    using memory_space = typename SliceType::memory_space;
    return LinkedCellList<memory_space, Scalar, NumSpaceDim>(
        positions, begin, end, grid_delta, grid_min, grid_max );
}

/*!
  \brief Creation function for linked cell list with custom cutoff radius and
  cell ratio.
  \return LinkedCellList.
*/
template <class SliceType, class Scalar, std::size_t NumSpaceDim = 3>
auto createLinkedCellList( SliceType positions,
                           const Scalar grid_delta[NumSpaceDim],
                           const Scalar grid_min[NumSpaceDim],
                           const Scalar grid_max[NumSpaceDim],
                           const Scalar neighborhood_radius,
                           const Scalar cell_size_ratio = 1.0 )
{
    using memory_space = typename SliceType::memory_space;
    return LinkedCellList<memory_space, Scalar, NumSpaceDim>(
        positions, grid_delta, grid_min, grid_max, neighborhood_radius,
        cell_size_ratio );
}

/*!
  \brief Creation function for linked cell list with partial range and custom
  cutoff radius and/or cell ratio.
  \return LinkedCellList.
*/
template <class SliceType, class Scalar, std::size_t NumSpaceDim = 3>
auto createLinkedCellList( SliceType positions, const std::size_t begin,
                           const std::size_t end,
                           const Scalar grid_delta[NumSpaceDim],
                           const Scalar grid_min[NumSpaceDim],
                           const Scalar grid_max[NumSpaceDim],
                           const Scalar neighborhood_radius,
                           const Scalar cell_size_ratio = 1.0 )
{
    using memory_space = typename SliceType::memory_space;
    return LinkedCellList<memory_space, Scalar, NumSpaceDim>(
        positions, begin, end, grid_delta, grid_min, grid_max,
        neighborhood_radius, cell_size_ratio );
}

//---------------------------------------------------------------------------//
//! \cond Impl
template <typename>
struct is_linked_cell_list_impl : public std::false_type
{
};

template <typename MemorySpace, typename Scalar>
struct is_linked_cell_list_impl<LinkedCellList<MemorySpace, Scalar>>
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
    LinkedCellListType& linked_cell_list, AoSoA_t& aosoa,
    typename std::enable_if<( is_linked_cell_list<LinkedCellListType>::value &&
                              is_aosoa<AoSoA_t>::value ),
                            int>::type* = 0 )
{
    permute( linked_cell_list.binningData(), aosoa );

    // Update internal state.
    linked_cell_list.update( true );

    linked_cell_list.storeParticleBins();
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
    LinkedCellListType& linked_cell_list, SliceType& slice,
    typename std::enable_if<( is_linked_cell_list<LinkedCellListType>::value &&
                              is_slice<SliceType>::value ),
                            int>::type* = 0 )
{
    permute( linked_cell_list.binningData(), slice );

    // Update internal state.
    linked_cell_list.update( true );

    linked_cell_list.storeParticleBins();
}

//---------------------------------------------------------------------------//
//! LinkedCellList NeighborList interface.
template <class MemorySpace, typename Scalar, std::size_t NumSpaceDim>
class NeighborList<LinkedCellList<MemorySpace, Scalar, NumSpaceDim>>
{
  public:
    //! Kokkos memory space.
    using memory_space = MemorySpace;
    //! Neighbor list type.
    using list_type = LinkedCellList<MemorySpace, Scalar, NumSpaceDim>;
    //! Spatial dimension
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    //! Get the maximum number of neighbors per particle.
    KOKKOS_INLINE_FUNCTION static std::size_t
    totalNeighbor( const list_type& list )
    {
        std::size_t total_n = 0;
        // Sum neighbors across all particles.
        for ( std::size_t p = list.getParticleBegin();
              p < list.getParticleEnd(); p++ )
            total_n += numNeighbor( list, p );
        return total_n;
    }

    //! Get the maximum number of neighbors across all particles.
    KOKKOS_INLINE_FUNCTION
    static std::size_t maxNeighbor( const list_type& list )
    {
        std::size_t max_n = 0;
        for ( std::size_t p = list.getParticleBegin();
              p < list.getParticleEnd(); p++ )
            if ( numNeighbor( list, p ) > max_n )
                max_n = numNeighbor( list, p );
        return max_n;
    }

    //! Get the number of neighbors for a given particle index.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION static std::enable_if_t<3 == NSD, std::size_t>
    numNeighbor( const list_type& list, const std::size_t particle_index )
    {
        int total_count = 0;
        Kokkos::Array<int, 3> min;
        Kokkos::Array<int, 3> max;
        list.getStencilCells( list.getParticleBin( particle_index ), min, max );

        Kokkos::Array<int, 3> ijk;
        for ( int i = min[0]; i < max[0]; ++i )
            for ( int j = min[1]; j < max[1]; ++j )
                for ( int k = min[2]; k < max[2]; ++k )
                {
                    ijk = { i, j, k };
                    total_count += list.binSize( ijk );
                }

        return total_count;
    }

    //! Get the number of neighbors for a given particle index.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION static std::enable_if_t<2 == NSD, std::size_t>
    numNeighbor( const list_type& list, const std::size_t particle_index )
    {
        int total_count = 0;
        Kokkos::Array<int, 2> min;
        Kokkos::Array<int, 2> max;
        list.getStencilCells( list.getParticleBin( particle_index ), min, max );

        Kokkos::Array<int, 2> ij;
        for ( int i = min[0]; i < max[0]; ++i )
            for ( int j = min[1]; j < max[1]; ++j )
            {
                ij = { i, j };
                total_count += list.binSize( ij );
            }

        return total_count;
    }

    //! Get the id for a neighbor for a given particle index and the index of
    //! the neighbor relative to the particle.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION static std::enable_if_t<3 == NSD, std::size_t>
    getNeighbor( const list_type& list, const std::size_t particle_index,
                 const std::size_t neighbor_index )
    {
        std::size_t total_count = 0;
        std::size_t previous_count = 0;
        Kokkos::Array<int, 3> min;
        Kokkos::Array<int, 3> max;
        list.getStencilCells( list.getParticleBin( particle_index ), min, max );

        Kokkos::Array<int, 3> ijk;
        for ( int i = min[0]; i < max[0]; ++i )
            for ( int j = min[1]; j < max[1]; ++j )
                for ( int k = min[2]; k < max[2]; ++k )
                {
                    ijk = { i, j, k };

                    total_count += list.binSize( ijk );
                    // This neighbor is in this bin.
                    if ( total_count > neighbor_index )
                    {
                        int particle_id = list.binOffset( ijk ) +
                                          ( neighbor_index - previous_count );
                        return list.getParticle( particle_id );
                    }
                    // Update previous to all bins so far.
                    previous_count = total_count;
                }

        assert( total_count <= totalNeighbor( list ) );

        // Should never make it to this point.
        return 0;
    }

    //! Get the id for a neighbor for a given particle index and the index of
    //! the neighbor relative to the particle.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION static std::enable_if_t<2 == NSD, std::size_t>
    getNeighbor( const list_type& list, const std::size_t particle_index,
                 const std::size_t neighbor_index )
    {
        std::size_t total_count = 0;
        std::size_t previous_count = 0;
        Kokkos::Array<int, 2> min;
        Kokkos::Array<int, 2> max;
        list.getStencilCells( list.getParticleBin( particle_index ), min, max );

        // Loop over the cell stencil.
        Kokkos::Array<int, 2> ij;
        for ( int i = min[0]; i < max[0]; ++i )
            for ( int j = min[1]; j < max[1]; ++j )
            {
                ij = { i, j };

                total_count += list.binSize( ij );
                // This neighbor is in this bin.
                if ( total_count > neighbor_index )
                {
                    int particle_id = list.binOffset( ij ) +
                                      ( neighbor_index - previous_count );
                    return list.getParticle( particle_id );
                }
                // Update previous to all bins so far.
                previous_count = total_count;
            }

        assert( total_count <= totalNeighbor( list ) );

        // Should never make it to this point.
        return 0;
    }
};

} // end namespace Cabana

#endif // end CABANA_SORT_HPP
