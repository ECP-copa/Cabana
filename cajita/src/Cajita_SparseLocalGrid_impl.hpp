/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_LOCALGRID_SPARSE_IMPL_HPP
#define CAJITA_LOCALGRID_SPARSE_IMPL_HPP

#include <type_traits>
namespace Cajita
{
namespace Experimental
{
//---------------------------------------------------------------------------//
// Constructor
template <class Scalar, std::size_t NumSpaceDim>
LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::LocalGrid(
    const std::shared_ptr<GlobalGrid<mesh_type>>& global_grid,
    const int halo_cell_width, const long cell_num_per_tile_dim )
    : _global_grid( global_grid )
    , _cell_num_per_tile_dim( cell_num_per_tile_dim )
{
    static_assert( 3 == num_space_dim, "SparseMesh supports only 3D" );
    // ensure integer number of halo tiles
    _halo_cell_width =
        static_cast<int>( ( halo_cell_width + cell_num_per_tile_dim - 1 ) /
                          cell_num_per_tile_dim ) *
        cell_num_per_tile_dim;
}

//---------------------------------------------------------------------------//
// Get the global grid that owns the local grid.
template <class Scalar, std::size_t NumSpaceDim>
const GlobalGrid<SparseMesh<Scalar, NumSpaceDim>>&
LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::globalGrid() const
{
    return *_global_grid;
}

//---------------------------------------------------------------------------//
// Get a mutable version of the global grid that own the local grid.
template <class Scalar, std::size_t NumSpaceDim>
GlobalGrid<SparseMesh<Scalar, NumSpaceDim>>&
LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::globalGrid()
{
    return *_global_grid;
}

//---------------------------------------------------------------------------//
// Get the halo size. (unit in cell)
template <class Scalar, std::size_t NumSpaceDim>
int LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::haloCellWidth() const
{
    return _halo_cell_width;
}

//---------------------------------------------------------------------------//
// Get the halo size. (unit in tile)
template <class Scalar, std::size_t NumSpaceDim>
int LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::haloTileWidth() const
{
    return _halo_cell_width / _cell_num_per_tile_dim;
}

//---------------------------------------------------------------------------//
// Get the total number of local cells in a given dimension (owned + halo).
template <class Scalar, std::size_t NumSpaceDim>
int LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::totalNumCell(
    const int d ) const
{
    return _global_grid->ownedNumCell( d ) + 2 * _halo_cell_width;
}

//---------------------------------------------------------------------------//
// Get the total number of local cells in current rank (owned + halo).
template <class Scalar, std::size_t NumSpaceDim>
int LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::totalNumCell() const
{
    int total_num_cell = 1;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        total_num_cell *= totalNumCell( static_cast<int>( d ) );

    return total_num_cell;
}

//---------------------------------------------------------------------------//
// Given the relative offsets of a neighbor rank relative to this local grid's
// indices get the of the neighbor.
template <class Scalar, std::size_t NumSpaceDim>
int LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::neighborRank(
    const std::array<int, num_space_dim>& off_ijk ) const
{
    std::array<int, num_space_dim> nijk;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        nijk[d] = _global_grid->dimBlockId( d ) + off_ijk[d];
    return _global_grid->blockRank( nijk );
}

template <class Scalar, std::size_t NumSpaceDim>
int LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::neighborRank(
    const int off_i, const int off_j, const int off_k ) const
{
    std::array<int, num_space_dim> off_ijk = { off_i, off_j, off_k };
    return neighborRank( off_ijk );
}

//! \cond Impl
//---------------------------------------------------------------------------//
// Get the index space for a  given combination of decomposition, entity, and
// index types.
template <class Scalar, std::size_t NumSpaceDim>
template <class DecompositionTag, class EntityType, class IndexType>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpace(
    DecompositionTag t1, EntityType t2, IndexType t3 ) const
    -> IndexSpace<num_space_dim>
{
    return indexSpaceImpl( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
// Given the relative offsets of a neighbor rank relative to this local
// grid's indices get the set of local entity indices shared with that
// neighbor in the given decomposition. Optionally provide a halo width
// for the shared space. This halo width must be less than or equal to the
// halo width of the local grid. The default behavior is to use the halo
// width of the local grid.
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim, class DecompositionTag,
          class EntityType>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpace(
    DecompositionTag t1, EntityType t2,
    const std::array<int, num_space_dim>& off_ijk, const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    static constexpr unsigned long long cell_num_per_tile_dim =
        1 << cellBitsPerTileDim;

    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;
    hw = static_cast<int>( ( hw + cell_num_per_tile_dim - 1 ) /
                           cell_num_per_tile_dim ) *
         cell_num_per_tile_dim;

    // Check that the offsets are valid.
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        if ( off_ijk[d] < -1 || 1 < off_ijk[d] )
            throw std::logic_error( "Neighbor indices out of bounds" );

    // Check that the requested halo width is valid.
    if ( hw > _halo_cell_width )
        throw std::logic_error(
            "Requested halo width larger than local grid halo" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_ijk ) < 0 )
    {
        std::array<long, num_space_dim> zero_size;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            zero_size[d] = 0;
        return TileIndexSpace<num_space_dim, cellBitsPerTileDim>( zero_size,
                                                                  zero_size );
    }

    // Call the underlying implementation.
    return sharedTileIndexSpaceImpl<cellBitsPerTileDim>( t1, t2, off_ijk, hw );
}

//---------------------------------------------------------------------------//
// Get the global shared tile index space according to input tags
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim, class DecompositionTag,
          class EntityType, std::size_t NSD>
std::enable_if_t<3 == NSD, TileIndexSpace<3, cellBitsPerTileDim>>
LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpace(
    DecompositionTag t1, EntityType t2, const int off_i, const int off_j,
    const int off_k, const int halo_width ) const
{
    std::array<int, 3> off_ijk = { off_i, off_j, off_k };
    return sharedTileIndexSpace<cellBitsPerTileDim>( t1, t2, off_ijk,
                                                     halo_width );
}

//---------------------------------------------------------------------------//
//----------------------------- Node/Cell -----------------------------------//
//---------------------------------------------------------------------------//
// Get the local index space of the owned nodes or cell centers (unit in cell)
template <class Scalar, std::size_t NumSpaceDim>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl( Own,
                                                                 Local ) const
    -> IndexSpace<num_space_dim>
{
    // Compute the lower bound.
    std::array<long, num_space_dim> min;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        min[d] = _halo_cell_width;

    // Compute the upper bound.
    std::array<long, num_space_dim> max;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        max[d] = min[d] + _global_grid->ownedNumCell( d );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted nodes or cell centers
// (unit in cell)
template <class Scalar, std::size_t NumSpaceDim>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl( Ghost,
                                                                 Local ) const
    -> IndexSpace<num_space_dim>
{
    // Compute the size.
    std::array<long, num_space_dim> size;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        size[d] = totalNumCell( d );
    }

    return IndexSpace<num_space_dim>( size );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned nodes or cell centers (unit in cell)
template <class Scalar, std::size_t NumSpaceDim>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl( Own,
                                                                 Global ) const
    -> IndexSpace<num_space_dim>
{
    // Compute the lower bound.
    std::array<long, num_space_dim> min;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        min[d] = _global_grid->globalOffset( d );

    // Compute the upper bound.
    std::array<long, num_space_dim> max;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        max[d] = min[d] + _global_grid->ownedNumCell( d );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor, get the set of global
// tile indices we own that we share with that neighbor to use as ghosts.
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpaceImpl(
    Own, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    auto owned_cell_space = indexSpaceImpl( Own(), Node(), Global() );
    // Compute the lower bound.
    std::array<long, num_space_dim> min;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        // Lower neighbor.
        if ( -1 == off_ijk[d] )
            min[d] = owned_cell_space.min( d ) >> cellBitsPerTileDim;

        // Middle neighbor.
        else if ( 0 == off_ijk[d] )
            min[d] = owned_cell_space.min( d ) >> cellBitsPerTileDim;

        // Upper neighbor.
        else if ( 1 == off_ijk[d] )
            min[d] = ( owned_cell_space.max( d ) - halo_width ) >>
                     cellBitsPerTileDim;
        else
            throw std::runtime_error( "Neighbor offset must be 1, 0, or -1" );
    }

    // Compute the upper bound.
    std::array<long, num_space_dim> max;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        // Lower neighbor.
        if ( -1 == off_ijk[d] )
            max[d] = ( owned_cell_space.min( d ) + halo_width ) >>
                     cellBitsPerTileDim;

        // Middle neighbor.
        else if ( 0 == off_ijk[d] )
            max[d] = owned_cell_space.max( d ) >> cellBitsPerTileDim;

        // Upper neighbor.
        else if ( 1 == off_ijk[d] )
            max[d] = owned_cell_space.max( d ) >> cellBitsPerTileDim;
        else
            throw std::runtime_error( "Neighbor offset must be 1, 0, or -1" );
    }

    return TileIndexSpace<num_space_dim, cellBitsPerTileDim>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor, get set of global tile
// indices owned by that neighbor that are shared with us to use as ghosts.
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpaceImpl(
    Ghost, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    // Get the owned local index space.
    auto owned_cell_space = indexSpaceImpl( Own(), Node(), Global() );

    // Compute the lower bound.
    std::array<long, num_space_dim> min;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        // Lower neighbor.
        if ( -1 == off_ijk[d] )
            min[d] = ( owned_cell_space.min( d ) - halo_width ) >>
                     cellBitsPerTileDim;

        // Middle neighbor
        else if ( 0 == off_ijk[d] )
            min[d] = owned_cell_space.min( d ) >> cellBitsPerTileDim;

        // Upper neighbor.
        else if ( 1 == off_ijk[d] )
            min[d] = owned_cell_space.max( d ) >> cellBitsPerTileDim;
        else
            throw std::runtime_error( "Neighbor offset must be 1, 0, or -1" );
    }

    // Compute the upper bound.
    std::array<long, num_space_dim> max;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        // Lower neighbor.
        if ( -1 == off_ijk[d] )
            max[d] = owned_cell_space.min( d ) >> cellBitsPerTileDim;

        // Middle neighbor
        else if ( 0 == off_ijk[d] )
            max[d] = owned_cell_space.max( d ) >> cellBitsPerTileDim;

        // Upper neighbor.
        else if ( 1 == off_ijk[d] )
            max[d] = ( owned_cell_space.max( d ) + halo_width ) >>
                     cellBitsPerTileDim;
        else
            throw std::runtime_error( "Neighbor offset must be 1, 0, or -1" );
    }

    return TileIndexSpace<num_space_dim, cellBitsPerTileDim>( min, max );
}

//---------------------------------------------------------------------------//
//-------------------------------- Node -------------------------------------//
//---------------------------------------------------------------------------//
// Implementations for node indices
template <class Scalar, std::size_t NumSpaceDim>
template <class DecompositionTag, class IndexType>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl(
    DecompositionTag t1, Node, IndexType t3 ) const -> IndexSpace<num_space_dim>
{
    return indexSpaceImpl( t1, t3 );
}

// Implementation for node-related shared index space
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpaceImpl(
    DecompositionTag t1, Node, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    return sharedTileIndexSpaceImpl<cellBitsPerTileDim>( t1, off_ijk,
                                                         halo_width );
}

//---------------------------------------------------------------------------//
//-------------------------------- Cell -------------------------------------//
//---------------------------------------------------------------------------//
// Implementations for cell indices
template <class Scalar, std::size_t NumSpaceDim>
template <class DecompositionTag, class IndexType>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl(
    DecompositionTag t1, Cell, IndexType t3 ) const -> IndexSpace<num_space_dim>
{
    return indexSpaceImpl( t1, t3 );
}

// Implementation for cell-related shared index space
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpaceImpl(
    DecompositionTag t1, Cell, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    return sharedTileIndexSpaceImpl<cellBitsPerTileDim>( t1, off_ijk,
                                                         halo_width );
}

//---------------------------------------------------------------------------//
//-------------------------------- Face -------------------------------------//
//---------------------------------------------------------------------------//
// Implementations for Face<Dim::I> indices
template <class Scalar, std::size_t NumSpaceDim>
template <class DecompositionTag, class IndexType>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl(
    DecompositionTag t1, Face<Dim::I>, IndexType t3 ) const
    -> IndexSpace<num_space_dim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Face entities so far" );
    return indexSpaceImpl( t1, t3 );
}

// Implementations for Face<Dim::J> indices
template <class Scalar, std::size_t NumSpaceDim>
template <class DecompositionTag, class IndexType>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl(
    DecompositionTag t1, Face<Dim::J>, IndexType t3 ) const
    -> IndexSpace<num_space_dim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Face entities so far" );
    return indexSpaceImpl( t1, t3 );
}

// Implementations for Face<Dim::K> indices
template <class Scalar, std::size_t NumSpaceDim>
template <class DecompositionTag, class IndexType>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl(
    DecompositionTag t1, Face<Dim::K>, IndexType t3 ) const
    -> IndexSpace<num_space_dim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Face entities so far" );
    return indexSpaceImpl( t1, t3 );
}

// Implementation for face<Dim::I>-related shared index space
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpaceImpl(
    DecompositionTag t1, Face<Dim::I>,
    const std::array<int, num_space_dim>& off_ijk, const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Face entities so far" );
    return sharedTileIndexSpaceImpl<cellBitsPerTileDim>( t1, off_ijk,
                                                         halo_width );
}

// Implementation for face<Dim::J>-related shared index space
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpaceImpl(
    DecompositionTag t1, Face<Dim::J>,
    const std::array<int, num_space_dim>& off_ijk, const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Face entities so far" );
    return sharedTileIndexSpaceImpl<cellBitsPerTileDim>( t1, off_ijk,
                                                         halo_width );
}
// Implementation for face<Dim::K>-related shared index space
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpaceImpl(
    DecompositionTag t1, Face<Dim::K>,
    const std::array<int, num_space_dim>& off_ijk, const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Face entities so far" );
    return sharedTileIndexSpaceImpl<cellBitsPerTileDim>( t1, off_ijk,
                                                         halo_width );
}

//---------------------------------------------------------------------------//
//-------------------------------- Edge -------------------------------------//
//---------------------------------------------------------------------------//
// Implementations for edge<Dim::I> indices
template <class Scalar, std::size_t NumSpaceDim>
template <class DecompositionTag, class IndexType>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl(
    DecompositionTag t1, Edge<Dim::I>, IndexType t3 ) const
    -> IndexSpace<num_space_dim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Edge entities so far" );
    return indexSpaceImpl( t1, t3 );
}

// Implementations for edge<Dim::J> indices
template <class Scalar, std::size_t NumSpaceDim>
template <class DecompositionTag, class IndexType>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl(
    DecompositionTag t1, Edge<Dim::J>, IndexType t3 ) const
    -> IndexSpace<num_space_dim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Edge entities so far" );
    return indexSpaceImpl( t1, t3 );
}

// Implementations for edge<Dim::K> indices
template <class Scalar, std::size_t NumSpaceDim>
template <class DecompositionTag, class IndexType>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::indexSpaceImpl(
    DecompositionTag t1, Edge<Dim::K>, IndexType t3 ) const
    -> IndexSpace<num_space_dim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Edge entities so far" );
    return indexSpaceImpl( t1, t3 );
}

// Implementation for Edge<Dim::I>-related shared index space
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpaceImpl(
    DecompositionTag t1, Edge<Dim::I>,
    const std::array<int, num_space_dim>& off_ijk, const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Edge entities so far" );
    return sharedTileIndexSpaceImpl<cellBitsPerTileDim>( t1, off_ijk,
                                                         halo_width );
}

// Implementation for Edge<Dim::I>-related shared index space
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpaceImpl(
    DecompositionTag t1, Edge<Dim::J>,
    const std::array<int, num_space_dim>& off_ijk, const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Edge entities so far" );
    return sharedTileIndexSpaceImpl<cellBitsPerTileDim>( t1, off_ijk,
                                                         halo_width );
}

// Implementation for Edge<Dim::I>-related shared index space
template <class Scalar, std::size_t NumSpaceDim>
template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
auto LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::sharedTileIndexSpaceImpl(
    DecompositionTag t1, Edge<Dim::K>,
    const std::array<int, num_space_dim>& off_ijk, const int halo_width ) const
    -> TileIndexSpace<num_space_dim, cellBitsPerTileDim>
{
    std::runtime_error(
        "Sparse grid implementation doesn't support Edge entities so far" );
    return sharedTileIndexSpaceImpl<cellBitsPerTileDim>( t1, off_ijk,
                                                         halo_width );
}
//! \endcond

//---------------------------------------------------------------------------//
} // namespace Experimental
} // namespace Cajita

#endif // end CAJITA_LOCALGRID_SPARSE_IMPL_HPP
