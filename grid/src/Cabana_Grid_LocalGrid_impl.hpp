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

#ifndef CABANA_GRID_LOCALGRID_IMPL_HPP
#define CABANA_GRID_LOCALGRID_IMPL_HPP

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
// Constructor.
template <class MeshType>
LocalGrid<MeshType>::LocalGrid(
    const std::shared_ptr<GlobalGrid<MeshType>>& global_grid,
    const int halo_cell_width )
    : _global_grid( global_grid )
    , _halo_cell_width( halo_cell_width )
{
}

//---------------------------------------------------------------------------//
// Get the global grid that owns the local grid.
template <class MeshType>
const GlobalGrid<MeshType>& LocalGrid<MeshType>::globalGrid() const
{
    return *_global_grid;
}

//---------------------------------------------------------------------------//
// Get a mutable version of the global grid that own the local grid.
template <class MeshType>
GlobalGrid<MeshType>& LocalGrid<MeshType>::globalGrid()
{
    return *_global_grid;
}

//---------------------------------------------------------------------------//
// Get the halo size.
template <class MeshType>
int LocalGrid<MeshType>::haloCellWidth() const
{
    return _halo_cell_width;
}

//---------------------------------------------------------------------------//
// Get the total number of local cells in a given dimension (owned + halo).
template <class MeshType>
int LocalGrid<MeshType>::totalNumCell( const int d ) const
{
    return _global_grid->ownedNumCell( d ) + 2 * _halo_cell_width;
}

//---------------------------------------------------------------------------//
// Given the relative offsets of a neighbor rank relative to this local grid's
// indices get the of the neighbor.
template <class MeshType>
int LocalGrid<MeshType>::neighborRank(
    const std::array<int, num_space_dim>& off_ijk ) const
{
    std::array<int, num_space_dim> nijk;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        nijk[d] = _global_grid->dimBlockId( d ) + off_ijk[d];
    return _global_grid->blockRank( nijk );
}

template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
LocalGrid<MeshType>::neighborRank( const int off_i, const int off_j,
                                   const int off_k ) const
{
    std::array<int, num_space_dim> off_ijk = { off_i, off_j, off_k };
    return neighborRank( off_ijk );
}

template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<2 == NSD, int>
LocalGrid<MeshType>::neighborRank( const int off_i, const int off_j ) const
{
    std::array<int, num_space_dim> off_ijk = { off_i, off_j };
    return neighborRank( off_ijk );
}

//! \cond Impl
//---------------------------------------------------------------------------//
// Get the index space for a given combination of decomposition, entity, and
// index types.
template <class MeshType>
template <class DecompositionTag, class EntityType, class IndexType>
auto LocalGrid<MeshType>::indexSpace( DecompositionTag t1, EntityType t2,
                                      IndexType t3 ) const
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
template <class MeshType>
template <class DecompositionTag, class EntityType>
auto LocalGrid<MeshType>::sharedIndexSpace(
    DecompositionTag t1, EntityType t2,
    const std::array<int, num_space_dim>& off_ijk, const int halo_width ) const
    -> IndexSpace<num_space_dim>
{
    // If we got the default halo width of -1 this means we want to use the
    // default of the entire halo.
    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;

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
        return IndexSpace<num_space_dim>( zero_size, zero_size );
    }

    // Call the underlying implementation.
    return sharedIndexSpaceImpl( t1, t2, off_ijk, hw );
}

template <class MeshType>
template <class DecompositionTag, class EntityType, std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>> LocalGrid<MeshType>::sharedIndexSpace(
    DecompositionTag t1, EntityType t2, const int off_i, const int off_j,
    const int off_k, const int halo_width ) const
{
    std::array<int, 3> off_ijk = { off_i, off_j, off_k };
    return sharedIndexSpace( t1, t2, off_ijk, halo_width );
}

template <class MeshType>
template <class DecompositionTag, class EntityType, std::size_t NSD>
std::enable_if_t<2 == NSD, IndexSpace<2>>
LocalGrid<MeshType>::sharedIndexSpace( DecompositionTag t1, EntityType t2,
                                       const int off_i, const int off_j,
                                       const int halo_width ) const
{
    std::array<int, 2> off_ijk = { off_i, off_j };
    return sharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
// Given the relative offsets of a boundary relative to this local grid's
// indices get the set of local entity indices associated with that boundary
// in the given decomposition. Optionally provide a halo width for the shared
// space. This halo width must be less than or equal to the halo width of the
// local grid. The default behavior is to use the halo width of the local
// grid. For example, if the Own decomposition is used, the interior entities
// that would be affected by a boundary operation are provided whereas if the
// Ghost decomposition is used the halo entities on the boundary are provided.
template <class MeshType>
template <class DecompositionTag, class EntityType>
auto LocalGrid<MeshType>::boundaryIndexSpace(
    DecompositionTag t1, EntityType t2,
    const std::array<int, num_space_dim>& off_ijk, const int halo_width ) const
    -> IndexSpace<num_space_dim>
{
    // If we got the default halo width of -1 this means we want to use the
    // default of the entire halo.
    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;

    // Check that the offsets are valid.
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        if ( off_ijk[d] < -1 || 1 < off_ijk[d] )
            throw std::logic_error( "Boundary indices out of bounds" );

    // Check that the requested halo width is valid.
    if ( hw > _halo_cell_width )
        throw std::logic_error(
            "Requested halo width larger than local grid halo" );

    // Check to see if this is not a communication neighbor. If it is, return
    // a boundary space of size 0 because there is no boundary.
    if ( neighborRank( off_ijk ) >= 0 )
    {
        std::array<long, num_space_dim> zero_size;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            zero_size[d] = 0;
        return IndexSpace<num_space_dim>( zero_size, zero_size );
    }

    return boundaryIndexSpaceImpl( t1, t2, off_ijk, hw );
}

template <class MeshType>
template <class DecompositionTag, class EntityType, std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::boundaryIndexSpace( DecompositionTag t1, EntityType t2,
                                         const int off_i, const int off_j,
                                         const int off_k,
                                         const int halo_width ) const
{
    std::array<int, 3> off_ijk = { off_i, off_j, off_k };
    return boundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

template <class MeshType>
template <class DecompositionTag, class EntityType, std::size_t NSD>
std::enable_if_t<2 == NSD, IndexSpace<2>>
LocalGrid<MeshType>::boundaryIndexSpace( DecompositionTag t1, EntityType t2,
                                         const int off_i, const int off_j,
                                         const int halo_width ) const
{
    std::array<int, 2> off_ijk = { off_i, off_j };
    return boundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned cells.
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Own, Cell, Local ) const
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
// Get the local index space of the owned and ghosted cells.
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Ghost, Cell, Local ) const
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
// Get the global index space of the owned cells.
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Own t1, Cell t2, Global ) const
    -> IndexSpace<num_space_dim>
{
    return globalIndexSpace( t1, t2 );
}

// Return the bound for a shared/boundary index space given an owned space.
template <class MeshType>
template <class OwnedIndexSpace>
auto LocalGrid<MeshType>::getBound(
    OwnedIndexSpace owned_space, const int upper_lower,
    const std::array<int, num_space_dim>& off_ijk, const int lower_shift,
    const int upper_shift ) const
{
    // Compute the lower bound.
    std::array<long, num_space_dim> minmax;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        // Lower neighbor.
        if ( -1 == off_ijk[d] )
        {
            minmax[d] = owned_space.min( d ) + lower_shift;
        }
        // Middle neighbor.
        else if ( 0 == off_ijk[d] )
        {
            if ( upper_lower < 0 )
                minmax[d] = owned_space.min( d );
            else if ( upper_lower > 0 )
                minmax[d] = owned_space.max( d );
            else
                throw std::runtime_error( "Cabana::Grid::LocalGrid::getBound::"
                                          "Impl: Only 1 or -1 allowed" );
        }
        // Upper neighbor.
        else if ( 1 == off_ijk[d] )
        {
            minmax[d] = owned_space.max( d ) + upper_shift;
        }
        else
        {
            throw std::runtime_error( "Cabana::Grid::LocalGrid::getBound: "
                                      "Neighbor offset must be 1, 0, or -1" );
        }
    }
    return minmax;
}

// Return the bound for a shared/boundary index space given an owned space.
// Overload for face/edge with different range per dimension.
template <class MeshType>
template <int Dir, class OwnedIndexSpace>
auto LocalGrid<MeshType>::getBound(
    OwnedIndexSpace owned_space, const int upper_lower,
    const std::array<int, num_space_dim>& off_ijk, const int lower_shift_dir,
    const int lower_shift, const int upper_shift_dir,
    const int upper_shift ) const
{
    // Compute the lower bound.
    std::array<long, num_space_dim> minmax;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        // Lower neighbor.
        if ( -1 == off_ijk[d] )
        {
            minmax[d] = ( Dir == d ) ? owned_space.min( d ) + lower_shift_dir
                                     : owned_space.min( d ) + lower_shift;
        }
        // Middle neighbor.
        else if ( 0 == off_ijk[d] )
        {
            if ( upper_lower < 0 )
                minmax[d] = owned_space.min( d );
            else if ( upper_lower > 0 )
                minmax[d] = owned_space.max( d );
            else
                throw std::runtime_error( "Cabana::Grid::LocalGrid::getBound::"
                                          "Impl: Only 1 or -1 allowed" );
        }
        // Upper neighbor.
        else if ( 1 == off_ijk[d] )
        {
            minmax[d] = ( Dir == d ) ? owned_space.max( d ) + upper_shift_dir
                                     : owned_space.max( d ) + upper_shift;
        }
        else
        {
            throw std::runtime_error( "Cabana::Grid::LocalGrid::getBound: "
                                      "Neighbor offset must be 1, 0, or -1" );
        }
    }
    return minmax;
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local cell
// indices we own that we share with that neighbor to use as ghosts.
template <class MeshType>
auto LocalGrid<MeshType>::sharedIndexSpaceImpl(
    Own, Cell, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Cell(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, 0, -halo_width );
    auto max = getBound( owned_space, 1, off_ijk, halo_width, 0 );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get set of local cell
// indices owned by that neighbor that are shared with us to use as
// ghosts.
template <class MeshType>
auto LocalGrid<MeshType>::sharedIndexSpaceImpl(
    Ghost, Cell, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Cell(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, -halo_width, 0 );
    auto max = getBound( owned_space, 1, off_ijk, 0, halo_width );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Given the relative offset of a boundary relative to this local grid's
// get the set of local cell indices we own that we share with that boundary
// to use as ghosts. This case is equivalent to sharedIndexSpaceImpl.
template <class MeshType>
auto LocalGrid<MeshType>::boundaryIndexSpaceImpl(
    Own, Cell, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    return sharedIndexSpaceImpl( Own(), Cell(), off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
// Given the relative offset of a boundary relative to this local grid's
// get the set of local cell indices owned by that boundary that are shared
// with us to use as ghosts. This case is equivalent to sharedIndexSpaceImpl.
template <class MeshType>
auto LocalGrid<MeshType>::boundaryIndexSpaceImpl(
    Ghost, Cell, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    return sharedIndexSpaceImpl( Ghost(), Cell(), off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned nodes.
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Own, Node, Local ) const
    -> IndexSpace<num_space_dim>
{
    // Compute the lower bound.
    std::array<long, num_space_dim> min;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        min[d] = _halo_cell_width;

    // Compute the upper bound. Resolve the shared node if the dimension is
    // periodic.
    std::array<long, num_space_dim> max;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        max[d] = ( _global_grid->isPeriodic( d ) ||
                   _global_grid->dimBlockId( d ) <
                       _global_grid->dimNumBlock( d ) - 1 )
                     ? min[d] + _global_grid->ownedNumCell( d )
                     : min[d] + _global_grid->ownedNumCell( d ) + 1;

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted nodes.
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Ghost, Node, Local ) const
    -> IndexSpace<num_space_dim>
{
    // Compute the size.
    std::array<long, num_space_dim> size;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        size[d] = totalNumCell( d ) + 1;
    }

    return IndexSpace<num_space_dim>( size );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned nodes.
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Own t1, Node t2, Global ) const
    -> IndexSpace<num_space_dim>
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local node
// indices we own that we share with that neighbor to use as ghosts.
template <class MeshType>
auto LocalGrid<MeshType>::sharedIndexSpaceImpl(
    Own, Node, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Node(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, 0, -halo_width );
    auto max = getBound( owned_space, 1, off_ijk, halo_width + 1, 0 );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get set of local node
// indices owned by that neighbor that are shared with us to use as
// ghosts.
template <class MeshType>
auto LocalGrid<MeshType>::sharedIndexSpaceImpl(
    Ghost, Node, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Node(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, -halo_width, 0 );
    auto max = getBound( owned_space, 1, off_ijk, 0, halo_width + 1 );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Given the relative offset of a boundary relative to this local grid's
// get the set of local node indices we own that we share with that boundary
// to use as ghosts.
template <class MeshType>
auto LocalGrid<MeshType>::boundaryIndexSpaceImpl(
    Own, Node, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Node(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, 0, -halo_width - 1 );
    auto max = getBound( owned_space, 1, off_ijk, halo_width + 1, 0 );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Given the relative offset of a boundary relative to this local grid's
// get the set of local node indices owned by that boundary that are shared
// with us to use as ghosts.
template <class MeshType>
auto LocalGrid<MeshType>::boundaryIndexSpaceImpl(
    Ghost, Node, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Node(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, -halo_width, 0 );
    auto max = getBound( owned_space, 1, off_ijk, 0, halo_width );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Own t1, Face<Dim::I> t2,
                                          Local t3 ) const
    -> IndexSpace<num_space_dim>
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Ghost t1, Face<Dim::I> t2,
                                          Local t3 ) const
    -> IndexSpace<num_space_dim>
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Own t1, Face<Dim::I> t2,
                                          Global t3 ) const
    -> IndexSpace<num_space_dim>
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::sharedIndexSpaceImpl(
    Own t1, Face<Dim::I> t2, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    return faceSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::sharedIndexSpaceImpl(
    Ghost t1, Face<Dim::I> t2, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    return faceSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::boundaryIndexSpaceImpl(
    Own t1, Face<Dim::I> t2, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    return faceBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::boundaryIndexSpaceImpl(
    Ghost t1, Face<Dim::I> t2, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    return faceBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Own t1, Face<Dim::J> t2,
                                          Local t3 ) const
    -> IndexSpace<num_space_dim>
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Ghost t1, Face<Dim::J> t2,
                                          Local t3 ) const
    -> IndexSpace<num_space_dim>
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::indexSpaceImpl( Own t1, Face<Dim::J> t2,
                                          Global t3 ) const
    -> IndexSpace<num_space_dim>
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::sharedIndexSpaceImpl(
    Own t1, Face<Dim::J> t2, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    return faceSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::sharedIndexSpaceImpl(
    Ghost t1, Face<Dim::J> t2, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    return faceSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::boundaryIndexSpaceImpl(
    Own t1, Face<Dim::J> t2, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    return faceBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
auto LocalGrid<MeshType>::boundaryIndexSpaceImpl(
    Ghost t1, Face<Dim::J> t2, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    return faceBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Own t1, Face<Dim::K> t2, Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Ghost t1, Face<Dim::K> t2, Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Own t1, Face<Dim::K> t2, Global t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::sharedIndexSpaceImpl( Own t1, Face<Dim::K> t2,
                                           const std::array<int, 3>& off_ijk,
                                           const int halo_width ) const
{
    return faceSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::sharedIndexSpaceImpl( Ghost t1, Face<Dim::K> t2,
                                           const std::array<int, 3>& off_ijk,
                                           const int halo_width ) const
{
    return faceSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::boundaryIndexSpaceImpl( Own t1, Face<Dim::K> t2,
                                             const std::array<int, 3>& off_ijk,
                                             const int halo_width ) const
{
    return faceBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::boundaryIndexSpaceImpl( Ghost t1, Face<Dim::K> t2,
                                             const std::array<int, 3>& off_ijk,
                                             const int halo_width ) const
{
    return faceBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Own t1, Edge<Dim::I> t2, Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Ghost t1, Edge<Dim::I> t2, Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Own t1, Edge<Dim::I> t2, Global t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::sharedIndexSpaceImpl( Own t1, Edge<Dim::I> t2,
                                           const std::array<int, 3>& off_ijk,
                                           const int halo_width ) const
{
    return edgeSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::sharedIndexSpaceImpl( Ghost t1, Edge<Dim::I> t2,
                                           const std::array<int, 3>& off_ijk,
                                           const int halo_width ) const
{
    return edgeSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::boundaryIndexSpaceImpl( Own t1, Edge<Dim::I> t2,
                                             const std::array<int, 3>& off_ijk,
                                             const int halo_width ) const
{
    return edgeBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::boundaryIndexSpaceImpl( Ghost t1, Edge<Dim::I> t2,
                                             const std::array<int, 3>& off_ijk,
                                             const int halo_width ) const
{
    return edgeBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Own t1, Edge<Dim::J> t2, Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Ghost t1, Edge<Dim::J> t2, Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Own t1, Edge<Dim::J> t2, Global t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::sharedIndexSpaceImpl( Own t1, Edge<Dim::J> t2,
                                           const std::array<int, 3>& off_ijk,
                                           const int halo_width ) const
{
    return edgeSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::sharedIndexSpaceImpl( Ghost t1, Edge<Dim::J> t2,
                                           const std::array<int, 3>& off_ijk,
                                           const int halo_width ) const
{
    return edgeSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::boundaryIndexSpaceImpl( Own t1, Edge<Dim::J> t2,
                                             const std::array<int, 3>& off_ijk,
                                             const int halo_width ) const
{
    return edgeBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::boundaryIndexSpaceImpl( Ghost t1, Edge<Dim::J> t2,
                                             const std::array<int, 3>& off_ijk,
                                             const int halo_width ) const
{
    return edgeBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}
//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Own t1, Edge<Dim::K> t2, Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Ghost t1, Edge<Dim::K> t2, Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::indexSpaceImpl( Own t1, Edge<Dim::K> t2, Global t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::sharedIndexSpaceImpl( Own t1, Edge<Dim::K> t2,
                                           const std::array<int, 3>& off_ijk,
                                           const int halo_width ) const
{
    return edgeSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::sharedIndexSpaceImpl( Ghost t1, Edge<Dim::K> t2,
                                           const std::array<int, 3>& off_ijk,
                                           const int halo_width ) const
{
    return edgeSharedIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::boundaryIndexSpaceImpl( Own t1, Edge<Dim::K> t2,
                                             const std::array<int, 3>& off_ijk,
                                             const int halo_width ) const
{
    return edgeBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::boundaryIndexSpaceImpl( Ghost t1, Edge<Dim::K> t2,
                                             const std::array<int, 3>& off_ijk,
                                             const int halo_width ) const
{
    return edgeBoundaryIndexSpace( t1, t2, off_ijk, halo_width );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned entities.
template <class MeshType>
template <class EntityType>
auto LocalGrid<MeshType>::globalIndexSpace( Own, EntityType ) const
    -> IndexSpace<num_space_dim>
{
    auto local_space = indexSpaceImpl( Own(), EntityType(), Local() );
    std::array<long, num_space_dim> min;
    std::array<long, num_space_dim> max;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        min[d] = _global_grid->globalOffset( d );
        max[d] = min[d] + local_space.extent( d );
    }

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned Dir-direction faces.
template <class MeshType>
template <int Dir>
auto LocalGrid<MeshType>::faceIndexSpace( Own, Face<Dir>, Local ) const
    -> IndexSpace<num_space_dim>
{
    static_assert( Dir < num_space_dim, "Spatial dimension out of bounds" );

    // Compute the lower bound.
    std::array<long, num_space_dim> min;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        min[d] = _halo_cell_width;

    // Compute the upper bound.
    std::array<long, num_space_dim> max;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        if ( Dir == d )
        {
            max[d] = ( _global_grid->isPeriodic( d ) ||
                       _global_grid->dimBlockId( d ) <
                           _global_grid->dimNumBlock( d ) - 1 )
                         ? min[d] + _global_grid->ownedNumCell( d )
                         : min[d] + _global_grid->ownedNumCell( d ) + 1;
        }
        else
        {
            max[d] = min[d] + _global_grid->ownedNumCell( d );
        }
    }

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted Dir-direction faces.
template <class MeshType>
template <int Dir>
auto LocalGrid<MeshType>::faceIndexSpace( Ghost, Face<Dir>, Local ) const
    -> IndexSpace<num_space_dim>
{
    static_assert( Dir < num_space_dim, "Spatial dimension out of bounds" );

    // Compute the size.
    std::array<long, num_space_dim> size;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        if ( Dir == d )
        {
            size[d] = totalNumCell( d ) + 1;
        }
        else
        {
            size[d] = totalNumCell( d );
        }
    }

    return IndexSpace<num_space_dim>( size );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned Dir-direction faces.
template <class MeshType>
template <int Dir>
auto LocalGrid<MeshType>::faceIndexSpace( Own t1, Face<Dir> t2, Global ) const
    -> IndexSpace<num_space_dim>
{
    static_assert( Dir < num_space_dim, "Spatial dimension out of bounds" );
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local
// Dir-direction face indices we own that we share with that neighbor to use
// as ghosts.
template <class MeshType>
template <int Dir>
auto LocalGrid<MeshType>::faceSharedIndexSpace(
    Own, Face<Dir>, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    static_assert( Dir < num_space_dim, "Spatial dimension out of bounds" );

    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Face<Dir>(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, 0, -halo_width );
    auto max = getBound<Dir>( owned_space, 1, off_ijk, halo_width + 1,
                              halo_width, 0, 0 );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get set of local
// Dir-direction face indices owned by that neighbor that are shared with us
// to use as ghosts.
template <class MeshType>
template <int Dir>
auto LocalGrid<MeshType>::faceSharedIndexSpace(
    Ghost, Face<Dir>, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    static_assert( Dir < num_space_dim, "Spatial dimension out of bounds" );

    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Face<Dir>(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, -halo_width, 0 );
    auto max = getBound<Dir>( owned_space, 1, off_ijk, 0, 0, halo_width + 1,
                              halo_width );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Given the relative offset of a boundary relative to this local grid's
// get the set of local Dir-direction face indices we own that we share
// with that boundary to use as ghosts.
template <class MeshType>
template <int Dir>
auto LocalGrid<MeshType>::faceBoundaryIndexSpace(
    Own, Face<Dir>, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    static_assert( Dir < num_space_dim, "Spatial dimension out of bounds" );

    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Face<Dir>(), Local() );

    auto min = getBound<Dir>( owned_space, -1, off_ijk, 0, 0, -halo_width - 1,
                              -halo_width );
    auto max = getBound<Dir>( owned_space, 1, off_ijk, halo_width + 1,
                              halo_width, 0, 0 );

    return IndexSpace<num_space_dim>( min, max );
}

//---------------------------------------------------------------------------//
// Given the relative offset of a boundary relative to this local grid's
// get the set of local Dir-direction face indices owned by that boundary
// that are shared with us to use as ghosts.
template <class MeshType>
template <int Dir>
auto LocalGrid<MeshType>::faceBoundaryIndexSpace(
    Ghost, Face<Dir>, const std::array<int, num_space_dim>& off_ijk,
    const int halo_width ) const -> IndexSpace<num_space_dim>
{
    static_assert( Dir < num_space_dim, "Spatial dimension out of bounds" );

    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Face<Dir>(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, -halo_width, 0 );
    auto max = getBound( owned_space, 1, off_ijk, 0, halo_width );

    return IndexSpace<num_space_dim>( min, max );
}
//---------------------------------------------------------------------------//
// Get the local index space of the owned Dir-direction edges.
template <class MeshType>
template <int Dir, std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::edgeIndexSpace( Own, Edge<Dir>, Local ) const
{
    // Compute the lower bound.
    std::array<long, 3> min;
    for ( std::size_t d = 0; d < 3; ++d )
        min[d] = _halo_cell_width;

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( std::size_t d = 0; d < 3; ++d )
    {
        if ( Dir == d )
        {
            max[d] = min[d] + _global_grid->ownedNumCell( d );
        }
        else
        {
            max[d] = ( _global_grid->isPeriodic( d ) ||
                       _global_grid->dimBlockId( d ) <
                           _global_grid->dimNumBlock( d ) - 1 )
                         ? min[d] + _global_grid->ownedNumCell( d )
                         : min[d] + _global_grid->ownedNumCell( d ) + 1;
        }
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted Dir-direction edges.
template <class MeshType>
template <int Dir, std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::edgeIndexSpace( Ghost, Edge<Dir>, Local ) const
{
    // Compute the size.
    std::array<long, 3> size;
    for ( std::size_t d = 0; d < 3; ++d )
    {
        if ( Dir == d )
        {
            size[d] = totalNumCell( d );
        }
        else
        {
            size[d] = totalNumCell( d ) + 1;
        }
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned nodes.
template <class MeshType>
template <int Dir, std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::edgeIndexSpace( Own t1, Edge<Dir> t2, Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local
// Dir-direction edge indices we own that we share with that neighbor to use
// as ghosts.
template <class MeshType>
template <int Dir, std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::edgeSharedIndexSpace( Own, Edge<Dir>,
                                           const std::array<int, 3>& off_ijk,
                                           const int halo_width ) const
{
    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Edge<Dir>(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, 0, -halo_width );
    auto max = getBound<Dir>( owned_space, 1, off_ijk, halo_width,
                              halo_width + 1, 0, 0 );

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get set of local
// Dir-direction edge indices owned by that neighbor that are shared with us
// to use as ghosts.
template <class MeshType>
template <int Dir, std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::edgeSharedIndexSpace( Ghost, Edge<Dir>,
                                           const std::array<int, 3>& off_ijk,
                                           const int halo_width ) const
{
    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Edge<Dir>(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, -halo_width, 0 );
    auto max = getBound<Dir>( owned_space, 1, off_ijk, 0, 0, halo_width,
                              halo_width + 1 );

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Given the relative offset of a boundary relative to this local grid's
// get the set of local Dir-direction edge indices we own that we share
// with that boundary to use as ghosts.
template <class MeshType>
template <int Dir, std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::edgeBoundaryIndexSpace( Own, Edge<Dir>,
                                             const std::array<int, 3>& off_ijk,
                                             const int halo_width ) const
{
    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Edge<Dir>(), Local() );

    auto min = getBound<Dir>( owned_space, -1, off_ijk, 0, 0, -halo_width,
                              -halo_width - 1 );
    auto max = getBound<Dir>( owned_space, 1, off_ijk, halo_width,
                              halo_width + 1, 0, 0 );

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Given the relative offset of a boundary relative to this local grid's
// get the set of local Dir-direction edge indices owned by that boundary
// that are shared with us to use as ghosts.
template <class MeshType>
template <int Dir, std::size_t NSD>
std::enable_if_t<3 == NSD, IndexSpace<3>>
LocalGrid<MeshType>::edgeBoundaryIndexSpace( Ghost, Edge<Dir>,
                                             const std::array<int, 3>& off_ijk,
                                             const int halo_width ) const
{
    // Get the owned local index space.
    auto owned_space = indexSpaceImpl( Own(), Edge<Dir>(), Local() );

    auto min = getBound( owned_space, -1, off_ijk, -halo_width, 0 );
    auto max = getBound( owned_space, 1, off_ijk, 0, halo_width );

    return IndexSpace<3>( min, max );
}
//! \endcond

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_LOCALGRID_IMPL_HPP
