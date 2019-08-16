/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita_Block.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Constructor.
Block::Block( const std::shared_ptr<GlobalGrid> &global_grid,
              const int halo_cell_width )
    : _global_grid( global_grid )
    , _halo_cell_width( halo_cell_width )
{
}

//---------------------------------------------------------------------------//
// Get the global grid that owns the block.
const GlobalGrid &Block::globalGrid() const { return *_global_grid; }

//---------------------------------------------------------------------------//
// Get the physical coordinates of the low corner of the grid in a given
// dimension in the owned decomposition.
template <>
double Block::lowCorner( Own, const int dim ) const
{
    return _global_grid->domain().lowCorner( dim ) +
           _global_grid->cellSize() * _global_grid->globalOffset( dim );
}

//---------------------------------------------------------------------------//
// Get the physical coordinates of the high corner of the grid in a given
// dimension in the owned decomposition.
template <>
double Block::highCorner( Own, const int dim ) const
{
    return _global_grid->domain().lowCorner( dim ) +
           _global_grid->cellSize() * ( _global_grid->globalOffset( dim ) +
                                        _global_grid->ownedNumCell( dim ) );
}

//---------------------------------------------------------------------------//
// Get the physical coordinates of the low corner of the grid in a given
// dimension in the ghosted decomposition.
template <>
double Block::lowCorner( Ghost, const int dim ) const
{
    return ( _global_grid->domain().isPeriodic( dim ) ||
             _global_grid->dimBlockId( dim ) > 0 )
               ? lowCorner( Own(), dim ) -
                     _halo_cell_width * _global_grid->cellSize()
               : lowCorner( Own(), dim );
}

//---------------------------------------------------------------------------//
// Get the physical coordinates of the high corner of the grid in a given
// dimension in the ghosted decomposition.
template <>
double Block::highCorner( Ghost, const int dim ) const
{
    return ( _global_grid->domain().isPeriodic( dim ) ||
             _global_grid->dimBlockId( dim ) <
                 _global_grid->dimNumBlock( dim ) - 1 )
               ? highCorner( Own(), dim ) +
                     _halo_cell_width * _global_grid->cellSize()
               : highCorner( Own(), dim );
}

//---------------------------------------------------------------------------//
// Get the halo size.
int Block::haloWidth() const { return _halo_cell_width; }

//---------------------------------------------------------------------------//
// Given the relative offsets of a neighbor rank relative to this block's
// indices get the of the neighbor. If the neighbor rank is out of bounds
// return -1. Note that in the case of periodic boundaries out of bounds
// indices are allowed as the indices will be wrapped around the periodic
// boundary.
int Block::neighborRank( const int off_i, const int off_j,
                         const int off_k ) const
{
    return _global_grid->blockRank( _global_grid->dimBlockId( Dim::I ) + off_i,
                                    _global_grid->dimBlockId( Dim::J ) + off_j,
                                    _global_grid->dimBlockId( Dim::K ) +
                                        off_k );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned cells.
template <>
IndexSpace<3> Block::indexSpace( Own, Cell, Local ) const
{
    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
        min[d] = ( _global_grid->domain().isPeriodic( d ) ||
                   _global_grid->dimBlockId( d ) > 0 )
                     ? _halo_cell_width
                     : 0;

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
        max[d] = min[d] + _global_grid->ownedNumCell( d );

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted cells.
template <>
IndexSpace<3> Block::indexSpace( Ghost, Cell, Local ) const
{
    // Compute the size.
    std::array<long, 3> size;
    for ( int d = 0; d < 3; ++d )
    {
        // Start with the local number of cells.
        size[d] = _global_grid->ownedNumCell( d );

        // Add the lower halo.
        if ( _global_grid->domain().isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) > 0 )
            size[d] += _halo_cell_width;

        // Add the upper halo.
        if ( _global_grid->domain().isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) <
                 _global_grid->dimNumBlock( d ) - 1 )
            size[d] += _halo_cell_width;
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned cells.
template <>
IndexSpace<3> Block::indexSpace( Own t1, Cell t2, Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned and ghosted cells.
template <>
IndexSpace<3> Block::indexSpace( Ghost t1, Cell t2, Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local cell
// indices we own that we share with that neighbor to use as ghosts.
template <>
IndexSpace<3> Block::sharedIndexSpace( Own, Cell, const int off_i,
                                       const int off_j, const int off_k ) const
{
    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( {0, 0, 0}, {0, 0, 0} );

    // Wrap the indices.
    std::array<long, 3> nid = {off_i, off_j, off_k};

    // Get the owned local index space.
    auto owned_space = indexSpace( Own(), Cell(), Local() );

    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = owned_space.min( d );

        // Middle neighbor.
        else if ( 0 == nid[d] )
            min[d] = owned_space.min( d );

        // Upper neighbor.
        else if ( 1 == nid[d] )
            min[d] = owned_space.max( d ) - _halo_cell_width;
    }

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = owned_space.min( d ) + _halo_cell_width;

        // Middle neighbor.
        else if ( 0 == nid[d] )
            max[d] = owned_space.max( d );

        // Upper neighbor.
        else if ( 1 == nid[d] )
            max[d] = owned_space.max( d );
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get set of local cell
// indices owned by that neighbor that are shared with us to use as
// ghosts.
template <>
IndexSpace<3> Block::sharedIndexSpace( Ghost, Cell tag, const int off_i,
                                       const int off_j, const int off_k ) const
{
    return ghostedSharedIndexSpace( tag, off_i, off_j, off_k );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned nodes.
template <>
IndexSpace<3> Block::indexSpace( Own, Node, Local ) const
{
    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
        min[d] = ( _global_grid->domain().isPeriodic( d ) ||
                   _global_grid->dimBlockId( d ) > 0 )
                     ? _halo_cell_width
                     : 0;

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
        max[d] = ( _global_grid->domain().isPeriodic( d ) ||
                   _global_grid->dimBlockId( d ) <
                       _global_grid->dimNumBlock( d ) - 1 )
                     ? min[d] + _global_grid->ownedNumCell( d )
                     : min[d] + _global_grid->ownedNumCell( d ) + 1;

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted nodes.
template <>
IndexSpace<3> Block::indexSpace( Ghost, Node, Local ) const
{
    // Compute the size.
    std::array<long, 3> size;
    for ( int d = 0; d < 3; ++d )
    {
        // Start with the local number of nodes.
        size[d] = _global_grid->ownedNumCell( d ) + 1;

        // Add the lower halo.
        if ( _global_grid->domain().isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) > 0 )
            size[d] += _halo_cell_width;

        // Add the upper halo.
        if ( _global_grid->domain().isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) <
                 _global_grid->dimNumBlock( d ) - 1 )
            size[d] += _halo_cell_width;
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned nodes.
template <>
IndexSpace<3> Block::indexSpace( Own t1, Node t2, Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned and ghosted nodes.
template <>
IndexSpace<3> Block::indexSpace( Ghost t1, Node t2, Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local node
// indices we own that we share with that neighbor to use as ghosts.
template <>
IndexSpace<3> Block::sharedIndexSpace( Own, Node, const int off_i,
                                       const int off_j, const int off_k ) const
{
    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( {0, 0, 0}, {0, 0, 0} );

    // Wrap the indices.
    std::array<long, 3> nid = {off_i, off_j, off_k};

    // Get the owned local index space.
    auto owned_space = indexSpace( Own(), Node(), Local() );

    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = owned_space.min( d );

        // Middle neighbor.
        else if ( 0 == nid[d] )
            min[d] = owned_space.min( d );

        // Upper neighbor.
        else if ( 1 == nid[d] )
            min[d] = owned_space.max( d ) - _halo_cell_width;
    }

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = owned_space.min( d ) + _halo_cell_width + 1;

        // Middle neighbor.
        else if ( 0 == nid[d] )
            max[d] = owned_space.max( d );

        // Upper neighbor.
        else if ( 1 == nid[d] )
            max[d] = owned_space.max( d );
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get set of local node
// indices owned by that neighbor that are shared with us to use as
// ghosts.
template <>
IndexSpace<3> Block::sharedIndexSpace( Ghost, Node tag, const int off_i,
                                       const int off_j, const int off_k ) const
{
    return ghostedSharedIndexSpace( tag, off_i, off_j, off_k );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Own t1, Face<Dim::I> t2, Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Ghost t1, Face<Dim::I> t2, Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Own t1, Face<Dim::I> t2, Global t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Ghost t1, Face<Dim::I> t2, Global t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::sharedIndexSpace( Own t1, Face<Dim::I> t2, const int i,
                                       const int j, const int k ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::sharedIndexSpace( Ghost t1, Face<Dim::I> t2, const int i,
                                       const int j, const int k ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Own t1, Face<Dim::J> t2, Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Ghost t1, Face<Dim::J> t2, Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Own t1, Face<Dim::J> t2, Global t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Ghost t1, Face<Dim::J> t2, Global t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::sharedIndexSpace( Own t1, Face<Dim::J> t2, const int i,
                                       const int j, const int k ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::sharedIndexSpace( Ghost t1, Face<Dim::J> t2, const int i,
                                       const int j, const int k ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Own t1, Face<Dim::K> t2, Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Ghost t1, Face<Dim::K> t2, Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Own t1, Face<Dim::K> t2, Global t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::indexSpace( Ghost t1, Face<Dim::K> t2, Global t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::sharedIndexSpace( Own t1, Face<Dim::K> t2, const int i,
                                       const int j, const int k ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k );
}

//---------------------------------------------------------------------------//
template <>
IndexSpace<3> Block::sharedIndexSpace( Ghost t1, Face<Dim::K> t2, const int i,
                                       const int j, const int k ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned cells.
template <class EntityType>
IndexSpace<3> Block::globalIndexSpace( Own, EntityType ) const
{
    auto local_space = indexSpace( Own(), EntityType(), Local() );
    std::array<long, 3> min;
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        min[d] = _global_grid->globalOffset( d );
        max[d] = min[d] + local_space.extent( d );
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned and ghosted cells.
template <class EntityType>
IndexSpace<3> Block::globalIndexSpace( Ghost, EntityType ) const
{
    auto own_local_space = indexSpace( Own(), EntityType(), Local() );
    auto ghost_local_space = indexSpace( Ghost(), EntityType(), Local() );
    std::array<long, 3> min;
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        min[d] = _global_grid->globalOffset( d ) - own_local_space.min( d );
        max[d] = min[d] + ghost_local_space.extent( d );
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the ghosted shared index space of the block.
template <class EntityType>
IndexSpace<3> Block::ghostedSharedIndexSpace( EntityType, const int off_i,
                                              const int off_j,
                                              const int off_k ) const
{
    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( {0, 0, 0}, {0, 0, 0} );

    // Wrap the indices.
    std::array<long, 3> nid = {off_i, off_j, off_k};

    // Get the owned local index space.
    auto owned_space = indexSpace( Own(), EntityType(), Local() );

    // Get the ghosted local index space.
    auto ghosted_space = indexSpace( Ghost(), EntityType(), Local() );

    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = ghosted_space.min( d );

        // Middle neighbor
        else if ( 0 == nid[d] )
            min[d] = owned_space.min( d );

        // Upper neighbor.
        else if ( 1 == nid[d] )
            min[d] = owned_space.max( d );
    }

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = owned_space.min( d );

        // Middle neighbor
        else if ( 0 == nid[d] )
            max[d] = owned_space.max( d );

        // Upper neighbor.
        else if ( 1 == nid[d] )
            max[d] = ghosted_space.max( d );
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned Dir-direction faces.
template <int Dir>
IndexSpace<3> Block::faceIndexSpace( Own, Face<Dir>, Local ) const
{
    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
        min[d] = ( _global_grid->domain().isPeriodic( d ) ||
                   _global_grid->dimBlockId( d ) > 0 )
                     ? _halo_cell_width
                     : 0;

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dir == d )
        {
            max[d] = ( _global_grid->domain().isPeriodic( d ) ||
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

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted Dir-direction faces.
template <int Dir>
IndexSpace<3> Block::faceIndexSpace( Ghost, Face<Dir>, Local ) const
{
    // Compute the size.
    std::array<long, 3> size;
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dir == d )
        {
            size[d] = _global_grid->ownedNumCell( d ) + 1;
        }
        else
        {
            size[d] = _global_grid->ownedNumCell( d );
        }
    }

    for ( int d = 0; d < 3; ++d )
    {
        // Add the lower halo.
        if ( _global_grid->domain().isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) > 0 )
            size[d] += _halo_cell_width;

        // Add the upper halo.
        if ( _global_grid->domain().isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) <
                 _global_grid->dimNumBlock( d ) - 1 )
            size[d] += _halo_cell_width;
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned nodes.
template <int Dir>
IndexSpace<3> Block::faceIndexSpace( Own t1, Face<Dir> t2, Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned and ghosted nodes.
template <int Dir>
IndexSpace<3> Block::faceIndexSpace( Ghost t1, Face<Dir> t2, Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local
// Dir-direction face indices we own that we share with that neighbor to use
// as ghosts.
template <int Dir>
IndexSpace<3> Block::faceSharedIndexSpace( Own, Face<Dir>, const int off_i,
                                           const int off_j,
                                           const int off_k ) const
{
    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( {0, 0, 0}, {0, 0, 0} );

    // Wrap the indices.
    std::array<long, 3> nid = {off_i, off_j, off_k};

    // Get the owned local index space.
    auto owned_space = indexSpace( Own(), Face<Dir>(), Local() );

    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = owned_space.min( d );

        // Middle neighbor.
        else if ( 0 == nid[d] )
            min[d] = owned_space.min( d );

        // Upper neighbor.
        else if ( 1 == nid[d] )
            min[d] = owned_space.max( d ) - _halo_cell_width;
    }

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = ( Dir == d ) ? owned_space.min( d ) + _halo_cell_width + 1
                                  : owned_space.min( d ) + _halo_cell_width;

        // Middle neighbor.
        else if ( 0 == nid[d] )
            max[d] = owned_space.max( d );

        // Upper neighbor.
        else if ( 1 == nid[d] )
            max[d] = owned_space.max( d );
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get set of local
// Dir-direction face indices owned by that neighbor that are shared with us
// to use as ghosts.
template <int Dir>
IndexSpace<3> Block::faceSharedIndexSpace( Ghost, Face<Dir> tag,
                                           const int off_i, const int off_j,
                                           const int off_k ) const
{
    return ghostedSharedIndexSpace( tag, off_i, off_j, off_k );
}

//---------------------------------------------------------------------------//
// Global indexing explicit instantiations
//---------------------------------------------------------------------------//
template IndexSpace<3> Block::globalIndexSpace( Own, Cell ) const;

template IndexSpace<3> Block::globalIndexSpace( Ghost, Cell ) const;

template IndexSpace<3> Block::globalIndexSpace( Own, Node ) const;

template IndexSpace<3> Block::globalIndexSpace( Ghost, Node ) const;

template IndexSpace<3> Block::globalIndexSpace( Own, Face<Dim::I> ) const;

template IndexSpace<3> Block::globalIndexSpace( Ghost, Face<Dim::I> ) const;

template IndexSpace<3> Block::globalIndexSpace( Own, Face<Dim::J> ) const;

template IndexSpace<3> Block::globalIndexSpace( Ghost, Face<Dim::J> ) const;

template IndexSpace<3> Block::globalIndexSpace( Own, Face<Dim::K> ) const;

template IndexSpace<3> Block::globalIndexSpace( Ghost, Face<Dim::K> ) const;

//---------------------------------------------------------------------------//
// Ghosted shared indexing explicit instantiations.
//---------------------------------------------------------------------------//
template IndexSpace<3>
Block::ghostedSharedIndexSpace( Cell, const int, const int, const int ) const;

template IndexSpace<3>
Block::ghostedSharedIndexSpace( Node, const int, const int, const int ) const;

template IndexSpace<3> Block::ghostedSharedIndexSpace( Face<Dim::I>, const int,
                                                       const int,
                                                       const int ) const;

template IndexSpace<3> Block::ghostedSharedIndexSpace( Face<Dim::J>, const int,
                                                       const int,
                                                       const int ) const;

template IndexSpace<3> Block::ghostedSharedIndexSpace( Face<Dim::K>, const int,
                                                       const int,
                                                       const int ) const;

//---------------------------------------------------------------------------//
// Face indexing explicit instantiations
//---------------------------------------------------------------------------//
template IndexSpace<3> Block::faceIndexSpace( Own, Face<Dim::I>, Local ) const;

template IndexSpace<3> Block::faceIndexSpace( Own, Face<Dim::I>, Global ) const;

template IndexSpace<3> Block::faceIndexSpace( Ghost, Face<Dim::I>,
                                              Local ) const;

template IndexSpace<3> Block::faceIndexSpace( Ghost, Face<Dim::I>,
                                              Global ) const;

template IndexSpace<3> Block::faceIndexSpace( Own, Face<Dim::J>, Local ) const;

template IndexSpace<3> Block::faceIndexSpace( Own, Face<Dim::J>, Global ) const;

template IndexSpace<3> Block::faceIndexSpace( Ghost, Face<Dim::J>,
                                              Local ) const;

template IndexSpace<3> Block::faceIndexSpace( Ghost, Face<Dim::J>,
                                              Global ) const;

template IndexSpace<3> Block::faceIndexSpace( Own, Face<Dim::K>, Local ) const;

template IndexSpace<3> Block::faceIndexSpace( Own, Face<Dim::K>, Global ) const;

template IndexSpace<3> Block::faceIndexSpace( Ghost, Face<Dim::K>,
                                              Local ) const;

template IndexSpace<3> Block::faceIndexSpace( Ghost, Face<Dim::K>,
                                              Global ) const;

template IndexSpace<3> Block::faceSharedIndexSpace( Own, Face<Dim::I>,
                                                    const int, const int,
                                                    const int ) const;

template IndexSpace<3> Block::faceSharedIndexSpace( Ghost, Face<Dim::I>,
                                                    const int, const int,
                                                    const int ) const;

template IndexSpace<3> Block::faceSharedIndexSpace( Own, Face<Dim::J>,
                                                    const int, const int,
                                                    const int ) const;

template IndexSpace<3> Block::faceSharedIndexSpace( Ghost, Face<Dim::J>,
                                                    const int, const int,
                                                    const int ) const;

template IndexSpace<3> Block::faceSharedIndexSpace( Own, Face<Dim::K>,
                                                    const int, const int,
                                                    const int ) const;

template IndexSpace<3> Block::faceSharedIndexSpace( Ghost, Face<Dim::K>,
                                                    const int, const int,
                                                    const int ) const;

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
std::shared_ptr<Block>
createBlock( const std::shared_ptr<GlobalGrid> &global_grid,
             const int halo_cell_width )
{
    return std::make_shared<Block>( global_grid, halo_cell_width );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita
