/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita_LocalGrid.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Constructor.
template <class MeshType>
LocalGrid<MeshType>::LocalGrid(
    const std::shared_ptr<GlobalGrid<MeshType>> &global_grid,
    const int halo_cell_width )
    : _global_grid( global_grid )
    , _halo_cell_width( halo_cell_width )
{
}

//---------------------------------------------------------------------------//
// Get the global grid that owns the local grid.
template <class MeshType>
const GlobalGrid<MeshType> &LocalGrid<MeshType>::globalGrid() const
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
// Given the relative offsets of a neighbor rank relative to this local grid's
// indices get the of the neighbor. If the neighbor rank is out of bounds
// return -1. Note that in the case of periodic boundaries out of bounds
// indices are allowed as the indices will be wrapped around the periodic
// boundary.
template <class MeshType>
int LocalGrid<MeshType>::neighborRank( const int off_i, const int off_j,
                                       const int off_k ) const
{
    return _global_grid->blockRank( _global_grid->dimBlockId( Dim::I ) + off_i,
                                    _global_grid->dimBlockId( Dim::J ) + off_j,
                                    _global_grid->dimBlockId( Dim::K ) +
                                        off_k );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned cells.
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own, Cell, Local ) const
{
    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
        min[d] = ( _global_grid->isPeriodic( d ) ||
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
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Ghost, Cell, Local ) const
{
    // Compute the size.
    std::array<long, 3> size;
    for ( int d = 0; d < 3; ++d )
    {
        // Start with the local number of cells.
        size[d] = _global_grid->ownedNumCell( d );

        // Add the lower halo.
        if ( _global_grid->isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) > 0 )
            size[d] += _halo_cell_width;

        // Add the upper halo.
        if ( _global_grid->isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) <
                 _global_grid->dimNumBlock( d ) - 1 )
            size[d] += _halo_cell_width;
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned cells.
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Cell t2, Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local cell
// indices we own that we share with that neighbor to use as ghosts.
template <class MeshType>
IndexSpace<3>
LocalGrid<MeshType>::sharedIndexSpace( Own, Cell, const int off_i,
                                       const int off_j, const int off_k,
                                       const int halo_width ) const
{
    // If we got the default halo width of -1 this means we want to use the
    // default of the entire halo.
    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;

    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check that the requested halo width is valid.
    if ( hw > _halo_cell_width )
        throw std::logic_error(
            "Requested halo width larger than local grid halo" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( { 0, 0, 0 }, { 0, 0, 0 } );

    // Wrap the indices.
    std::array<long, 3> nid = { off_i, off_j, off_k };

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
            min[d] = owned_space.max( d ) - hw;
    }

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = owned_space.min( d ) + hw;

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
template <class MeshType>
IndexSpace<3>
LocalGrid<MeshType>::sharedIndexSpace( Ghost, Cell, const int off_i,
                                       const int off_j, const int off_k,
                                       const int halo_width ) const
{
    // If we got the default halo width of -1 this means we want to use the
    // default of the entire halo.
    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;

    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check that the requested halo width is valid.
    if ( hw > _halo_cell_width )
        throw std::logic_error(
            "Requested halo width larger than local grid halo" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( { 0, 0, 0 }, { 0, 0, 0 } );

    // Wrap the indices.
    std::array<long, 3> nid = { off_i, off_j, off_k };

    // Get the owned local index space.
    auto owned_space = indexSpace( Own(), Cell(), Local() );

    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = owned_space.min( d ) - hw;

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
            max[d] = owned_space.max( d ) + hw;
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned nodes.
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own, Node, Local ) const
{
    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
        min[d] = ( _global_grid->isPeriodic( d ) ||
                   _global_grid->dimBlockId( d ) > 0 )
                     ? _halo_cell_width
                     : 0;

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
        max[d] = ( _global_grid->isPeriodic( d ) ||
                   _global_grid->dimBlockId( d ) <
                       _global_grid->dimNumBlock( d ) - 1 )
                     ? min[d] + _global_grid->ownedNumCell( d )
                     : min[d] + _global_grid->ownedNumCell( d ) + 1;

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted nodes.
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Ghost, Node, Local ) const
{
    // Compute the size.
    std::array<long, 3> size;
    for ( int d = 0; d < 3; ++d )
    {
        // Start with the local number of nodes.
        size[d] = _global_grid->ownedNumCell( d ) + 1;

        // Add the lower halo.
        if ( _global_grid->isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) > 0 )
            size[d] += _halo_cell_width;

        // Add the upper halo.
        if ( _global_grid->isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) <
                 _global_grid->dimNumBlock( d ) - 1 )
            size[d] += _halo_cell_width;
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned nodes.
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Node t2, Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local node
// indices we own that we share with that neighbor to use as ghosts.
template <class MeshType>
IndexSpace<3>
LocalGrid<MeshType>::sharedIndexSpace( Own, Node, const int off_i,
                                       const int off_j, const int off_k,
                                       const int halo_width ) const
{
    // If we got the default halo width of -1 this means we want to use the
    // default of the entire halo.
    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;

    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check that the requested halo width is valid.
    if ( hw > _halo_cell_width )
        throw std::logic_error(
            "Requested halo width larger than local grid halo" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( { 0, 0, 0 }, { 0, 0, 0 } );

    // Wrap the indices.
    std::array<long, 3> nid = { off_i, off_j, off_k };

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
            min[d] = owned_space.max( d ) - hw;
    }

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = owned_space.min( d ) + hw + 1;

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
template <class MeshType>
IndexSpace<3>
LocalGrid<MeshType>::sharedIndexSpace( Ghost, Node, const int off_i,
                                       const int off_j, const int off_k,
                                       const int halo_width ) const
{
    // If we got the default halo width of -1 this means we want to use the
    // default of the entire halo.
    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;

    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check that the requested halo width is valid.
    if ( hw > _halo_cell_width )
        throw std::logic_error(
            "Requested halo width larger than local grid halo" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( { 0, 0, 0 }, { 0, 0, 0 } );

    // Wrap the indices.
    std::array<long, 3> nid = { off_i, off_j, off_k };

    // Get the owned local index space.
    auto owned_space = indexSpace( Own(), Node(), Local() );

    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = owned_space.min( d ) - hw;

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
            max[d] = owned_space.max( d ) + hw + 1;
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Face<Dim::I> t2,
                                               Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Ghost t1, Face<Dim::I> t2,
                                               Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Face<Dim::I> t2,
                                               Global t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Own t1, Face<Dim::I> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Ghost t1, Face<Dim::I> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Face<Dim::J> t2,
                                               Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Ghost t1, Face<Dim::J> t2,
                                               Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Face<Dim::J> t2,
                                               Global t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Own t1, Face<Dim::J> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Ghost t1, Face<Dim::J> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Face<Dim::K> t2,
                                               Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Ghost t1, Face<Dim::K> t2,
                                               Local t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Face<Dim::K> t2,
                                               Global t3 ) const
{
    return faceIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Own t1, Face<Dim::K> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Ghost t1, Face<Dim::K> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return faceSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Edge<Dim::I> t2,
                                               Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Ghost t1, Edge<Dim::I> t2,
                                               Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Edge<Dim::I> t2,
                                               Global t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Own t1, Edge<Dim::I> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return edgeSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Ghost t1, Edge<Dim::I> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return edgeSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Edge<Dim::J> t2,
                                               Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Ghost t1, Edge<Dim::J> t2,
                                               Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Edge<Dim::J> t2,
                                               Global t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Own t1, Edge<Dim::J> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return edgeSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Ghost t1, Edge<Dim::J> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return edgeSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Edge<Dim::K> t2,
                                               Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Ghost t1, Edge<Dim::K> t2,
                                               Local t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::indexSpace( Own t1, Edge<Dim::K> t2,
                                               Global t3 ) const
{
    return edgeIndexSpace( t1, t2, t3 );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Own t1, Edge<Dim::K> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return edgeSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
template <class MeshType>
IndexSpace<3> LocalGrid<MeshType>::sharedIndexSpace( Ghost t1, Edge<Dim::K> t2,
                                                     const int i, const int j,
                                                     const int k,
                                                     const int hw ) const
{
    return edgeSharedIndexSpace( t1, t2, i, j, k, hw );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned entities.
template <class MeshType>
template <class EntityType>
IndexSpace<3> LocalGrid<MeshType>::globalIndexSpace( Own, EntityType ) const
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
// Get the local index space of the owned Dir-direction faces.
template <class MeshType>
template <int Dir>
IndexSpace<3> LocalGrid<MeshType>::faceIndexSpace( Own, Face<Dir>, Local ) const
{
    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
        min[d] = ( _global_grid->isPeriodic( d ) ||
                   _global_grid->dimBlockId( d ) > 0 )
                     ? _halo_cell_width
                     : 0;

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
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

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned and ghosted Dir-direction faces.
template <class MeshType>
template <int Dir>
IndexSpace<3> LocalGrid<MeshType>::faceIndexSpace( Ghost, Face<Dir>,
                                                   Local ) const
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
        if ( _global_grid->isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) > 0 )
            size[d] += _halo_cell_width;

        // Add the upper halo.
        if ( _global_grid->isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) <
                 _global_grid->dimNumBlock( d ) - 1 )
            size[d] += _halo_cell_width;
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned nodes.
template <class MeshType>
template <int Dir>
IndexSpace<3> LocalGrid<MeshType>::faceIndexSpace( Own t1, Face<Dir> t2,
                                                   Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local
// Dir-direction face indices we own that we share with that neighbor to use
// as ghosts.
template <class MeshType>
template <int Dir>
IndexSpace<3>
LocalGrid<MeshType>::faceSharedIndexSpace( Own, Face<Dir>, const int off_i,
                                           const int off_j, const int off_k,
                                           const int halo_width ) const
{
    // If we got the default halo width of -1 this means we want to use the
    // default of the entire halo.
    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;

    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check that the requested halo width is valid.
    if ( hw > _halo_cell_width )
        throw std::logic_error(
            "Requested halo width larger than local grid halo" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( { 0, 0, 0 }, { 0, 0, 0 } );

    // Wrap the indices.
    std::array<long, 3> nid = { off_i, off_j, off_k };

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
            min[d] = owned_space.max( d ) - hw;
    }

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = ( Dir == d ) ? owned_space.min( d ) + hw + 1
                                  : owned_space.min( d ) + hw;

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
template <class MeshType>
template <int Dir>
IndexSpace<3>
LocalGrid<MeshType>::faceSharedIndexSpace( Ghost, Face<Dir>, const int off_i,
                                           const int off_j, const int off_k,
                                           const int halo_width ) const
{
    // If we got the default halo width of -1 this means we want to use the
    // default of the entire halo.
    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;

    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check that the requested halo width is valid.
    if ( hw > _halo_cell_width )
        throw std::logic_error(
            "Requested halo width larger than local grid halo" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( { 0, 0, 0 }, { 0, 0, 0 } );

    // Wrap the indices.
    std::array<long, 3> nid = { off_i, off_j, off_k };

    // Get the owned local index space.
    auto owned_space = indexSpace( Own(), Face<Dir>(), Local() );

    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = owned_space.min( d ) - hw;

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
            max[d] = ( Dir == d ) ? owned_space.max( d ) + hw + 1
                                  : owned_space.max( d ) + hw;
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Get the local index space of the owned Dir-direction edges.
template <class MeshType>
template <int Dir>
IndexSpace<3> LocalGrid<MeshType>::edgeIndexSpace( Own, Edge<Dir>, Local ) const
{
    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
        min[d] = ( _global_grid->isPeriodic( d ) ||
                   _global_grid->dimBlockId( d ) > 0 )
                     ? _halo_cell_width
                     : 0;

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
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
template <int Dir>
IndexSpace<3> LocalGrid<MeshType>::edgeIndexSpace( Ghost, Edge<Dir>,
                                                   Local ) const
{
    // Compute the size.
    std::array<long, 3> size;
    for ( int d = 0; d < 3; ++d )
    {
        if ( Dir == d )
        {
            size[d] = _global_grid->ownedNumCell( d );
        }
        else
        {
            size[d] = _global_grid->ownedNumCell( d ) + 1;
        }
    }

    for ( int d = 0; d < 3; ++d )
    {
        // Add the lower halo.
        if ( _global_grid->isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) > 0 )
            size[d] += _halo_cell_width;

        // Add the upper halo.
        if ( _global_grid->isPeriodic( d ) ||
             _global_grid->dimBlockId( d ) <
                 _global_grid->dimNumBlock( d ) - 1 )
            size[d] += _halo_cell_width;
    }

    return IndexSpace<3>( size );
}

//---------------------------------------------------------------------------//
// Get the global index space of the owned nodes.
template <class MeshType>
template <int Dir>
IndexSpace<3> LocalGrid<MeshType>::edgeIndexSpace( Own t1, Edge<Dir> t2,
                                                   Global ) const
{
    return globalIndexSpace( t1, t2 );
}

//---------------------------------------------------------------------------//
// Given a relative set of indices of a neighbor get the set of local
// Dir-direction edge indices we own that we share with that neighbor to use
// as ghosts.
template <class MeshType>
template <int Dir>
IndexSpace<3>
LocalGrid<MeshType>::edgeSharedIndexSpace( Own, Edge<Dir>, const int off_i,
                                           const int off_j, const int off_k,
                                           const int halo_width ) const
{
    // If we got the default halo width of -1 this means we want to use the
    // default of the entire halo.
    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;

    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check that the requested halo width is valid.
    if ( hw > _halo_cell_width )
        throw std::logic_error(
            "Requested halo width larger than local grid halo" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( { 0, 0, 0 }, { 0, 0, 0 } );

    // Wrap the indices.
    std::array<long, 3> nid = { off_i, off_j, off_k };

    // Get the owned local index space.
    auto owned_space = indexSpace( Own(), Edge<Dir>(), Local() );

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
            min[d] = owned_space.max( d ) - hw;
    }

    // Compute the upper bound.
    std::array<long, 3> max;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            max[d] = ( Dir == d ) ? owned_space.min( d ) + hw
                                  : owned_space.min( d ) + hw + 1;

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
// Dir-direction edge indices owned by that neighbor that are shared with us
// to use as ghosts.
template <class MeshType>
template <int Dir>
IndexSpace<3>
LocalGrid<MeshType>::edgeSharedIndexSpace( Ghost, Edge<Dir>, const int off_i,
                                           const int off_j, const int off_k,
                                           const int halo_width ) const
{
    // If we got the default halo width of -1 this means we want to use the
    // default of the entire halo.
    int hw = ( -1 == halo_width ) ? _halo_cell_width : halo_width;

    // Check that the offsets are valid.
    if ( off_i < -1 || 1 < off_i || off_j < -1 || 1 < off_j || off_k < -1 ||
         1 < off_k )
        throw std::logic_error( "Neighbor indices out of bounds" );

    // Check that the requested halo width is valid.
    if ( hw > _halo_cell_width )
        throw std::logic_error(
            "Requested halo width larger than local grid halo" );

    // Check to see if this is a valid neighbor. If not, return a shared space
    // of size 0.
    if ( neighborRank( off_i, off_j, off_k ) < 0 )
        return IndexSpace<3>( { 0, 0, 0 }, { 0, 0, 0 } );

    // Wrap the indices.
    std::array<long, 3> nid = { off_i, off_j, off_k };

    // Get the owned local index space.
    auto owned_space = indexSpace( Own(), Edge<Dir>(), Local() );

    // Compute the lower bound.
    std::array<long, 3> min;
    for ( int d = 0; d < 3; ++d )
    {
        // Lower neighbor.
        if ( -1 == nid[d] )
            min[d] = owned_space.min( d ) - hw;

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
            max[d] = ( Dir == d ) ? owned_space.max( d ) + hw
                                  : owned_space.max( d ) + hw + 1;
    }

    return IndexSpace<3>( min, max );
}

//---------------------------------------------------------------------------//
// Global indexing explicit instantiations
//---------------------------------------------------------------------------//
#define CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( MESH, FP, DECOMP, ENTITY )     \
    template IndexSpace<3> LocalGrid<MESH<FP>>::globalIndexSpace(              \
        DECOMP, ENTITY ) const;

CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( UniformMesh, float, Own, Cell )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( UniformMesh, float, Own, Node )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( UniformMesh, float, Own, Face<Dim::I> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( UniformMesh, float, Own, Face<Dim::J> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( UniformMesh, float, Own, Face<Dim::K> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( UniformMesh, double, Own, Cell )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( UniformMesh, double, Own, Node )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( UniformMesh, double, Own, Face<Dim::I> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( UniformMesh, double, Own, Face<Dim::J> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( UniformMesh, double, Own, Face<Dim::K> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( NonUniformMesh, float, Own, Cell )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( NonUniformMesh, float, Own, Node )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( NonUniformMesh, float, Own,
                                        Face<Dim::I> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( NonUniformMesh, float, Own,
                                        Face<Dim::J> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( NonUniformMesh, float, Own,
                                        Face<Dim::K> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( NonUniformMesh, double, Own, Cell )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( NonUniformMesh, double, Own, Node )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( NonUniformMesh, double, Own,
                                        Face<Dim::I> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( NonUniformMesh, double, Own,
                                        Face<Dim::J> )
CAJITA_INST_LOCALGRID_GLOBALINDEXSPACE( NonUniformMesh, double, Own,
                                        Face<Dim::K> )

//---------------------------------------------------------------------------//
// Face indexing explicit instantiations
//---------------------------------------------------------------------------//
#define CAJITA_INST_LOCALGRID_FACEINDEXSPACE( MESH, FP, DECOMP, DIM, INDEX )   \
    template IndexSpace<3> LocalGrid<MESH<FP>>::faceIndexSpace(                \
        DECOMP, Face<DIM>, INDEX ) const;

CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, float, Own, Dim::I, Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, float, Own, Dim::I, Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, float, Ghost, Dim::I, Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, float, Own, Dim::J, Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, float, Own, Dim::J, Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, float, Ghost, Dim::J, Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, float, Own, Dim::K, Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, float, Own, Dim::K, Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, float, Ghost, Dim::K, Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, double, Own, Dim::I, Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, double, Own, Dim::I, Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, double, Ghost, Dim::I,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, double, Own, Dim::J, Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, double, Own, Dim::J, Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, double, Ghost, Dim::J,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, double, Own, Dim::K, Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, double, Own, Dim::K, Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( UniformMesh, double, Ghost, Dim::K,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, float, Own, Dim::I,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, float, Own, Dim::I,
                                      Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, float, Ghost, Dim::I,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, float, Own, Dim::J,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, float, Own, Dim::J,
                                      Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, float, Ghost, Dim::J,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, float, Own, Dim::K,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, float, Own, Dim::K,
                                      Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, float, Ghost, Dim::K,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, double, Own, Dim::I,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, double, Own, Dim::I,
                                      Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, double, Ghost, Dim::I,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, double, Own, Dim::J,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, double, Own, Dim::J,
                                      Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, double, Ghost, Dim::J,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, double, Own, Dim::K,
                                      Local )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, double, Own, Dim::K,
                                      Global )
CAJITA_INST_LOCALGRID_FACEINDEXSPACE( NonUniformMesh, double, Ghost, Dim::K,
                                      Local )

#define CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( MESH, FP, DECOMP, DIM )    \
    template IndexSpace<3> LocalGrid<MESH<FP>>::faceSharedIndexSpace(          \
        DECOMP, Face<DIM>, const int, const int, const int, const int ) const;

CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, double, Own, Dim::I )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, double, Ghost, Dim::I )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, double, Own, Dim::J )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, double, Ghost, Dim::J )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, double, Own, Dim::K )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, double, Ghost, Dim::K )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, float, Own, Dim::I )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, float, Ghost, Dim::I )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, float, Own, Dim::J )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, float, Ghost, Dim::J )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, float, Own, Dim::K )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( UniformMesh, float, Ghost, Dim::K )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, double, Own,
                                            Dim::I )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, double, Ghost,
                                            Dim::I )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, double, Own,
                                            Dim::J )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, double, Ghost,
                                            Dim::J )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, double, Own,
                                            Dim::K )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, double, Ghost,
                                            Dim::K )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, float, Own, Dim::I )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, float, Ghost,
                                            Dim::I )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, float, Own, Dim::J )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, float, Ghost,
                                            Dim::J )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, float, Own, Dim::K )
CAJITA_INST_LOCALGRID_FACESHAREDINDEXSPACE( NonUniformMesh, float, Ghost,
                                            Dim::K )

//---------------------------------------------------------------------------//
// Edge indexing explicit instantiations
//---------------------------------------------------------------------------//
#define CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( MESH, FP, DECOMP, DIM, INDEX )   \
    template IndexSpace<3> LocalGrid<MESH<FP>>::edgeIndexSpace(                \
        DECOMP, Edge<DIM>, INDEX ) const;

CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, float, Own, Dim::I, Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, float, Own, Dim::I, Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, float, Ghost, Dim::I, Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, float, Own, Dim::J, Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, float, Own, Dim::J, Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, float, Ghost, Dim::J, Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, float, Own, Dim::K, Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, float, Own, Dim::K, Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, float, Ghost, Dim::K, Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, double, Own, Dim::I, Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, double, Own, Dim::I, Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, double, Ghost, Dim::I,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, double, Own, Dim::J, Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, double, Own, Dim::J, Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, double, Ghost, Dim::J,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, double, Own, Dim::K, Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, double, Own, Dim::K, Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( UniformMesh, double, Ghost, Dim::K,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, float, Own, Dim::I,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, float, Own, Dim::I,
                                      Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, float, Ghost, Dim::I,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, float, Own, Dim::J,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, float, Own, Dim::J,
                                      Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, float, Ghost, Dim::J,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, float, Own, Dim::K,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, float, Own, Dim::K,
                                      Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, float, Ghost, Dim::K,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, double, Own, Dim::I,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, double, Own, Dim::I,
                                      Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, double, Ghost, Dim::I,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, double, Own, Dim::J,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, double, Own, Dim::J,
                                      Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, double, Ghost, Dim::J,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, double, Own, Dim::K,
                                      Local )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, double, Own, Dim::K,
                                      Global )
CAJITA_INST_LOCALGRID_EDGEINDEXSPACE( NonUniformMesh, double, Ghost, Dim::K,
                                      Local )

#define CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( MESH, FP, DECOMP, DIM )    \
    template IndexSpace<3> LocalGrid<MESH<FP>>::edgeSharedIndexSpace(          \
        DECOMP, Edge<DIM>, const int, const int, const int, const int ) const;

CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, double, Own, Dim::I )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, double, Ghost, Dim::I )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, double, Own, Dim::J )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, double, Ghost, Dim::J )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, double, Own, Dim::K )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, double, Ghost, Dim::K )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, float, Own, Dim::I )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, float, Ghost, Dim::I )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, float, Own, Dim::J )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, float, Ghost, Dim::J )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, float, Own, Dim::K )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( UniformMesh, float, Ghost, Dim::K )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, double, Own,
                                            Dim::I )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, double, Ghost,
                                            Dim::I )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, double, Own,
                                            Dim::J )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, double, Ghost,
                                            Dim::J )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, double, Own,
                                            Dim::K )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, double, Ghost,
                                            Dim::K )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, float, Own, Dim::I )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, float, Ghost,
                                            Dim::I )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, float, Own, Dim::J )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, float, Ghost,
                                            Dim::J )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, float, Own, Dim::K )
CAJITA_INST_LOCALGRID_EDGESHAREDINDEXSPACE( NonUniformMesh, float, Ghost,
                                            Dim::K )

//---------------------------------------------------------------------------//
// Class explicit instantiations.

template class LocalGrid<UniformMesh<float>>;
template class LocalGrid<UniformMesh<double>>;

template class LocalGrid<NonUniformMesh<float>>;
template class LocalGrid<NonUniformMesh<double>>;

//---------------------------------------------------------------------------//

} // end namespace Cajita
