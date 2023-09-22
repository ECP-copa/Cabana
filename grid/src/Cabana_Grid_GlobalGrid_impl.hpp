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

#ifndef CABANA_GRID_GLOBALGRID_IMPL_HPP
#define CABANA_GRID_GLOBALGRID_IMPL_HPP

#include <algorithm>
#include <cmath>
#include <limits>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
// Constructor.
template <class MeshType>
GlobalGrid<MeshType>::GlobalGrid(
    MPI_Comm comm, const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
    const std::array<bool, num_space_dim>& periodic,
    const BlockPartitioner<num_space_dim>& partitioner )
    : _global_mesh( global_mesh )
    , _periodic( periodic )
{
    // Partition the problem.
    std::array<int, num_space_dim> global_num_cell;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        global_num_cell[d] = _global_mesh->globalNumCell( d );
    _ranks_per_dim = partitioner.ranksPerDimension( comm, global_num_cell );

    // Extract the periodicity of the boundary as integers.
    std::array<int, num_space_dim> periodic_dims;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        periodic_dims[d] = _periodic[d];

    // Generate a communicator with a Cartesian topology.
    int reorder_cart_ranks = 1;

    auto num_space_dim_copy = num_space_dim;
    auto ranks_per_dim_copy = _ranks_per_dim.data();
    auto periodic_dims_copy = periodic_dims.data();
    _cart_comm_ptr.reset(
        // Duplicate the communicator and store in a std::shared_ptr so that
        // all copies point to the same object
        [comm, num_space_dim_copy, ranks_per_dim_copy, periodic_dims_copy,
         reorder_cart_ranks]()
        {
            auto p = std::make_unique<MPI_Comm>();
            MPI_Cart_create( comm, num_space_dim_copy, ranks_per_dim_copy,
                             periodic_dims_copy, reorder_cart_ranks, p.get() );
            return p.release();
        }(),
        // Custom deleter to mark the communicator for deallocation
        []( MPI_Comm* p )
        {
            MPI_Comm_free( p );
            delete p;
        } );

    // Get the Cartesian topology index of this rank.
    int linear_rank;
    MPI_Comm_rank( this->comm(), &linear_rank );
    MPI_Cart_coords( this->comm(), linear_rank, num_space_dim,
                     _cart_rank.data() );

    // Get the cells per dimension and the remainder.
    partitioner.ownedCellInfo( this->comm(), global_num_cell, _owned_num_cell,
                               _global_cell_offset );

    // Determine if a block is on the low or high boundaries.
    for ( std::size_t d = 0; d < num_space_dim; ++d )
    {
        _boundary_lo[d] = ( 0 == _cart_rank[d] );
        _boundary_hi[d] = ( _ranks_per_dim[d] - 1 == _cart_rank[d] );
    }
}

//---------------------------------------------------------------------------//
// Destructor.
template <class MeshType>
GlobalGrid<MeshType>::~GlobalGrid()
{
}

//---------------------------------------------------------------------------//
// Get the grid communicator.
template <class MeshType>
MPI_Comm GlobalGrid<MeshType>::comm() const
{
    return *_cart_comm_ptr;
}

//---------------------------------------------------------------------------//
// Get the global mesh data.
template <class MeshType>
const GlobalMesh<MeshType>& GlobalGrid<MeshType>::globalMesh() const
{
    return *_global_mesh;
}

//---------------------------------------------------------------------------//
// Get whether a given dimension is periodic.
template <class MeshType>
bool GlobalGrid<MeshType>::isPeriodic( const int dim ) const
{
    return _periodic[dim];
}

//---------------------------------------------------------------------------//
// Determine if this block is on a low boundary in the given dimension.
template <class MeshType>
bool GlobalGrid<MeshType>::onLowBoundary( const int dim ) const
{
    return _boundary_lo[dim];
}

//---------------------------------------------------------------------------//
// Determine if this block is on a high boundary in the given dimension.
template <class MeshType>
bool GlobalGrid<MeshType>::onHighBoundary( const int dim ) const
{
    return _boundary_hi[dim];
}

//---------------------------------------------------------------------------//
// Get the number of blocks in each dimension in the global mesh.
template <class MeshType>
int GlobalGrid<MeshType>::dimNumBlock( const int dim ) const
{
    return _ranks_per_dim[dim];
}

//---------------------------------------------------------------------------//
// Get the total number of blocks.
template <class MeshType>
int GlobalGrid<MeshType>::totalNumBlock() const
{
    int comm_size;
    MPI_Comm_size( this->comm(), &comm_size );
    return comm_size;
}

//---------------------------------------------------------------------------//
// Get the id of this block in a given dimension.
template <class MeshType>
int GlobalGrid<MeshType>::dimBlockId( const int dim ) const
{
    return _cart_rank[dim];
}

//---------------------------------------------------------------------------//
// Get the id of this block.
template <class MeshType>
int GlobalGrid<MeshType>::blockId() const
{
    int comm_rank;
    MPI_Comm_rank( this->comm(), &comm_rank );
    return comm_rank;
}

//---------------------------------------------------------------------------//
// Get the MPI rank of a block with the given indices. If the rank is out
// of bounds and the boundary is not periodic, return -1 to indicate an
// invalid rank.
template <class MeshType>
int GlobalGrid<MeshType>::blockRank(
    const std::array<int, num_space_dim>& ijk ) const
{
    // Check for invalid indices. An index is invalid if it is out of bounds
    // and the dimension is not periodic. An out of bound index in a periodic
    // dimension is valid because it will wrap around to a valid index.
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        if ( !_periodic[d] && ( ijk[d] < 0 || _ranks_per_dim[d] <= ijk[d] ) )
            return -1;

    // If we have indices get their rank.
    int lr;
    MPI_Cart_rank( this->comm(), ijk.data(), &lr );
    return lr;
}

//---------------------------------------------------------------------------//
// Get the MPI rank of a block with the given indices. If the rank is out
// of bounds and the boundary is not periodic, return -1 to indicate an
// invalid rank.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
GlobalGrid<MeshType>::blockRank( const int i, const int j, const int k ) const
{
    std::array<int, 3> cr = { i, j, k };
    return blockRank( cr );
}

//---------------------------------------------------------------------------//
// Get the MPI rank of a block with the given indices. If the rank is out
// of bounds and the boundary is not periodic, return -1 to indicate an
// invalid rank.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<2 == NSD, int>
GlobalGrid<MeshType>::blockRank( const int i, const int j ) const
{
    std::array<int, 2> cr = { i, j };
    return blockRank( cr );
}

//---------------------------------------------------------------------------//
// Get the global number of cells in a given dimension.
template <class MeshType>
int GlobalGrid<MeshType>::globalNumEntity( Cell, const int dim ) const
{
    return _global_mesh->globalNumCell( dim );
}

//---------------------------------------------------------------------------//
// Get the global number of nodes in a given dimension.
template <class MeshType>
int GlobalGrid<MeshType>::globalNumEntity( Node, const int dim ) const
{
    // If this dimension is periodic that last node in the dimension is
    // repeated across the periodic boundary.
    if ( _periodic[dim] )
        return globalNumEntity( Cell(), dim );
    else
        return globalNumEntity( Cell(), dim ) + 1;
}

//---------------------------------------------------------------------------//
// Get the global number of I-faces in a given dimension.
template <class MeshType>
int GlobalGrid<MeshType>::globalNumEntity( Face<Dim::I>, const int dim ) const
{
    return ( Dim::I == dim ) ? globalNumEntity( Node(), dim )
                             : globalNumEntity( Cell(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of J-faces in a given dimension.
template <class MeshType>
int GlobalGrid<MeshType>::globalNumEntity( Face<Dim::J>, const int dim ) const
{
    return ( Dim::J == dim ) ? globalNumEntity( Node(), dim )
                             : globalNumEntity( Cell(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of K-faces in a given dimension.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
GlobalGrid<MeshType>::globalNumEntity( Face<Dim::K>, const int dim ) const
{
    return ( Dim::K == dim ) ? globalNumEntity( Node(), dim )
                             : globalNumEntity( Cell(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of I-edges in a given dimension.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
GlobalGrid<MeshType>::globalNumEntity( Edge<Dim::I>, const int dim ) const
{
    return ( Dim::I == dim ) ? globalNumEntity( Cell(), dim )
                             : globalNumEntity( Node(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of J-edges in a given dimension.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
GlobalGrid<MeshType>::globalNumEntity( Edge<Dim::J>, const int dim ) const
{
    return ( Dim::J == dim ) ? globalNumEntity( Cell(), dim )
                             : globalNumEntity( Node(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of K-edges in a given dimension.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
GlobalGrid<MeshType>::globalNumEntity( Edge<Dim::K>, const int dim ) const
{
    return ( Dim::K == dim ) ? globalNumEntity( Cell(), dim )
                             : globalNumEntity( Node(), dim );
}

//---------------------------------------------------------------------------//
// Get the owned number of cells in a given dimension.
template <class MeshType>
int GlobalGrid<MeshType>::ownedNumCell( const int dim ) const
{
    return _owned_num_cell[dim];
}

//---------------------------------------------------------------------------//
// Get the global offset in a given dimension for the entity of a given
// type. This is where our block starts in the global indexing scheme.
template <class MeshType>
int GlobalGrid<MeshType>::globalOffset( const int dim ) const
{
    return _global_cell_offset[dim];
}

//---------------------------------------------------------------------------//
// Set the number of owned cells and global offset. Make sure this is
// consistent across all ranks.
template <class MeshType>
void GlobalGrid<MeshType>::setNumCellAndOffset(
    const std::array<int, num_space_dim>& num_cell,
    const std::array<int, num_space_dim>& offset )
{
    std::copy( std::begin( num_cell ), std::end( num_cell ),
               std::begin( _owned_num_cell ) );
    std::copy( std::begin( offset ), std::end( offset ),
               std::begin( _global_cell_offset ) );
}

//---------------------------------------------------------------------------//
} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_GLOBALGRID_IMPL_HPP
