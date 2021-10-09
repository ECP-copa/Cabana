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

#ifndef CAJITA_GLOBALGRID_IMPL_NEW_HPP
#define CAJITA_GLOBALGRID_IMPL_NEW_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
//--------------------------- GlobalGridBase --------------------------------//
// Constructor.
template <class MeshType>
GlobalGridBase<MeshType>::GlobalGridBase(
    MPI_Comm comm, const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
    const std::array<bool, num_space_dim>& periodic,
    const std::shared_ptr<BlockPartitioner<num_space_dim>>& partitioner )
    : _global_mesh( global_mesh )
    , _periodic( periodic )
{
    // Partition the problem.
    std::array<int, num_space_dim> global_num_cell;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        global_num_cell[d] = _global_mesh->globalNumCell( d );
    _ranks_per_dim = partitioner->ranksPerDimension( comm, global_num_cell );

    // Extract the periodicity of the boundary as integers.
    std::array<int, num_space_dim> periodic_dims;
    for ( std::size_t d = 0; d < num_space_dim; ++d )
        periodic_dims[d] = _periodic[d];

    // Generate a communicator with a Cartesian topology.
    int reorder_cart_ranks = 1;
    MPI_Cart_create( comm, num_space_dim, _ranks_per_dim.data(),
                     periodic_dims.data(), reorder_cart_ranks, &_cart_comm );

    // Get the Cartesian topology index of this rank.
    int linear_rank;
    MPI_Comm_rank( _cart_comm, &linear_rank );
    MPI_Cart_coords( _cart_comm, linear_rank, num_space_dim,
                     _cart_rank.data() );

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
GlobalGridBase<MeshType>::~GlobalGridBase()
{
    MPI_Comm_free( &_cart_comm );
}

//---------------------------------------------------------------------------//
// Get the grid communicator.
template <class MeshType>
MPI_Comm GlobalGridBase<MeshType>::comm() const
{
    return _cart_comm;
}

//---------------------------------------------------------------------------//
// Get the global mesh data.
template <class MeshType>
const GlobalMesh<MeshType>& GlobalGridBase<MeshType>::globalMesh() const
{
    return *_global_mesh;
}

//---------------------------------------------------------------------------//
// Get whether a given dimension is periodic.
template <class MeshType>
bool GlobalGridBase<MeshType>::isPeriodic( const int dim ) const
{
    return _periodic[dim];
}

//---------------------------------------------------------------------------//
// Determine if this block is on a low boundary in the given dimension.
template <class MeshType>
bool GlobalGridBase<MeshType>::onLowBoundary( const int dim ) const
{
    return _boundary_lo[dim];
}

//---------------------------------------------------------------------------//
// Determine if this block is on a high boundary in the given dimension.
template <class MeshType>
bool GlobalGridBase<MeshType>::onHighBoundary( const int dim ) const
{
    return _boundary_hi[dim];
}

//---------------------------------------------------------------------------//
// Get the number of blocks in each dimension in the global mesh.
template <class MeshType>
int GlobalGridBase<MeshType>::dimNumBlock( const int dim ) const
{
    return _ranks_per_dim[dim];
}

//---------------------------------------------------------------------------//
// Get the total number of blocks.
template <class MeshType>
int GlobalGridBase<MeshType>::totalNumBlock() const
{
    int comm_size;
    MPI_Comm_size( _cart_comm, &comm_size );
    return comm_size;
}

//---------------------------------------------------------------------------//
// Get the id of this block in a given dimension.
template <class MeshType>
int GlobalGridBase<MeshType>::dimBlockId( const int dim ) const
{
    return _cart_rank[dim];
}

//---------------------------------------------------------------------------//
// Get the id of this block.
template <class MeshType>
int GlobalGridBase<MeshType>::blockId() const
{
    int comm_rank;
    MPI_Comm_rank( _cart_comm, &comm_rank );
    return comm_rank;
}

//---------------------------------------------------------------------------//
// Get the MPI rank of a block with the given indices. If the rank is out
// of bounds and the boundary is not periodic, return -1 to indicate an
// invalid rank.
template <class MeshType>
int GlobalGridBase<MeshType>::blockRank(
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
    MPI_Cart_rank( _cart_comm, ijk.data(), &lr );
    return lr;
}

//---------------------------------------------------------------------------//
// Get the MPI rank of a block with the given indices. If the rank is out
// of bounds and the boundary is not periodic, return -1 to indicate an
// invalid rank.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
GlobalGridBase<MeshType>::blockRank( const int i, const int j,
                                     const int k ) const
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
GlobalGridBase<MeshType>::blockRank( const int i, const int j ) const
{
    std::array<int, 2> cr = { i, j };
    return blockRank( cr );
}

//---------------------------------------------------------------------------//
// Get the global number of cells in a given dimension.
template <class MeshType>
int GlobalGridBase<MeshType>::globalNumEntity( Cell, const int dim ) const
{
    return _global_mesh->globalNumCell( dim );
}

//---------------------------------------------------------------------------//
// Get the global number of nodes in a given dimension.
template <class MeshType>
int GlobalGridBase<MeshType>::globalNumEntity( Node, const int dim ) const
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
int GlobalGridBase<MeshType>::globalNumEntity( Face<Dim::I>,
                                               const int dim ) const
{
    return ( Dim::I == dim ) ? globalNumEntity( Node(), dim )
                             : globalNumEntity( Cell(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of J-faces in a given dimension.
template <class MeshType>
int GlobalGridBase<MeshType>::globalNumEntity( Face<Dim::J>,
                                               const int dim ) const
{
    return ( Dim::J == dim ) ? globalNumEntity( Node(), dim )
                             : globalNumEntity( Cell(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of K-faces in a given dimension.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
GlobalGridBase<MeshType>::globalNumEntity( Face<Dim::K>, const int dim ) const
{
    return ( Dim::K == dim ) ? globalNumEntity( Node(), dim )
                             : globalNumEntity( Cell(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of I-edges in a given dimension.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
GlobalGridBase<MeshType>::globalNumEntity( Edge<Dim::I>, const int dim ) const
{
    return ( Dim::I == dim ) ? globalNumEntity( Cell(), dim )
                             : globalNumEntity( Node(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of J-edges in a given dimension.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
GlobalGridBase<MeshType>::globalNumEntity( Edge<Dim::J>, const int dim ) const
{
    return ( Dim::J == dim ) ? globalNumEntity( Cell(), dim )
                             : globalNumEntity( Node(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of K-edges in a given dimension.
template <class MeshType>
template <std::size_t NSD>
std::enable_if_t<3 == NSD, int>
GlobalGridBase<MeshType>::globalNumEntity( Edge<Dim::K>, const int dim ) const
{
    return ( Dim::K == dim ) ? globalNumEntity( Cell(), dim )
                             : globalNumEntity( Node(), dim );
}

//---------------------------------------------------------------------------//
//-------------------------------- GlobalGrid -------------------------------//
// Constructor.
template <class MeshType>
GlobalGrid<MeshType>::GlobalGrid(
    MPI_Comm comm, const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
    const std::array<bool, num_space_dim>& periodic,
    std::shared_ptr<BlockPartitioner<num_space_dim>>& partitioner )
    : _GlobalGridBase( comm, global_mesh, periodic, partitioner )
    , _partitioner( partitioner )
{
    // // Get the cells per dimension and the remainder.
    // std::array<int, num_space_dim> cells_per_dim;
    // std::array<int, num_space_dim> dim_remainder;
    // for ( std::size_t d = 0; d < num_space_dim; ++d )
    // {
    //     cells_per_dim[d] = global_num_cell[d] / _ranks_per_dim[d];
    //     dim_remainder[d] = global_num_cell[d] % _ranks_per_dim[d];
    // }

    // // Compute the global cell offset and the local low corner on this rank by
    // // computing the starting global cell index via exclusive scan.
    // for ( std::size_t d = 0; d < num_space_dim; ++d )
    // {
    //     _global_cell_offset[d] = 0;
    //     for ( int r = 0; r < _cart_rank[d]; ++r )
    //     {
    //         _global_cell_offset[d] += cells_per_dim[d];
    //         if ( dim_remainder[d] > r )
    //             ++_global_cell_offset[d];
    //     }
    // }

    // // Compute the number of local cells in this rank in each dimension.
    // for ( std::size_t d = 0; d < num_space_dim; ++d )
    // {
    //     _owned_num_cell[d] = cells_per_dim[d];
    //     if ( dim_remainder[d] > _cart_rank[d] )
    //         ++_owned_num_cell[d];
    // }
}

} // end namespace Cajita

#endif // !CAJITA_GLOBAL_GRID_IMPL_NEW_HPP