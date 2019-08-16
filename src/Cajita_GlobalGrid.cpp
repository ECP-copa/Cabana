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

#include <Cajita_GlobalGrid.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Constructor.
GlobalGrid::GlobalGrid( MPI_Comm comm, const std::shared_ptr<Domain> &domain,
                        const Partitioner &partitioner, const double cell_size )
    : _domain( domain )
    , _cell_size( cell_size )
{
    // Compute how many cells are in each dimension.
    _global_num_cell.resize( 3 );
    for ( int d = 0; d < 3; ++d )
        _global_num_cell[d] = std::rint( ( _domain->extent( d ) ) / cell_size );

    // Check the cell size.
    for ( int d = 0; d < 3; ++d )
        if ( std::abs( _global_num_cell[d] * cell_size -
                       _domain->extent( d ) ) >
             10.0 * std::numeric_limits<double>::epsilon() )
            throw std::invalid_argument(
                "Dimension not divisible by cell size" );

    // Partition the problem.
    _ranks_per_dim = partitioner.ranksPerDimension( comm, _global_num_cell );

    // Extract the periodicity of the boundary as integers.
    std::vector<int> periodic_dims = {_domain->isPeriodic( Dim::I ),
                                      _domain->isPeriodic( Dim::J ),
                                      _domain->isPeriodic( Dim::K )};

    // Generate a communicator with a Cartesian topology.
    int reorder_cart_ranks = 1;
    MPI_Cart_create( comm, 3, _ranks_per_dim.data(), periodic_dims.data(),
                     reorder_cart_ranks, &_cart_comm );

    // Get the Cartesian topology index of this rank.
    int linear_rank;
    MPI_Comm_rank( _cart_comm, &linear_rank );
    _cart_rank.resize( 3 );
    MPI_Cart_coords( _cart_comm, linear_rank, 3, _cart_rank.data() );

    // Get the cells per dimension and the remainder.
    std::vector<int> cells_per_dim( 3 );
    std::vector<int> dim_remainder( 3 );
    for ( int d = 0; d < 3; ++d )
    {
        cells_per_dim[d] = _global_num_cell[d] / _ranks_per_dim[d];
        dim_remainder[d] = _global_num_cell[d] % _ranks_per_dim[d];
    }

    // Compute the global cell offset and the local low corner on this rank by
    // computing the starting global cell index via exclusive scan.
    _global_cell_offset.assign( 3, 0 );
    for ( int d = 0; d < 3; ++d )
    {
        for ( int r = 0; r < _cart_rank[d]; ++r )
        {
            _global_cell_offset[d] += cells_per_dim[d];
            if ( dim_remainder[d] > r )
                ++_global_cell_offset[d];
        }
    }

    // Compute the number of local cells in this rank in each dimension.
    _owned_num_cell.resize( 3 );
    for ( int d = 0; d < 3; ++d )
    {
        _owned_num_cell[d] = cells_per_dim[d];
        if ( dim_remainder[d] > _cart_rank[d] )
            ++_owned_num_cell[d];
    }
}

//---------------------------------------------------------------------------//
// Get the grid communicator.
MPI_Comm GlobalGrid::comm() const { return _cart_comm; }

//---------------------------------------------------------------------------//
// Get the domain of the grid.
const Domain &GlobalGrid::domain() const { return *_domain; }

//---------------------------------------------------------------------------//
// Get the number of blocks in each dimension in the global mesh.
int GlobalGrid::dimNumBlock( const int dim ) const
{
    return _ranks_per_dim[dim];
}

//---------------------------------------------------------------------------//
// Get the total number of blocks.
int GlobalGrid::totalNumBlock() const
{
    int comm_size;
    MPI_Comm_size( _cart_comm, &comm_size );
    return comm_size;
}

//---------------------------------------------------------------------------//
// Get the id of this block in a given dimension.
int GlobalGrid::dimBlockId( const int dim ) const { return _cart_rank[dim]; }

//---------------------------------------------------------------------------//
// Get the id of this block.
int GlobalGrid::blockId() const
{
    int comm_rank;
    MPI_Comm_rank( _cart_comm, &comm_rank );
    return comm_rank;
}

//---------------------------------------------------------------------------//
// Get the MPI rank of a block with the given indices. If the rank is out
// of bounds and the boundary is not periodic, return -1 to indicate an
// invalid rank.
int GlobalGrid::blockRank( const int i, const int j, const int k ) const
{
    // Get the indices.
    std::vector<int> cr = {i, j, k};

    // Check for invalid indices. An index is invalid if it is out of bounds
    // and the dimension is not periodic. An out of bound index in a periodic
    // dimension is valid because it will wrap around to a valid index.
    for ( int d = 0; d < 3; ++d )
        if ( !_domain->isPeriodic( d ) &&
             ( cr[d] < 0 || _ranks_per_dim[d] <= cr[d] ) )
            return -1;

    // If we have indices get their rank.
    int lr;
    MPI_Cart_rank( _cart_comm, cr.data(), &lr );
    return lr;
}

//---------------------------------------------------------------------------//
// Get the global number of cells in a given dimension.
template <>
int GlobalGrid::globalNumEntity( Cell, const int dim ) const
{
    return _global_num_cell[dim];
}

//---------------------------------------------------------------------------//
// Get the global number of nodes in a given dimension.
template <>
int GlobalGrid::globalNumEntity( Node, const int dim ) const
{
    // If this dimension is periodic that last node in the dimension is
    // repeated across the periodic boundary.
    if ( _domain->isPeriodic( dim ) )
        return _global_num_cell[dim];
    else
        return _global_num_cell[dim] + 1;
}

//---------------------------------------------------------------------------//
// Get the global number of I-faces in a given dimension.
template <>
int GlobalGrid::globalNumEntity( Face<Dim::I>, const int dim ) const
{
    return ( Dim::I == dim ) ? globalNumEntity( Node(), dim )
                             : globalNumEntity( Cell(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of J-faces in a given dimension.
template <>
int GlobalGrid::globalNumEntity( Face<Dim::J>, const int dim ) const
{
    return ( Dim::J == dim ) ? globalNumEntity( Node(), dim )
                             : globalNumEntity( Cell(), dim );
}

//---------------------------------------------------------------------------//
// Get the global number of K-faces in a given dimension.
template <>
int GlobalGrid::globalNumEntity( Face<Dim::K>, const int dim ) const
{
    return ( Dim::K == dim ) ? globalNumEntity( Node(), dim )
                             : globalNumEntity( Cell(), dim );
}

//---------------------------------------------------------------------------//
// Get the owned number of cells in a given dimension.
int GlobalGrid::ownedNumCell( const int dim ) const
{
    return _owned_num_cell[dim];
}

//---------------------------------------------------------------------------//
// Get the global offset in a given dimension for the entity of a given
// type. This is where our block starts in the global indexing scheme.
int GlobalGrid::globalOffset( const int dim ) const
{
    return _global_cell_offset[dim];
}

//---------------------------------------------------------------------------//
// Get the cell size.
double GlobalGrid::cellSize() const { return _cell_size; }

//---------------------------------------------------------------------------//
std::shared_ptr<GlobalGrid>
createGlobalGrid( MPI_Comm comm, const std::shared_ptr<Domain> domain,
                  const Partitioner &partitioner, const double cell_size )
{
    return std::make_shared<GlobalGrid>( comm, domain, partitioner, cell_size );
}

//---------------------------------------------------------------------------//
std::shared_ptr<GlobalGrid>
createGlobalGrid( MPI_Comm comm, const Partitioner &partitioner,
                  const std::vector<bool> &periodic,
                  const std::vector<double> &global_low_corner,
                  const std::vector<double> &global_high_corner,
                  const double cell_size )
{
    auto domain =
        createDomain( global_low_corner, global_high_corner, periodic );
    return std::make_shared<GlobalGrid>( comm, domain, partitioner, cell_size );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita
