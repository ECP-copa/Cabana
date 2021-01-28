#include <Cajita_SparseDimPartitioner.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
SparseDimPartitioner::SparseDimPartitioner( MPI_Comm comm )
{ 
  // compute the available rank number (in each dimension)
  ranksPerDimension(comm);
  // init partitioner

}

SparseDimPartitioner::SparseDimPartitioner( const std::array<int, 4>& ranks_per_dim )
    : _ranks_per_dim( ranks_per_dim )
{ 
  // init partitioner
}

//---------------------------------------------------------------------------//
std::array<int, 3>
SparseDimPartitioner::ranksPerDimension( MPI_Comm comm) const
{
    int comm_size;
    MPI_Comm_size( comm, &comm_size );
    MPI_Dims_create( comm_size, 3, _ranks_per_dim.data() );

    return _ranks_per_dim;
}

//---------------------------------------------------------------------------//
std::array<int,3>
UniformDimPartitioner::ownedCellsPerDimension(
    MPI_Comm cart_comm,
    const std::array<int, 3>& ) const
{
    // Get the Cartesian topology index of this rank.
    std::array<int,3> cart_rank;
    int linear_rank;
    MPI_Comm_rank( cart_comm, &linear_rank );
    MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

    // Get the cells per dimension and the remainder.
    std::array<int, 3> cells_per_dim;
    for ( int d = 0; d < 3; ++d )
      cells_per_dim[d] = _rectangle_partition[d][cart_rank[d]+1] - _rectangle_partition[d][cart_rank[d]];

    return cells_per_dim;
}

//---------------------------------------------------------------------------//

} // end namespace Cajita