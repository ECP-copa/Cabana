#include <Cajita_SparseDimPartitioner.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
template <typename MemorySpace>
SparseDimPartitioner::SparseDimPartitioner( MPI_Comm comm, 
                                            const std::array<int, 3>& global_cells_per_dim )
    : _workload_prefix_sum {Kokkos::View<int***, MemorySpace>( "rectangle_partition", 
                                          global_cells_per_dim[0], 
                                          global_cells_per_dim[1], 
                                          global_cells_per_dim[2])},
      _workload_per_cell {Kokkos::View<>( "workload", 
                                         global_cells_per_dim[0], 
                                         global_cells_per_dim[1], 
                                         global_cells_per_dim[2])}
{ 
  // compute the available rank number (in each dimension)
  ranksPerDimension(comm);
  // init partitioner

}

template <typename MemorySpace>
SparseDimPartitioner::SparseDimPartitioner( const std::array<int, 4>& ranks_per_dim, 
                                            const std::array<int, 3>& global_cells_per_dim )
    : _ranks_per_dim( ranks_per_dim ),
      _workload_prefix_sum {Kokkos::View( "rectangle_partition", 
                                          global_cells_per_dim[0], 
                                          global_cells_per_dim[1], 
                                          global_cells_per_dim[2])},
      _workload_per_cell {Kokkos::View( "workload", 
                                         global_cells_per_dim[0], 
                                         global_cells_per_dim[1], 
                                         global_cells_per_dim[2])}
{ 
  // init partitioner
}

//---------------------------------------------------------------------------//
std::array<int, 3>
SparseDimPartitioner::ranksPerDimension( MPI_Comm comm ) const
{
    int comm_size;
    MPI_Comm_size( comm, &comm_size );
    MPI_Dims_create( comm_size, 3, _ranks_per_dim.data() );

    return _ranks_per_dim;
}

//---------------------------------------------------------------------------//
std::array<int,3>
SparseDimPartitioner::ownedCellsPerDimension(
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
template <template ExecSpace, class ParticlePosViewType, template CellUnit>
void SparseDimPartitioner::computeLocalWorkLoad(ParticlePosViewType &view, int particle_num, CellUnit dx);
{
    Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace>(0, particle_num),
      KOKKOS_LAMBDA(cosnt int i){
        int cell_i = static_cast<int>(view(i, 0) / dx - 0.5);
        int cell_j = static_cast<int>(view(i, 1) / dx - 0.5);
        int cell_k = static_cast<int>(view(i, 2) / dx - 0.5);
        atomic_increment(&_workload_per_cell(cell_i, cell_j, cell_k)); 
      });
}

} // end namespace Cajita