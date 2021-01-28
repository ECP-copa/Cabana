#ifndef CAJITA_SPARSEDIMPARTITIONER_HPP
#define CAJITA_SPARSEDIMPARTITIONER_HPP

#include <Cajita_Partitioner.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <vector>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
template<typename MemorySpace>
class SparseDimPartitioner : public Partitioner
{
  public: 
    SparseDimPartitioner( MPI_Comm comm, const std::array<int, 3>& global_cells_per_dim );
    SparseDimPartitioner( const std::array<int, 3>& ranks_per_dim, 
                          const std::array<int, 3>& global_cells_per_dim );

    std::array<int, 3> ranksPerDimension(
        MPI_Comm comm ) const override;

    std::array<int,3> ownedCellsPerDimension(
        MPI_Comm cart_comm,
        const std::array<int, 3>& global_cells_per_dim ) const override;

    template <class ParticlePosViewType>
    void computeLocalWorkload(ParticlePosViewType &pos_view);

    void computeFullPredixSum();

    void optimizePartition();

    void greedyPartition();

  private:
    //! represent the rectangle partition in each dimension 
    //! with form [0, p_1, ..., p_n, cell_num], n = rank-num-in-current-dimension
    //! partition in this dimension would be [0, p_1), [p_1, p_2) ... [p_n, cellNum]
    std::array<std::vector<int>, 3> _rectangle_partition;
    //! 3d prefix sum of the workload of each cell on current 
    //! [TODO] remove to corresponding implementation?
    Kokkos::View<int***, MemorySpace> _workload_prefix_sum;
    Kokkos::View<int***, MemorySpace> _workload_per_cell;
    //! ranks per dimension
    std::array<int, 3> _ranks_per_dim;
    //! workload_threshold_coeff
    float _max_workload_coeff;
    //! workload_threshold
    int _workload_threshold;
};
} // end namespace Cajita

#endif // end CAJITA_SPARSEDIMPARTITIONER_HPP