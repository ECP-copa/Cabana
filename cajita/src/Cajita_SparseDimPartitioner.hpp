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
template <typename MemorySpace, unsigned long long CellPerTileDim = 4>
class SparseDimPartitioner : public Partitioner
{
  public:
    //! Number of bits (per dimension) needed to index the cells inside a tile
    static constexpr unsigned long long cell_bits_per_tile_dim =
        bitCount( CellPerTileDim );
    //! Number of cells inside each tile (per dimension), tile size reset to
    //! power of 2
    static constexpr unsigned long long cell_num_per_tile_dim =
        1 << cell_bits_per_tile_dim;

    SparseDimPartitioner( MPI_Comm comm, float max_workload_coeff,
                          int particle_num, int num_step_rebalance,
                          const std::array<int, 3>& global_cells_per_dim );
    SparseDimPartitioner( float max_workload_coeff, int particle_num,
                          int num_step_rebalance,
                          const std::array<int, 3>& ranks_per_dim,
                          const std::array<int, 3>& global_cells_per_dim );

    std::array<int, 3> ranksPerDimension( MPI_Comm comm ) const override;
    std::array<int, 3> ranksPerDimension() const override;

    std::array<int, 3>
    ownedTilesPerDimension( MPI_Comm cart_comm,
                            const std::array<int, 3>& global_cells_per_dim );
    std::array<int, 3> ownedCellsPerDimension(
        MPI_Comm cart_comm,
        const std::array<int, 3>& global_cells_per_dim ) const override;

    void initialize_rec_partition( std::vector<int>& rec_partition_i,
                                   std::vector<int>& rec_partition_j,
                                   std::vector<int>& rec_partition_k );

    // sparse_map (where there are particles), Kokkos_array weight - tile =>
    // workload cell > workload > partition > particle init template <class
    // ParticlePosViewType> void initialize_rec_partition( ParticlePosViewType&
    // pos_view );

    // to compute the tileweight, assume tile_weight = 1 at the first place
    template <template ExecSpace, class ParticlePosViewType, template CellUnit>
    void computeLocalWorkLoad( ParticlePosViewType& view, int particle_num,
                               CellUnit dx );
    template <template ExecSpace, class TileWeightViewType>
    void computeLocalWorkload( TileWeightViewType& tile_weight,
                               SparseMapType& sparseMap );

    template <typename MemorySpace>
    void computeFullPrefixSum( MPI_Comm comm );

    template <typename MemorySpace, typename ExecSpace>
    void optimiazePartition();

    // void greedyPartition();

    bool adaptive_load_balance();

  private:
    // ! workload_threshold_coeff
    // float _max_workload_coeff;
    //! workload_threshold
    int _workload_threshold;
    //! default check point for re-balance
    int _num_step_rebalance;
    //! represent the rectangle partition in each dimension
    //! with form [0, p_1, ..., p_n, cell_num], n =
    //! rank-num-in-current-dimension partition in this dimension would be [0,
    //! p_1), [p_1, p_2) ... [p_n, cellNum] (unit: tile)
    Kokkos::View<int*[3], MemorySpace> _rectangle_partition_dev;
    //! 3d prefix sum of the workload of each cell on current
    // current pre-set size: global_tile_per_dim * global_tile_per_dim*
    // global_tile_per_dim
    Kokkos::View<int***, MemorySpace> _workload_prefix_sum;
    // current pre-set size: global_tile_per_dim * global_tile_per_dim*
    // global_tile_per_dim
    Kokkos::View<int***, MemorySpace> _workload_per_tile;
    // std::array<Kokkos::View<int***, MemorySpace>> _workload_buffer;
    //! ranks per dimension
    Kokkos::Array<int, 3> _ranks_per_dim;
};
} // end namespace Cajita

#endif // end CAJITA_SPARSEDIMPARTITIONER_HPP