#include "Cajita_SparseDimPartitioner.hpp"
#include <cmath>

namespace Cajita
{
//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
SparseDimPartitioner<MemorySpace, ExecSpace, CellPerTileDim>::
    SparseDimPartitioner( MPI_Comm comm, float max_workload_coeff,
                          int particle_num, int num_step_rebalance,
                          const std::array<int, 3>& global_cells_per_dim )
    : _workload_threshold(
          static_cast<int>( max_workload_coeff * particle_num ) )
    , _num_step_rebalance( num_step_rebalance )
    , _workload_prefix_sum( Kokkos::View<int***, MemorySpace>(
          "workload_prefix_sum",
          global_cells_per_dim[0] >> cell_bits_per_tile_dim,
          global_cells_per_dim[1] >> cell_bits_per_tile_dim,
          global_cells_per_dim[2] >> cell_bits_per_tile_dim ) )
    , _workload_per_tile( Kokkos::View<int***, MemorySpace>(
          "workload", global_cells_per_dim[0] >> cell_bits_per_tile_dim,
          global_cells_per_dim[1] >> cell_bits_per_tile_dim,
          global_cells_per_dim[2] >> cell_bits_per_tile_dim ) )
{
    // compute the available rank number (in each dimension)
    ranksPerDimension( comm );
}

template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
SparseDimPartitioner<MemorySpace, ExecSpace, CellPerTileDim>::
    SparseDimPartitioner( float max_workload_coeff, int particle_num,
                          int num_step_rebalance,
                          const std::array<int, 3>& ranks_per_dim,
                          const std::array<int, 3>& global_cells_per_dim )
    : _workload_threshold(
          static_cast<int>( max_workload_coeff * particle_num ) )
    , _num_step_rebalance( num_step_rebalance )
    , _workload_prefix_sum( Kokkos::View<int***, MemorySpace>(
          "workload_prefix_sum",
          global_cells_per_dim[0] >> cell_bits_per_tile_dim,
          global_cells_per_dim[1] >> cell_bits_per_tile_dim,
          global_cells_per_dim[2] >> cell_bits_per_tile_dim ) )
    , _workload_per_tile( Kokkos::View<int***, MemorySpace>(
          "workload", global_cells_per_dim[0] >> cell_bits_per_tile_dim,
          global_cells_per_dim[1] >> cell_bits_per_tile_dim,
          global_cells_per_dim[2] >> cell_bits_per_tile_dim ) )
{
    std::copy( ranks_per_dim.begin(), ranks_per_dim.end(),
               _ranks_per_dim.data() );
}

//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
std::array<int, 3>
SparseDimPartitioner<MemorySpace, ExecSpace, CellPerTileDim>::ranksPerDimension(
    MPI_Comm comm, const std::array<int, 3>& ) const
{
    std::array<int, 3> ranks_per_dim = { _ranks_per_dim[0], _ranks_per_dim[1],
                                         _ranks_per_dim[2] };
    return ranks_per_dim;
}
//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
std::array<int, 3>
SparseDimPartitioner<MemorySpace, ExecSpace, CellPerTileDim>::ranksPerDimension(
    MPI_Comm comm )
{
    int comm_size;
    MPI_Comm_size( comm, &comm_size );
    MPI_Dims_create( comm_size, 3, _ranks_per_dim.data() );

    std::array<int, 3> ranks_per_dim = { _ranks_per_dim[0], _ranks_per_dim[1],
                                         _ranks_per_dim[2] };
    return ranks_per_dim;
}

//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
std::array<int, 3>
SparseDimPartitioner<MemorySpace, ExecSpace, CellPerTileDim>::
    ownedTilesPerDimension( MPI_Comm cart_comm ) const
{
    // Get the Cartesian topology index of this rank.
    std::array<int, 3> cart_rank;
    int linear_rank;
    MPI_Comm_rank( cart_comm, &linear_rank );
    MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

    // Get the tiles per dimension and the remainder.
    std::array<int, 3> tiles_per_dim;
    auto rec_mirror = Kokkos::create_mirror_view( Kokkos::HostSpace(),
                                                  _rectangle_partition_dev );
    for ( int d = 0; d < 3; ++d )
        tiles_per_dim[d] =
            rec_mirror( cart_rank[d] + 1, d ) - rec_mirror( cart_rank[d], d );
    return tiles_per_dim;
}

//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
std::array<int, 3>
SparseDimPartitioner<MemorySpace, ExecSpace, CellPerTileDim>::
    ownedCellsPerDimension(
        MPI_Comm cart_comm,
        const std::array<int, 3>& global_cells_per_dim ) const
{
    auto tiles_per_dim = ownedTilesPerDimension( cart_comm );
    for ( int i = 0; i < 3; ++i )
    {
        // compute cells_per_dim from tiles_per_dim
        tiles_per_dim[i] <<= cell_bits_per_tile_dim;
    }
    return tiles_per_dim;
}

//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
void SparseDimPartitioner<MemorySpace, ExecSpace, CellPerTileDim>::
    initialize_rec_partition( std::vector<int>& rec_partition_i,
                              std::vector<int>& rec_partition_j,
                              std::vector<int>& rec_partition_k )
{
    int max_size = 0;
    for ( int d = 0; d < 3; ++d )
        max_size = max_size < _ranks_per_dim[d] ? _ranks_per_dim[d] : max_size;

    _rectangle_partition_dev = Kokkos::View<int* [3], MemorySpace>(
        "_rectangle_partition_dev", max_size );
    auto rec_mirror = Kokkos::create_mirror_view( Kokkos::HostSpace(),
                                                  _rectangle_partition_dev );
    for ( int id = 0; id < _ranks_per_dim[0]; ++id )
        rec_mirror( id, 0 ) = rec_partition_i[id];

    for ( int id = 0; id < _ranks_per_dim[1]; ++id )
        rec_mirror( id, 1 ) = rec_partition_j[id];

    for ( int id = 0; id < _ranks_per_dim[2]; ++id )
        rec_mirror( id, 2 ) = rec_partition_k[id];

    Kokkos::deep_copy( _rectangle_partition_dev, rec_mirror );
}

//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
template <class ParticlePosViewType, typename CellUnit>
void SparseDimPartitioner<MemorySpace, ExecSpace, CellPerTileDim>::
    computeLocalWorkLoad( ParticlePosViewType& view, int particle_num,
                          CellUnit dx )
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>( 0, particle_num ),
        KOKKOS_LAMBDA( const int i ) {
            int ti = static_cast<int>( view( i, 0 ) / dx - 0.5 ) >>
                     cell_bits_per_tile_dim;
            int tj = static_cast<int>( view( i, 1 ) / dx - 0.5 ) >>
                     cell_bits_per_tile_dim;
            int tz = static_cast<int>( view( i, 2 ) / dx - 0.5 ) >>
                     cell_bits_per_tile_dim;
            Kokkos::atomic_increment( &_workload_per_tile( ti, tj, tz ) );
        } );
}

//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
template <class SparseMapType>
void SparseDimPartitioner<MemorySpace, ExecSpace, CellPerTileDim>::
    computeLocalWorkLoad( SparseMapType& sparseMap )
{
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>( 0, sparseMap.capacity() ),
        KOKKOS_LAMBDA( uint32_t i ) {
            if ( sparseMap.valid_at( i ) )
            {
                auto key = sparseMap.key_at( i );
                int ti, tj, tk;
                sparseMap.key2ijk( key, ti, tj, tk );
                Kokkos::atomic_increment( &_workload_per_tile( ti, tj, tk ) );
            }
        } );
}

//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
void SparseDimPartitioner<MemorySpace, ExecSpace,
                          CellPerTileDim>::computeFullPrefixSum( MPI_Comm comm )
{
    // all reduce the _workload_per_cell => _workloard_prefix_sum
    int total_size = _workload_per_tile.extend_0() *
                     _workload_per_tile.extend_1() *
                     _workload_per_tile.extend_2();
    MPI_Allreduce( _workload_per_tile, _workload_prefix_sum, total_size,
                   MPI_INT, MPI_SUM, comm );

    // compute the prefix sum
    for ( int j = 0; j < _workload_prefix_sum.extent( 1 ); ++j )
        for ( int k = 0; k < _workload_prefix_sum.extent( 2 ); ++k )
            Kokkos::parallel_scan(
                _workload_prefix_sum.extent( 0 ),
                KOKKOS_LAMBDA( const int i, int& update, const bool final ) {
                    const float val_i = _workload_prefix_sum( i, j, k );
                    update += val_i;
                    if ( final )
                    {
                        _workload_prefix_sum( i, j, k ) = update;
                    }
                } );

    for ( int i = 0; i < _workload_prefix_sum.extent( 0 ); ++i )
        for ( int k = 0; k < _workload_prefix_sum.extent( 2 ); ++k )
            Kokkos::parallel_scan(
                _workload_prefix_sum.extent( 1 ),
                KOKKOS_LAMBDA( const int j, int& update, const bool final ) {
                    const float val_i = _workload_prefix_sum( i, j, k );
                    update += val_i;
                    if ( final )
                    {
                        _workload_prefix_sum( i, j, k ) = update;
                    }
                } );

    for ( int i = 0; i < _workload_prefix_sum.extent( 0 ); ++i )
        for ( int j = 0; j < _workload_prefix_sum.extent( 1 ); ++j )
            Kokkos::parallel_scan(
                _workload_prefix_sum.extent( 2 ),
                KOKKOS_LAMBDA( const int k, int& update, const bool final ) {
                    const float val_i = _workload_prefix_sum( i, j, k );
                    update += val_i;
                    if ( final )
                    {
                        _workload_prefix_sum( i, j, k ) = update;
                    }
                } );
}

//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
KOKKOS_INLINE_FUNCTION int
SparseDimPartitioner<MemorySpace, ExecSpace,
                     CellPerTileDim>::compute_sub_workload( int dim_j, int j,
                                                            int dim_k, int k )
{
    Kokkos::Array<int, 3> end, start;
    int dim_i = 0;
    while ( dim_i == dim_j || dim_i == dim_k )
    {
        dim_i = ( dim_i + 1 ) % 3;
    }
    end[dim_i] = _ranks_per_dim[dim_i];
    end[dim_j] = _rectangle_partition_dev( j, dim_j );
    end[dim_k] = _rectangle_partition_dev( k, dim_k );

    start[dim_i] = _ranks_per_dim[dim_i];
    start[dim_j] = ( j > 0 ) ? _rectangle_partition_dev( j - 1, dim_j ) : 0;
    start[dim_k] = ( k > 0 ) ? _rectangle_partition_dev( k - 1, dim_k ) : 0;

    // S[i][j][k] = S[i-1][j][k] + S[i][j-1][k] + S[i][j][k-1] - S[i-1][j-1][k]
    // - S[i][j-1][k-1] - S[i-1][j][k-1] + S[i-1][j-1][k-1] + a[i][j][k]

    return _workload_prefix_sum( end[0], end[1], end[2] )     // S[i][j][k]
           - _workload_prefix_sum( start[0], end[1], end[2] ) // S[i-1][j][k]
           - _workload_prefix_sum( end[0], start[1], end[2] ) // S[i][j-1][k]
           - _workload_prefix_sum( end[0], end[1], start[2] ) // S[i][j][k-1]
           + _workload_prefix_sum( start[0], start[1],
                                   end[2] ) // S[i-1][j-1][k]
           + _workload_prefix_sum( end[0], start[1],
                                   start[2] ) // S[i][j-1][k-1]
           + _workload_prefix_sum( start[0], end[1],
                                   start[2] ) // S[i-1][j][k-1]
           - _workload_prefix_sum( start[0], start[1],
                                   start[2] ); // S[i-1][j-1][k-1]
}

template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
KOKKOS_INLINE_FUNCTION int
SparseDimPartitioner<MemorySpace, ExecSpace,
                     CellPerTileDim>::compute_sub_workload( int dim_i, int i,
                                                            int dim_j, int j,
                                                            int dim_k, int k )
{
    Kokkos::Array<int, 3> end, start;
    end[dim_i] = _rectangle_partition_dev( i, dim_i );
    end[dim_j] = _rectangle_partition_dev( j, dim_j );
    end[dim_k] = _rectangle_partition_dev( k, dim_k );

    start[dim_i] = ( i > 0 ) ? _rectangle_partition_dev( i - 1, dim_i ) : 0;
    start[dim_j] = ( j > 0 ) ? _rectangle_partition_dev( j - 1, dim_j ) : 0;
    start[dim_k] = ( k > 0 ) ? _rectangle_partition_dev( k - 1, dim_k ) : 0;

    // S[i][j][k] = S[i-1][j][k] + S[i][j-1][k] + S[i][j][k-1] - S[i-1][j-1][k]
    // - S[i][j-1][k-1] - S[i-1][j][k-1] + S[i-1][j-1][k-1] + a[i][j][k]

    return _workload_prefix_sum( end[0], end[1], end[2] )     // S[i][j][k]
           - _workload_prefix_sum( start[0], end[1], end[2] ) // S[i-1][j][k]
           - _workload_prefix_sum( end[0], start[1], end[2] ) // S[i][j-1][k]
           - _workload_prefix_sum( end[0], end[1], start[2] ) // S[i][j][k-1]
           + _workload_prefix_sum( start[0], start[1],
                                   end[2] ) // S[i-1][j-1][k]
           + _workload_prefix_sum( end[0], start[1],
                                   start[2] ) // S[i][j-1][k-1]
           + _workload_prefix_sum( start[0], end[1],
                                   start[2] ) // S[i-1][j][k-1]
           - _workload_prefix_sum( start[0], start[1],
                                   start[2] ); // S[i-1][j-1][k-1]
}

//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
void SparseDimPartitioner<MemorySpace, ExecSpace,
                          CellPerTileDim>::optimizePartition()
{
    for ( int di = 0; di < 3; ++di )
    {
        auto& rank = _ranks_per_dim[di];

        int dj = ( di + 1 ) % 3;
        int dk = ( di + 2 ) % 3;

        Kokkos::View<int*, MemorySpace> ave_workload(
            "ave_workload", _ranks_per_dim[dj] * _ranks_per_dim[dk] );

        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecSpace>( 0, _ranks_per_dim[dj] *
                                                   _ranks_per_dim[dk] ),
            KOKKOS_LAMBDA( uint32_t jnk ) {
                int j = static_cast<int>( jnk / _ranks_per_dim[dk] );
                int k = static_cast<int>( jnk % _ranks_per_dim[dk] );
                ave_workload( jnk ) =
                    compute_sub_workload( dj, j, dk, k ) / rank;
            } );

        int sum_ave_workload = 0;
        Kokkos::parallel_reduce(
            "Reduction", _ranks_per_dim[dj] * _ranks_per_dim[dk],
            KOKKOS_LAMBDA( const int idx, int& update ) {
                update += ave_workload( idx );
            },
            sum_ave_workload );

        int point_i = 1;
        Kokkos::View<int*, MemorySpace> current_workload(
            "current_workload", _ranks_per_dim[dj] * _ranks_per_dim[dk] );
        for ( int current_rank = 0; current_rank < rank - 1; current_rank++ )
        {
            int last_diff = __INT_MAX__;
            while ( true )
            {
                Kokkos::parallel_for(
                    Kokkos::RangePolicy<ExecSpace>( 0, _ranks_per_dim[dj] *
                                                           _ranks_per_dim[dk] ),
                    KOKKOS_LAMBDA( uint32_t jnk ) {
                        int j = static_cast<int>( jnk / _ranks_per_dim[dk] );
                        int k = static_cast<int>( jnk % _ranks_per_dim[dk] );
                        current_workload( jnk ) =
                            compute_sub_workload( di, point_i, dj, j, dk, k ) /
                            rank;
                    } );
                int sum_current_workload;
                Kokkos::parallel_reduce(
                    "Reduction", _ranks_per_dim[dj] * _ranks_per_dim[dk],
                    KOKKOS_LAMBDA( const int idx, int& update ) {
                        update += current_workload( idx );
                    },
                    sum_current_workload );
                auto diff = std::abs( sum_current_workload - sum_ave_workload );
                if ( diff < last_diff )
                {
                    last_diff = diff;
                    point_i++;
                }
                else
                {
                    _rectangle_partition_dev( current_rank, di ) = point_i - 1;
                    break;
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
// void SparseDimPartitioner::greedyPartition()
// {
//     // pass
// }

//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim>
bool SparseDimPartitioner<MemorySpace, ExecSpace,
                          CellPerTileDim>::adaptive_load_balance()
{
    // pass
}

} // end namespace Cajita