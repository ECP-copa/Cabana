#ifndef CAJITA_SPARSEDIMPARTITIONER_HPP
#define CAJITA_SPARSEDIMPARTITIONER_HPP

#include <Cajita_Partitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <vector>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
template <typename MemorySpace, typename ExecSpace,
          unsigned long long CellPerTileDim = 4>
class SparseDimPartitioner : public BlockPartitioner<3>
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
                          int max_optimize_iteration,
                          const std::array<int, 3>& global_cells_per_dim )
        : _workload_threshold(
              static_cast<int>( max_workload_coeff * particle_num ) )
        , _num_step_rebalance( num_step_rebalance )
        , _max_optimize_iteration( max_optimize_iteration )
        , _workload_prefix_sum(
              "workload_prefix_sum",
              ( global_cells_per_dim[0] >> cell_bits_per_tile_dim ) + 1,
              ( global_cells_per_dim[1] >> cell_bits_per_tile_dim ) + 1,
              ( global_cells_per_dim[2] >> cell_bits_per_tile_dim ) + 1 )
        , _workload_per_tile(
              "workload",
              ( global_cells_per_dim[0] >> cell_bits_per_tile_dim ) + 1,
              ( global_cells_per_dim[1] >> cell_bits_per_tile_dim ) + 1,
              ( global_cells_per_dim[2] >> cell_bits_per_tile_dim ) + 1 )
    {
        // _workload_prefix_sum = Kokkos::View<int***, MemorySpace>(
        //     "workload_prefix_sum",
        //     ( global_cells_per_dim[0] >> cell_bits_per_tile_dim ) + 1,
        //     ( global_cells_per_dim[1] >> cell_bits_per_tile_dim ) + 1,
        //     ( global_cells_per_dim[2] >> cell_bits_per_tile_dim ) + 1 );
        // _workload_per_tile = Kokkos::View<int***, MemorySpace>(
        //     "workload",
        //     ( global_cells_per_dim[0] >> cell_bits_per_tile_dim ) + 1,
        //     ( global_cells_per_dim[1] >> cell_bits_per_tile_dim ) + 1,
        //     ( global_cells_per_dim[2] >> cell_bits_per_tile_dim ) + 1 );
        // compute the available rank number( in each dimension )
        ranksPerDimension( comm );
    }

    SparseDimPartitioner( float max_workload_coeff, int particle_num,
                          int num_step_rebalance, int max_optimize_iteration,
                          const std::array<int, 3>& ranks_per_dim,
                          const std::array<int, 3>& global_cells_per_dim )
        : _workload_threshold(
              static_cast<int>( max_workload_coeff * particle_num ) )
        , _num_step_rebalance( num_step_rebalance )
        , _max_optimize_iteration( max_optimize_iteration )
        , _workload_prefix_sum(
              "workload_prefix_sum",
              ( global_cells_per_dim[0] >> cell_bits_per_tile_dim ) + 1,
              ( global_cells_per_dim[1] >> cell_bits_per_tile_dim ) + 1,
              ( global_cells_per_dim[2] >> cell_bits_per_tile_dim ) + 1 )
        , _workload_per_tile(
              "workload",
              ( global_cells_per_dim[0] >> cell_bits_per_tile_dim ) + 1,
              ( global_cells_per_dim[1] >> cell_bits_per_tile_dim ) + 1,
              ( global_cells_per_dim[2] >> cell_bits_per_tile_dim ) + 1 )
    {
        // _workload_prefix_sum = Kokkos::View<int***, MemorySpace>(
        //     "workload_prefix_sum",
        //     ( global_cells_per_dim[0] >> cell_bits_per_tile_dim ) + 1,
        //     ( global_cells_per_dim[1] >> cell_bits_per_tile_dim ) + 1,
        //     ( global_cells_per_dim[2] >> cell_bits_per_tile_dim ) + 1 );
        // _workload_per_tile = Kokkos::View<int***, MemorySpace>(
        //     "workload",
        //     ( global_cells_per_dim[0] >> cell_bits_per_tile_dim ) + 1,
        //     ( global_cells_per_dim[1] >> cell_bits_per_tile_dim ) + 1,
        //     ( global_cells_per_dim[2] >> cell_bits_per_tile_dim ) + 1 );
        std::copy( ranks_per_dim.begin(), ranks_per_dim.end(),
                   _ranks_per_dim.data() );
    }

    // [TODO arguments in the virtual func]
    std::array<int, 3> ranksPerDimension( MPI_Comm comm )
    {
        int comm_size;
        MPI_Comm_size( comm, &comm_size );

        std::array<int, 3> ranks_per_dim;
        for ( std::size_t d = 0; d < 3; ++d )
            ranks_per_dim[d] = 0;
        MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

        std::copy( ranks_per_dim.begin(), ranks_per_dim.end(),
                   _ranks_per_dim.data() );

        return ranks_per_dim;
    }

    std::array<int, 3> ranksPerDimension(
        MPI_Comm comm,
        const std::array<int, 3>& global_cells_per_dim ) const override
    {
        std::array<int, 3> ranks_per_dim = {
            _ranks_per_dim[0], _ranks_per_dim[1], _ranks_per_dim[2] };
        return ranks_per_dim;
    }

    std::array<int, 3> ownedTilesPerDimension( MPI_Comm cart_comm ) const
    {
        // Get the Cartesian topology index of this rank.
        std::array<int, 3> cart_rank;
        int linear_rank;
        MPI_Comm_rank( cart_comm, &linear_rank );
        MPI_Cart_coords( cart_comm, linear_rank, 3, cart_rank.data() );

        // Get the tiles per dimension and the remainder.
        std::array<int, 3> tiles_per_dim;
        auto rec_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _rectangle_partition_dev );
        for ( int d = 0; d < 3; ++d )
            tiles_per_dim[d] = rec_mirror( cart_rank[d] + 1, d ) -
                               rec_mirror( cart_rank[d], d );
        return tiles_per_dim;
    }

    std::array<int, 3> ownedCellsPerDimension( MPI_Comm cart_comm,
                                               const std::array<int, 3>& ) const
    {
        auto tiles_per_dim = ownedTilesPerDimension( cart_comm );
        for ( int i = 0; i < 3; ++i )
        {
            // compute cells_per_dim from tiles_per_dim
            tiles_per_dim[i] <<= cell_bits_per_tile_dim * 3;
        }
        return tiles_per_dim;
    }

    void initialize_rec_partition( std::vector<int>& rec_partition_i,
                                   std::vector<int>& rec_partition_j,
                                   std::vector<int>& rec_partition_k )
    {
        int max_size = 0;
        for ( int d = 0; d < 3; ++d )
            max_size =
                max_size < _ranks_per_dim[d] ? _ranks_per_dim[d] : max_size;

        _rectangle_partition_dev = Kokkos::View<int* [3], MemorySpace>(
            "_rectangle_partition_dev", max_size + 1 );
        auto rec_mirror = Kokkos::create_mirror_view(
            Kokkos::HostSpace(), _rectangle_partition_dev );
        for ( int id = 0; id < _ranks_per_dim[0] + 1; ++id )
            rec_mirror( id, 0 ) = rec_partition_i[id];

        for ( int id = 0; id < _ranks_per_dim[1] + 1; ++id )
            rec_mirror( id, 1 ) = rec_partition_j[id];

        for ( int id = 0; id < _ranks_per_dim[2] + 1; ++id )
            rec_mirror( id, 2 ) = rec_partition_k[id];

        Kokkos::deep_copy( _rectangle_partition_dev, rec_mirror );
    }

    std::array<std::vector<int>, 3> get_current_partition()
    {
        std::array<std::vector<int>, 3> rec_part;
        auto rec_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _rectangle_partition_dev );
        for ( int d = 0; d < 3; ++d )
        {
            rec_part[d].resize( _ranks_per_dim[d] );
            for ( int id = 0; id < _ranks_per_dim[d] + 1; ++id )
            {
                rec_part[d][id] = rec_mirror( id, d );
            }
        }
        return rec_part;
    }

    // sparse_map (where there are particles), Kokkos_array weight - tile =>
    // workload cell > workload > partition > particle init template <class
    // ParticlePosViewType> void initialize_rec_partition( ParticlePosViewType&
    // pos_view );

    // to compute the tileweight, assume tile_weight = 1 at the first place
    template <class ParticlePosViewType, typename CellUnit>
    void computeLocalWorkLoad( ParticlePosViewType& view, int particle_num,
                               CellUnit dx )
    {
        auto workload = _workload_per_tile;
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecSpace>( 0, particle_num ),
            KOKKOS_LAMBDA( const int i ) {
                int ti = static_cast<int>( view( i, 0 ) / dx - 0.5 ) >>
                         cell_bits_per_tile_dim;
                int tj = static_cast<int>( view( i, 1 ) / dx - 0.5 ) >>
                         cell_bits_per_tile_dim;
                int tz = static_cast<int>( view( i, 2 ) / dx - 0.5 ) >>
                         cell_bits_per_tile_dim;
                Kokkos::atomic_increment( &workload( ti + 1, tj + 1, tz + 1 ) );
            } );
    }

    // template <HashTypes Hash = HashTypes::Naive, typename Key = uint64_t,
    //           typename Value = uint64_t>
    template <class SparseMapType>
    void computeLocalWorkLoad( const SparseMapType& sparseMap )
    // SparseMap<ExecSpace, CellPerTileDim, Hash, Key, Value>& sparseMap )
    {
        auto workload = _workload_per_tile;
        Kokkos::parallel_for(
            Kokkos::RangePolicy<ExecSpace>( 0, sparseMap.capacity() ),
            KOKKOS_LAMBDA( uint32_t i ) {
                if ( sparseMap.valid_at( i ) )
                {
                    auto key = sparseMap.key_at( i );
                    int ti, tj, tk;
                    sparseMap.key2ijk( key, ti, tj, tk );
                    Kokkos::atomic_increment(
                        &workload( ti + 1, tj + 1, tk + 1 ) );
                }
            } );
    }

    void computeFullPrefixSum( MPI_Comm comm )
    {
        // all reduce the _workload_per_cell => _workloard_prefix_sum
        int total_size = _workload_per_tile.extent( 0 ) *
                         _workload_per_tile.extent( 1 ) *
                         _workload_per_tile.extent( 2 );
        // printf( "total_size = %d, _wo_per_tile = %p, _workload_prefix_sum = "
        //         "%p, MPI_INT = %p, MPI_SUM = %p, comm = %p \n",
        //         total_size, _workload_per_tile.data(),
        //         _workload_prefix_sum.data(), MPI_INT, MPI_SUM, comm );
        auto workload = _workload_per_tile;
        auto prefix_sum = _workload_prefix_sum;
        // for test
        // Kokkos::parallel_for(
        //     Kokkos::RangePolicy<ExecSpace>( 0, 5 ),
        //     KOKKOS_LAMBDA( uint32_t i ) {
        //         for ( int j = 0; j < 5; ++j )
        //             for ( int k = 0; k < 5; ++k )
        //             {
        //                 // printf( "_workload_per_tile(%d, %d, %d) = %d, "
        //                 // "_workload_prefix_sum = %d\n",
        //                 // i, j, k, _workload_per_tile( i, j, k ),
        //                 // _workload_prefix_sum( i, j, k ) );
        //                 printf( "_workload_per_tile(%d, %d, %d) = %d, "
        //                         "_workload_prefix_sum = %d\n",
        //                         i, j, k, workload( i, j, k ),
        //                         prefix_sum( i, j, k ) );
        //             }
        //     } );
        // printf( "== print ==\n" );
        // for -test -only
        // MPI_Allreduce( _workload_per_tile.data(),
        // _workload_prefix_sum.data(),
        //    total_size, MPI_INT, MPI_SUM, comm );
        MPI_Allreduce( workload.data(), prefix_sum.data(), total_size, MPI_INT,
                       MPI_SUM, comm );
        // printf( "finish allreduce\n" );

        // for ( int i = 0; i < _workload_per_tile.extent( 0 ); ++i )
        //     for ( int j = 0; j < _workload_per_tile.extent( 1 ); ++j )
        //         for ( int k = 0; k < _workload_per_tile.extent( 2 ); ++k )
        //         {
        //             printf( "i = %d, j = %d, k = %d, _workload_per_tile =
        //             %d,"
        //                     "_workload_prefix_sum = %d\n",
        //                     i, j, k, _workload_per_tile( i, j, k ),
        //                     _workload_prefix_sum( i, j, k ) );
        //         }

        // compute the prefix sum
        for ( int j = 0;
              j < static_cast<int>( _workload_prefix_sum.extent( 1 ) ); ++j )
            for ( int k = 0;
                  k < static_cast<int>( _workload_prefix_sum.extent( 2 ) );
                  ++k )
                Kokkos::parallel_scan(
                    Kokkos::RangePolicy<ExecSpace>(
                        0, _workload_prefix_sum.extent( 0 ) ),
                    KOKKOS_LAMBDA( const int i, int& update,
                                   const bool final ) {
                        // const float val_i = _workload_prefix_sum( i, j, k );
                        const float val_i = prefix_sum( i, j, k );
                        update += val_i;
                        if ( final )
                        {
                            prefix_sum( i, j, k ) = update;
                            // _workload_prefix_sum( i, j, k ) = update;
                        }
                    } );

        for ( int i = 0;
              i < static_cast<int>( _workload_prefix_sum.extent( 0 ) ); ++i )
            for ( int k = 0;
                  k < static_cast<int>( _workload_prefix_sum.extent( 2 ) );
                  ++k )
                Kokkos::parallel_scan(
                    Kokkos::RangePolicy<ExecSpace>(
                        0, _workload_prefix_sum.extent( 1 ) ),
                    KOKKOS_LAMBDA( const int j, int& update,
                                   const bool final ) {
                        // const float val_i = _workload_prefix_sum( i, j, k );
                        const float val_i = prefix_sum( i, j, k );
                        update += val_i;
                        if ( final )
                        {
                            prefix_sum( i, j, k ) = update;
                            // _workload_prefix_sum( i, j, k ) = update;
                        }
                    } );

        for ( int i = 0;
              i < static_cast<int>( _workload_prefix_sum.extent( 0 ) ); ++i )
            for ( int j = 0;
                  j < static_cast<int>( _workload_prefix_sum.extent( 1 ) );
                  ++j )
                Kokkos::parallel_scan(
                    Kokkos::RangePolicy<ExecSpace>(
                        0, _workload_prefix_sum.extent( 2 ) ),
                    KOKKOS_LAMBDA( const int k, int& update,
                                   const bool final ) {
                        const float val_i = prefix_sum( i, j, k );
                        // const float val_i = _workload_prefix_sum( i, j, k );
                        update += val_i;
                        if ( final )
                        {
                            prefix_sum( i, j, k ) = update;
                            // _workload_prefix_sum( i, j, k ) = update;
                        }
                    } );

        // for ( int i = 0; i < _workload_per_tile.extent( 0 ); ++i )
        //     for ( int j = 0; j < _workload_per_tile.extent( 1 ); ++j )
        //         for ( int k = 0; k < _workload_per_tile.extent( 2 ); ++k )
        //         {
        //             printf( "i = %d, j = %d, k = %d, _workload_per_tile =
        //             %d,"
        //                     "_workload_prefix_sum = %d\n",
        //                     i, j, k, _workload_per_tile( i, j, k ),
        //                     _workload_prefix_sum( i, j, k ) );
        //         }
        // Kokkos::deep_copy( _workload_prefix_sum, prefix_sum );
    }

    void optimizePartition()
    {
        bool is_changed = false;
        for ( int i = 0; i < _max_optimize_iteration; ++i )
        {
            optimizePartition( is_changed );
            if ( !is_changed )
                return;
        }
    }

    void optimizePartition( bool& is_changed )
    {
        is_changed = false;
        for ( int di = 0; di < 3; ++di )
        {
            auto rank = _ranks_per_dim[di];
            auto rec_partition = _rectangle_partition_dev;
            auto rec_mirror = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), _rectangle_partition_dev );
            auto prefix_sum = _workload_prefix_sum;

            int dj = ( di + 1 ) % 3;
            int dk = ( di + 2 ) % 3;
            auto rank_j = _ranks_per_dim[dj];
            auto rank_k = _ranks_per_dim[dk];
            // compute average workload
            Kokkos::View<int*, MemorySpace> ave_workload(
                "ave_workload", _ranks_per_dim[dj] * _ranks_per_dim[dk] );
            // printf( "In optimizePartition: before parallel_for\n" );
            Kokkos::parallel_for(
                Kokkos::RangePolicy<ExecSpace>( 0, _ranks_per_dim[dj] *
                                                       _ranks_per_dim[dk] ),
                KOKKOS_LAMBDA( uint32_t jnk ) {
                    // printf(
                    //     "in parallel_for, before calling "
                    //     "compute_sub_workload to compute ave_workload - 0\n"
                    //     );
                    int j = static_cast<int>( jnk / rank_k );
                    int k = static_cast<int>( jnk % rank_k );
                    // printf(
                    //     "in parallel_for, before calling "
                    //     "compute_sub_workload to compute ave_workload - 1\n"
                    //     );
                    ave_workload( jnk ) =
                        compute_sub_workload( dj, j, dk, k, rank, rec_partition,
                                              prefix_sum ) /
                        rank;
                    // printf( "ave_workload(%d) = %d, di = %d, rank = %d\n",
                    // jnk,
                    //         ave_workload( jnk ), di, rank );
                } );
            // printf( "In optimizePartition: after parallel_for\n" );

            int point_i = 1;
            int last_point = 0;
            Kokkos::View<int*, MemorySpace> current_workload(
                "current_workload", _ranks_per_dim[dj] * _ranks_per_dim[dk] );
            for ( int current_rank = 1; current_rank < rank; current_rank++ )
            {
                int last_diff = __INT_MAX__;
                while ( true )
                {
                    // compute current workload
                    Kokkos::parallel_for(
                        Kokkos::RangePolicy<ExecSpace>(
                            0, _ranks_per_dim[dj] * _ranks_per_dim[dk] ),
                        KOKKOS_LAMBDA( uint32_t jnk ) {
                            int j = static_cast<int>( jnk / rank_k );
                            int k = static_cast<int>( jnk % rank_k );
                            current_workload( jnk ) = compute_sub_workload(
                                di, last_point, point_i, dj, j, dk, k,
                                rec_partition, prefix_sum );
                            // printf( "point_i = %d, last_point = %d, "
                            //         "cur_workload(%d) = %d\n",
                            //         point_i, last_point, jnk,
                            //         current_workload( jnk ) );
                        } );
                    // compute the (w_jk^ave - w_jk^{previ:i})
                    Kokkos::parallel_for(
                        Kokkos::RangePolicy<ExecSpace>(
                            0, _ranks_per_dim[dj] * _ranks_per_dim[dk] ),
                        KOKKOS_LAMBDA( uint32_t jnk ) {
                            // int j =
                            //     static_cast<int>( jnk / rank_k );
                            // int k =
                            //     static_cast<int>( jnk % rank_k );
                            auto wl =
                                current_workload( jnk ) - ave_workload( jnk );
                            wl *= wl;
                            current_workload( jnk ) = wl;
                            // printf(
                            //     "point_i = %d, last_point = %d, "
                            //     "[cur_workload(%d)-ave_workload(%d)]^2 = %d\n
                            //     ", point_i, last_point, jnk, jnk,
                            //     current_workload( jnk ) );
                        } );

                    int diff;
                    Kokkos::parallel_reduce(
                        Kokkos::RangePolicy<ExecSpace>(
                            0, _ranks_per_dim[dj] * _ranks_per_dim[dk] ),
                        KOKKOS_LAMBDA( const int idx, int& update ) {
                            update += current_workload( idx );
                        },
                        diff );
                    // printf( "point_i = %d, last_point = %d, "
                    //         "diff_sqr = %d\n",
                    //         point_i, last_point, diff );
                    if ( diff <= last_diff )
                    {
                        // printf( "new optimal: last_diff = %d, diff = %d, "
                        //         "point_i = %d, rank = %d\n",
                        //         last_diff, diff, point_i, rank );
                        if ( point_i == rec_mirror( rank, di ) )
                        {
                            rec_mirror( current_rank, di ) = point_i;
                            break;
                        }
                        last_diff = diff;
                        point_i++;
                    }
                    else
                    {
                        // printf( "find optimal(+1): di = %d, last_diff = %d, "
                        //         "diff = %d,"
                        //         "point_i = %d, rank = %d, "
                        //         "rec_mirror(current_rank, di) = %d\n",
                        //         di, last_diff, diff, point_i, rank,
                        //         rec_mirror( current_rank, di ) );
                        if ( rec_mirror( current_rank, di ) != point_i - 1 )
                        {
                            rec_mirror( current_rank, di ) = point_i - 1;
                            is_changed = true;
                        }
                        last_point = point_i;
                        break;
                    }
                } // end while
            }
            Kokkos::deep_copy( _rectangle_partition_dev, rec_mirror );
        }
    }

    // void greedyPartition();

    // bool adaptive_load_balance();
  private:
    template <typename PartitionView, typename WorkloadView>
    KOKKOS_INLINE_FUNCTION int
    compute_sub_workload( int dim_j, int j, int dim_k, int k, int rank_i,
                          PartitionView& rec_partition,
                          WorkloadView& prefix_sum )
    {
        // auto prefix_sum = _workload_prefix_sum;
        // printf( "Inside compute_sub_workload 1\n" );
        // Kokkos::Array<int, 3> end, start;
        int end[3], start[3];
        int dim_i = 0;
        while ( dim_i == dim_j || dim_i == dim_k )
        {
            dim_i = ( dim_i + 1 ) % 3;
        }
        end[dim_i] = rec_partition( rank_i, dim_i );
        // end[dim_i] = rec_partition( _ranks_per_dim[dim_i], dim_i );
        end[dim_j] = rec_partition( j + 1, dim_j );
        end[dim_k] = rec_partition( k + 1, dim_k );

        start[dim_i] = 0;
        start[dim_j] = rec_partition( j, dim_j );
        start[dim_k] = rec_partition( k, dim_k );

        // S[i][j][k] = S[i-1][j][k] + S[i][j-1][k] + S[i][j][k-1] -
        // S[i-1][j-1][k]
        // - S[i][j-1][k-1] - S[i-1][j][k-1] + S[i-1][j-1][k-1] + a[i][j][k]
        // printf(
        //     "1 - dim_j = %d, j = %d, dim_k = %d, k = %d; start = [%d, %d,%d],
        //     " "end = [%d, %d, %d]: %d, %d, %d, %d, %d, %d, %d, %d\n", dim_j,
        //     j, dim_k, k, start[dim_i], start[dim_j], start[dim_k],
        //     end[dim_i], end[dim_j], end[dim_k],
        //     prefix_sum( end[0], end[1], end[2] ),
        //     prefix_sum( start[0], end[1], end[2] ),
        //     prefix_sum( end[0], start[1], end[2] ),
        //     prefix_sum( end[0], end[1], start[2] ),
        //     prefix_sum( start[0], start[1], end[2] ),
        //     prefix_sum( end[0], start[1], start[2] ),
        //     prefix_sum( start[0], end[1], start[2] ),
        //     prefix_sum( start[0], start[1], start[2] ) );

        return prefix_sum( end[0], end[1], end[2] )     //    S[i][j][k]
               - prefix_sum( start[0], end[1], end[2] ) //  S[i-1][j][k]
               - prefix_sum( end[0], start[1], end[2] ) //  S[i][j-1][k]
               - prefix_sum( end[0], end[1], start[2] ) //  S[i][j][k-1]
               + prefix_sum( start[0], start[1],
                             end[2] ) // S[i-1][j-1][k]
               + prefix_sum( end[0], start[1],
                             start[2] ) // S[i][j-1][k-1]
               + prefix_sum( start[0], end[1],
                             start[2] ) // S[i-1][j][k-1]
               - prefix_sum( start[0], start[1],
                             start[2] ); // S[i-1][j-1][k-1]
        // return _workload_prefix_sum( end[0], end[1], end[2] );
    }

    template <typename PartitionView, typename WorkloadView>
    KOKKOS_INLINE_FUNCTION int
    compute_sub_workload( int dim_i, int i_start, int i_end, int dim_j, int j,
                          int dim_k, int k, PartitionView& rec_partition,
                          WorkloadView& prefix_sum )
    {
        // auto prefix_sum = _workload_prefix_sum;
        // printf( "Inside compute_sub_workload 2\n" );
        int end[3], start[3];
        // Kokkos::Array<int, 3> end, start;
        end[dim_i] = i_end;
        end[dim_j] = rec_partition( j + 1, dim_j );
        end[dim_k] = rec_partition( k + 1, dim_k );

        start[dim_i] = i_start;
        start[dim_j] = rec_partition( j, dim_j );
        start[dim_k] = rec_partition( k, dim_k );

        // S[i][j][k] = S[i-1][j][k] + S[i][j-1][k] + S[i][j][k-1] -
        // S[i-1][j-1][k]
        // - S[i][j-1][k-1] - S[i-1][j][k-1] + S[i-1][j-1][k-1] + a[i][j][k]

        // printf(
        //     "2 - dim_j = %d, j = %d, dim_k = %d, k = %d; start = [%d, %d,
        //     %d], " "end = [%d, %d, %d]: %d, %d, %d, %d, %d, %d, %d, %d\n",
        //     dim_j, j, dim_k, k, start[dim_i], start[dim_j], start[dim_k],
        //     end[dim_i], end[dim_j], end[dim_k],
        //     prefix_sum( end[0], end[1], end[2] ),
        //     prefix_sum( start[0], end[1], end[2] ),
        //     prefix_sum( end[0], start[1], end[2] ),
        //     prefix_sum( end[0], end[1], start[2] ),
        //     prefix_sum( start[0], start[1], end[2] ),
        //     prefix_sum( end[0], start[1], start[2] ),
        //     prefix_sum( start[0], end[1], start[2] ),
        //     prefix_sum( start[0], start[1], start[2] ) );
        return prefix_sum( end[0], end[1], end[2] )     // S[i][j][k]
               - prefix_sum( start[0], end[1], end[2] ) // S[i-1][j][k]
               - prefix_sum( end[0], start[1], end[2] ) // S[i][j-1][k]
               - prefix_sum( end[0], end[1], start[2] ) // S[i][j][k-1]
               + prefix_sum( start[0], start[1],
                             end[2] ) // S[i-1][j-1][k]
               + prefix_sum( end[0], start[1],
                             start[2] ) // S[i][j-1][k-1]
               + prefix_sum( start[0], end[1],
                             start[2] ) // S[i-1][j][k-1]
               - prefix_sum( start[0], start[1],
                             start[2] ); // S[i-1][j-1][k-1]
    }

  private:
    // ! workload_threshold_coeff
    // float _max_workload_coeff;
    //! workload_threshold
    int _workload_threshold;
    //! default check point for re-balance
    int _num_step_rebalance;
    //! max_optimize iterations
    int _max_optimize_iteration;
    //! represent the rectangle partition in each dimension
    //! with form [p_1, ..., p_n, cell_num], n =
    //! rank-num-in-current-dimension partition in this dimension would be [0,
    //! p_1), [p_1, p_2) ... [p_n, cellNum] (unit: tile)
    Kokkos::View<int* [3], MemorySpace> _rectangle_partition_dev;
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