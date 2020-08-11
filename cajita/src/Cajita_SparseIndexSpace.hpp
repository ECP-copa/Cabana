#ifndef CAJITA_SPARSE_INDEXSPACE_HPP
#define CAJITA_SPARSE_INDEXSPACE_HPP

#include <Kokkos_UnorderedMap.hpp>

#include <array>
#include <string>

namespace Cajita
{

enum class HashTypes : unsigned char
{
    Naive = 0,
    Morton = 1
};

//---------------------------------------------------------------------------//
// bit operations for Morton Code computing
template <typename Integer>
constexpr Integer bit_length( Integer N ) noexcept
{
    if ( N )
        return bit_length( N >> 1 ) + static_cast<Integer>( 1 );
    else
        return 0;
}

template <typename Integer>
constexpr Integer bit_count( Integer N ) noexcept
{
    return bit_length( N - 1 );
}

template <typename Integer>
constexpr Integer binary_reverse( Integer data,
                                  char loc = sizeof( Integer ) * 8 - 1 )
{
    if ( data == 0 )
        return 0;
    return ( ( data & 1 ) << loc ) | binary_reverse( data >> 1, loc - 1 );
}

template <typename Integer>
constexpr unsigned count_leading_zeros( Integer data )
{
    unsigned res{0};
    data = binary_reverse( data );
    if ( data == 0 )
        return sizeof( Integer ) * 8;
    while ( ( data & 1 ) == 0 )
        res++, data >>= 1;
    return res;
}

constexpr int bit_pack( const uint64_t mask, const uint64_t data )
{
    uint64_t slresult = 0;
    uint64_t &ulresult{slresult};
    uint64_t uldata = data;
    int count = 0;
    ulresult = 0;

    uint64_t rmask = binary_reverse( mask );
    unsigned char lz{0};

    while ( rmask )
    {
        lz = count_leading_zeros( rmask );
        uldata >>= lz;
        ulresult <<= 1;
        count++;
        ulresult |= ( uldata & 1 );
        uldata >>= 1;
        rmask <<= lz + 1;
    }
    ulresult <<= 64 - count; // 64 bits (maybe not use a constant 64 ...?)
    ulresult = binary_reverse( ulresult );
    return (int)slresult;
}

constexpr uint64_t bit_spread( const uint64_t mask, const int data )
{
    uint64_t rmask = binary_reverse( mask );
    int dat = data;
    uint64_t result = 0;
    unsigned char lz{0};
    while ( rmask )
    {
        lz = count_leading_zeros( rmask ) + 1;
        result = result << lz | ( dat & 1 );
        dat >>= 1, rmask <<= lz;
    }
    result = binary_reverse( result ) >> count_leading_zeros( mask );
    return result;
}
//---------------------------------------------------------------------------//

// Tile Hash Key (hash) <=> TileID
template <typename Key, HashTypes HashT>
struct TileID2HashKey;

template <typename Key, HashTypes HashT>
struct HashKey2TileID;

// functions to compute hash key
// can be rewriten in a recursive way
template <typename Key>
struct TileID2HashKey<Key, HashTypes::Naive>
{
    TileID2HashKey( std::initializer_list<int> &&tnum )
    {
        std::copy( tnum.begin(), tnum.end(), _tilenum.data() );
    }
    TileID2HashKey( int i, int j, int k )
    {
        _tilenum[0] = i;
        _tilenum[1] = j;
        _tilenum[2] = k;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr auto operator()( int tile_i, int tile_j, int tile_k ) -> Key
    {
        return tile_i * _tilenum[1] * _tilenum[2] + tile_j * _tilenum[2] +
               tile_k;
    }

  private:
    Kokkos::Array<int, 3> _tilenum;
};

template <typename Key>
struct TileID2HashKey<Key, HashTypes::Morton>
{
    TileID2HashKey( std::initializer_list<int> &&tnum ) {} // ugly
    TileID2HashKey( int, int, int ) {}

    enum : uint64_t
    { // hand-coded now, can be improved by iterative func
        page_kmask = ( 0x9249249249249249UL ),
        page_jmask = ( 0x2492492492492492UL ),
        page_imask = ( 0x4924924924924924UL )
    };

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr auto operator()( int tile_i, int tile_j, int tile_k ) -> Key
    {
        return bit_spread( page_kmask, tile_k ) |
               bit_spread( page_jmask, tile_j ) |
               bit_spread( page_imask, tile_i );
    }
};

// can be rewriten in a recursive way
template <typename Key>
struct HashKey2TileID<Key, HashTypes::Naive>
{
    HashKey2TileID( std::initializer_list<int> &&tnum )
        : _tilenum()
    {
        std::copy( tnum.begin(), tnum.end(), _tilenum.data() );
    }
    HashKey2TileID( int i, int j, int k )
    {
        _tilenum[0] = i;
        _tilenum[1] = j;
        _tilenum[2] = k;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr void operator()( Key tilekey, int &tile_i, int &tile_j,
                               int &tile_k )
    {
        tile_k = tilekey % _tilenum[2];
        tile_j = static_cast<Key>( tilekey / _tilenum[2] ) % _tilenum[1];
        tile_i = static_cast<Key>( tilekey / _tilenum[2] / _tilenum[1] ) %
                 _tilenum[0];
    }

  private:
    Kokkos::Array<int, 3> _tilenum;
};

template <typename Key>
struct HashKey2TileID<Key, HashTypes::Morton>
{
    HashKey2TileID( std::initializer_list<int> &&tnum ) {} // ugly
    HashKey2TileID( int, int, int ) {}
    enum : uint64_t
    { // hand-coded now, can be improved by iterative func
        page_kmask = ( 0x9249249249249249UL ),
        page_jmask = ( 0x2492492492492492UL ),
        page_imask = ( 0x4924924924924924UL )
    };

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr void operator()( Key tilekey, int &tile_i, int &tile_j,
                               int &tile_k )
    {
        tile_k = bit_pack( page_kmask, tilekey );
        tile_j = bit_pack( page_jmask, tilekey );
        tile_i = bit_pack( page_imask, tilekey );
    }
};

//---------------------------------------------------------------------------//
template <typename MemorySpace, int N, unsigned long long CBits,
          unsigned long long CNumPerDim, unsigned long long CNumPerTile,
          HashTypes Hash, typename Key, typename Value>
class BlockIndexSpace;

template <int CBits, int CNumPerDim, int CNumPerTile>
class TileIndexSpace;
/*!
  \class SparseIndexSpace
  \brief Sparse index space, hierarchical structure (cell->tile->block)
  \ ValueType : tileNo type
 */
template <typename MemorySpace, int N = 3,
          unsigned long long CellPerTileDim = 4,
          HashTypes Hash = HashTypes::Naive, typename Key = uint64_t,
          typename Value = uint64_t>
class SparseIndexSpace
{
  public:
    //! Number of dimensions, 3 = ijk, or 4 = ijk + ch
    static constexpr int Rank = N;
    //! Number of cells inside each tile, tile size reset to power of 2
    static constexpr unsigned long long CellBitsPerTileDim =
        bit_count( CellPerTileDim );
    static constexpr unsigned long long CellNumPerTileDim =
        1 << CellBitsPerTileDim;
    static constexpr unsigned long long CellMaskPerTileDim =
        ( 1 << CellBitsPerTileDim ) - 1;
    static constexpr unsigned long long CellBitsPerTile =
        CellBitsPerTileDim + CellBitsPerTileDim + CellBitsPerTileDim;
    static constexpr unsigned long long CellNumPerTile =
        CellNumPerTileDim * CellNumPerTileDim * CellNumPerTileDim;
    //! Types
    using KeyType = Key;                        // tile hash key type
    using ValueType = Value;                    // tile value type
    static constexpr HashTypes HashType = Hash; // hash table type

    // size should be global
    SparseIndexSpace( const std::array<int, N> size, const int capacity,
                      const float rehash_factor )
        : _blkIdSpace( size, 1 << bit_count( capacity ), rehash_factor )
    {
        std::fill( _min.data(), _min.data() + Rank, 0 );
        std::copy( size.begin(), size.end(), _max.data() );
    }

    //! given a cell ijk, insert the tile such cell reside in
    KOKKOS_FORCEINLINE_FUNCTION
    ValueType insert_cell( int cell_i, int cell_j, int cell_k )
    {
        return insert_tile( cell_i >> CellBitsPerTileDim,
                            cell_j >> CellBitsPerTileDim,
                            cell_k >> CellBitsPerTileDim );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    ValueType insert_tile( int cell_i, int cell_j, int cell_k )
    {
        return _blkIdSpace.insert( cell_i, cell_j, cell_k );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    ValueType query_cell( int cell_i, int cell_j, int cell_k )
    {
        auto tileid = _blkIdSpace.find( cell_i >> CellBitsPerTileDim,
                                        cell_j >> CellBitsPerTileDim,
                                        cell_k >> CellBitsPerTileDim );
        auto cellid = _tileIdSpace.coord_to_offset(
            cell_i & CellMaskPerTileDim, cell_j & CellMaskPerTileDim,
            cell_k & CellMaskPerTileDim );
        return ( ValueType )( ( tileid << CellBitsPerTile ) & cellid );
    }

  private:
    //! block index space, map tile ijk to tile No
    BlockIndexSpace<MemorySpace, Rank, CellBitsPerTileDim, CellNumPerTileDim,
                    CellNumPerTile, HashType, KeyType, ValueType>
        _blkIdSpace;
    //! tile index space, map cell ijk to cell local No inside a tile
    TileIndexSpace<CellBitsPerTileDim, CellNumPerTileDim, CellNumPerTile>
        _tileIdSpace;
    //! space size (global), channel size
    Kokkos::Array<int, Rank> _min;
    Kokkos::Array<int, Rank> _max;
};

//---------------------------------------------------------------------------//
template <typename MemorySpace, int N, unsigned long long CBits,
          unsigned long long CNumPerDim, unsigned long long CNumPerTile,
          HashTypes Hash, typename Key, typename Value>
class BlockIndexSpace
{
  public:
    //! Number of cells inside each tile
    static constexpr unsigned long long CellBitsPerTileDim = CBits;
    static constexpr unsigned long long CellNumPerTileDim = CNumPerDim;
    static constexpr unsigned long long CellNumPerTile = CNumPerTile;
    //! Types
    using KeyType = Key;                        // tile hash key type
    using ValueType = Value;                    // tile value type
    static constexpr HashTypes HashType = Hash; // hash table type

    BlockIndexSpace( const std::array<int, N> size, const int capacity,
                     const float rehash_factor )
        : _tile_num( 0 )
        , _tile_capacity( capacity )
        , _rehash_coeff( rehash_factor )
        , _tile_table( capacity )
        , _op_ijk2key( size[0], size[1], size[2] )
        , _op_key2ijk( size[0], size[1], size[2] )
    {
        _tile_table.clear();
    }

    KOKKOS_FORCEINLINE_FUNCTION
    ValueType insert( int tile_i, int tile_j, int tile_k )
    {
        if ( _tile_table.size() >= _tile_capacity )
        {
            _tile_capacity = static_cast<int>( _tile_capacity * _rehash_coeff );
            _tile_table.rehash( _tile_capacity );
            // TODO: view reallocate needed
        }
        const KeyType tileKey = _op_ijk2key( tile_i, tile_j, tile_k );
        ValueType tileNo;
        if ( ( tileNo = _tile_table.find( tileKey ) ) ==
             _tile_table.invalid_index )
        {
            // current impl: use inserting sequence as id
            // rethink: sort the id .
            tileNo = _tile_table.size();
            _tile_table.insert( tileKey, tileNo );
        }
        return tileNo;
    }

    KOKKOS_FORCEINLINE_FUNCTION
    void sortTile()
    {
        // pass
    }

    // query
    KOKKOS_FORCEINLINE_FUNCTION
    ValueType find( int tile_i, int tile_j, int tile_k )
    {
        return _tile_table.find( _op_ijk2key( tile_i, tile_j, tile_k ) );
    }

  private:
    //! valid block number
    int _tile_num;
    //! pre-allocated size
    int _tile_capacity;
    //! default factor, rehash by which when need more capacity of the hash
    //! table
    float _rehash_coeff;
    //! hash table (tile IJK <=> tile No)
    Kokkos::UnorderedMap<KeyType, ValueType, MemorySpace> _tile_table;
    //! Op: tile IJK <=> tile No
    TileID2HashKey<KeyType, HashType> _op_ijk2key;
    HashKey2TileID<KeyType, HashType> _op_key2ijk;
};

//---------------------------------------------------------------------------//
template <int CBits, int CNumPerDim, int CNumPerTile>
class TileIndexSpace
{
  public:
    //! Number of cells inside each tile
    static constexpr int CellBitsPerTileDim = CBits;
    static constexpr int CellNumPerTileDim = CNumPerDim;
    static constexpr int CellNumPerTile = CNumPerTile;
    static constexpr int Rank = 3;

    //! Coord  <=> Offset Computations
    struct Coord2OffsetDim
    {
        template <typename Coord>
        constexpr auto operator()( int dimNo, Coord &&i ) -> uint64_t
        {
            uint64_t result = (uint64_t)i;
            for ( int i = 0; i < dimNo; i++ )
                result *= CellNumPerTileDim;
            return result;
        }
    };

    struct Offset2CoordDim
    {
        template <typename Coord>
        constexpr auto operator()( int dimNo, Coord &i, const uint64_t &offset )
            -> uint64_t
        {
            i = offset % CellNumPerTileDim;
            return static_cast<uint64_t>( offset / CellNumPerTileDim );
        }
    };

    //! Cell ijk <=> Cell Id
    template <typename... Coords>
    static constexpr auto coord_to_offset( Coords &&... coords ) -> uint64_t
    {
        return from_coord<Coord2OffsetDim>( std::forward<Coords>( coords )... );
    }

    template <typename Key, typename... Coords>
    static constexpr void offset_to_coord( Key &&key, Coords &... coords )
    {
        to_coord<Offset2CoordDim>( std::forward<Key>( key ),
                                   std::forward<Coords>( coords )... );
    }

  protected:
    template <typename Func, typename... Coords>
    static constexpr auto from_coord( Coords &&... coords ) -> uint64_t
    {
        if ( sizeof...( Coords ) != Rank )
            throw std::invalid_argument( "Dimension of coordinate mismatch" );
        using Integer = std::common_type_t<Coords...>;
        if ( !( std::is_integral<Integer>::value ) )
            throw std::invalid_argument( "Coordinate is not integral type" );
        return from_coord_impl<Func>( 0, coords... );
    }

    template <typename Func, typename Key, typename... Coords>
    static constexpr void to_coord( Key &&key, Coords &... coords )
    {
        if ( sizeof...( Coords ) != Rank )
            throw std::invalid_argument( "Dimension of coordinate mismatch" );
        using Integer = std::common_type_t<Coords...>;
        if ( !( std::is_integral<Integer>::value ) )
            throw std::invalid_argument( "Coordinate is not integral type" );
        return to_coord_impl<Func>( 0, key, coords... );
    }

    template <typename Func, typename Coord>
    static constexpr auto from_coord_impl( int &&dimNo, Coord &&i )
    {
        return Func()( dimNo, std::forward<Coord>( i ) );
    }

    template <typename Func, typename Coord, typename... Coords>
    static constexpr auto from_coord_impl( int &&dimNo, Coord &&i,
                                           Coords &&... is )
    {
        auto result = Func()( dimNo, std::forward<Coord>( i ) );
        if ( dimNo + 1 < Rank )
            result += from_coord_impl<Func>( dimNo + 1,
                                             std::forward<Coords>( is )... );
        return result;
    }

    template <typename Func, typename Key, typename Coord>
    static constexpr void to_coord_impl( int &&dimNo, Key &&key, Coord &i )
    {
        Func()( dimNo, i, std::forward<Key>( key ) );
    }

    template <typename Func, typename Key, typename Coord, typename... Coords>
    static constexpr void to_coord_impl( int &&dimNo, Key &&key, Coord &i,
                                         Coords &... is )
    {
        Key newKey = Func()( dimNo, i, std::forward<Key>( key ) );
        if ( dimNo + 1 < Rank )
            to_coord_impl<Func>( dimNo + 1, std::forward<Key>( newKey ),
                                 is... );
    }
};

} // end namespace Cajita
#endif ///< !CAJITA_SPARSE_INDEXSPACE_HPP