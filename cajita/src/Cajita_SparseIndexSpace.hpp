#ifnedf CAJITA_SPARSE_INDEXSPACE_HPP
#define CAJITA_SPARSE_INDEXSPACE_HPP

#include <Kokkos_Core.hpp>

#include <array>
#include <string>

namespace Cajita
{

struct HashTypes
{
    enum Values
    {
        Naive = 0,
        Morton = 1,
    };
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
template <typename Key, typename HashType>
struct TileID2HashKey;

template <typename Key, typename HashType>
struct HashKey2TileID;

// functions to compute hash key
// can be rewriten in a recursive way
template <typename Key>
struct TileID2HashKey<Key, HashTypes::Naive>
{
    TileID2HashKey( std::array<int, 3> &&tnum )
        : _tilenum( tnum )
    {
    }
    TileID2HashKey( int i, int j, int k )
        : _tilenum( i, j, k )
    {
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr auto operator()( std::array<int, 3> tileid ) -> Key
    {
        return tileid[0] * _tilenum[1] * _tilenum[2] + tileid[1] * _tilenum[2] +
               tileid[2];
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr auto operator()( int tileid_x, int tileid_y, int tileid_z ) -> Key
    {
        return tileid_x * _tilenum[1] * _tilenum[2] + tileid_y * _tilenum[2] +
               tileid_z;
    }

  private:
    std::array<int, 3> _tilenum;
};

template <typename Key>
struct TileID2HashKey<Key, HashTypes::Morton>
{
    TileID2HashKey( std::array<int, 3> && ) {} // ugly
    TileID2HashKey( int, int, int ) {}

    enum : uint64_t
    { // hand-coded now, can be improved by iterative func
        page_zmask = ( 0x9249249249249249UL ),
        page_ymask = ( 0x2492492492492492UL ),
        page_xmask = ( 0x4924924924924924UL )
    };
    KOKKOS_FORCEINLINE_FUNCTION
    constexpr auto operator()( std::array<int, 3> tileid ) -> Key
    {
        return bit_spread( page_zmask, tileid[2] ) |
               bit_spread( page_ymask, tileid[1] ) |
               bit_spread( page_xmask, tileid[0] );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr auto operator()( int tileid_x, int tileid_y, int tileid_z ) -> Key
    {
        return bit_spread( page_zmask, tileid_z ) |
               bit_spread( page_ymask, tileid_y ) |
               bit_spread( page_xmask, tileid_x );
    }
};

// can be rewriten in a recursive way
template <typename Key>
struct HashKey2TileID<Key, HashTypes::Naive>
{
    HashKey2TileID( std::array<int, 3> &&tnum )
        : _tilenum( tnum )
    {
    }
    HashKey2TileID( int i, int j, int k )
        : _tilenum( i, j, k )
    {
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr void operator()( Key tilekey, std::array<int, 3> &tileid )
    {
        tileid[2] = tilekey % _tilenum[2];
        tileid[1] = static_cast<Key>( tilekey / _tilenum[2] ) % _tilenum[1];
        tileid[0] = static_cast<Key>( tilekey / _tilenum[2] / _tilenum[1] ) %
                    _tilenum[0];
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr void operator()( Key tilekey, int &tileid_x, int &tileid_y,
                               int &tileid_z )
    {
        tileid_z = tilekey % _tilenum[2];
        tileid_y = static_cast<Key>( tilekey / _tilenum[2] ) % _tilenum[1];
        tileid_x = static_cast<Key>( tilekey / _tilenum[2] / _tilenum[1] ) %
                   _tilenum[0];
    }

  private:
    std::array<int, 3> _tilenum;
};

template <typename Key>
struct HashKey2TileID<Key, HashTypes::Morton>
{
    HashKey2TileID( std::array<int, 3> && ) {} // ugly
    HashKey2TileID( int, int, int ) {}
    enum : uint64_t
    { // hand-coded now, can be improved by iterative func
        page_zmask = ( 0x9249249249249249UL ),
        page_ymask = ( 0x2492492492492492UL ),
        page_xmask = ( 0x4924924924924924UL )
    };

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr void operator()( Key tilekey, std::array<int, 3> &tileid )
    {
        tileid[2] = bit_pack( page_zmask, tilekey );
        tileid[1] = bit_pack( page_ymask, tilekey );
        tileid[0] = bit_pack( page_xmask, tilekey );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr void operator()( Key tilekey, int &tileid_x, int &tileid_y,
                               int &tileid_z )
    {
        tileid_z = bit_pack( page_zmask, tilekey );
        tileid_y = bit_pack( page_ymask, tilekey );
        tileid_x = bit_pack( page_xmask, tilekey );
    }
};

//---------------------------------------------------------------------------//
/*!
  \class SparseIndexSpace
  \brief Sparse index space, hierarchical structure (cell->tile->block)
  \ ValueType : tileNo type
 */
template <int N = 3, unsigned long long TileNPerDim = 4,
          typename Hash = HashTypes::Naive,
          typename Device = Kokkos::DefaultExecutionSpace,
          typename Key = uint64_t, typename Value = uint64_t>
class SparseIndexSpace
{
  public:
    //! Number of dimensions, 3 = ijk, or 4 = ijk + ch
    static constexpr int Rank = N;
    //! Number of cells inside each tile, tile size reset to power of 2
    static constexpr unsigned long long TileBitsPerDim =
        bit_count( TileNPerDim );
    static constexpr unsigned long long TileSizePerDim = 1 << TileBitsPerDim;
    static constexpr unsigned long long TileMaskPerDim =
        ( 1 << TileBitsPerDim ) - 1;
    static constexpr unsigned long long TileBits =
        TileBitsPerDim + TileBitsPerDim + TileBitsPerDim;
    static constexpr unsigned long long TileSize =
        TileSizePerDim * TileSizePerDim * TileSizePerDim;
    //! Types
    using KeyType = Key;     // tile hash key type
    using ValueType = Value; // tile value type
    using HashType = Hash;   // hash table type

    // size should be global
    SparseIndexSpace( const std::array<int, N> &size, int capacity,
                      float rehash_factor )
        : _blkIdSpace( 1 << bit_count( capacity ), rehash_factor, size )
        , _tileIdSpace()
    {
        std::fill( _min.data(), _min.data() + Rank, 0 );
        std::copy( size.begin(), size.end(), _max.data() );
    }

    //! given a cell ijk, insert the tile such cell reside in
    KOKKOS_FORCEINLINE_FUNCTION
    ValueType insert_cell( int cellid_x, int cellid_y, int cellid_z )
    {
        return _blkIdSpace.insert( cellid_x >> TileBitsPerDim,
                                   cellid_y >> TileBitsPerDim,
                                   cellid_z >> TileBitsPerDim );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    ValueType insert_tile( int tileid_x, int tileid_y, int tileid_z )
    {
        return _blkIdSpace.insert( tileid_x, tileid_y, tileid_z );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    ValueType query_cell( int cellid_x, int cellid_y, int cellid_z )
    {
        auto tileid = _blkIdSpace.find( cellid_x >> TileBitsPerDim,
                                        cellid_y >> TileBitsPerDim,
                                        cellid_z >> TileBitsPerDim );
        auto cellid = _tileIdSpace.coord_to_offset(
            cellid_x & TileMask, cellid_y & TileMask, cellid_z & TileMask );
        return ( ValueType )( ( tileid << TileBits ) & cellid );
    }

  private:
    //! block index space, map tile ijk to tile No
    BlockIndexSpace<TileBitsPerDim, TileSizePerDim, TSize, HashType, Device,
                    KeyType, ValueType>
        _blkIdSpace;
    //! tile index space, map cell ijk to cell local No inside a tile
    TileIndexSpace<TileBitsPerDim, TileSizePerDim, TSize> _tileIdSpace;
    //! space size (global), channel size
    Kokkos::Array<int, Rank> _min;
    Kokkos::Array<int, Rank> _max;
};

//---------------------------------------------------------------------------//
template <unsigned long long TBits, unsigned long long TSizePerDim,
          unsigned long long TSize, typename Hash, typename Device,
          typename Key, typename Value>
class BlockIndexSpace
{
  public:
    //! Number of cells inside each tile
    static constexpr unsigned long long TileBitsPerDim = TBits;
    static constexpr unsigned long long TileSizePerDim = TSizePerDim;
    static constexpr unsigned long long TileSize = TSize;
    //! Types
    using KeyType = Key;     // tile hash key type
    using ValueType = Value; // tile value type
    using HashType = Hash;   // hash table type

    BlockIndexSpace( int capacity, float rehash_factor,
                     std::array<int, N> &size )
        : _tile_num( 0 )
        , _tile_capacity( capacity )
        , _rehash_coeff( rehash_factor )
        , _tile_table( capacity )
        , _op_ijk2key( size[0], size[1], size[2] )
        , _op_key2ijk( size[0], size[1], size[2] )
    {
        _block_table.clear();
        _value_table.clear();
    }

    KOKKOS_FORCEINLINE_FUNCTION
    ValueType insert( int tileid_x, int tileid_y, int tileid_z )
    {
        if ( _tile_table.size() >= _tile_capacity )
        {
            _tile_capacity = static_cast<int>( _tile_capacity * _rehash_coeff );
            _tile_table.rehash( _tile_capacity );
            // TODO: view reallocate needed
        }
        const KeyType tileKey = _op_ijk2key( tileid_x, tileid_y, tileid_z );
        ValueType tileNo;
        if ( ( tileNo = _key_table.find( tileKey ) ) ==
             Kokkos::UnorderedMap::invalid_index )
        {
            // current impl: use inserting sequence as id
            // rethink: sort the id .
            tileNo = _blk_table.size();
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
    ValueType find( int tileid_x, int tileid_y, int tileid_z )
    {
        return _blk_table.find( _op_ijk2key( tileid_x, tileid_y, tileid_z ) );
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
    Kokkos::UnorderedMap<KeyType, ValueType, Device> _tile_table;
    //! Op: tile IJK <=> tile No
    TileID2HashKey<KeyType, HashType> _op_ijk2key;
    HashKey2TileID<KeyType, HashType> _op_key2ijk;
};

//---------------------------------------------------------------------------//
template <int TBits, int TSizePerDim, int TSize>
class TileIndexSpace
{
  public:
    //! Number of cells inside each tile
    static constexpr int TileBitsPerDim = TBits;
    static constexpr int TileSizePerDim = TSizePerDim;
    static constexpr int TileSize = TSize;

    //! Coord  <=> Offset Computations
    struct Coord2OffsetDim
    {
        template <typename Coord>
        constexpr auto operator()( int dimNo, Coord &&i ) -> uint64_t
        {
            uint64_t result = (uint64_t)i;
            for ( int i = 0; i < dimNo; i++ )
                result *= TileSizePerDim;
            return result;
        }
    };

    struct Offset2CoordDim
    {
        template <typename Coord>
        constexpr auto operator()( int dimNo, Coord &i, const uint64_t &offset )
            -> uint64_t
        {
            i = offset % BlockSizePerDim;
            return static_cast<uint64_t>( offset / TileSizePerDim );
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
                                   std::forward<Coords>( coords ) );
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
    static constexpr void to_coord( Key &&key, Coords &... coords )->uint64_t
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
            to_coord_impl<Func>( dimNo + 1, std::forward<Key>( Newkey ),
                                 is... );
    }
};

} // end namespace Cajita
#endif ///< !CAJITA_SPARSE_INDEXSPACE_HPP