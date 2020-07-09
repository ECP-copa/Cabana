#ifndef CAJITA_SPARSE_INDEXSPACE_HPP
#define CAJITA_SPARSE_INDEXSPACE_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <string>

#include <Kokkos_UnorderedMap.hpp>

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

// BlockKey (hash) <=> BlockID
template <typename Key, typename HashType>
struct BlockID2HashKey;

template <typename Key, typename HashType>
struct HashKey2BlockID;

// functions to compute hash key
// can be rewriten in a recursive way
template <typename Key>
struct BlockID2HashKey<Key, HashTypes::Naive>
{
    KOKKOS_FORCEINLINE_FUNCTION
    constexpr auto operator()( std::array<int, 3> blockid ) -> Key
    {
        return blockid[0] * _blocknum[1] * _blocknum[2] +
               blockid[1] * _blocknum[2] + blockid[2];
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr auto operator()( int blockid_x, int blockid_y, int blockid_z )
        -> Key
    {
        return blockid_x * _blocknum[1] * _blocknum[2] +
               blockid_y * _blocknum[2] + blockid_z;
    }

  private:
    std::array<int, 3> _blocknum;
};

template <typename Key>
struct BlockID2HashKey<Key, HashTypes::Morton>
{
    BlockID2HashKey( std::array<int, 3> ) {} // ugly

    enum : uint64_t
    { // hand-coded now, can be improved by iterative func
        page_zmask = ( 0x9249249249249249UL ),
        page_ymask = ( 0x2492492492492492UL ),
        page_xmask = ( 0x4924924924924924UL )
    };
    KOKKOS_FORCEINLINE_FUNCTION
    constexpr auto operator()( std::array<int, 3> blockid ) -> Key
    {
        return bit_spread( page_zmask, blockid[2] ) |
               bit_spread( page_ymask, blockid[1] ) |
               bit_spread( page_xmask, blockid[0] );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr auto operator()( int blockid_x, int blockid_y, int blockid_z )
        -> Key
    {
        return bit_spread( page_zmask, blockid_z ) |
               bit_spread( page_ymask, blockid_y ) |
               bit_spread( page_xmask, blockid_x );
    }
};

// can be rewriten in a recursive way
template <typename Key>
struct HashKey2BlockID<Key, HashTypes::Naive>
{
    KOKKOS_FORCEINLINE_FUNCTION
    constexpr void operator()( Key blockkey, std::array<int, 3> &blockid )
    {
        blockid[2] = blockkey % _blocknum[2];
        blockid[1] = static_cast<Key>( blockkey / _blocknum[2] ) % _blocknum[1];
        blockid[0] =
            static_cast<Key>( blockkey / _blocknum[2] / _blocknum[1] ) %
            _blocknum[0];
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr void operator()( Key blockkey, int &blockid_x, int &blockid_y,
                               int &blockid_z )
    {
        blockid_z = blockkey % _blocknum[2];
        blockid_y = static_cast<Key>( blockkey / _blocknum[2] ) % _blocknum[1];
        blockid_x = static_cast<Key>( blockkey / _blocknum[2] / _blocknum[1] ) %
                    _blocknum[0];
    }

  private:
    std::array<int, 3> _blocknum;
};

template <typename Key>
struct HashKey2BlockID<Key, HashTypes::Morton>
{
    HashKey2BlockID( std::array<int, 3> ) {} // ugly

    enum : uint64_t
    { // hand-coded now, can be improved by iterative func
        page_zmask = ( 0x9249249249249249UL ),
        page_ymask = ( 0x2492492492492492UL ),
        page_xmask = ( 0x4924924924924924UL )
    };

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr void operator()( Key blockkey, std::array<int, 3> &blockid )
    {
        blockid[2] = bit_pack( page_zmask, blockkey );
        blockid[1] = bit_pack( page_ymask, blockkey );
        blockid[0] = bit_pack( page_xmask, blockkey );
    }

    KOKKOS_FORCEINLINE_FUNCTION
    constexpr void operator()( Key blockkey, int &blockid_x, int &blockid_y,
                               int &blockid_z )
    {
        blockid_z = bit_pack( page_zmask, blockkey );
        blockid_y = bit_pack( page_ymask, blockkey );
        blockid_x = bit_pack( page_xmask, blockkey );
    }
};

//---------------------------------------------------------------------------//
/*!
  \class SparseIndexSpace
  \brief Sparse index space, hierarchical structure (cell->block->whole domain)
  \ ValueType : blockNo type
 */
template <long N, int BlkNPerDim = 4, typename Hash = HashTypes::Naive,
          typename Device = Kokkos::DefaultExecutionSpace,
          typename Key = uint64_t, typename Value = uint32_t>
class SparseIndexSpace
{
  public:
    //! Number of dimensions, should be 2
    static constexpr long Rank = N;
    //! Number of cells inside each block per dim
    static constexpr int BlockSizePerDim = BlkNPerDim;
    static constexpr int BlockSizeTotal = BlkNPerDim * BlkNPerDim * BlkNPerDim;
    //! Number of bits account for block ID info
    // If we can make more constrictions on BlockSizePerDim (=2^n)
    // we can use bit operations to compute the blockNo/cellNo
    static constexpr int BlockBits = bit_count( BlockSizePerDim );
    //! Key type, for blockkey and blockhashedkey
    using KeyType = Key;
    //! Value type, for the blockno
    using ValueType = Value;
    //! Hash Type
    using HashType = Hash;
    //! When need more capacity of the hash table, then rehash by factor 2
    static constexpr Rehash_Coeff = 2;
    //! Total number of cells inside each block
    static constexpr int BlockSizeTotal
    {
        int BS = 1;
        for ( decltype( Rank ) curDim = 0; curDim < Rank; curDim++ )
            BS *= BlockSize;
        return BS;
    }
    using LocalIndex = SparseIndexSpace<Rank, BlockSizePerDim>;

    SparseIndexSpace( int capacity, std::array<int, 3> blocknum )
        : _capacity_hint( 1 << bit_count( capacity ) )
        , _block_table( _capacity_hint )
        , _value_table( _capacity_hint )
        , _valid_block_num( 0 )
        , _op_blk2key( blocknum )
        , _op_key2blk( blocknum )
        , _min_ch( -1 )
        , _max_ch( -1 )
    {
        _block_table.clear();
        _value_table.clear();
    }

    // set ch min/max
    void set_ch_min( const int ch_min ) { _min_ch = ch_min; }
    void set_ch_max( const int ch_max ) { _max_ch = ch_max; }

    // get ch min/max
    int chmin() const { return _min_ch; }
    int chmax() const { return _max_ch; }

    // insert, return the blockNo
    // Need to rethink this part, if blockNo is determined by the sequence of
    // inserting there should be no need to identify the naive/morton coding of
    // the block Need Ordered_Map to make use of the naive/morton codring
    // results
    KOKKOS_FORCEINLINE_FUNCTION
    ValueType insert( std::array<int, 3> &&blockid )
    {
        if ( _blk_table.size() >= _capacity_hint )
        {
            _capacity_hint *= Rehash_Coeff;
            _key_table.rehash( _capacity_hint );
            _value_table.rehash( _capacity_hint );
            // TODO: view reallocate needed
        }
        const KeyType blockKey =
            _op_blk2key( std::forward<std::array<int, 3>>( blockid ) );
        ValueType blockNo;
        if ( ( blockNo = _key_table.find( blockKey ) ) ==
             Kokkos::UnorderedMap::invalid_index )
        {
            blockNo = _blk_table.size();
            _blk_table.insert( blockKey, blockNo );
        }
        return blockNo;
    }

    // query
    KOKKOS_FORCEINLINE_FUNCTION
    ValueType find( std::array<int, 3> &&blockid )
    {
        return _blk_table.find(
            _op_blk2key( std::forward<std::array<int, 3>>( blockid ) ) );
    }

    // compare operations?
    // get min/max operations?

  private:
    // hash table (blockId -> blockNo)
    // Perhaps need another table, two options:
    // 1. Two table, one for owned blk, one for ghost blk
    // 2. Two table, one for all blk, one for owned/ghost label (same key)
    Kokkos::UnorderedMap<KeyType, ValueType, Device> _blk_table;
    // Valid block number
    int _valid_block_num;
    // allocated size
    int _capacity_hint;
    // BlockID <=> BlockKey Op
    BlockID2HashKey<KeyType, HashType> _op_blk2key;
    HashKey2BlockID<KeyType, HashType> _op_key2blk;
    // min and max for the ch space (temporal setting)
    int _min_ch;
    int _max_ch;
}; // end class SparseIndexSpace

//---------------------------------------------------------------------------//
/*!
  \class SparseBlockLocalIndexSpace
  \brief Block local index space, hierarchical structure (cell->block->whole
  domain); CellNo : uint64_t; CellId : (int...)
  \brief Lexicographical order is used to traverse the interior of this
  small-sized blocks
 */
template <long N, int BlkSizePerDim>
class SparseBlockLocalIndexSpace
{
  public:
    //! Number of dimensions
    static constexpr long Rank = N;
    //! Number of cells inside each block per dim
    static constexpr int BlockSizePerDim = BlkSizePerDim;
    //! Total number of cells inside each block
    static constexpr int BlockSizeTotal
    {
        int BS = 1;
        for ( decltype( Rank ) curDim = 0; curDim < Rank; curDim++ )
            BS *= BlockSize;
        return BS;
    }
    //! coord <=> offset computations
    struct Coord2OffsetDim
    {
        template <typename Coord>
        constexpr auto operator()( int dimNo, Coord &&i ) -> uint64_t
        {
            uint64_t result = (uint64_t)i;
            for ( int i = 0; i < dimNo; i++ )
                result *= BlockSizePerDim;
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
            return static_cast<uint64_t>( offset / BlockSizePerDim );
        }
    };

    template <typename... Coords>
    static constexpr auto coord2Offset( Coords &&... coords ) -> uint64_t
    {
        return from_coord<Coord2OffsetDim>( std::forward<Coords>( coords )... );
    }

    template <typename Key, typename... Coords>
    static constexpr void offset2Coord( Key &&key, Coords &... coords )
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
}; // end SparseBlockLocalIndexSpace

//---------------------------------------------------------------------------//
// execution policies
// range over all possible blocks

// range over all possible cells inside a block
template <class ExecutionSpace, typename SparseIndexSpace_t,
          int rank = SparseIndexSpace_t::Rank,
          ypename std::enable_if_t<rank == 2> * = nullptr>
Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<SparseIndexSpace_t::Rank>>
createExecutionPolicy( const SparseIndexSpace_t &index_space,
                       std::array<int, 3> &&blockid, const ExecutionSpace & )
{
    auto blockNo = index_space.find( blockid );
    Kokkos::Array<long, 2> min, max;
    min[0] = blockNo;
    max[0] = blockNo + index_space.BlockSizeTotal;
    min[1] = index_space.chmin();
    max[1] = index_space.chmax();

    return Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<rank>>( min,
                                                                      max );
}

//---------------------------------------------------------------------------//
// create view

// create subview

// appendDimension

} // end namespace Cajita
#endif ///< ! CAJITA_SPARSE_INDEXSPACE_HPP