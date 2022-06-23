/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cajita_SparseIndexSpace.hpp
  \brief Sparse grid mapping
*/
#ifndef CAJITA_SPARSE_INDEXSPACE_HPP
#define CAJITA_SPARSE_INDEXSPACE_HPP

#include <Cajita_GlobalMesh.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <array>
#include <memory>
#include <string>

//---------------------------------------------------------------------------//
// Naming convension:
//  Tile ID / Tile ijk = 3D tile indexing, i.e., (i, j, k)
//  Tile No. = 1D number indicating the tile position in the allocated array
//  Tile key / Hash key = 1D number, computed with the given indexing method and
//  used as hash key
//---------------------------------------------------------------------------//

namespace Cajita
{

//---------------------------------------------------------------------------//
//! Hash table type tag.
enum class HashTypes : unsigned char
{
    Naive = 0, // Lexicographical Order
    Morton = 1 // Morton Curve
};

//---------------------------------------------------------------------------//
// bit operations for Morton Code computing
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
/*!
  \brief (Host/Device) Compute the least bit number/length needed to represent
  the given input integer
  \param input_int integer that is going to be evaluated
*/
template <typename Integer>
KOKKOS_INLINE_FUNCTION constexpr Integer bitLength( Integer input_int ) noexcept
{
    if ( input_int )
        return bitLength( input_int >> 1 ) + static_cast<Integer>( 1 );
    else
        return 0;
}

//---------------------------------------------------------------------------//
/*!
  \brief (Host/Device) Compute the lease bit number needed to index input
  integer
  \param input_int integer that is going to be evaluated
*/
template <typename Integer>
KOKKOS_INLINE_FUNCTION constexpr Integer bitCount( Integer input_int ) noexcept
{
    return bitLength( input_int - 1 );
}

//---------------------------------------------------------------------------//
/*!
  \brief (Host/Device) Given a integer, reverse the corresponding binary string,
  return the resulting integer.
  \param input_int integer that is going to be evaluated
  \param loc location for next round reverse
*/
template <typename Integer>
KOKKOS_INLINE_FUNCTION constexpr Integer
binaryReverse( Integer input_int, char loc = sizeof( Integer ) * 8 - 1 )
{
    if ( input_int == 0 )
        return 0;
    return ( ( input_int & 1 ) << loc ) |
           binaryReverse( input_int >> 1, loc - 1 );
}

//---------------------------------------------------------------------------//
/*!
  \brief (Host/Device) Count the leading zeros in the corresponding binary
  string of the input integer
  \param input_int integer that is going to be evaluated
*/
template <typename Integer>
KOKKOS_INLINE_FUNCTION constexpr unsigned countLeadingZeros( Integer input_int )
{
    unsigned res{ 0 };
    input_int = binaryReverse( input_int );
    if ( input_int == 0 )
        return sizeof( Integer ) * 8;
    while ( ( input_int & 1 ) == 0 )
        res++, input_int >>= 1;
    return res;
}

//---------------------------------------------------------------------------//
/*!
  \brief (Host/Device) Pack up the data bits where the corresponding bit of the
  mask is 1
  \param mask mask value
  \param data integer to be packed
*/
KOKKOS_INLINE_FUNCTION
constexpr int bitPack( const uint64_t mask, const uint64_t data )
{
    uint64_t slresult = 0;
    uint64_t& ulresult{ slresult };
    uint64_t uldata = data;
    int count = 0;
    ulresult = 0;

    uint64_t rmask = binaryReverse( mask );
    unsigned char lz{ 0 };

    while ( rmask )
    {
        lz = countLeadingZeros( rmask );
        uldata >>= lz;
        ulresult <<= 1;
        count++;
        ulresult |= ( uldata & 1 );
        uldata >>= 1;
        rmask <<= lz + 1;
    }
    ulresult <<= 64 - count; // 64 bits (maybe not use a constant 64 ...?)
    ulresult = binaryReverse( ulresult );
    return (int)slresult;
}

//---------------------------------------------------------------------------//
/*!
  \brief (Host/Device) Spread out the data bits where the corresponding bit of
  the mask is 1
  \param mask mask value
  \param data integer to be spreaded
*/
KOKKOS_INLINE_FUNCTION
constexpr uint64_t bitSpread( const uint64_t mask, const int data )
{
    uint64_t rmask = binaryReverse( mask );
    int dat = data;
    uint64_t result = 0;
    unsigned char lz{ 0 };
    while ( rmask )
    {
        lz = countLeadingZeros( rmask ) + 1;
        result = result << lz | ( dat & 1 );
        dat >>= 1, rmask <<= lz;
    }
    result = binaryReverse( result ) >> countLeadingZeros( mask );
    return result;
}

//---------------------------------------------------------------------------//
// Tile Hash Key (hash) <=> TileID.
//---------------------------------------------------------------------------//
template <typename Key, HashTypes HashT>
struct TileID2HashKey;

template <typename Key, HashTypes HashT>
struct HashKey2TileID;

/*!
  \brief Compute the hash key from the 3D tile ijk

  Lexicographical order specialization
  Can be rewrite in a recursive way
  \tparam Key key type
*/
template <typename Key>
struct TileID2HashKey<Key, HashTypes::Naive>
{
    //! ID to hash conversion type.
    using tid_to_key_type = TileID2HashKey<Key, HashTypes::Naive>;
    //! Hash type.
    static constexpr HashTypes hash_type = HashTypes::Naive;

    //! Constructor (Host) from a given initializer list
    TileID2HashKey( std::initializer_list<int> tnum )
    {
        std::copy( tnum.begin(), tnum.end(), _tile_num.data() );
        _tile_num_j_mul_k = _tile_num[1] * _tile_num[2];
    }
    //! Constructor (Host) from three integers
    TileID2HashKey( int i, int j, int k )
    {
        _tile_num[0] = i;
        _tile_num[1] = j;
        _tile_num[2] = k;
        _tile_num_j_mul_k = j * k;
    }

    /*!
      \brief (Device) Compute the 1D hash key of tile in a lexicographical way
      \param tile_i, tile_j, tile_k Tile ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    Key operator()( int tile_i, int tile_j, int tile_k ) const
    {
        return tile_i * _tile_num_j_mul_k + tile_j * _tile_num[2] + tile_k;
    }

  private:
    // The index bounds of the tiles on the current MPI rank
    Kokkos::Array<int, 3> _tile_num;
    int _tile_num_j_mul_k;
};

/*!
  \brief Compute the hash key from the 3D tile ijk

  Morton code specialization
*/
template <typename Key>
struct TileID2HashKey<Key, HashTypes::Morton>
{
    //! ID to hash conversion type.
    using tid_to_key_type = TileID2HashKey<Key, HashTypes::Morton>;
    //! Hash type.
    static constexpr HashTypes hash_type = HashTypes::Morton;

    //! Constructor (Host) from a given initializer list
    TileID2HashKey( std::initializer_list<int> tnum )
    {
        std::copy( tnum.begin(), tnum.end(), _tile_num.data() );
    }
    //! Constructor (Host) from three integers
    TileID2HashKey( int i, int j, int k )
    {

        _tile_num[0] = i;
        _tile_num[1] = j;
        _tile_num[2] = k;
    }
    //! Page mask used for computing the morton code
    enum : uint64_t
    { // hand-coded now, can be improved by iterative func
        page_kmask = ( 0x9249249249249249UL ),
        page_jmask = ( 0x2492492492492492UL ),
        page_imask = ( 0x4924924924924924UL )
    };

    /*!
      \brief (Device) Compute the 1D hash key of tile in a morton way
      \param tile_i, tile_j, tile_k Tile ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    Key operator()( int tile_i, int tile_j, int tile_k ) const
    {
        return bitSpread( page_kmask, tile_k ) |
               bitSpread( page_jmask, tile_j ) |
               bitSpread( page_imask, tile_i );
    }

  private:
    // The index bounds of the tiles on the current MPI rank
    Kokkos::Array<int, 3> _tile_num;
};

/*!
  \brief Compute the 3D tile ijk from the hash key

  Lexicographical order specialization
  Can be rewrite in a recursive way
*/
template <typename Key>
struct HashKey2TileID<Key, HashTypes::Naive>
{
    //! ID to hash conversion type.
    using key_to_tid_type = HashKey2TileID<Key, HashTypes::Naive>;
    //! Hash type.
    static constexpr HashTypes hash_type = HashTypes::Naive;

    //! Constructor (Host) from a given initializer list
    HashKey2TileID( std::initializer_list<int> tnum )
        : _tile_num()
    {
        std::copy( tnum.begin(), tnum.end(), _tile_num.data() );
    }
    //! Constructor (Host) from three integers
    HashKey2TileID( int i, int j, int k )
    {
        _tile_num[0] = i;
        _tile_num[1] = j;
        _tile_num[2] = k;
    }

    /*!
      \brief (Device) Compute the tile ijk from the lexicographical hash key
      \param tile_key input - tile hash key number
      \param tile_i, tile_j, tile_k output - Tile ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    void operator()( Key tile_key, int& tile_i, int& tile_j, int& tile_k ) const
    {
        tile_k = tile_key % _tile_num[2];
        tile_j = static_cast<Key>( tile_key / _tile_num[2] ) % _tile_num[1];
        tile_i = static_cast<Key>( tile_key / _tile_num[2] / _tile_num[1] ) %
                 _tile_num[0];
    }

  private:
    // The index bounds of the tiles on the current MPI rank
    Kokkos::Array<int, 3> _tile_num;
};

/*!
  \brief Compute the 3D tile ijk from the hash key

  Morton code specialization
*/
template <typename Key>
struct HashKey2TileID<Key, HashTypes::Morton>
{
    //! ID to hash conversion type.
    using key_to_tid_type = HashKey2TileID<Key, HashTypes::Morton>;
    //! Hash type.
    static constexpr HashTypes hash_type = HashTypes::Morton;

    //! Constructor (Host) from a given initializer list
    HashKey2TileID( std::initializer_list<int> tnum )
    {
        std::copy( tnum.begin(), tnum.end(), _tile_num.data() );
    }
    //! Constructor (Host) from three integers
    HashKey2TileID( int i, int j, int k )
    {
        _tile_num[0] = i;
        _tile_num[1] = j;
        _tile_num[2] = k;
    }
    //! Page mask used for computing the morton code
    enum : uint64_t
    { // hand-coded now, can be improved by iterative func
        page_kmask = ( 0x9249249249249249UL ),
        page_jmask = ( 0x2492492492492492UL ),
        page_imask = ( 0x4924924924924924UL )
    };

    /*!
      \brief (Device) Compute the tile ijk from the lexicographical hash key
      \param tile_key input - tile hash key number
      \param tile_i, tile_j, tile_k output - Tile ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    void operator()( Key tile_key, int& tile_i, int& tile_j, int& tile_k ) const
    {
        tile_k = bitPack( page_kmask, tile_key );
        tile_j = bitPack( page_jmask, tile_key );
        tile_i = bitPack( page_imask, tile_key );
    }

  private:
    // The index bounds of the tiles on the current MPI rank
    Kokkos::Array<int, 3> _tile_num;
};

//---------------------------------------------------------------------------//
// Hierarchical index spaces
// SparseMap <- BlockMap <- TileMap
// Naming:
//      Block = MPI Rank
//      Tile = Sub-block with several cells
//      Cell = Basic grid unit
//---------------------------------------------------------------------------//

//! Declaration of BlockMap
template <typename MemorySpace, unsigned long long CBits,
          unsigned long long CNumPerDim, unsigned long long CNumPerTile,
          HashTypes Hash, typename Key, typename Value>
class BlockMap;

//! Declaration of TileMap
template <int CBits, int CNumPerDim, int CNumPerTile>
class TileMap;

/*!
  \brief Sparse index space, with a hierarchical structure (cell->tile->block)
  \tparam MemorySpace Memory space to store the Map(Hash Table)
  \tparam CellPerTileDim Cell number inside each tile per dimension
  \tparam Hash Hash type (lexicographical or morton)
  \tparam Key Type of the tile/cell hash key
  \tparam Value Type of the tile/cell No.
 */
template <typename MemorySpace, unsigned long long CellPerTileDim = 4,
          HashTypes Hash = HashTypes::Naive, typename Key = uint64_t,
          typename Value = uint64_t>
class SparseMap
{
  public:
    //! Number of dimensions, 3 = ijk
    static constexpr int rank = 3;
    //! Number of bits (per dimension) needed to index the cells inside a tile
    static constexpr unsigned long long cell_bits_per_tile_dim =
        bitCount( CellPerTileDim );
    //! Number of cells inside each tile (per dimension), tile size reset to
    //! power of 2
    static constexpr unsigned long long cell_num_per_tile_dim =
        1 << cell_bits_per_tile_dim;
    //! Cell mask (per dimension), indicating which part of the tile
    //! address(binary bits) will index the cells inside each tile
    static constexpr unsigned long long cell_mask_per_tile_dim =
        ( 1 << cell_bits_per_tile_dim ) - 1;
    //! Number of bits (total) needed to index the cells inside a tile
    static constexpr unsigned long long cell_bits_per_tile =
        cell_bits_per_tile_dim + cell_bits_per_tile_dim +
        cell_bits_per_tile_dim;
    //! Number of cells (total) inside each tile
    static constexpr unsigned long long cell_num_per_tile =
        cell_num_per_tile_dim * cell_num_per_tile_dim * cell_num_per_tile_dim;
    //! Tile hash key type.
    using key_type = Key;
    //! Tile number type.
    using value_type = Value;
    //! Hash table type.
    static constexpr HashTypes hash_type = Hash;

    /*!
      \brief (Host) Constructor
      \param size The size of the block (MPI rank) (Unit: cell)
      \param pre_alloc_size Expected capacity of the allocator to store the
      tiles when tile nums exceed the capacity
    */
    SparseMap( const std::array<int, rank> size,
               const unsigned int pre_alloc_size )
        : _block_id_space( size[0] >> cell_bits_per_tile_dim,
                           size[1] >> cell_bits_per_tile_dim,
                           size[2] >> cell_bits_per_tile_dim,
                           1 << bitCount( pre_alloc_size ) )
    {
        std::fill( _min.data(), _min.data() + rank, 0 );
        std::copy( size.begin(), size.end(), _max.data() );
    }

    /*!
      \brief (Device) Insert a cell
      \param cell_i, cell_j, cell_k Cell ID in each dimension

      Given a cell ijk, insert the tile where the cell reside in to hash table;
      Note that the ijk should be global
    */
    KOKKOS_INLINE_FUNCTION
    void insertCell( int cell_i, int cell_j, int cell_k ) const
    {
        insertTile( cell_i >> cell_bits_per_tile_dim,
                    cell_j >> cell_bits_per_tile_dim,
                    cell_k >> cell_bits_per_tile_dim );
    }

    /*!
      \brief (Device) Insert a tile (to hash table); Note that the tile ijk
      should be global
      \param tile_i, tile_j, tile_k Tile ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    void insertTile( int tile_i, int tile_j, int tile_k ) const
    {
        _block_id_space.insert( tile_i, tile_j, tile_k );
    }

    /*!
      \brief (Device) Query the 1D tile key from the 3D cell ijk
      \param cell_i, cell_j, cell_k Cell ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    value_type queryTile( int cell_i, int cell_j, int cell_k ) const
    {
        // query the tile No.
        auto tile_id = _block_id_space.find( cell_i >> cell_bits_per_tile_dim,
                                             cell_j >> cell_bits_per_tile_dim,
                                             cell_k >> cell_bits_per_tile_dim );
        return tile_id;
    }

    /*!
      \brief (Device) Query the 1D cell key from the 3D cell ijk
      \param cell_i, cell_j, cell_k Cell ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    value_type queryCell( int cell_i, int cell_j, int cell_k ) const
    {
        // query the tile No.
        auto tile_id = queryTile( cell_i, cell_j, cell_k );
        auto cell_id = _tile_id_space.coordToOffset(
            cell_i & cell_mask_per_tile_dim, cell_j & cell_mask_per_tile_dim,
            cell_k & cell_mask_per_tile_dim );
        return static_cast<value_type>( ( tile_id << cell_bits_per_tile ) |
                                        cell_id );
    }

    /*!
      \brief (Host) Clear tile hash table (required by unordered map clear())
    */
    void clear() { _block_id_space.clear(); }

    /*!
      \brief (Host) Set new capacity lower bound on the unordered map
      \param capacity New capacity lower bound
    */
    bool reserve( const value_type capacity )
    {
        return _block_id_space.reserve( capacity );
    }

    /*!
     \brief (Host/Device) Require capacity of the index hash table
    */
    KOKKOS_INLINE_FUNCTION
    uint32_t capacity() const { return _block_id_space.capacity(); }

    /*!
      \brief (Host) Valid tile number inside current block (MPI rank)
    */
    value_type size() const { return _block_id_space.validTileNumHost(); }

    /*!
      \brief (Device) Valid block at index
      \param index index number in Kokkos unordered_map
    */
    KOKKOS_INLINE_FUNCTION
    bool valid_at( uint32_t index ) const
    {
        return _block_id_space.valid_at( index );
    }

    /*!
      \brief (Device) Get block key at index
      \param index index number in Kokkos unordered_map
    */
    KOKKOS_INLINE_FUNCTION
    key_type key_at( uint32_t index ) const
    {
        return _block_id_space.key_at( index );
    }

    /*!
      \brief (Device) get tile id value at index
      \param index index value in kokkos unordered map
    */
    KOKKOS_INLINE_FUNCTION
    value_type value_at( uint32_t index ) const
    {
        return _block_id_space.value_at( index );
    }

    /*!
      \brief (Device) Transfer block hash key to block ijk
      \param key Tile hash key
      \param tile_i, tile_j, tile_k Tile ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    void key2ijk( key_type& key, int& tile_i, int& tile_j, int& tile_k ) const
    {
        return _block_id_space.key2ijk( key, tile_i, tile_j, tile_k );
    }

  private:
    //! block index space, map tile ijk to tile
    BlockMap<MemorySpace, cell_bits_per_tile_dim, cell_num_per_tile_dim,
             cell_num_per_tile, hash_type, key_type, value_type>
        _block_id_space;
    //! tile index space, map cell ijk to cell local No inside a tile
    TileMap<cell_bits_per_tile_dim, cell_num_per_tile_dim, cell_num_per_tile>
        _tile_id_space;
    //! space size (global), channel size
    Kokkos::Array<int, rank> _min;
    Kokkos::Array<int, rank> _max;
};

//---------------------------------------------------------------------------//
//! Creation function for SparseMap from GlobalMesh<SparseMesh>
template <typename MemorySpace, class Scalar,
          unsigned long long CellPerTileDim = 4,
          HashTypes Hash = HashTypes::Naive, typename Key = uint64_t,
          typename Value = uint64_t>
SparseMap<MemorySpace, CellPerTileDim, Hash, Key, Value> createSparseMap(
    const std::shared_ptr<GlobalMesh<SparseMesh<Scalar>>>& global_mesh,
    int pre_alloc_size )
{
    return SparseMap<MemorySpace, CellPerTileDim, Hash, Key, Value>(
        { global_mesh->globalNumCell( Dim::I ),
          global_mesh->globalNumCell( Dim::J ),
          global_mesh->globalNumCell( Dim::K ) },
        pre_alloc_size );
}

//---------------------------------------------------------------------------//
/*!
  \brief Block index space, mapping tile ijks to tile No. through a hash table
  (Kokkos unordered map), note that the ijks should be global
  \tparam CBits Number of bits (per dimension) to index the cells inside a tile
  \tparam CNumPerDim Number of cells (per dimension) inside each tile
  \tparam CNumPerTile Number of cells (total) inside each tile
  \tparam Hash Hash type (lexicographical or morton)
  \tparam Key Type of the tile/cell hash key
  \tparam Value Type of the tile/cell No.
*/
template <typename MemorySpace, unsigned long long CBits,
          unsigned long long CNumPerDim, unsigned long long CNumPerTile,
          HashTypes Hash, typename Key, typename Value>
class BlockMap
{
  public:
    //! Number of bits (per dimension) needed to index the cells inside a tile
    static constexpr unsigned long long cell_bits_per_tile_dim = CBits;
    //! Number of cells (per dimension) inside each tile
    static constexpr unsigned long long cell_num_per_tile_dim = CNumPerDim;
    //! Number of cells (total) inside each tile
    static constexpr unsigned long long cell_num_per_tile = CNumPerTile;
    //! Tile hash key type.
    using key_type = Key;
    //! Tile number type.
    using value_type = Value;
    //! Hash table type.
    static constexpr HashTypes hash_type = Hash;
    //! Self type.
    using bis_Type =
        BlockMap<MemorySpace, cell_bits_per_tile_dim, cell_num_per_tile_dim,
                 cell_num_per_tile, hash_type, key_type, value_type>; // itself

    /*!
      \brief (Host) Constructor
      \param size_x, size_y, size_z The size of the block (MPI rank) in each
      dimension (Unit: tile)
      \param pre_alloc_size Expected capacity of the
      allocator to store the tiles when tile nums exceed the capcity
    */
    BlockMap( const int size_x, const int size_y, const int size_z,
              const value_type pre_alloc_size )
        : _tile_table_info( "hash_table_info" )
        , _tile_table( size_x * size_y * size_z )
        , _op_ijk2key( size_x, size_y, size_z )
        , _op_key2ijk( size_x, size_y, size_z )
    {
        // hash table related init
        auto tile_table_info_mirror =
            Kokkos::create_mirror_view( Kokkos::HostSpace(), _tile_table_info );
        tile_table_info_mirror( 0 ) = 0;
        tile_table_info_mirror( 1 ) = pre_alloc_size;
        Kokkos::deep_copy( _tile_table_info, tile_table_info_mirror );

        // size related init
        _block_size[0] = size_x;
        _block_size[1] = size_y;
        _block_size[2] = size_z;
    }

    /*!
      \brief (Host) Clear tile hash table (Host, required by unordered map
      clear())
    */
    void clear()
    {
        _tile_table.clear(); // clear hash table
        auto tile_table_info_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _tile_table_info );
        tile_table_info_mirror( 0 ) = 0;
        Kokkos::deep_copy( _tile_table_info,
                           tile_table_info_mirror ); // clear valid size record
    }

    /*!
      \brief (Host) Set new capacity lower bound on the unordered map
             (Only for hash table, not for capacity of allocation)
      \param capacity New capacity lower bound of the unordered map
    */
    bool reserve( const value_type capacity )
    {
        return _tile_table.rehash( capacity );
    }

    /*!
     \brief (Host/Device) Require capacity of the index hash table
    */
    KOKKOS_INLINE_FUNCTION
    uint32_t capacity() const
    {
        return _tile_table.capacity(); // hash_table capacity
    }

    /*!
      \brief (Device) Valid tile at index.
      \param index index number in Kokkos unordered map
    */
    KOKKOS_INLINE_FUNCTION
    bool valid_at( uint32_t index ) const
    {
        return _tile_table.valid_at( index );
    }

    /*!
      \brief (Device) Get tile key at index.
      \param index index number in Kokkos unordered map
    */
    KOKKOS_INLINE_FUNCTION
    key_type key_at( uint32_t index ) const
    {
        return _tile_table.key_at( index );
    }

    /*!
      \brief (Device) get block id value at index
      \param index index value in kokkos unordered map
    */
    KOKKOS_INLINE_FUNCTION
    key_type value_at( uint32_t index ) const
    {
        return _tile_table.value_at( index );
    }

    /*!
      \brief (Device) Valid tile number inside current block (MPI rank)
    */
    KOKKOS_INLINE_FUNCTION
    value_type validTileNumDev() const { return _tile_table_info( 0 ); }

    /*!
      \brief (Host) Valid tile number inside current block (MPI rank)
    */
    value_type validTileNumHost() const
    {
        auto tile_table_info_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), _tile_table_info );
        return tile_table_info_mirror( 0 );
    }

    /*!
      \brief (Device) Insert a tile into the hash table
      \param tile_i, tile_j, tile_k Tile ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    void insert( int tile_i, int tile_j, int tile_k ) const
    {
        // Note: reallocation will be performed after insert
        // tile ijk => tile key
        key_type tile_key = ijk2key( tile_i, tile_j, tile_k );
        // Try to insert the tile key into the map. Give it a dummy value for
        // now.
        auto insert_result = _tile_table.insert( tile_key, 0 );
        // If the tile key was actually inserted, atomically increment the
        // counter Only threads that actually did a successful insert will call
        // it
        if ( !insert_result.existing() )
            _tile_table.value_at( insert_result.index() ) =
                Kokkos::atomic_fetch_add( &( _tile_table_info( 0 ) ), 1 );
    }

    /*!
      \brief (Device) Query the tile No. from the hash table
      \param tile_i, tile_j, tile_k Tile ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    value_type find( int tile_i, int tile_j, int tile_k ) const
    {
        return _tile_table.value_at(
            _tile_table.find( ijk2key( tile_i, tile_j, tile_k ) ) );
    }

    /*!
      \brief (Device) Transfer tile ijk to tile hash key
      \param tile_i, tile_j, tile_k Tile ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    key_type ijk2key( int tile_i, int tile_j, int tile_k ) const
    {
        return _op_ijk2key( tile_i, tile_j, tile_k );
    }

    /*!
      \brief (Device) Transfer tile hash key to tile ijk
      \param key Tile hash key
      \param tile_i, tile_j, tile_k Tile ID in each dimension
    */
    KOKKOS_INLINE_FUNCTION
    void key2ijk( key_type& key, int& tile_i, int& tile_j, int& tile_k ) const
    {
        return _op_key2ijk( key, tile_i, tile_j, tile_k );
    }

  private:
    //! [0] current valid table size, [1] preserved for pre-allocated tile size
    Kokkos::View<int[2], MemorySpace> _tile_table_info;
    //! current number of tiles inserted to the hash table
    Kokkos::Array<int, 3> _block_size;
    //! hash table (tile hash key => tile No)
    Kokkos::UnorderedMap<key_type, value_type, MemorySpace> _tile_table;
    //! Ops: transfer between tile ijk <=> tile hash key
    TileID2HashKey<key_type, hash_type> _op_ijk2key;
    HashKey2TileID<key_type, hash_type> _op_key2ijk;
};

//---------------------------------------------------------------------------//
/*!
  \brief Tile index space, inside each local tile, mapping cell ijks to cell
  No.(Lexicographical Order)
  \tparam CBits Bits number (per dimension) to index cells inside each tile
  \tparam Cbits Number of bits (per dimension) to index the cells inside a tile
  \tparam CNumPerDim Cell number (per dimension) inside each tile
  \tparam CNumPerTile Cel number (total) inside each tile
*/
template <int CBits, int CNumPerDim, int CNumPerTile>
class TileMap
{
  public:
    //! Number of bits (per dimension) needed to index the cells inside a tile
    static constexpr int cell_bits_per_tile_dim = CBits;
    //! Number of cells (per dimension) inside each tile
    static constexpr int cell_num_per_tile_dim = CNumPerDim;
    //! Number of cells (total) inside each tile
    static constexpr int cell_num_per_tile = CNumPerTile;
    //! Indexing dimension, currently only serves 3D representations
    static constexpr int rank = 3;

    //! Cell ijk <=> Cell Id
    /*!
      \brief (Host/Device) Compute from coordinate to offset
      \param coords coordinates used to compute the offset
    */
    template <typename... Coords>
    KOKKOS_INLINE_FUNCTION static constexpr auto
    coordToOffset( Coords&&... coords ) -> uint64_t
    {
        return fromCoord<Coord2OffsetDim>( std::forward<Coords>( coords )... );
    }

    /*!
      \brief (Host/Device) Compute from offset to coordinates
      \param key The given single number
      \param coords The output coordinates
    */
    template <typename Key, typename... Coords>
    KOKKOS_INLINE_FUNCTION static constexpr void
    offsetToCoord( Key&& key, Coords&... coords )
    {
        toCoord<Offset2CoordDim>( std::forward<Key>( key ), coords... );
    }

  private:
    //! Coord  <=> Offset Computations
    /*!
      \brief Transfer function: tile ijk to tile key
      \param dim_no dimension id
      \param i coordinate in dimension dim_no
    */
    struct Coord2OffsetDim
    {
        template <typename Coord>
        KOKKOS_INLINE_FUNCTION constexpr auto operator()( int dim_no,
                                                          Coord&& i )
            -> uint64_t
        {
            uint64_t result = static_cast<uint64_t>( i );
            for ( int i = 0; i < dim_no; i++ )
                result *= cell_num_per_tile_dim;
            return result;
        }
    };

    /*!
      \brief Transfer function: tile key to tile ijk
      \param i coordinate to be computed
      \param offset input offset number
    */
    struct Offset2CoordDim
    {
        template <typename Coord, typename Key>
        KOKKOS_INLINE_FUNCTION constexpr auto operator()( Coord& i,
                                                          Key&& offset )
            -> uint64_t
        {
            i = offset % cell_num_per_tile_dim;
            return ( offset / cell_num_per_tile_dim );
        }
    };

    /*!
      \brief Compute a single number from given coordinates
      \tparam Func Transfer fuctions from coords to the single number
      \param coords Input coordinates
    */
    template <typename Func, typename... Coords>
    KOKKOS_INLINE_FUNCTION static constexpr auto fromCoord( Coords&&... coords )
        -> uint64_t
    {
        static_assert( sizeof...( Coords ) == rank,
                       "Dimension of coordinate mismatch" );
        using integer = std::common_type_t<Coords...>;
        static_assert( std::is_integral<integer>::value,
                       "Coordinate should be integral type" );
        return fromCoordImpl<Func>( 0, coords... );
    }

    /*!
      \brief Compute the coordinates from a given single number;
             The dimension of the coordinate is determined by the input param
      number
      \tparam Func Transfer fuctions from the single number to coords
      \param key The given single number
      \param coords The output coordinates
    */
    template <typename Func, typename Key, typename... Coords>
    KOKKOS_INLINE_FUNCTION static constexpr void toCoord( Key&& key,
                                                          Coords&... coords )
    {
        static_assert( sizeof...( Coords ) == rank,
                       "Dimension of coordinate mismatch" );
        using integer = std::common_type_t<Coords...>;
        static_assert( std::is_integral<integer>::value,
                       "Coordinate should be integral type" );
        return toCoordImpl<Func>( 0, std::forward<Key>( key ), coords... );
    }

    /*!
      \brief Implementation (coords => single number)
      \param dim_no Current coord dimension
      \param i The coord value
    */
    template <typename Func, typename Coord>
    KOKKOS_INLINE_FUNCTION static constexpr auto fromCoordImpl( int&& dim_no,
                                                                Coord&& i )
    {
        return Func()( dim_no, std::forward<Coord>( i ) );
    }

    /*!
      \brief Implementation (coords => single number)
      \param dim_no Current coord dimension
      \param i The coord value
      \param is The rest of the coord values
    */
    template <typename Func, typename Coord, typename... Coords>
    KOKKOS_INLINE_FUNCTION static constexpr auto
    fromCoordImpl( int&& dim_no, Coord&& i, Coords&&... is )
    {
        auto result = Func()( dim_no, std::forward<Coord>( i ) );
        if ( dim_no + 1 < rank )
            result += fromCoordImpl<Func>( dim_no + 1,
                                           std::forward<Coords>( is )... );
        return result;
    }

    /*!
      \brief Implementation (single number => coords)
      \param dim_no Current coord dimension
      \param key The input single number
      \param i The output coord value
    */
    template <typename Func, typename Key, typename Coord>
    KOKKOS_INLINE_FUNCTION static constexpr void
    toCoordImpl( int&& dim_no, Key&& key, Coord& i ) noexcept
    {
        if ( dim_no < rank )
            Func()( i, std::forward<Key>( key ) );
    }

    /*!
      \brief Implementation (single number => coords)
      \param dim_no Current coord dimension
      \param key The input single number
      \param i The output coord value
      \param is The rest of the coord values
    */
    template <typename Func, typename Key, typename Coord, typename... Coords>
    KOKKOS_INLINE_FUNCTION static constexpr void
    toCoordImpl( int&& dim_no, Key&& key, Coord& i, Coords&... is ) noexcept
    {
        auto new_key = Func()( i, std::forward<Key>( key ) );
        if ( dim_no + 1 < rank )
            toCoordImpl<Func>( dim_no + 1, new_key, is... );
    }
};

template <long N>
class IndexSpace;

//---------------------------------------------------------------------------//
/*!
  \brief Index space with tile as unit; _min and _max forms the tile range.
         Note this is for sparse grid only, mainly used in sparse halo impl.
*/
template <std::size_t N, unsigned long long cellBitsPerTileDim,
          typename std::enable_if_t<( N == 3 ), bool> = true>
class TileIndexSpace : public IndexSpace<N>
{
  public:
    //! dimension
    static constexpr std::size_t Rank = N;
    //! number of bits to represent the local cell id in each tile in each
    //! dimension
    static constexpr unsigned long long cell_bits_per_tile_dim =
        cellBitsPerTileDim;
    //! number of local cells in each tile in each dimension
    static constexpr unsigned long long cell_num_per_tile =
        1 << ( cell_bits_per_tile_dim * Rank );

    //! brief Default constructor.
    TileIndexSpace()
        : IndexSpace<N>()
    {
    }

    //! Other constructors.
    TileIndexSpace( const std::initializer_list<long>& size )
        : IndexSpace<N>( size )
    {
    }

    //! Other constructors.
    TileIndexSpace( const std::initializer_list<long>& min,
                    const std::initializer_list<long>& max )
        : IndexSpace<N>( min, max )
    {
    }

    //! Other constructors.
    template <typename... Params>
    TileIndexSpace( Params&&... pars )
        : IndexSpace<N>( std::forward<Params>( pars )... )
    {
    }

    //! Get the minimum index in a given dimension.
    KOKKOS_INLINE_FUNCTION
    long min( const long dim ) const { return IndexSpace<N>::min( dim ); }

    //! Get the maximum index in a given dimension.
    KOKKOS_INLINE_FUNCTION
    long max( const long dim ) const { return IndexSpace<N>::max( dim ); }

    //! Get the minimum indices in all dimensions.
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<long, Rank> min() const { return IndexSpace<N>::min(); }

    //! Get the maximum indices in all dimensions.
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<long, Rank> max() const { return IndexSpace<N>::max(); }

    //! Get the total number of tiles of the index space.
    KOKKOS_FORCEINLINE_FUNCTION
    long sizeTile() const { return IndexSpace<N>::size(); }

    //! Get the total number of cells of the index space.
    KOKKOS_FORCEINLINE_FUNCTION
    long sizeCell() const { return ( sizeTile() * cell_num_per_tile ); }

    //! Determine if given tile indices is within the range of the index space.
    template <int NSD = N>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<3 == NSD, bool>
    tileInRange( const long tile_i, const long tile_j, const long tile_k ) const
    {
        long index[N] = { tile_i, tile_j, tile_k };
        return IndexSpace<N>::inRange( index );
    }

    //! Determine if given cell indices is within the range of the index space.
    template <int NSD = N>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<3 == NSD, bool>
    cellInRange( const long cell_i, const long cell_j, const long cell_k ) const
    {
        return tileInRange( cell_i >> cell_bits_per_tile_dim,
                            cell_j >> cell_bits_per_tile_dim,
                            cell_k >> cell_bits_per_tile_dim );
    }
};

} // end namespace Cajita
#endif ///< !CAJITA_SPARSE_INDEXSPACE_HPP
