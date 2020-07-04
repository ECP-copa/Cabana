#ifndef CAJITA_SPARSE_INDEXSPACE_HPP
#define CAJITA_SPARSE_INDEXSPACE_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <string>

namespace Cajita
{
//---------------------------------------------------------------------------//
/*!
  \class SparseIndexSpace
  \brief Sparse index space, hierarchical structure (cell->block->whole domain)
 */
template <long N, int BlkSizePerDim = 4>
class SparseIndexSpace
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
        for ( decltype( N ) curDim = 0; curDim < N; curDim++ )
            BS *= BlockSize;
        return BS;
    }

    SparseIndexSpace() {}

    // Should support
    // insert
    // query
    // query and insert
    // compare operations?
    // get min/max operations?

  private:
    // hash table (blockId -> blockNo)
    // Valid block number
    // Valid block Ids
    // allocated size
}; // end class SparseIndexSpace

//---------------------------------------------------------------------------//
/*!
  \class SparseBlockLocalIndexSpace
  \brief Block local index space, hierarchical structure (cell->block->whole
  domain) \brief CellNo : uint64_t \brief CellId : (int...)
 */

template <long N, int BlkSizePerDim>
class SparseBlockLocalIndexSpace
{
    // coord <=> offset computations
    struct Coord2OffsetDim
    {
        template <typename Coord>
        constexpr auto operator()( int dimNo, Coord &&i ) -> uint64_t
        {
            uint64_t result = (uint64_t)i;
            for ( int i = 0; i < dimNo; i++ )
                result *= BlkSizePerDim;
            return result;
        }
    };

    struct Offset2CoordDim
    {
        template <typename Coord>
        constexpr auto operator()( int dimNo, Coord &i, const uint64_t &offset )
            -> uint64_t
        {
            i = offset % BlkSizePerDim;
            return static_cast<uint64_t>( offset / BlkSizePerDim );
        }
    };

  protected:
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
        if ( dimNo + 1 < N )
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
        if ( dimNo + 1 < N )
            to_coord_impl<Func>( dimNo + 1, std::forward<Key>( Newkey ),
                                 is... );
    }
}

} // end namespace Cajita

//---------------------------------------------------------------------------//
// execution policies
// range over all possible blocks
// range over all possible cells inside a block

//---------------------------------------------------------------------------//
// create view

// create subview

// appendDimension

#endif ///< ! CAJITA_SPARSE_INDEXSPACE_HPP