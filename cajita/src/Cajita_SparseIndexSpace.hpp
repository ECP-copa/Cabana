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
template <long N>
class SparseIndexSpace
{
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