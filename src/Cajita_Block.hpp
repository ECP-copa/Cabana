/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_BLOCK_HPP
#define CAJITA_BLOCK_HPP

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <vector>
#include <memory>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Local Cartesian grid block.
//---------------------------------------------------------------------------//
class Block
{
  public:

    /*!
      \brief Constructor.
      \param global_grid The global grid from which the block will be
      constructed.
      \param halo_cell_width The number of halo cells surrounding the locally
      owned cells.
    */
    Block( const std::shared_ptr<GlobalGrid>& global_grid,
           const int halo_cell_width );

    // Get the global grid that owns the block.
    const GlobalGrid& globalGrid() const;

    // Get the physical coordinates of the low corner of the block in a given
    // dimension in the given decomposition.
    template<class DecompositionTag>
    double lowCorner( DecompositionTag, const int dim ) const;

    // Get the physical coordinates of the high corner of the block in a given
    // dimension in the given decomposition.
    template<class DecompositionTag>
    double highCorner( DecompositionTag, const int dim ) const;

    // Get the number of cells in the halo.
    int haloWidth() const;

    // Given the relative offsets of a neighbor rank relative to this block's
    // indices get the of the neighbor. If the neighbor rank is out of bounds
    // return -1. Note that in the case of periodic boundaries out of bounds
    // indices are allowed as the indices will be wrapped around the periodic
    // boundary.
    int neighborRank( const int off_i, const int off_j, const int off_k ) const;

    // Get the index space of the block.
    template<class DecompositionTag, class EntityType, class IndexType>
    IndexSpace<3> indexSpace( DecompositionTag, EntityType, IndexType ) const;

    // Given a relative set of indices of a neighbor get the set of local
    // entity indices shared with that neighbor in the given decomposition.
    template<class DecompositionTag, class EntityType>
    IndexSpace<3> sharedIndexSpace( DecompositionTag,
                                    EntityType,
                                    const int off_i,
                                    const int off_j,
                                    const int off_k ) const;

  private:

    // Get the global index space of the block.
    template<class EntityType>
    IndexSpace<3> globalIndexSpace( Own, EntityType ) const;
    template<class EntityType>
    IndexSpace<3> globalIndexSpace( Ghost, EntityType ) const;

    // Get the ghosted shared index space of the block.
    template<class EntityType>
    IndexSpace<3> ghostedSharedIndexSpace( EntityType,
                                           const int off_i,
                                           const int off_j,
                                           const int off_k ) const;

    // Get the face index space of the block.
    template<int Dir>
    IndexSpace<3> faceIndexSpace( Own, Face<Dir>, Local ) const;
    template<int Dir>
    IndexSpace<3> faceIndexSpace( Own, Face<Dir>, Global ) const;
    template<int Dir>
    IndexSpace<3> faceIndexSpace( Ghost, Face<Dir>, Local ) const;
    template<int Dir>
    IndexSpace<3> faceIndexSpace( Ghost, Face<Dir>, Global ) const;

    // Given a relative set of indices of a neighbor get the set of local
    // face indices shared with that neighbor in the given decomposition.
    template<int Dir>
    IndexSpace<3> faceSharedIndexSpace( Own,
                                        Face<Dir>,
                                        const int off_,
                                        const int off_j,
                                        const int off_k ) const;
    template<int Dir>
    IndexSpace<3> faceSharedIndexSpace( Ghost,
                                        Face<Dir>,
                                        const int off_i,
                                        const int off_j,
                                        const int off_k ) const;

  private:

    std::shared_ptr<GlobalGrid> _global_grid;
    int _halo_cell_width;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a block.
  \param global_grid The global grid from which the block will be
  constructed.
  \param halo_cell_width The number of halo cells surrounding the locally
  owned cells.
*/
std::shared_ptr<Block> createBlock(
    const std::shared_ptr<GlobalGrid>& global_grid,
    const int halo_cell_width );

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_BLOCK_HPP
