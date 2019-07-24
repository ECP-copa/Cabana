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

#ifndef CAJITA_GLOBALGRID_HPP
#define CAJITA_GLOBALGRID_HPP

#include <Cajita_Domain.hpp>
#include <Cajita_Partitioner.hpp>
#include <Cajita_Types.hpp>

#include <vector>
#include <memory>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Global Cartesian grid.
//---------------------------------------------------------------------------//
class GlobalGrid
{
  public:

    /*!
     \brief Constructor.
     \param comm The communicator over which to define the grid.
     \param domain The domain of the grid.
     \param partitioner The grid partitioner.
     \param cell_size The size of the cells in the grid.
    */
    GlobalGrid( MPI_Comm comm,
                const std::shared_ptr<Domain>& domain,
                const Partitioner& partitioner,
                const double cell_size );

    // Get the communicator. This communicator was generated with a Cartesian
    // topology.
    MPI_Comm comm() const;

    // Get the domain of the grid.
    const Domain& domain() const;

    // Get the number of blocks in each dimension in the global mesh.
    int dimNumBlock( const int dim ) const;

    // Get the total number of blocks.
    int totalNumBlock() const;

    // Get the id of this block in a given dimension.
    int dimBlockId( const int dim ) const;

    // Get the id of this block.
    int blockId() const;

    // Get the MPI rank of a block with the given indices. If the rank is out
    // of bounds and the boundary is not periodic, return -1 to indicate an
    // invalid rank.
    int blockRank( const int i, const int j, const int k ) const;

    // Get the global number of entities in a given dimension.
    template<class EntityType>
    int globalNumEntity( EntityType, const int dim ) const;

    // Get the owned number of cells in a given dimension of this block.
    int ownedNumCell( const int dim ) const;

    // Get the global offset in a given dimension. This is where our block
    // starts in the global indexing scheme.
    int globalOffset( const int dim ) const;

    // Get the cell size.
    double cellSize() const;

  private:

    MPI_Comm _cart_comm;
    std::shared_ptr<Domain> _domain;
    std::vector<int> _ranks_per_dim;
    std::vector<int> _cart_rank;
    std::vector<int> _owned_num_cell;
    std::vector<int> _global_num_cell;
    std::vector<int> _global_cell_offset;
    double _cell_size;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a global grid.
  \param comm The communicator over which to define the grid.
  \param domain The domain of the grid.
  \param partitioner The grid partitioner.
  \param cell_size The size of the cells in the grid.
*/
std::shared_ptr<GlobalGrid>
createGlobalGrid( MPI_Comm comm,
                  const std::shared_ptr<Domain> domain,
                  const Partitioner& partitioner,
                  const double cell_size );

//---------------------------------------------------------------------------//
/*!
  \brief Create a global grid.
  \param comm The communicator over which to define the grid.
  \param partitioner The grid partitioner.
  \param global_low_corner The low corner of the domain in physical space.
  \param global_high_corner The high corner of the domain in physical space.
  \param periodic Whether each logical dimension is periodic.
  \param cell_size The size of the cells in the grid.
*/
std::shared_ptr<GlobalGrid>
createGlobalGrid( MPI_Comm comm,
                  const Partitioner& partitioner,
                  const std::vector<bool>& periodic,
                  const std::vector<double>& global_low_corner,
                  const std::vector<double>& global_high_corner,
                  const double cell_size );

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_GLOBALGRID_HPP
