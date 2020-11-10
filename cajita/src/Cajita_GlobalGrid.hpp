/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_GLOBALGRID_HPP
#define CAJITA_GLOBALGRID_HPP

#include <Cajita_GlobalMesh.hpp>
#include <Cajita_Partitioner.hpp>
#include <Cajita_Types.hpp>

#include <array>
#include <memory>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Global logical grid.
//---------------------------------------------------------------------------//
template <class MeshType>
class GlobalGrid
{
  public:
    // Mesh type.
    using mesh_type = MeshType;

    /*!
     \brief Constructor.
     \param comm The communicator over which to define the grid.
     \param global_mesh The global mesh data.
     \param periodic Whether each logical dimension is periodic.
     \param partitioner The grid partitioner.
    */
    GlobalGrid( MPI_Comm comm,
                const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
                const std::array<bool, 3>& periodic,
                const Partitioner& partitioner );

    // Destructor.
    ~GlobalGrid();

    // Get the communicator. This communicator was generated with a Cartesian
    // topology.
    MPI_Comm comm() const;

    // Get the global mesh data.
    const GlobalMesh<MeshType>& globalMesh() const;

    // Get whether a given dimension is periodic.
    bool isPeriodic( const int dim ) const;

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
    int globalNumEntity( Cell, const int dim ) const;
    int globalNumEntity( Node, const int dim ) const;
    int globalNumEntity( Face<Dim::I>, const int dim ) const;
    int globalNumEntity( Face<Dim::J>, const int dim ) const;
    int globalNumEntity( Face<Dim::K>, const int dim ) const;
    int globalNumEntity( Edge<Dim::I>, const int dim ) const;
    int globalNumEntity( Edge<Dim::J>, const int dim ) const;
    int globalNumEntity( Edge<Dim::K>, const int dim ) const;

    // Get the owned number of cells in a given dimension of this block.
    int ownedNumCell( const int dim ) const;

    // Get the global offset in a given dimension. This is where our block
    // starts in the global indexing scheme.
    int globalOffset( const int dim ) const;

  private:
    MPI_Comm _cart_comm;
    std::shared_ptr<GlobalMesh<MeshType>> _global_mesh;
    std::array<bool, 3> _periodic;
    std::array<int, 3> _ranks_per_dim;
    std::array<int, 3> _cart_rank;
    std::array<int, 3> _owned_num_cell;
    std::array<int, 3> _global_cell_offset;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a global grid.
  \param comm The communicator over which to define the grid.
  \param global_mesh The global mesh data.
  \param periodic Whether each logical dimension is periodic.
  \param partitioner The grid partitioner.
*/
template <class MeshType>
std::shared_ptr<GlobalGrid<MeshType>> createGlobalGrid(
    MPI_Comm comm, const std::shared_ptr<GlobalMesh<MeshType>>& global_mesh,
    const std::array<bool, 3>& periodic, const Partitioner& partitioner )
{
    return std::make_shared<GlobalGrid<MeshType>>( comm, global_mesh, periodic,
                                                   partitioner );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_GLOBALGRID_HPP
