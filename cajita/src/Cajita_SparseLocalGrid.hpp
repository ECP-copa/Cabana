/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cajita_SparseLocalGrid.hpp
  \brief Local grid
*/
#ifndef CAJITA_SPARSELOCALGRID_HPP
#define CAJITA_SPARSELOCALGRID_HPP

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <vector>

namespace Cajita
{
template <class MeshType>
class LocalGrid;

//---------------------------------------------------------------------------//
/*!
  \brief Local logical grid - specialization for sparse grid
  \tparam SparseMesh
*/
template <class Scalar, std::size_t NumSpaceDim>
class LocalGrid<SparseMesh<Scalar, NumSpaceDim>>
{
    //! Mesh type.
    using mesh_type = SparseMesh<Scalar, NumSpaceDim>;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    // Constructor
    /*!
         \brief Constructor.
         \param global_grid The global grid from which the local grid will be
         constructed.
         \param halo_cell_width The number of halo cells surrounding the locally
         owned cells.
       */
    LocalGrid( const std::shared_ptr<GlobalGrid<mesh_type>>& global_grid,
               const int halo_cell_width );

    //! \brief Get the global grid that owns the local grid.
    const GlobalGrid<mesh_type>& globalGrid() const;

    //! \brief Get a mutable version of the global grid that own the local grid
    GlobalGrid<mesh_type>& globalGrid();

    //! \brief Get the number of cells in the halo.
    int haloCellWidth() const;

    //! \brief Get the total number of local cells per dimension (owned + halo).
    //! \param d Spatial dimension.
    int totalNumCell( const int d ) const;

    // neighbor Rank

    // index space

    // shared index space

    // boundary index space

  private:
    // indexSpaceImpl
    // 3D and 3D entity types

    // 3D only entity types

    // sharedIndexSpaceImpl
    // 3D and 3D entity types

    // 3D only entity types

    // globalIndexSpace

    // faceIndexSpace

    // faceSharedIndexSpace

    // edgeIndexSpace

    // edgeSharedIndexSpace

  private:
    std::shared_ptr<GlobalGrid<mesh_type>> _global_grid;
    int _halo_cell_width;
}; // class LocalGrid<SparseMesh<Scalar, NumSpaceDim>>

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a local grid (sparse grid specialization).
  \param global_grid The global grid from which the local grid will be
  constructed, the gloabl_grid should be the specialization for sparse girds
  \param halo_cell_width The number of halo cells surrounding the locally
  owned cells.
*/
template <class Scalar, std::size_t NumSpaceDim = 3>
std::shared_ptr<LocalGrid<SparseMap<Scalar, NumSpaceDim>>>
createSparseLocalGrid(
    const std::shared_ptr<GlobalGrid<SparseMap<Scalar, NumSpaceDim>>>&
        global_grid,
    const int halo_cell_width )
{
    return std::make_shared<LocalGrid<SparseMap<Scalar, NumSpaceDim>>>(
        global_grid, halo_cell_width );
}

//---------------------------------------------------------------------------//

}; // end namespace Cajita

//---------------------------------------------------------------------------//
// Template implementations
//---------------------------------------------------------------------------//

#include <Cajita_SparseLocalGrid_impl.hpp>

#endif // ! CAJITA_SPARSELOCALGRID_HPP