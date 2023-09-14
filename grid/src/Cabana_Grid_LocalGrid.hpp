/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Grid_LocalGrid.hpp
  \brief Local grid
*/
#ifndef CABANA_GRID_LOCALGRID_HPP
#define CABANA_GRID_LOCALGRID_HPP

#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Grid_Types.hpp>
#include <Cabana_Utils.hpp> // FIXME: remove after next release.

#include <array>
#include <memory>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
/*!
  \brief Local logical grid.
  \tparam MeshType Mesh type: UniformMesh, NonUniformMesh, or SparseMesh
*/
template <class MeshType>
class LocalGrid
{
  public:
    //! Mesh type.
    using mesh_type = MeshType;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    /*!
      \brief Constructor.
      \param global_grid The global grid from which the local grid will be
      constructed.
      \param halo_cell_width The number of halo cells surrounding the locally
      owned cells.
    */
    LocalGrid( const std::shared_ptr<GlobalGrid<MeshType>>& global_grid,
               const int halo_cell_width );

    //! \brief Get the global grid that owns the local grid.
    const GlobalGrid<MeshType>& globalGrid() const;

    //! \brief Get a mutable version of the global grid that own the local grid
    GlobalGrid<MeshType>& globalGrid();

    //! \brief Get the number of cells in the halo.
    int haloCellWidth() const;

    //! \brief Get the total number of local cells per dimension (owned + halo).
    //! \param d Spatial dimension.
    int totalNumCell( const int d ) const;

    /*!
      \brief Get the global index of a neighbor given neighbor rank offsets
      relative to this local grid.

      \param off_ijk %Array of neighbor offset indices.

      If the neighbor rank is out of bounds return -1. Note that in the case of
      periodic boundaries out of bounds indices are allowed as the indices will
      be wrapped around the periodic boundary.
    */
    int neighborRank( const std::array<int, num_space_dim>& off_ijk ) const;

    /*!
      \brief Get the global index of a neighbor given neighbor rank offsets
      relative to this local grid.

      \param off_i, off_j, off_k Neighbor offset index in a given dimension.

      If the neighbor rank is out of bounds return -1. Note that in the case of
      periodic boundaries out of bounds indices are allowed as the indices will
      be wrapped around the periodic boundary.
    */
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int>
    neighborRank( const int off_i, const int off_j, const int off_k ) const;

    /*!
      \brief Get the global index of a neighbor given neighbor rank offsets
      relative to this local grid.

      \param off_i, off_j Neighbor offset index in a given dimension.

      If the neighbor rank is out of bounds return -1. Note that in the case of
      periodic boundaries out of bounds indices are allowed as the indices will
      be wrapped around the periodic boundary.
    */
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, int> neighborRank( const int off_i,
                                                  const int off_j ) const;

    /*!
      \brief Given a decomposition type, entity type, and index type, get the
      contiguous set of indices that span the space of those entities in the
      local domain

      \param t1 Decomposition type: Own or Ghost
      \param t2 Entity type: Cell, Node, Edge, or Face
      \param t3 Index type: Local or Global
    */
    template <class DecompositionTag, class EntityType, class IndexType>
    IndexSpace<num_space_dim> indexSpace( DecompositionTag t1, EntityType t2,
                                          IndexType t3 ) const;

    /*!
      \brief Given the relative offsets of a neighbor rank relative to this
      local grid's indices get the set of local entity indices shared with that
      neighbor in the given decomposition.

      \param t1 Decomposition type: Own or Ghost
      \param t2 Entity: Cell, Node, Edge, or Face
      \param off_ijk %Array of neighbor offset indices.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.
    */
    template <class DecompositionTag, class EntityType>
    IndexSpace<num_space_dim>
    sharedIndexSpace( DecompositionTag t1, EntityType t2,
                      const std::array<int, num_space_dim>& off_ijk,
                      const int halo_width = -1 ) const;

    /*!
      \brief Given the relative offsets of a neighbor rank relative to this
      local grid's indices get the set of local entity indices shared with that
      neighbor in the given decomposition.

      \tparam DecompositionTag Decomposition type: Own or Ghost
      \tparam EntityType Entity: Cell, Node, Edge, or Face

      \param t1 Decomposition type: Own or Ghost
      \param t2 Entity: Cell, Node, Edge, or Face
      \param off_i, off_j, off_k Neighbor offset index in a given dimension.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.
    */
    template <class DecompositionTag, class EntityType,
              std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    sharedIndexSpace( DecompositionTag t1, EntityType t2, const int off_i,
                      const int off_j, const int off_k,
                      const int halo_width = -1 ) const;

    /*!
      \brief Given the relative offsets of a neighbor rank relative to this
      local grid's indices get the set of local entity indices shared with that
      neighbor in the given decomposition.

      \tparam DecompositionTag Decomposition type: Own or Ghost
      \tparam EntityType Entity: Cell, Node, Edge, or Face

      \param t1 Decomposition type: Own or Ghost
      \param t2 Entity: Cell, Node, Edge, or Face
      \param off_i, off_j Neighbor offset index in a given dimension.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.
    */
    template <class DecompositionTag, class EntityType,
              std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, IndexSpace<2>>
    sharedIndexSpace( DecompositionTag t1, EntityType t2, const int off_i,
                      const int off_j, const int halo_width = -1 ) const;

    /*!
      \brief Given the relative offsets of a boundary relative to this local
      grid's indices get the set of local entity indices associated with that
      boundary in the given decomposition.

      \param t1 Decomposition type: Own or Ghost
      \param t2 Entity: Cell, Node, Edge, or Face
      \param off_ijk %Array of neighbor offset indices.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.

      For example, if the Own decomposition is used, the interior entities that
      would be affected by a boundary operation are provided whereas if the
      Ghost decomposition is used the halo entities on the boundary are
      provided.
    */
    template <class DecompositionTag, class EntityType>
    IndexSpace<num_space_dim>
    boundaryIndexSpace( DecompositionTag t1, EntityType t2,
                        const std::array<int, num_space_dim>& off_ijk,
                        const int halo_width = -1 ) const;

    /*!
      \brief Given the relative offsets of a boundary relative to this local
      grid's indices get the set of local entity indices associated with that
      boundary in the given decomposition.

      \param t1 Decomposition type: Own or Ghost
      \param t2 Entity: Cell, Node, Edge, or Face
      \param off_i, off_j, off_k Neighbor offset index in a given dimension.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.

      For example, if the Own decomposition is used, the interior entities that
      would be affected by a boundary operation are provided whereas if the
      Ghost decomposition is used the halo entities on the boundary are
      provided.
    */
    template <class DecompositionTag, class EntityType,
              std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    boundaryIndexSpace( DecompositionTag t1, EntityType t2, const int off_i,
                        const int off_j, const int off_k,
                        const int halo_width = -1 ) const;

    /*!
      \brief Given the relative offsets of a boundary relative to this local
      grid's indices get the set of local entity indices associated with that
      boundary in the given decomposition.

      \param t1 Decomposition type: Own or Ghost
      \param t2 Entity: Cell, Node, Edge, or Face
      \param off_i, off_j Neighbor offset index in a given dimension.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.

      For example, if the Own decomposition is used, the interior entities that
      would be affected by a boundary operation are provided whereas if the
      Ghost decomposition is used the halo entities on the boundary are
      provided.
    */
    template <class DecompositionTag, class EntityType,
              std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, IndexSpace<2>>
    boundaryIndexSpace( DecompositionTag t1, EntityType t2, const int off_i,
                        const int off_j, const int halo_width = -1 ) const;

  private:
    // Helper functions
    template <class OwnedIndexSpace>
    auto getBound( OwnedIndexSpace owned_space, const int upper_lower,
                   const std::array<int, num_space_dim>& off_ijk,
                   const int lower_shift, const int upper_shift ) const;
    template <int Dir, class OwnedIndexSpace>
    auto getBound( OwnedIndexSpace owned_space, const int upper_lower,
                   const std::array<int, num_space_dim>& off_ijk,
                   const int lower_shift_dir, const int lower_shift,
                   const int upper_shift_dir, const int upper_shift ) const;

    // 3D and 2D entity types
    IndexSpace<num_space_dim> indexSpaceImpl( Own, Cell, Local ) const;
    IndexSpace<num_space_dim> indexSpaceImpl( Ghost, Cell, Local ) const;
    IndexSpace<num_space_dim> indexSpaceImpl( Own, Cell, Global ) const;

    IndexSpace<num_space_dim> indexSpaceImpl( Own, Node, Local ) const;
    IndexSpace<num_space_dim> indexSpaceImpl( Ghost, Node, Local ) const;
    IndexSpace<num_space_dim> indexSpaceImpl( Own, Node, Global ) const;

    IndexSpace<num_space_dim> indexSpaceImpl( Own, Face<Dim::I>, Local ) const;
    IndexSpace<num_space_dim> indexSpaceImpl( Ghost, Face<Dim::I>,
                                              Local ) const;
    IndexSpace<num_space_dim> indexSpaceImpl( Own, Face<Dim::I>, Global ) const;

    IndexSpace<num_space_dim> indexSpaceImpl( Own, Face<Dim::J>, Local ) const;
    IndexSpace<num_space_dim> indexSpaceImpl( Ghost, Face<Dim::J>,
                                              Local ) const;
    IndexSpace<num_space_dim> indexSpaceImpl( Own, Face<Dim::J>, Global ) const;

    // 3D-only entity types.
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> indexSpaceImpl( Own, Face<Dim::K>,
                                                              Local ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
        indexSpaceImpl( Ghost, Face<Dim::K>, Local ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> indexSpaceImpl( Own, Face<Dim::K>,
                                                              Global ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> indexSpaceImpl( Own, Edge<Dim::I>,
                                                              Local ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
        indexSpaceImpl( Ghost, Edge<Dim::I>, Local ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> indexSpaceImpl( Own, Edge<Dim::I>,
                                                              Global ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> indexSpaceImpl( Own, Edge<Dim::J>,
                                                              Local ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
        indexSpaceImpl( Ghost, Edge<Dim::J>, Local ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> indexSpaceImpl( Own, Edge<Dim::J>,
                                                              Global ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> indexSpaceImpl( Own, Edge<Dim::K>,
                                                              Local ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
        indexSpaceImpl( Ghost, Edge<Dim::K>, Local ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> indexSpaceImpl( Own, Edge<Dim::K>,
                                                              Global ) const;

    // 3D and 2D entity types.
    IndexSpace<num_space_dim>
    sharedIndexSpaceImpl( Own, Cell,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    sharedIndexSpaceImpl( Ghost, Cell,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    sharedIndexSpaceImpl( Own, Node,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    sharedIndexSpaceImpl( Ghost, Node,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    sharedIndexSpaceImpl( Own, Face<Dim::I>,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    sharedIndexSpaceImpl( Ghost, Face<Dim::I>,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    sharedIndexSpaceImpl( Own, Face<Dim::J>,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    sharedIndexSpaceImpl( Ghost, Face<Dim::J>,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width = -1 ) const;

    // 3D-only entity types
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    sharedIndexSpaceImpl( Own, Face<Dim::K>, const std::array<int, 3>& off_ijk,
                          const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    sharedIndexSpaceImpl( Ghost, Face<Dim::K>,
                          const std::array<int, 3>& off_ijk,
                          const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    sharedIndexSpaceImpl( Own, Edge<Dim::I>, const std::array<int, 3>& off_ijk,
                          const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    sharedIndexSpaceImpl( Ghost, Edge<Dim::I>,
                          const std::array<int, 3>& off_ijk,
                          const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    sharedIndexSpaceImpl( Own, Edge<Dim::J>, const std::array<int, 3>& off_ijk,
                          const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    sharedIndexSpaceImpl( Ghost, Edge<Dim::J>,
                          const std::array<int, 3>& off_ijk,
                          const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    sharedIndexSpaceImpl( Own, Edge<Dim::K>, const std::array<int, 3>& off_ijk,
                          const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    sharedIndexSpaceImpl( Ghost, Edge<Dim::K>,
                          const std::array<int, 3>& off_ijk,
                          const int halo_width = -1 ) const;

    // 3D and 2D entity types.
    IndexSpace<num_space_dim>
    boundaryIndexSpaceImpl( Own, Cell,
                            const std::array<int, num_space_dim>& off_ijk,
                            const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    boundaryIndexSpaceImpl( Ghost, Cell,
                            const std::array<int, num_space_dim>& off_ijk,
                            const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    boundaryIndexSpaceImpl( Own, Node,
                            const std::array<int, num_space_dim>& off_ijk,
                            const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    boundaryIndexSpaceImpl( Ghost, Node,
                            const std::array<int, num_space_dim>& off_ijk,
                            const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    boundaryIndexSpaceImpl( Own, Face<Dim::I>,
                            const std::array<int, num_space_dim>& off_ijk,
                            const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    boundaryIndexSpaceImpl( Ghost, Face<Dim::I>,
                            const std::array<int, num_space_dim>& off_ijk,
                            const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    boundaryIndexSpaceImpl( Own, Face<Dim::J>,
                            const std::array<int, num_space_dim>& off_ijk,
                            const int halo_width = -1 ) const;

    IndexSpace<num_space_dim>
    boundaryIndexSpaceImpl( Ghost, Face<Dim::J>,
                            const std::array<int, num_space_dim>& off_ijk,
                            const int halo_width = -1 ) const;

    // 3D-only entity types
    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    boundaryIndexSpaceImpl( Own, Face<Dim::K>,
                            const std::array<int, 3>& off_ijk,
                            const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    boundaryIndexSpaceImpl( Ghost, Face<Dim::K>,
                            const std::array<int, 3>& off_ijk,
                            const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    boundaryIndexSpaceImpl( Own, Edge<Dim::I>,
                            const std::array<int, 3>& off_ijk,
                            const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    boundaryIndexSpaceImpl( Ghost, Edge<Dim::I>,
                            const std::array<int, 3>& off_ijk,
                            const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    boundaryIndexSpaceImpl( Own, Edge<Dim::J>,
                            const std::array<int, 3>& off_ijk,
                            const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    boundaryIndexSpaceImpl( Ghost, Edge<Dim::J>,
                            const std::array<int, 3>& off_ijk,
                            const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    boundaryIndexSpaceImpl( Own, Edge<Dim::K>,
                            const std::array<int, 3>& off_ijk,
                            const int halo_width = -1 ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    boundaryIndexSpaceImpl( Ghost, Edge<Dim::K>,
                            const std::array<int, 3>& off_ijk,
                            const int halo_width = -1 ) const;

    // Get the global index space of the local grid.
    template <class EntityType>
    IndexSpace<num_space_dim> globalIndexSpace( Own, EntityType ) const;

    // Get the face index space of the local grid.
    template <int Dir>
    IndexSpace<num_space_dim> faceIndexSpace( Own, Face<Dir>, Local ) const;
    template <int Dir>
    IndexSpace<num_space_dim> faceIndexSpace( Own, Face<Dir>, Global ) const;
    template <int Dir>
    IndexSpace<num_space_dim> faceIndexSpace( Ghost, Face<Dir>, Local ) const;

    // Given a relative set of indices of a neighbor get the set of local
    // face indices shared with that neighbor in the given decomposition.
    template <int Dir>
    IndexSpace<num_space_dim>
    faceSharedIndexSpace( Own, Face<Dir>,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width ) const;

    template <int Dir>
    IndexSpace<num_space_dim>
    faceSharedIndexSpace( Ghost, Face<Dir>,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width ) const;

    // Given the relative offset of a boundary relative to this local grid's
    // get the set of local Dir-direction face indices shared with that
    // boundary in the given decomposition.
    template <int Dir>
    IndexSpace<num_space_dim>
    faceBoundaryIndexSpace( Own, Face<Dir>,
                            const std::array<int, num_space_dim>& off_ijk,
                            const int halo_width ) const;

    template <int Dir>
    IndexSpace<num_space_dim>
    faceBoundaryIndexSpace( Ghost, Face<Dir>,
                            const std::array<int, num_space_dim>& off_ijk,
                            const int halo_width ) const;

    // Get the edge index space of the local grid.
    template <int Dir, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> edgeIndexSpace( Own, Edge<Dir>,
                                                              Local ) const;

    template <int Dir, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> edgeIndexSpace( Own, Edge<Dir>,
                                                              Global ) const;

    template <int Dir, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>> edgeIndexSpace( Ghost, Edge<Dir>,
                                                              Local ) const;

    // Given a relative set of indices of a neighbor get the set of local
    // edge indices shared with that neighbor in the given decomposition.
    template <int Dir, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    edgeSharedIndexSpace( Own, Edge<Dir>, const std::array<int, 3>& off_ijk,
                          const int halo_width ) const;

    template <int Dir, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    edgeSharedIndexSpace( Ghost, Edge<Dir>, const std::array<int, 3>& off_ijk,
                          const int halo_width ) const;

    // Given the relative offset of a boundary relative to this local grid's
    // get the set of local Dir-direction edge indices shared with that
    // boundary in the given decomposition.
    template <int Dir, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    edgeBoundaryIndexSpace( Own, Edge<Dir>, const std::array<int, 3>& off_ijk,
                            const int halo_width ) const;

    template <int Dir, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    edgeBoundaryIndexSpace( Ghost, Edge<Dir>, const std::array<int, 3>& off_ijk,
                            const int halo_width ) const;

  private:
    std::shared_ptr<GlobalGrid<MeshType>> _global_grid;
    int _halo_cell_width;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a local grid.
  \param global_grid The global grid from which the local grid will be
  constructed.
  \param halo_cell_width The number of halo cells surrounding the locally
  owned cells.
  \return Shared pointer to a LocalGrid.
*/
template <class MeshType>
std::shared_ptr<LocalGrid<MeshType>>
createLocalGrid( const std::shared_ptr<GlobalGrid<MeshType>>& global_grid,
                 const int halo_cell_width )
{
    return std::make_shared<LocalGrid<MeshType>>( global_grid,
                                                  halo_cell_width );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

namespace Cajita
{
//! \cond Deprecated
template <class MeshType>
using LocalGrid CAJITA_DEPRECATED = Cabana::Grid::LocalGrid<MeshType>;

template <class... Args>
CAJITA_DEPRECATED auto createLocalGrid( Args&&... args )
{
    return Cabana::Grid::createLocalGrid( std::forward<Args>( args )... );
}
//! \endcond
} // namespace Cajita

//---------------------------------------------------------------------------//
// Template implementations
//---------------------------------------------------------------------------//

#include <Cabana_Grid_LocalGrid_impl.hpp>

//---------------------------------------------------------------------------//

#endif // end CABANA_GRID_LOCALGRID_HPP
