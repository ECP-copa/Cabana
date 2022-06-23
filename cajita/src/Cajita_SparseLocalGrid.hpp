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
  \brief Sparse Local grid and related implementations
*/
#ifndef CAJITA_LOCALGRID_SPARSE_HPP
#define CAJITA_LOCALGRID_SPARSE_HPP

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <array>
#include <vector>

namespace Cajita
{
namespace Experimental
{
//---------------------------------------------------------------------------//
template <class MeshType>
class LocalGrid;

//---------------------------------------------------------------------------//
/*!
  \brief Local logical grid specialized for sparse grid.
  \tparam MeshType Mesh type: SparseMesh
  \tparam NumSpaceDim Space Dimension, needed by SparseMesh's template
*/
template <class Scalar, std::size_t NumSpaceDim>
class LocalGrid<SparseMesh<Scalar, NumSpaceDim>>
{
  public:
    //! Mesh type.
    using mesh_type = SparseMesh<Scalar, NumSpaceDim>;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    /*!
      \brief Constructor.
      \param global_grid The global grid from which the local grid will be
      constructed.
      \param halo_cell_width The number of halo cells surrounding the locally
      owned cells.
      \param cell_num_per_tile_dim The number of local cells in each tile in
      each dimension.
    */
    LocalGrid( const std::shared_ptr<GlobalGrid<mesh_type>>& global_grid,
               const int halo_cell_width, const long cell_num_per_tile_dim );

    //! \brief Get the global grid that owns the local grid.
    const GlobalGrid<mesh_type>& globalGrid() const;
    //! \brief Get a mutable version of the global grid that own the local grid
    GlobalGrid<mesh_type>& globalGrid();

    //! \brief Get the number of cells in the halo.
    int haloCellWidth() const;

    //! \brief Get the number of tiles in the halo.
    int haloTileWidth() const;

    //! \brief Get the total number of local cells per dimension (owned + halo).
    //! \param d Spatial dimension.
    int totalNumCell( const int d ) const;

    //! \brief Get the total number of local cells (owned + halo) on this MPI
    //! rank
    int totalNumCell() const;

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
    int neighborRank( const int off_i, const int off_j, const int off_k ) const;

    /*!
     \brief Given a decomposition type, entity type, and index type, get the
     contiguous set of indices that span the space of those entities in the
     local domain (return the general cell index space)

     \param t1 Decomposition type: Own or Ghost
     \param t2 Entity type: Cell, Node, Edge, or Face
     \param t3 Index type: Local or Global
   */
    template <class DecompositionTag, class EntityType, class IndexType>
    IndexSpace<num_space_dim> indexSpace( DecompositionTag t1, EntityType t2,
                                          IndexType t3 ) const;

    /*!
      \brief Given the relative offsets of a neighbor rank, get the set of
      global entity indices shared with that neighbor in the given
      decomposition.

      \param t1 Decomposition type: Own or Ghost
      \param t2 Entity: Cell or Node; TODO: Edge or Face
      \param off_ijk Array of neighbor offset indices.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.
    */
    template <unsigned long long cellBitsPerTileDim, class DecompositionTag,
              class EntityType>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpace( DecompositionTag t1, EntityType t2,
                          const std::array<int, num_space_dim>& off_ijk,
                          const int halo_width = -1 ) const;

    /*!
      \brief Given the relative offsets of a neighbor rank, get the set of
      global entity indices shared with that neighbor in the given
      decomposition.

      \tparam DecompositionTag Decomposition type: Own or Ghost
      \tparam EntityType Entity: Cell, Node, Edge, or Face

      \param t1 Decomposition type: Own or Ghost
      \param t2 Entity: Cell or Node; TODO: Edge or Face
      \param off_i, off_j, off_k Neighbor offset index in a given dimension.
      \param halo_width Optional depth of shared indices within the halo. Must
      be less than or equal to the halo width of the local grid. Default is to
      use the halo width of the local grid.
    */
    template <unsigned long long cellBitsPerTileDim, class DecompositionTag,
              class EntityType, std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, TileIndexSpace<3, cellBitsPerTileDim>>
    sharedTileIndexSpace( DecompositionTag t1, EntityType t2, const int off_i,
                          const int off_j, const int off_k,
                          const int halo_width = -1 ) const;

  private:
    // Node related index space implementations
    template <class DecompositionTag, class IndexType>
    IndexSpace<num_space_dim> indexSpaceImpl( DecompositionTag t1, Node,
                                              IndexType t3 ) const;

    template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpaceImpl( DecompositionTag t1, Node,
                              const std::array<int, num_space_dim>& off_ijk,
                              const int halo_width ) const;

    // Cell related index space implementations
    template <class DecompositionTag, class IndexType>
    IndexSpace<num_space_dim> indexSpaceImpl( DecompositionTag t1, Cell,
                                              IndexType t3 ) const;

    template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpaceImpl( DecompositionTag t1, Cell,
                              const std::array<int, num_space_dim>& off_ijk,
                              const int halo_width ) const;

    // Face related index space implementations
    template <class DecompositionTag, class IndexType>
    IndexSpace<num_space_dim> indexSpaceImpl( DecompositionTag t1, Face<Dim::I>,
                                              IndexType t3 ) const;
    template <class DecompositionTag, class IndexType>
    IndexSpace<num_space_dim> indexSpaceImpl( DecompositionTag t1, Face<Dim::J>,
                                              IndexType t3 ) const;
    template <class DecompositionTag, class IndexType>
    IndexSpace<num_space_dim> indexSpaceImpl( DecompositionTag t1, Face<Dim::K>,
                                              IndexType t3 ) const;

    template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpaceImpl( DecompositionTag t1, Face<Dim::I>,
                              const std::array<int, num_space_dim>& off_ijk,
                              const int halo_width ) const;
    template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpaceImpl( DecompositionTag t1, Face<Dim::J>,
                              const std::array<int, num_space_dim>& off_ijk,
                              const int halo_width ) const;
    template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpaceImpl( DecompositionTag t1, Face<Dim::K>,
                              const std::array<int, num_space_dim>& off_ijk,
                              const int halo_width ) const;
    // Edge related index space implementations1

    template <class DecompositionTag, class IndexType>
    IndexSpace<num_space_dim> indexSpaceImpl( DecompositionTag t1, Edge<Dim::I>,
                                              IndexType t3 ) const;
    template <class DecompositionTag, class IndexType>
    IndexSpace<num_space_dim> indexSpaceImpl( DecompositionTag t1, Edge<Dim::J>,
                                              IndexType t3 ) const;
    template <class DecompositionTag, class IndexType>
    IndexSpace<num_space_dim> indexSpaceImpl( DecompositionTag t1, Edge<Dim::K>,
                                              IndexType t3 ) const;

    template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpaceImpl( DecompositionTag t1, Edge<Dim::I>,
                              const std::array<int, num_space_dim>& off_ijk,
                              const int halo_width ) const;
    template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpaceImpl( DecompositionTag t1, Edge<Dim::J>,
                              const std::array<int, num_space_dim>& off_ijk,
                              const int halo_width ) const;
    template <unsigned long long cellBitsPerTileDim, class DecompositionTag>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpaceImpl( DecompositionTag t1, Edge<Dim::K>,
                              const std::array<int, num_space_dim>& off_ijk,
                              const int halo_width ) const;

    // More genearl implementations for node and cell center indices
    // Note: In Sparse grid, each cell considers the left-most node and its cell
    // center as belongings, i.e., Node and Cell share the same element-to-cell
    // mapping, thus sharse the same interface.
    IndexSpace<num_space_dim> indexSpaceImpl( Own, Local ) const;
    IndexSpace<num_space_dim> indexSpaceImpl( Ghost, Local ) const;
    IndexSpace<num_space_dim> indexSpaceImpl( Own, Global ) const;

    template <unsigned long long cellBitsPerTileDim>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpaceImpl( Own,
                              const std::array<int, num_space_dim>& off_ijk,
                              const int halo_width ) const;
    template <unsigned long long cellBitsPerTileDim>
    TileIndexSpace<num_space_dim, cellBitsPerTileDim>
    sharedTileIndexSpaceImpl( Ghost,
                              const std::array<int, num_space_dim>& off_ijk,
                              const int halo_width ) const;

  private:
    std::shared_ptr<GlobalGrid<mesh_type>> _global_grid;
    int _halo_cell_width;
    const long _cell_num_per_tile_dim;

}; // end LocalGrid<SparseMesh<Scalar, NumSpaceDim>>

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
/*!
  \brief Create a local grid.
  \param global_grid The global grid from which the local grid will be
  constructed.
  \param halo_cell_width The number of halo cells surrounding the locally
  owned cells.
  \param cell_num_per_tile_dim The number of local cells in each tile in each
  dimension.
*/
template <class MeshType>
std::shared_ptr<LocalGrid<MeshType>>
createSparseLocalGrid( const std::shared_ptr<GlobalGrid<MeshType>>& global_grid,
                       const int halo_cell_width,
                       const int cell_num_per_tile_dim )
{
    static_assert( isSparseMesh<MeshType>::value,
                   "createSparseLocalGrid supports only SparseMesh" );
    return std::make_shared<LocalGrid<MeshType>>( global_grid, halo_cell_width,
                                                  cell_num_per_tile_dim );
}

} // namespace Experimental
//---------------------------------------------------------------------------//
} // namespace Cajita

//---------------------------------------------------------------------------//
// Template implementations
//---------------------------------------------------------------------------//

#include <Cajita_SparseLocalGrid_impl.hpp>

//---------------------------------------------------------------------------//
#endif // end CAJITA_LOCALGRID_SPARSE_HPP
