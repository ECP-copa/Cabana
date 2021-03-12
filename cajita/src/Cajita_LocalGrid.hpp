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

#ifndef CAJITA_LOCALGRID_HPP
#define CAJITA_LOCALGRID_HPP

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <array>
#include <memory>
#include <type_traits>
#include <vector>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Local logical grid.
//---------------------------------------------------------------------------//
template <class MeshType>
class LocalGrid
{
  public:
    // Mesh type.
    using mesh_type = MeshType;

    // Spatial dimension.
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

    // Get the global grid that owns the local grid.
    const GlobalGrid<MeshType>& globalGrid() const;

    // Get the number of cells in the halo.
    int haloCellWidth() const;

    // Get the total number of local cells in a given dimension (owned +
    // halo).
    int totalNumCell( const int d ) const;

    // Given the relative offsets of a neighbor rank relative to this local
    // grid's indices get the of the neighbor. If the neighbor rank is out of
    // bounds return -1. Note that in the case of periodic boundaries out of
    // bounds indices are allowed as the indices will be wrapped around the
    // periodic boundary.
    int neighborRank( const std::array<int, num_space_dim>& off_ijk ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, int>
    neighborRank( const int off_i, const int off_j, const int off_k ) const;

    template <std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, int> neighborRank( const int off_i,
                                                  const int off_j ) const;

    // Get the index space for a given combination of decomposition, entity,
    // and index types.
    template <class DecompositionTag, class EntityType, class IndexType>
    IndexSpace<num_space_dim> indexSpace( DecompositionTag, EntityType,
                                          IndexType ) const;

    /*
       Given the relative offsets of a neighbor rank relative to this local
       grid's indices get the set of local entity indices shared with that
       neighbor in the given decomposition. Optionally provide a halo width
       for the shared space. This halo width must be less than or equal to the
       halo width of the local grid. The default behavior is to use the halo
       width of the local grid.
    */
    template <class DecompositionTag, class EntityType>
    IndexSpace<num_space_dim>
    sharedIndexSpace( DecompositionTag, EntityType,
                      const std::array<int, num_space_dim>& off_ijk,
                      const int halo_width = -1 ) const;

    template <class DecompositionTag, class EntityType,
              std::size_t NSD = num_space_dim>
    std::enable_if_t<3 == NSD, IndexSpace<3>>
    sharedIndexSpace( DecompositionTag, EntityType, const int off_i,
                      const int off_j, const int off_k,
                      const int halo_width = -1 ) const;

    template <class DecompositionTag, class EntityType,
              std::size_t NSD = num_space_dim>
    std::enable_if_t<2 == NSD, IndexSpace<2>>
    sharedIndexSpace( DecompositionTag, EntityType, const int off_i,
                      const int off_j, const int halo_width = -1 ) const;

  private:
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

} // end namespace Cajita

//---------------------------------------------------------------------------//
// Template implementations
//---------------------------------------------------------------------------//

#include <Cajita_LocalGrid_impl.hpp>

//---------------------------------------------------------------------------//

#endif // end CAJITA_LOCALGRID_HPP
