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

#ifndef CAJITA_LOCALGRID_HPP
#define CAJITA_LOCALGRID_HPP

#include <Cajita_GlobalGrid.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <memory>
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

    /*!
      \brief Constructor.
      \param global_grid The global grid from which the local grid will be
      constructed.
      \param halo_cell_width The number of halo cells surrounding the locally
      owned cells.
    */
    LocalGrid( const std::shared_ptr<GlobalGrid<MeshType>> &global_grid,
               const int halo_cell_width );

    // Get the global grid that owns the local grid.
    const GlobalGrid<MeshType> &globalGrid() const;

    // Get the number of cells in the halo.
    int haloCellWidth() const;

    // Given the relative offsets of a neighbor rank relative to this local
    // grid's indices get the of the neighbor. If the neighbor rank is out of
    // bounds return -1. Note that in the case of periodic boundaries out of
    // bounds indices are allowed as the indices will be wrapped around the
    // periodic boundary.
    int neighborRank( const int off_i, const int off_j, const int off_k ) const;

    /*
       Get the index space of the local grid.

       Interface has the same structure as:

       template<class DecompositionTag, class EntityType, class IndexType>
       IndexSpace<3>
       indexSpace( DecompositionTag, EntityType, IndexType ) const;
    */

    IndexSpace<3> indexSpace( Own, Cell, Local ) const;
    IndexSpace<3> indexSpace( Ghost, Cell, Local ) const;
    IndexSpace<3> indexSpace( Own, Cell, Global ) const;
    IndexSpace<3> indexSpace( Ghost, Cell, Global ) const;

    IndexSpace<3> indexSpace( Own, Node, Local ) const;
    IndexSpace<3> indexSpace( Ghost, Node, Local ) const;
    IndexSpace<3> indexSpace( Own, Node, Global ) const;
    IndexSpace<3> indexSpace( Ghost, Node, Global ) const;

    IndexSpace<3> indexSpace( Own, Face<Dim::I>, Local ) const;
    IndexSpace<3> indexSpace( Ghost, Face<Dim::I>, Local ) const;
    IndexSpace<3> indexSpace( Own, Face<Dim::I>, Global ) const;
    IndexSpace<3> indexSpace( Ghost, Face<Dim::I>, Global ) const;

    IndexSpace<3> indexSpace( Own, Face<Dim::J>, Local ) const;
    IndexSpace<3> indexSpace( Ghost, Face<Dim::J>, Local ) const;
    IndexSpace<3> indexSpace( Own, Face<Dim::J>, Global ) const;
    IndexSpace<3> indexSpace( Ghost, Face<Dim::J>, Global ) const;

    IndexSpace<3> indexSpace( Own, Face<Dim::K>, Local ) const;
    IndexSpace<3> indexSpace( Ghost, Face<Dim::K>, Local ) const;
    IndexSpace<3> indexSpace( Own, Face<Dim::K>, Global ) const;
    IndexSpace<3> indexSpace( Ghost, Face<Dim::K>, Global ) const;

    IndexSpace<3> indexSpace( Own, Edge<Dim::I>, Local ) const;
    IndexSpace<3> indexSpace( Ghost, Edge<Dim::I>, Local ) const;
    IndexSpace<3> indexSpace( Own, Edge<Dim::I>, Global ) const;
    IndexSpace<3> indexSpace( Ghost, Edge<Dim::I>, Global ) const;

    IndexSpace<3> indexSpace( Own, Edge<Dim::J>, Local ) const;
    IndexSpace<3> indexSpace( Ghost, Edge<Dim::J>, Local ) const;
    IndexSpace<3> indexSpace( Own, Edge<Dim::J>, Global ) const;
    IndexSpace<3> indexSpace( Ghost, Edge<Dim::J>, Global ) const;

    IndexSpace<3> indexSpace( Own, Edge<Dim::K>, Local ) const;
    IndexSpace<3> indexSpace( Ghost, Edge<Dim::K>, Local ) const;
    IndexSpace<3> indexSpace( Own, Edge<Dim::K>, Global ) const;
    IndexSpace<3> indexSpace( Ghost, Edge<Dim::K>, Global ) const;

    /*
       Given a relative set of indices of a neighbor get the set of local
       entity indices shared with that neighbor in the given
       decomposition. Optionally provide a halo width for the shared
       space. This halo width must be less than or equal to the halo width of
       the local grid. The default behavior is to use the halo width of the
       local grid.

       Interface has the same structure as:

       template<class DecompositionTag, class EntityType, class IndexType>
       IndexSpace<3> sharedIndexSpace( DecompositionTag, EntityType,
                                       const int off_i, const int off_j,
                                       const int off_k,
                                       const int halo_width = -1 ) const;
    */

    IndexSpace<3> sharedIndexSpace( Own, Cell, const int off_i, const int off_j,
                                    const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Ghost, Cell, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Own, Node, const int off_i, const int off_j,
                                    const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Ghost, Node, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Own, Face<Dim::I>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Ghost, Face<Dim::I>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Own, Face<Dim::J>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Ghost, Face<Dim::J>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Own, Face<Dim::K>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Ghost, Face<Dim::K>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Own, Edge<Dim::I>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Ghost, Edge<Dim::I>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Own, Edge<Dim::J>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Ghost, Edge<Dim::J>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Own, Edge<Dim::K>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

    IndexSpace<3> sharedIndexSpace( Ghost, Edge<Dim::K>, const int off_i,
                                    const int off_j, const int off_k,
                                    const int halo_width = -1 ) const;

  private:
    // Get the global index space of the local grid.
    template <class EntityType>
    IndexSpace<3> globalIndexSpace( Own, EntityType ) const;
    template <class EntityType>
    IndexSpace<3> globalIndexSpace( Ghost, EntityType ) const;

    // Get the face index space of the local grid.
    template <int Dir>
    IndexSpace<3> faceIndexSpace( Own, Face<Dir>, Local ) const;
    template <int Dir>
    IndexSpace<3> faceIndexSpace( Own, Face<Dir>, Global ) const;
    template <int Dir>
    IndexSpace<3> faceIndexSpace( Ghost, Face<Dir>, Local ) const;
    template <int Dir>
    IndexSpace<3> faceIndexSpace( Ghost, Face<Dir>, Global ) const;

    // Given a relative set of indices of a neighbor get the set of local
    // face indices shared with that neighbor in the given decomposition.
    template <int Dir>
    IndexSpace<3> faceSharedIndexSpace( Own, Face<Dir>, const int off_,
                                        const int off_j, const int off_k,
                                        const int halo_width ) const;
    template <int Dir>
    IndexSpace<3> faceSharedIndexSpace( Ghost, Face<Dir>, const int off_i,
                                        const int off_j, const int off_k,
                                        const int halo_width ) const;

    // Get the edge index space of the local grid.
    template <int Dir>
    IndexSpace<3> edgeIndexSpace( Own, Edge<Dir>, Local ) const;
    template <int Dir>
    IndexSpace<3> edgeIndexSpace( Own, Edge<Dir>, Global ) const;
    template <int Dir>
    IndexSpace<3> edgeIndexSpace( Ghost, Edge<Dir>, Local ) const;
    template <int Dir>
    IndexSpace<3> edgeIndexSpace( Ghost, Edge<Dir>, Global ) const;

    // Given a relative set of indices of a neighbor get the set of local
    // edge indices shared with that neighbor in the given decomposition.
    template <int Dir>
    IndexSpace<3> edgeSharedIndexSpace( Own, Edge<Dir>, const int off_,
                                        const int off_j, const int off_k,
                                        const int halo_width ) const;
    template <int Dir>
    IndexSpace<3> edgeSharedIndexSpace( Ghost, Edge<Dir>, const int off_i,
                                        const int off_j, const int off_k,
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
createLocalGrid( const std::shared_ptr<GlobalGrid<MeshType>> &global_grid,
                 const int halo_cell_width )
{
    return std::make_shared<LocalGrid<MeshType>>( global_grid,
                                                  halo_cell_width );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_LOCALGRID_HPP
