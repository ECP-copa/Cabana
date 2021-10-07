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
  \file Cajita_Array.hpp
  \brief Grid field arrays
*/
#ifndef CAJITA_ARRAY_HPP
#define CAJITA_ARRAY_HPP

#include <Cajita_Array.hpp>

#include <vector>

#include <mpi.h>

namespace Cajita
{

template <class EntityType, class MeshType>
class ArrayLayout;

//---------------------------------------------------------------------------//
/*!
  \brief ArrayLayout partial specialization for 3D sparse mesh.

  \tparam Scalar Mesh floating point type.

  Non-uniform meshes have a list of node locations for each spatial dimension
  which describe a rectilinear mesh that has arbitrary cell sizes - each cell
  can possibly be different.
*/
template <class EntityType, class Scalar>
class ArrayLayout<EntityType, SparseMesh<Scalar, 3>>
{
  public:
    //! Entity type.
    using entity_type = EntityType;

    //! Mesh type.
    using mesh_type = SparseMesh<Sccalar, 3>;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    /*!
      \brief Constructor.
      \param local_grid The local grid over which the layout will be
      constructed.
      \param dofs_per_entity The number of degrees-of-freedom per grid entity.
    */
    ArrayLayout( const std::shared_ptr<LocalGrid<mesh_type>>& local_gird,
                 const int dofs_per_entity )
        : _local_grid( local_grid )
        , _dofs_per_entity( _dofs_per_entity )
    {
    }
 
    //! Get the local grid over which this layout is defined.
    const std::shared_ptr<LocalGrid<MeshType>> localGrid() const
    {
        return _local_grid;
    }

    //! Get the number of degrees-of-freedom on each grid entity.
    int dofsPerEntity() const { return _dofs_per_entity; }

  private:
    std::shared_ptr<LocalGrid<mesh_type>> _local_grid;
    // std::shared_ptr<SparseMap>
    int _dofs_per_entity;

}; // end class SparseArrayLayout


} // end namespace Cajita

#endif