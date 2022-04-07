/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita.hpp>

#include <iostream>

//---------------------------------------------------------------------------//
// Types example.
//---------------------------------------------------------------------------//
void typesExample()
{
    /*
      Cajita types are simple structs to used build grid and mesh objects. This
      includes dimension tags (i,j,k) and mesh entity tags: cell, node, edge,
      and face. In addition there are simple types to differentiate types of
      indexing, whether an entity belongs to the current rank, and the supported
      meshes variants: uniform, non-uniform, and sparse.
    */
    std::cout << "Cajita Mesh Types Example\n" << std::endl;

    /*
      Dimension indexing can be done with type I,J,K for 3D and I,J for 2D.
      Direct integers can be used instead, e.g. within loops.
    */
    std::cout << "Indices types: " << Cajita::Dim::I << " " << Cajita::Dim::J
              << " " << Cajita::Dim::K << "\n"
              << std::endl;

    /*
      Entities types designate locations within the mesh to store properties
      or perform calculations. Each mesh cell has one "Cell" at the center and
      "Nodes" at each corner (8 for 3D and 4 for 2D).

      Type checkers allow computation only on specific locations in the mesh
      or general meta-programming.
    */
    std::cout << "Is Cell a Cell? " << Cajita::isCell<Cajita::Cell>()
              << std::endl;
    std::cout << "Is Node a Node? " << Cajita::isNode<Cajita::Node>()
              << std::endl;
    std::cout << "Is Cell a Node? " << Cajita::isNode<Cajita::Cell>() << "\n"
              << std::endl;

    /*
      Each mesh cell also has "Faces" (6 in 3D, 4 in 2D) and "Edges" (12 in 3D,
      but none in 2D) which are indexed in each dimension, with similar type
      checkers.
    */
    std::cout << "Is I Edge an Edge? "
              << Cajita::isEdge<Cajita::Edge<Cajita::Dim::I>>() << std::endl;
    std::cout << "Is J Face a Face? "
              << Cajita::isFace<Cajita::Face<Cajita::Dim::J>>() << std::endl;
    std::cout << "Is K Face an Edge? "
              << Cajita::isEdge<Cajita::Face<Cajita::Dim::K>>() << "\n"
              << std::endl;

    /*
      There are also type tags to denote whether a mesh cell belongs to the
      current MPI rank (it is owned by the current rank) or to neighboring rank;
      that is, it is a ghost.

      Cajita::Own
      Cajita::Ghost
    */

    /*
      Related, there are type tags to distinguish local and global indexing:
      local indicies are unique only within a given MPI rank, while global
      indicies are unique across all MPI ranks. For example, a given node might
      have a different local index on different MPI ranks if it is a ghost, but
      it will always have a unique global index on every MPI rank.

      Cajita::Local
      Cajita::Global
    */

    /*
      Cajita supports logically rectilinear meshes (discussed more in the
      following examples). Each type, uniform, non-uniform, and sparse, can be
      built with different scalar type precision and spatial dimension.
    */

    /*
      Create a uniform mesh object which is 3D by default.

      Note: this mesh is empty - the mesh creation functions in the next
      examples are the standard route to creating a filled mesh.
    */
    Cajita::UniformMesh<double> uniform;
    std::cout << "Uniform mesh with dimension " << uniform.num_space_dim
              << std::endl;

    Cajita::NonUniformMesh<float, 2> nonuniform;
    std::cout << "Non-uniform mesh with dimension " << nonuniform.num_space_dim
              << std::endl;

    Cajita::SparseMesh<double, 3> sparse;
    std::cout << "Sparse mesh with dimension " << sparse.num_space_dim
              << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    typesExample();

    return 0;
}

//---------------------------------------------------------------------------//
