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

#ifndef CAJTIA_GLOBALMESH_HPP
#define CAJTIA_GLOBALMESH_HPP

#include <Cajita_Types.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Forward declaration of global mesh.
template <class MeshType>
class GlobalMesh;

//---------------------------------------------------------------------------//
// Global mesh partial specialization for uniform mesh. Uniform meshes are
// rectilinear meshes where every cell in the mesh is identical. A cell is
// described by its width in each dimension.
template <class MeshType>
class GlobalMesh
{
  public:
    // Mesh type.
    using mesh_type = MeshType;

    // Scalar type.
    using scalar_type = typename mesh_type::scalar_type;

    // Cell size constructor - special case where all cell dimensions are the
    // same.
    GlobalMesh( const std::array<scalar_type, 3> &global_low_corner,
                const std::array<scalar_type, 3> &global_high_corner,
                const scalar_type cell_size )
        : _global_low_corner( global_low_corner )
        , _global_high_corner( global_high_corner )
        , _cell_size( { cell_size, cell_size, cell_size } )
    {
        // Check that the domain is evenly divisible by the cell size in each
        // dimension within round-off error.
        for ( int d = 0; d < 3; ++d )
        {
            scalar_type ext = globalNumCell( d ) * _cell_size[d];
            if ( std::abs( ext - extent( d ) ) >
                 scalar_type( 100.0 ) *
                     std::numeric_limits<scalar_type>::epsilon() )
                throw std::logic_error(
                    "Extent not evenly divisible by uniform cell size" );
        }
    }

    // Cell size constructor - each cell dimension can be different.
    GlobalMesh( const std::array<scalar_type, 3> &global_low_corner,
                const std::array<scalar_type, 3> &global_high_corner,
                const std::array<scalar_type, 3> &cell_size )
        : _global_low_corner( global_low_corner )
        , _global_high_corner( global_high_corner )
        , _cell_size( cell_size )
    {
        // Check that the domain is evenly divisible by the cell size in each
        // dimension within round-off error.
        for ( int d = 0; d < 3; ++d )
        {
            scalar_type ext = globalNumCell( d ) * _cell_size[d];
            if ( std::abs( ext - extent( d ) ) >
                 scalar_type( 100.0 ) *
                     std::numeric_limits<scalar_type>::epsilon() )
                throw std::logic_error(
                    "Extent not evenly divisible by uniform cell size" );
        }
    }

    // Number of global cells constructor.
    GlobalMesh( const std::array<scalar_type, 3> &global_low_corner,
                const std::array<scalar_type, 3> &global_high_corner,
                const std::array<int, 3> &global_num_cell )
        : _global_low_corner( global_low_corner )
        , _global_high_corner( global_high_corner )
    {
        // Compute the cell size in each dimension.
        for ( int d = 0; d < 3; ++d )
            _cell_size[d] = ( _global_high_corner[d] - _global_low_corner[d] ) /
                            global_num_cell[d];

        // Check that the domain is evenly divisible by the cell size in each
        // dimension within round-off error and that we got the expected
        // number of cells.
        for ( int d = 0; d < 3; ++d )
        {
            scalar_type ext = globalNumCell( d ) * _cell_size[d];
            if ( std::abs( ext - extent( d ) ) >
                 scalar_type( 100.0 ) *
                     std::numeric_limits<scalar_type>::epsilon() )
                throw std::logic_error(
                    "Extent not evenly divisible by uniform cell size" );
            if ( globalNumCell( d ) != global_num_cell[d] )
                throw std::logic_error( "Global number of cells mismatch" );
        }
    }

    // GLOBAL MESH INTERFACE

    // Get the global low corner of the mesh.
    scalar_type lowCorner( const int dim ) const
    {
        return _global_low_corner[dim];
    }

    // Get the global high corner of the mesh.
    scalar_type highCorner( const int dim ) const
    {
        return _global_high_corner[dim];
    }

    // Get the extent of a given dimension.
    scalar_type extent( const int dim ) const
    {
        return highCorner( dim ) - lowCorner( dim );
    }

    // Get the global numer of cells in a given dimension.
    int globalNumCell( const int dim ) const
    {
        return std::rint( extent( dim ) / _cell_size[dim] );
    }

    // UNIFORM MESH SPECIFIC

    // Get the uniform cell size in a given dimension.
    scalar_type cellSize( const int dim ) const { return _cell_size[dim]; }

  private:
    std::array<scalar_type, 3> _global_low_corner;
    std::array<scalar_type, 3> _global_high_corner;
    std::array<scalar_type, 3> _cell_size;
};

// Creation function for Uniform Mesh.
template <class Scalar>
std::shared_ptr<GlobalMesh<UniformMesh<Scalar>>>
createUniformGlobalMesh( const std::array<Scalar, 3> &global_low_corner,
                         const std::array<Scalar, 3> &global_high_corner,
                         const Scalar cell_size )
{
    return std::make_shared<GlobalMesh<UniformMesh<Scalar>>>(
        global_low_corner, global_high_corner, cell_size );
}

template <class Scalar>
std::shared_ptr<GlobalMesh<UniformMesh<Scalar>>>
createUniformGlobalMesh( const std::array<Scalar, 3> &global_low_corner,
                         const std::array<Scalar, 3> &global_high_corner,
                         const std::array<Scalar, 3> &cell_size )
{
    return std::make_shared<GlobalMesh<UniformMesh<Scalar>>>(
        global_low_corner, global_high_corner, cell_size );
}

template <class Scalar>
std::shared_ptr<GlobalMesh<UniformMesh<Scalar>>>
createUniformGlobalMesh( const std::array<Scalar, 3> &global_low_corner,
                         const std::array<Scalar, 3> &global_high_corner,
                         const std::array<int, 3> &global_num_cell )
{
    return std::make_shared<GlobalMesh<UniformMesh<Scalar>>>(
        global_low_corner, global_high_corner, global_num_cell );
}

// Creation functions for Sparse Mesh.
template <class Scalar>
std::shared_ptr<GlobalMesh<SparseMesh<Scalar>>>
createSparseGlobalMesh( const std::array<Scalar, 3> &global_low_corner,
                        const std::array<Scalar, 3> &global_high_corner,
                        const Scalar cell_size )
{
    return std::make_shared<GlobalMesh<SparseMesh<Scalar>>>(
        global_low_corner, global_high_corner, cell_size );
}

template <class Scalar>
std::shared_ptr<GlobalMesh<SparseMesh<Scalar>>>
createSparseGlobalMesh( const std::array<Scalar, 3> &global_low_corner,
                        const std::array<Scalar, 3> &global_high_corner,
                        const std::array<Scalar, 3> &cell_size )
{
    return std::make_shared<GlobalMesh<SparseMesh<Scalar>>>(
        global_low_corner, global_high_corner, cell_size );
}

template <class Scalar>
std::shared_ptr<GlobalMesh<SparseMesh<Scalar>>>
createSparseGlobalMesh( const std::array<Scalar, 3> &global_low_corner,
                        const std::array<Scalar, 3> &global_high_corner,
                        const std::array<int, 3> &global_num_cell )
{
    return std::make_shared<GlobalMesh<SparseMesh<Scalar>>>(
        global_low_corner, global_high_corner, global_num_cell );
}

//---------------------------------------------------------------------------//
// Global mesh partial specialization for non-uniform mesh. Non-uniform meshes
// have a list of node locations for each spatial dimension which describe a
// rectilinear mesh that has arbitrary cell sizes - each cell can possibly be
// different.
template <class Scalar>
class GlobalMesh<NonUniformMesh<Scalar>>
{
  public:
    // Mesh type.
    using mesh_type = NonUniformMesh<Scalar>;

    // Scalar type.
    using scalar_type = Scalar;

    // Constructor.
    GlobalMesh( const std::vector<Scalar> &i_edges,
                const std::vector<Scalar> &j_edges,
                const std::vector<Scalar> &k_edges )
        : _edges( { i_edges, j_edges, k_edges } )
    {
    }

    // GLOBAL MESH INTERFACE

    // Get the global low corner of the mesh.
    Scalar lowCorner( const int dim ) const { return _edges[dim].front(); }

    // Get the global high corner of the mesh.
    Scalar highCorner( const int dim ) const { return _edges[dim].back(); }

    // Get the extent of a given dimension.
    Scalar extent( const int dim ) const
    {
        return highCorner( dim ) - lowCorner( dim );
    }

    // Get the global numer of cells in a given dimension.
    int globalNumCell( const int dim ) const { return _edges[dim].size() - 1; }

    // NON-UNIFORM MESH SPECIFIC

    // Get the edge array in a given dimension.
    const std::vector<Scalar> &nonUniformEdge( const int dim ) const
    {
        return _edges[dim];
    }

  private:
    std::array<std::vector<Scalar>, 3> _edges;
};

// Creation function.
template <class Scalar>
std::shared_ptr<GlobalMesh<NonUniformMesh<Scalar>>>
createNonUniformGlobalMesh( const std::vector<Scalar> &i_edges,
                            const std::vector<Scalar> &j_edges,
                            const std::vector<Scalar> &k_edges )
{
    return std::make_shared<GlobalMesh<NonUniformMesh<Scalar>>>(
        i_edges, j_edges, k_edges );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJTIA_GLOBALMESH_HPP
