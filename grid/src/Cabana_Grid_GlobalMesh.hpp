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
  \file Cabana_Grid_GlobalMesh.hpp
  \brief Global mesh
*/
#ifndef CABANA_GRID_GLOBALMESH_HPP
#define CABANA_GRID_GLOBALMESH_HPP

#include <Cabana_Grid_Types.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace Grid
{
//---------------------------------------------------------------------------//
// Forward declaration of global mesh.
template <class MeshType>
class GlobalMesh;

//---------------------------------------------------------------------------//
/*!
  \brief Global mesh partial specialization for uniform mesh.

  Uniform meshes are rectilinear meshes where every cell in the mesh is
  identical. A cell is described by its width in each dimension.

  \tparam MeshType Mesh type: UniformMesh, SparseMesh
*/
template <class MeshType>
class GlobalMesh
{
  public:
    //! Mesh type.
    using mesh_type = MeshType;

    //! Scalar type.
    using scalar_type = typename mesh_type::scalar_type;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    //! \brief Cell size constructor where all cell dimensions are the same.
    GlobalMesh(
        const std::array<scalar_type, num_space_dim>& global_low_corner,
        const std::array<scalar_type, num_space_dim>& global_high_corner,
        const scalar_type cell_size )
        : _global_low_corner( global_low_corner )
        , _global_high_corner( global_high_corner )
    {
        // Check that the domain is evenly divisible by the cell size in each
        // dimension within round-off error.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            _cell_size[d] = cell_size;
            scalar_type ext = globalNumCell( d ) * _cell_size[d];
            if ( std::abs( ext - extent( d ) ) >
                 scalar_type( 100.0 ) *
                     std::numeric_limits<scalar_type>::epsilon() )
                throw std::logic_error(
                    "Cabana::Grid::GlocalMesh: Extent not evenly divisible by "
                    "uniform cell size" );
        }
    }

    //! \brief Cell size constructor - each cell dimension can be different.
    GlobalMesh(
        const std::array<scalar_type, num_space_dim>& global_low_corner,
        const std::array<scalar_type, num_space_dim>& global_high_corner,
        const std::array<scalar_type, num_space_dim>& cell_size )
        : _global_low_corner( global_low_corner )
        , _global_high_corner( global_high_corner )
        , _cell_size( cell_size )
    {
        // Check that the domain is evenly divisible by the cell size in each
        // dimension within round-off error.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            scalar_type ext = globalNumCell( d ) * _cell_size[d];
            if ( std::abs( ext - extent( d ) ) >
                 scalar_type( 100.0 ) *
                     std::numeric_limits<scalar_type>::epsilon() )
                throw std::logic_error(
                    "Cabana::Grid::GlocalMesh: Extent not evenly divisible by "
                    "uniform cell size" );
        }
    }

    //! \brief Number of global cells constructor.
    GlobalMesh(
        const std::array<scalar_type, num_space_dim>& global_low_corner,
        const std::array<scalar_type, num_space_dim>& global_high_corner,
        const std::array<int, num_space_dim>& global_num_cell )
        : _global_low_corner( global_low_corner )
        , _global_high_corner( global_high_corner )
    {
        // Compute the cell size in each dimension.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            _cell_size[d] = ( _global_high_corner[d] - _global_low_corner[d] ) /
                            global_num_cell[d];

        // Check that the domain is evenly divisible by the cell size in each
        // dimension within round-off error and that we got the expected
        // number of cells.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            scalar_type ext = globalNumCell( d ) * _cell_size[d];
            if ( std::abs( ext - extent( d ) ) >
                 scalar_type( 100.0 ) *
                     std::numeric_limits<scalar_type>::epsilon() )
                throw std::logic_error(
                    "Cabana::Grid::GlocalMesh: Extent not evenly divisible by "
                    "uniform cell size" );
            if ( globalNumCell( d ) != global_num_cell[d] )
                throw std::logic_error( "Cabana::Grid::GlocalMesh: Global "
                                        "number of cells mismatch" );
        }
    }

    // GLOBAL MESH INTERFACE

    //! \brief Get the global low corner of the mesh.
    //! \param dim Spatial dimension.
    scalar_type lowCorner( const std::size_t dim ) const
    {
        return _global_low_corner[dim];
    }

    //! \brief Get the global high corner of the mesh.
    //! \param dim Spatial dimension.
    scalar_type highCorner( const std::size_t dim ) const
    {
        return _global_high_corner[dim];
    }

    //! \brief Get the extent of a given dimension.
    //! \param dim Spatial dimension.
    scalar_type extent( const std::size_t dim ) const
    {
        return highCorner( dim ) - lowCorner( dim );
    }

    //! \brief Get the global number of cells in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumCell( const std::size_t dim ) const
    {
        return std::rint( extent( dim ) / _cell_size[dim] );
    }

    // UNIFORM MESH SPECIFIC

    //! \brief Get the uniform cell size in a given dimension.
    //! \param dim Spatial dimension.
    scalar_type cellSize( const std::size_t dim ) const
    {
        return _cell_size[dim];
    }

  private:
    std::array<scalar_type, num_space_dim> _global_low_corner;
    std::array<scalar_type, num_space_dim> _global_high_corner;
    std::array<scalar_type, num_space_dim> _cell_size;
};

/*!
  \brief Create uniform mesh with uniform cell size.

  \tparam Scalar Mesh floating point type.
  \tparam NumSpaceDim Spatial dimension.

  \param global_low_corner, global_high_corner Location of the mesh corner.
  \param cell_size Uniform cell size for every dimension.
  \return Shared pointer to a GlobalMesh.
*/
template <class Scalar, std::size_t NumSpaceDim>
std::shared_ptr<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>
createUniformGlobalMesh(
    const std::array<Scalar, NumSpaceDim>& global_low_corner,
    const std::array<Scalar, NumSpaceDim>& global_high_corner,
    const Scalar cell_size )
{
    return std::make_shared<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>(
        global_low_corner, global_high_corner, cell_size );
}

/*!
  \brief Create uniform mesh with uniform cell size.

  \tparam Scalar Mesh floating point type.
  \tparam NumSpaceDim Spatial dimension.

  \param global_low_corner, global_high_corner Location of the mesh corner.
  \param cell_size %Array ofuniform cell size per dimension.
  \return Shared pointer to a GlobalMesh.
*/
template <class Scalar, std::size_t NumSpaceDim>
std::shared_ptr<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>
createUniformGlobalMesh(
    const std::array<Scalar, NumSpaceDim>& global_low_corner,
    const std::array<Scalar, NumSpaceDim>& global_high_corner,
    const std::array<Scalar, NumSpaceDim>& cell_size )
{
    return std::make_shared<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>(
        global_low_corner, global_high_corner, cell_size );
}

/*!
  \brief Create uniform mesh with total number of cells.

  \tparam Scalar Mesh floating point type.
  \tparam NumSpaceDim Spatial dimension.

  \param global_low_corner, global_high_corner Location of the mesh corner.
  \param global_num_cell %Array ofnumber of cells per dimension.
  \return Shared pointer to a GlobalMesh.
*/
template <class Scalar, std::size_t NumSpaceDim>
std::shared_ptr<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>
createUniformGlobalMesh(
    const std::array<Scalar, NumSpaceDim>& global_low_corner,
    const std::array<Scalar, NumSpaceDim>& global_high_corner,
    const std::array<int, NumSpaceDim>& global_num_cell )
{
    return std::make_shared<GlobalMesh<UniformMesh<Scalar, NumSpaceDim>>>(
        global_low_corner, global_high_corner, global_num_cell );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create sparse mesh with uniform cell size.

  \tparam Scalar Mesh floating point type.
  \tparam NumSpaceDim Spatial dimension.

  \param global_low_corner, global_high_corner Location of the mesh corner.
  \param cell_size Uniform cell size for every dimension.
  \return Shared pointer to a GlobalMesh.
*/
template <class Scalar, std::size_t NumSpaceDim>
std::shared_ptr<GlobalMesh<SparseMesh<Scalar, NumSpaceDim>>>
createSparseGlobalMesh(
    const std::array<Scalar, NumSpaceDim>& global_low_corner,
    const std::array<Scalar, NumSpaceDim>& global_high_corner,
    const Scalar cell_size )
{
    return std::make_shared<GlobalMesh<SparseMesh<Scalar, NumSpaceDim>>>(
        global_low_corner, global_high_corner, cell_size );
}

/*!
  \brief Create sparse mesh with uniform cell size.

  \tparam Scalar Mesh floating point type.
  \tparam NumSpaceDim Spatial dimension.

  \param global_low_corner, global_high_corner Location of the mesh corner.
  \param cell_size %Array ofuniform cell size per dimension.
  \return Shared pointer to a GlobalMesh.
*/
template <class Scalar, std::size_t NumSpaceDim>
std::shared_ptr<GlobalMesh<SparseMesh<Scalar, NumSpaceDim>>>
createSparseGlobalMesh(
    const std::array<Scalar, NumSpaceDim>& global_low_corner,
    const std::array<Scalar, NumSpaceDim>& global_high_corner,
    const std::array<Scalar, NumSpaceDim>& cell_size )
{
    return std::make_shared<GlobalMesh<SparseMesh<Scalar, NumSpaceDim>>>(
        global_low_corner, global_high_corner, cell_size );
}

/*!
  \brief Create sparse mesh with total number of cells.

  \tparam Scalar Mesh floating point type.
  \tparam NumSpaceDim Spatial dimension.

  \param global_low_corner, global_high_corner Location of the mesh corner.
  \param global_num_cell %Array ofnumber of cells per dimension.
  \return Shared pointer to a GlobalMesh.
*/
template <class Scalar, std::size_t NumSpaceDim>
std::shared_ptr<GlobalMesh<SparseMesh<Scalar, NumSpaceDim>>>
createSparseGlobalMesh(
    const std::array<Scalar, NumSpaceDim>& global_low_corner,
    const std::array<Scalar, NumSpaceDim>& global_high_corner,
    const std::array<int, NumSpaceDim>& global_num_cell )
{
    return std::make_shared<GlobalMesh<SparseMesh<Scalar, NumSpaceDim>>>(
        global_low_corner, global_high_corner, global_num_cell );
}

//---------------------------------------------------------------------------//
/*!
  \brief Global mesh partial specialization for 3D non-uniform mesh.

  \tparam Scalar Mesh floating point type.

  Non-uniform meshes have a list of node locations for each spatial dimension
  which describe a rectilinear mesh that has arbitrary cell sizes - each cell
  can possibly be different.
*/
template <class Scalar>
class GlobalMesh<NonUniformMesh<Scalar, 3>>
{
  public:
    //! Mesh type.
    using mesh_type = NonUniformMesh<Scalar, 3>;

    //! Scalar type.
    using scalar_type = Scalar;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = 3;

    //! \brief 3D constructor.
    //! \param i_edges, j_edges, k_edges List of edges in each dimension.
    GlobalMesh( const std::vector<Scalar>& i_edges,
                const std::vector<Scalar>& j_edges,
                const std::vector<Scalar>& k_edges )
        : _edges( { i_edges, j_edges, k_edges } )
    {
    }

    // GLOBAL MESH INTERFACE

    //! \brief Get the global low corner of the mesh.
    //! \param dim Spatial dimension.
    Scalar lowCorner( const std::size_t dim ) const
    {
        return _edges[dim].front();
    }

    //! \brief Get the global high corner of the mesh.
    //! \param dim Spatial dimension.
    Scalar highCorner( const std::size_t dim ) const
    {
        return _edges[dim].back();
    }

    //! \brief Get the extent of a given dimension.
    //! \param dim Spatial dimension.
    Scalar extent( const std::size_t dim ) const
    {
        return highCorner( dim ) - lowCorner( dim );
    }

    //! \brief Get the global number of cells in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumCell( const std::size_t dim ) const
    {
        return _edges[dim].size() - 1;
    }

    // NON-UNIFORM MESH SPECIFIC

    //! \brief Get the edge array in a given dimension.
    //! \param dim Spatial dimension.
    const std::vector<Scalar>& nonUniformEdge( const std::size_t dim ) const
    {
        return _edges[dim];
    }

  private:
    std::array<std::vector<Scalar>, 3> _edges;
};

/*!
  \brief Create a non-uniform 3D mesh.
  \param i_edges, j_edges, k_edges List of edges defining the mesh in each
  dimension.
  \return Shared pointer to a GlobalMesh.
*/
template <class Scalar>
std::shared_ptr<GlobalMesh<NonUniformMesh<Scalar, 3>>>
createNonUniformGlobalMesh( const std::vector<Scalar>& i_edges,
                            const std::vector<Scalar>& j_edges,
                            const std::vector<Scalar>& k_edges )
{
    return std::make_shared<GlobalMesh<NonUniformMesh<Scalar, 3>>>(
        i_edges, j_edges, k_edges );
}

//---------------------------------------------------------------------------//
/*!
  \brief Global mesh partial specialization for 2D non-uniform mesh.

  \tparam Scalar Mesh floating point type.

  Non-uniform meshes have a list of node locations for each spatial dimension
  which describe a rectilinear mesh that has arbitrary cell sizes - each cell
  can possibly be different.
*/
template <class Scalar>
class GlobalMesh<NonUniformMesh<Scalar, 2>>
{
  public:
    //! Mesh type.
    using mesh_type = NonUniformMesh<Scalar, 2>;

    //! Scalar type.
    using scalar_type = Scalar;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = 2;

    //! \brief 2D constructor.
    //! \param i_edges, j_edges List of edges in each dimension.
    GlobalMesh( const std::vector<Scalar>& i_edges,
                const std::vector<Scalar>& j_edges )
        : _edges( { i_edges, j_edges } )
    {
    }

    // GLOBAL MESH INTERFACE

    //! \brief Get the global low corner of the mesh.
    //! \param dim Spatial dimension.
    Scalar lowCorner( const std::size_t dim ) const
    {
        return _edges[dim].front();
    }

    //! \brief Get the global high corner of the mesh.
    //! \param dim Spatial dimension.
    Scalar highCorner( const std::size_t dim ) const
    {
        return _edges[dim].back();
    }

    //! \brief Get the extent of a given dimension.
    //! \param dim Spatial dimension.
    Scalar extent( const std::size_t dim ) const
    {
        return highCorner( dim ) - lowCorner( dim );
    }

    //! \brief Get the global number of cells in a given dimension.
    //! \param dim Spatial dimension.
    int globalNumCell( const std::size_t dim ) const
    {
        return _edges[dim].size() - 1;
    }

    // NON-UNIFORM MESH SPECIFIC

    //! \brief Get the edge array in a given dimension.
    //! \param dim Spatial dimension.
    const std::vector<Scalar>& nonUniformEdge( const std::size_t dim ) const
    {
        return _edges[dim];
    }

  private:
    std::array<std::vector<Scalar>, 2> _edges;
};

/*!
  \brief Create a non-uniform 2D mesh.
  \tparam Scalar Mesh scalar type.
  \param i_edges, j_edges List of edges defining the mesh in each dimension.
  \return Shared pointer to a GlobalMesh.
*/
template <class Scalar>
std::shared_ptr<GlobalMesh<NonUniformMesh<Scalar, 2>>>
createNonUniformGlobalMesh( const std::vector<Scalar>& i_edges,
                            const std::vector<Scalar>& j_edges )
{
    return std::make_shared<GlobalMesh<NonUniformMesh<Scalar, 2>>>( i_edges,
                                                                    j_edges );
}

//---------------------------------------------------------------------------//

} // namespace Grid
} // namespace Cabana

#endif // end CABANA_GRID_GLOBALMESH_HPP
