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
  \file Cabana_Grid_Types.hpp
  \brief Grid type tags
*/
#ifndef CABANA_GRID_TYPES_HPP
#define CABANA_GRID_TYPES_HPP

#include <Cabana_Utils.hpp> // FIXME: remove after next release.

#include <type_traits>

namespace Cabana
{
namespace Grid
{

//---------------------------------------------------------------------------//
/*!
  \brief Logical dimension index.
*/
struct Dim
{
    //! Spatial dimension.
    enum Values
    {
        I = 0,
        J = 1,
        K = 2
    };
};

//---------------------------------------------------------------------------//
// Entity type tags.
//---------------------------------------------------------------------------//

/*!
  \brief Mesh cell tag.
*/
struct Cell
{
};

/*!
  \brief Mesh node tag.
*/
struct Node
{
};

/*!
  \brief Mesh face tag.
  \tparam D Dimension.
*/
template <int D>
struct Face;

//! I-face tag.
template <>
struct Face<Dim::I>
{
    //! Spatial dimension.
    static constexpr int dim = Dim::I;
};

//! J-face tag.
template <>
struct Face<Dim::J>
{
    //! Spatial dimension.
    static constexpr int dim = Dim::J;
};

//! K-face tag.
template <>
struct Face<Dim::K>
{
    //! Spatial dimension.
    static constexpr int dim = Dim::K;
};

/*!
  \brief Mesh edge tag.
  \tparam D Dimension.
*/
template <int D>
struct Edge;

//! I-edge tag.
template <>
struct Edge<Dim::I>
{
    //! Spatial dimension.
    static constexpr int dim = Dim::I;
};

//! J-edge tag.
template <>
struct Edge<Dim::J>
{
    //! Spatial dimension.
    static constexpr int dim = Dim::J;
};

//! K-edge tag.
template <>
struct Edge<Dim::K>
{
    //! Spatial dimension.
    static constexpr int dim = Dim::K;
};

// Type checkers.
template <class T>
struct isCell : public std::false_type
{
};

template <>
struct isCell<Cell> : public std::true_type
{
};

template <>
struct isCell<const Cell> : public std::true_type
{
};

template <class T>
struct isNode : public std::false_type
{
};

template <>
struct isNode<Node> : public std::true_type
{
};

template <>
struct isNode<const Node> : public std::true_type
{
};

template <class T>
struct isFace : public std::false_type
{
};

template <int Dir>
struct isFace<Face<Dir>> : public std::true_type
{
};

template <int Dir>
struct isFace<const Face<Dir>> : public std::true_type
{
};

template <class T>
struct isEdge : public std::false_type
{
};

template <int Dir>
struct isEdge<Edge<Dir>> : public std::true_type
{
};

template <int Dir>
struct isEdge<const Edge<Dir>> : public std::true_type
{
};

//---------------------------------------------------------------------------//
// Decomposition tags.
//---------------------------------------------------------------------------//

/*!
  \brief Owned decomposition tag.
*/
struct Own
{
};

/*!
  \brief Ghosted decomposition tag.
*/
struct Ghost
{
};

//---------------------------------------------------------------------------//
// Index type tags.
//---------------------------------------------------------------------------//

/*!
  \brief Local index tag.
*/
struct Local
{
};

/*!
  \brief Global index tag.
*/
struct Global
{
};

//---------------------------------------------------------------------------//
// Mesh type tags.
//---------------------------------------------------------------------------//

/*!
  \brief Uniform mesh tag.
*/
template <class Scalar, std::size_t NumSpaceDim = 3>
struct UniformMesh
{
    //! Scalar type for mesh floating point operations.
    using scalar_type = Scalar;

    //! Number of spatial dimensions.
    static constexpr std::size_t num_space_dim = NumSpaceDim;
};

/*!
  \brief Non-uniform mesh tag.
*/
template <class Scalar, std::size_t NumSpaceDim = 3>
struct NonUniformMesh
{
    //! Scalar type for mesh floating point operations.
    using scalar_type = Scalar;

    //! Number of spatial dimensions.
    static constexpr std::size_t num_space_dim = NumSpaceDim;
};

/*!
  \brief Sparse mesh tag.
*/
template <class Scalar, std::size_t NumSpaceDim = 3>
struct SparseMesh
{
    //! Scalar type for mesh floating point operations.
    using scalar_type = Scalar;

    //! Number of spatial dimensions.
    static constexpr std::size_t num_space_dim = NumSpaceDim;
};

// Type checker.
template <class T>
struct isMeshType : public std::false_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isMeshType<UniformMesh<Scalar, NumSpaceDim>> : public std::true_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isMeshType<const UniformMesh<Scalar, NumSpaceDim>>
    : public std::true_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isMeshType<NonUniformMesh<Scalar, NumSpaceDim>> : public std::true_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isMeshType<const NonUniformMesh<Scalar, NumSpaceDim>>
    : public std::true_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isMeshType<SparseMesh<Scalar, NumSpaceDim>> : public std::true_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isMeshType<const SparseMesh<Scalar, NumSpaceDim>> : public std::true_type
{
};

// Uniform mesh checker.
template <class T>
struct isUniformMesh : public std::false_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isUniformMesh<UniformMesh<Scalar, NumSpaceDim>> : public std::true_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isUniformMesh<const UniformMesh<Scalar, NumSpaceDim>>
    : public std::true_type
{
};

// Non-uniform mesh checker.
template <class T>
struct isNonUniformMesh : public std::false_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isNonUniformMesh<NonUniformMesh<Scalar, NumSpaceDim>>
    : public std::true_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isNonUniformMesh<const NonUniformMesh<Scalar, NumSpaceDim>>
    : public std::true_type
{
};

// Sparse mesh checker
template <class T>
struct isSparseMesh : public std::false_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isSparseMesh<SparseMesh<Scalar, NumSpaceDim>> : public std::true_type
{
};

template <class Scalar, std::size_t NumSpaceDim>
struct isSparseMesh<const SparseMesh<Scalar, NumSpaceDim>>
    : public std::true_type
{
};
//---------------------------------------------------------------------------//
} // namespace Grid
} // namespace Cabana

namespace Cajita
{
//! \cond Deprecated
using Dim CAJITA_DEPRECATED = Cabana::Grid::Dim;
using Own CAJITA_DEPRECATED = Cabana::Grid::Own;
using Ghost CAJITA_DEPRECATED = Cabana::Grid::Ghost;
using Local CAJITA_DEPRECATED = Cabana::Grid::Local;
using Global CAJITA_DEPRECATED = Cabana::Grid::Global;
using Cell CAJITA_DEPRECATED = Cabana::Grid::Cell;
using Node CAJITA_DEPRECATED = Cabana::Grid::Node;

template <int D>
using Face CAJITA_DEPRECATED = Cabana::Grid::Face<D>;
template <int D>
using Edge CAJITA_DEPRECATED = Cabana::Grid::Edge<D>;

template <class T>
using isCell CAJITA_DEPRECATED = Cabana::Grid::isCell<T>;
template <class T>
using isNode CAJITA_DEPRECATED = Cabana::Grid::isNode<T>;
template <class T>
using isFace CAJITA_DEPRECATED = Cabana::Grid::isFace<T>;
template <class T>
using isEdge CAJITA_DEPRECATED = Cabana::Grid::isEdge<T>;

template <class Scalar, std::size_t NumSpaceDim = 3>
using UniformMesh CAJITA_DEPRECATED =
    Cabana::Grid::UniformMesh<Scalar, NumSpaceDim>;
template <class Scalar, std::size_t NumSpaceDim = 3>
using NonUniformMesh CAJITA_DEPRECATED =
    Cabana::Grid::NonUniformMesh<Scalar, NumSpaceDim>;
template <class Scalar, std::size_t NumSpaceDim = 3>
using SparseMesh CAJITA_DEPRECATED =
    Cabana::Grid::SparseMesh<Scalar, NumSpaceDim>;

template <class T>
using isMeshType CAJITA_DEPRECATED = Cabana::Grid::isMeshType<T>;
template <class T>
using isUniformMesh CAJITA_DEPRECATED = Cabana::Grid::isUniformMesh<T>;
template <class T>
using isNonUniformMesh CAJITA_DEPRECATED = Cabana::Grid::isNonUniformMesh<T>;
template <class T>
using isSparseMesh CAJITA_DEPRECATED = Cabana::Grid::isSparseMesh<T>;
//! \endcond
} // namespace Cajita

#endif // CABANA_GRID_TYPES_HPP
