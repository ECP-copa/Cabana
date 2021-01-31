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

#ifndef CAJITA_TYPES_HPP
#define CAJITA_TYPES_HPP

#include <type_traits>

namespace Cajita
{

//---------------------------------------------------------------------------//
// Logical dimension index.
//---------------------------------------------------------------------------//
struct Dim
{
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

// Mesh cell tag.
struct Cell
{
};

// Mesh node tag.
struct Node
{
};

// Mesh face tags.
template <int D>
struct Face;

// I-face tag.
template <>
struct Face<Dim::I>
{
    static constexpr int dim = Dim::I;
};

// J-face tag.
template <>
struct Face<Dim::J>
{
    static constexpr int dim = Dim::J;
};

// K-face tag.
template <>
struct Face<Dim::K>
{
    static constexpr int dim = Dim::K;
};

// Mesh edge tags.
template <int D>
struct Edge;

// I-edge tag.
template <>
struct Edge<Dim::I>
{
    static constexpr int dim = Dim::I;
};

// J-edge tag.
template <>
struct Edge<Dim::J>
{
    static constexpr int dim = Dim::J;
};

// K-edge tag.
template <>
struct Edge<Dim::K>
{
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

// Owned decomposition tag.
struct Own
{
};

// Ghosted decomposition tag.
struct Ghost
{
};

//---------------------------------------------------------------------------//
// Index type tags.
//---------------------------------------------------------------------------//

// Local index tag.
struct Local
{
};

// Global index tag.
struct Global
{
};

//---------------------------------------------------------------------------//
// Mesh type tags.
//---------------------------------------------------------------------------//

// Uniform mesh tag.
template <class Scalar, std::size_t NumSpaceDim = 3>
struct UniformMesh
{
    // Scalar type for mesh floating point operations.
    using scalar_type = Scalar;

    // Number of spatial dimensions.
    static constexpr std::size_t num_space_dim = NumSpaceDim;
};

// Non-uniform mesh tag.
template <class Scalar, std::size_t NumSpaceDim = 3>
struct NonUniformMesh
{
    // Scalar type for mesh floating point operations.
    using scalar_type = Scalar;

    // Number of spatial dimensions.
    static constexpr std::size_t num_space_dim = NumSpaceDim;
};

// Sparse mesh tag
template <class Scalar, std::size_t NumSpaceDim = 3>
struct SparseMesh
{
    // Scalar type for mesh floating point operations.
    using scalar_type = Scalar;

    // Number of spatial dimensions.
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

} // end namespace Cajita

#endif // CAJITA_TYPES_HPP
