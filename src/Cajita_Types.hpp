/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
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
};

// J-face tag.
template <>
struct Face<Dim::J>
{
};

// K-face tag.
template <>
struct Face<Dim::K>
{
};

// Mesh edge tags.
template <int D>
struct Edge;

// I-edge tag.
template <>
struct Edge<Dim::I>
{
};

// J-edge tag.
template <>
struct Edge<Dim::J>
{
};

// K-edge tag.
template <>
struct Edge<Dim::K>
{
};

// Type checker.

template <class T>
struct isEntityType : public std::false_type
{
};

template <>
struct isEntityType<Cell> : public std::true_type
{
};

template <>
struct isEntityType<const Cell> : public std::true_type
{
};

template <>
struct isEntityType<Node> : public std::true_type
{
};

template <>
struct isEntityType<const Node> : public std::true_type
{
};

template <int Dir>
struct isEntityType<Face<Dir>> : public std::true_type
{
};

template <int Dir>
struct isEntityType<const Face<Dir>> : public std::true_type
{
};

template <int Dir>
struct isEntityType<Edge<Dir>> : public std::true_type
{
};

template <int Dir>
struct isEntityType<const Edge<Dir>> : public std::true_type
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
template <class Scalar>
struct UniformMesh
{
    // Scalar type for mesh floating point operations.
    using scalar_type = Scalar;
};

// Non-uniform mesh tag.
template <class Scalar>
struct NonUniformMesh
{
    // Scalar type for mesh floating point operations.
    using scalar_type = Scalar;
};

// Type checker.
template <class T>
struct isMeshType : public std::false_type
{
};

template <class Scalar>
struct isMeshType<UniformMesh<Scalar>> : public std::true_type
{
};

template <class Scalar>
struct isMeshType<const UniformMesh<Scalar>> : public std::true_type
{
};

template <class Scalar>
struct isMeshType<NonUniformMesh<Scalar>> : public std::true_type
{
};

template <class Scalar>
struct isMeshType<const NonUniformMesh<Scalar>> : public std::true_type
{
};

// Uniform mesh checker.
template <class T>
struct isUniformMesh : public std::false_type
{
};

template <class Scalar>
struct isUniformMesh<UniformMesh<Scalar>> : public std::true_type
{
};

template <class Scalar>
struct isUniformMesh<const UniformMesh<Scalar>> : public std::true_type
{
};

// Non-uniform mesh checker.
template <class T>
struct isNonUniformMesh : public std::false_type
{
};

template <class Scalar>
struct isNonUniformMesh<NonUniformMesh<Scalar>> : public std::true_type
{
};

template <class Scalar>
struct isNonUniformMesh<const NonUniformMesh<Scalar>> : public std::true_type
{
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // CAJITA_TYPES_HPP
