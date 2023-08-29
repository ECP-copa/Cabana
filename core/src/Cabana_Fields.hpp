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
  \file Cabana_Fields.hpp
  \brief Particle field types and common field examples.
*/
#ifndef CABANA_FIELDS_HPP
#define CABANA_FIELDS_HPP

namespace Cabana
{
namespace Field
{
//---------------------------------------------------------------------------//
// General field types.
//---------------------------------------------------------------------------//
// Forward declarations.
//! \cond Impl
template <class T>
struct Scalar;

template <class T, int D>
struct Vector;

template <class T, int D0, int D1>
struct Matrix;
//! \endcond

//! Scalar particle field type.
template <class T>
struct Scalar
{
    //! Field type.
    using value_type = T;
    //! Field rank.
    static constexpr int rank = 0;
    //! Field total size.
    static constexpr int size = 1;
    //! Scalar type.
    using data_type = value_type;
};

//! Vector (1D) particle field type.
template <class T, int D>
struct Vector
{
    //! Field type.
    using value_type = T;
    //! Field rank.
    static constexpr int rank = 1;
    //! Field total size.
    static constexpr int size = D;
    //! Field first dimension size.
    static constexpr int dim0 = D;
    //! Scalar type.
    using data_type = value_type[D];
};

//! Matrix (2D) particle field type.
template <class T, int D0, int D1>
struct Matrix
{
    //! Field type.
    using value_type = T;
    //! Field rank.
    static constexpr int rank = 2;
    //! Field total size.
    static constexpr int size = D0 * D1;
    //! Field first dimension size.
    static constexpr int dim0 = D0;
    //! Field second dimension size.
    static constexpr int dim1 = D1;
    //! Scalar type.
    using data_type = value_type[D0][D1];
};

//---------------------------------------------------------------------------//
// Common field types.
//---------------------------------------------------------------------------//
//! Particle position field type.
template <std::size_t NumSpaceDim>
struct Position : Vector<double, NumSpaceDim>
{
    //! Field label.
    static std::string label() { return "position"; }
};

} // namespace Field

//---------------------------------------------------------------------------//
// General type indexer.
//---------------------------------------------------------------------------//
//! \cond Impl
template <class T, int Size, int N, class Type, class... Types>
struct TypeIndexerImpl
{
    static constexpr std::size_t value =
        TypeIndexerImpl<T, Size, N - 1, Types...>::value *
        ( std::is_same<T, Type>::value ? Size - 1 - N : 1 );
};

template <class T, int Size, class Type, class... Types>
struct TypeIndexerImpl<T, Size, 0, Type, Types...>
{
    static constexpr std::size_t value =
        std::is_same<T, Type>::value ? Size - 1 : 1;
};
//! \endcond

//! Get the index of a field type within a particle type list.
template <class T, class... Types>
struct TypeIndexer
{
    //! Field index.
    static constexpr std::size_t index =
        TypeIndexerImpl<T, sizeof...( Types ), sizeof...( Types ) - 1,
                        Types...>::value;
};

} // namespace Cabana
#endif
