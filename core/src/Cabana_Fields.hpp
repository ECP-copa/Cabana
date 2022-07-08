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
template <class T>
struct Scalar;

template <class T, int D>
struct Vector;

template <class T, int D0, int D1>
struct Matrix;

template <class T>
struct Scalar
{
    using value_type = T;
    static constexpr int rank = 0;
    static constexpr int size = 1;
    using data_type = value_type;
};

template <class T, int D>
struct Vector
{
    using value_type = T;
    static constexpr int rank = 1;
    static constexpr int size = D;
    static constexpr int dim0 = D;
    using data_type = value_type[D];
};

template <class T, int D0, int D1>
struct Matrix
{
    using value_type = T;
    static constexpr int rank = 2;
    static constexpr int size = D0 * D1;
    static constexpr int dim0 = D0;
    static constexpr int dim1 = D1;
    using data_type = value_type[D0][D1];
};

//---------------------------------------------------------------------------//
// Common field types.
//---------------------------------------------------------------------------//
template <std::size_t NumSpaceDim>
struct Position : Vector<double, NumSpaceDim>
{
    static std::string label() { return "position"; }
};

} // namespace Field

//---------------------------------------------------------------------------//
// General type indexer.
//---------------------------------------------------------------------------//
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

template <class T, class... Types>
struct TypeIndexer
{
    static constexpr std::size_t index =
        TypeIndexerImpl<T, sizeof...( Types ), sizeof...( Types ) - 1,
                        Types...>::value;
};

} // namespace Cabana
#endif
