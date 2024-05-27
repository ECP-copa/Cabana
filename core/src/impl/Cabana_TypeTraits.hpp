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

#ifndef CABANA_TYPETRAITS_HPP
#define CABANA_TYPETRAITS_HPP

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
namespace Impl
{
//! \cond Impl

//---------------------------------------------------------------------------//
// Checks if an integer is a power of two. N must be greater than 0.
template <int N>
struct IsPowerOfTwo
{
    static_assert( N > 0, "Vector length must be greater than 0" );
    static constexpr bool value = ( ( N & ( N - 1 ) ) == 0 );
};

//---------------------------------------------------------------------------//
// Calculate the base-2 logarithm of an integer which must be a power of 2 and
// greater than 0.
template <int N>
struct LogBase2
{
    static_assert( IsPowerOfTwo<N>::value,
                   "Vector length must be a power of two" );
    static constexpr int value = 1 + LogBase2<( N >> 1U )>::value;
};

template <>
struct LogBase2<1>
{
    static constexpr int value = 0;
};

//---------------------------------------------------------------------------//
// Check that the provided vector length is valid.
template <int N>
struct IsVectorLengthValid
{
    static constexpr bool value = ( IsPowerOfTwo<N>::value && N > 0 );
};

//---------------------------------------------------------------------------//

//! \endcond
} // end namespace Impl
} // end namespace Cabana

#endif // CABANA_TYPETRAITS_HPP
