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

/*!
  \file Cabana_Utils.hpp
  \brief Cabana utilities.
*/
#ifndef CABANA_UTILS_HPP
#define CABANA_UTILS_HPP

#include <Cabana_Core_Config.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <type_traits>
namespace Cabana
{
//---------------------------------------------------------------------------//
// Array copies.
//---------------------------------------------------------------------------//
//! Copy std::array into Kokkos::Array for potential device use.
template <class Scalar, std::size_t Dim>
auto copyArray( const std::array<Scalar, Dim> input )
{
    Kokkos::Array<Scalar, Dim> output;
    for ( std::size_t d = 0; d < Dim; ++d )
        output[d] = input[d];

    return output;
}
//! Return original Kokkos::Array.
template <class Scalar, std::size_t Dim>
auto copyArray( const Kokkos::Array<Scalar, Dim> input )
{
    return input;
}
//! Copy c-array into Kokkos::Array for potential device use.
template <class Scalar, std::size_t Dim>
auto copyArray( const Scalar input[Dim] )
{
    Kokkos::Array<Scalar, Dim> output;
    for ( std::size_t d = 0; d < Dim; ++d )
        output[d] = input[d];

    return output;
}

namespace Impl
{
//! \cond Impl

// Custom warning for use within static_assert.
constexpr bool deprecated( std::false_type ) { return true; }

[[deprecated( "Cabana deprecation." )]] constexpr bool
deprecated( std::true_type )
{
    return true;
}

// Custom warning.
#ifdef Cabana_DISABLE_DEPRECATION_WARNINGS
#define CABANA_DEPRECATED
#else
#define CABANA_DEPRECATED [[deprecated( "Cabana deprecation." )]]
#endif

//! \endcond

} // namespace Impl
} // namespace Cabana

#endif
