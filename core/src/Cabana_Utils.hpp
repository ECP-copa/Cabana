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
namespace Impl
{
//! \cond Impl

//---------------------------------------------------------------------------//
// Array copies.
//---------------------------------------------------------------------------//
//! Copy std::array into Kokkos::Array for potential device use.
template <std::size_t Dim, class Scalar>
auto copyArray( const std::array<Scalar, Dim> corner )
{
    Kokkos::Array<Scalar, Dim> kokkos_corner;
    for ( std::size_t d = 0; d < Dim; ++d )
        kokkos_corner[d] = corner[d];

    return kokkos_corner;
}
//! Return original Kokkos::Array.
template <std::size_t Dim, class Scalar>
auto copyArray( const Kokkos::Array<Scalar, Dim> corner )
{
    return corner;
}
//! Copy c-array into Kokkos::Array for potential device use.
template <std::size_t Dim, class Scalar>
auto copyArray( const Scalar corner[Dim] )
{
    Kokkos::Array<Scalar, Dim> kokkos_corner;
    for ( std::size_t d = 0; d < Dim; ++d )
        kokkos_corner[d] = corner[d];

    return kokkos_corner;
}

// Custom warning for switch from device_type to memory_space.
constexpr bool deprecated( std::false_type ) { return true; }

[[deprecated(
    "Template parameter should be converted from Kokkos device type to "
    "Kokkos memory space." )]] constexpr bool
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
