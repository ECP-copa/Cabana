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

#include <type_traits>

namespace Cabana
{
namespace Impl
{
//! \cond Impl

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
