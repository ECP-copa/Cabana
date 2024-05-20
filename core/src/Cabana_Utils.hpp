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

// Custom warning for use within static_assert.
constexpr bool deprecated( std::false_type ) { return true; }

[[deprecated( "Cabana deprecation." )]] constexpr bool
deprecated( std::true_type )
{
    return true;
}

// Custom warning for switch from Cajita to Grid.
#ifdef Cabana_DISABLE_CAJITA_DEPRECATION_WARNINGS
#define CAJITA_DEPRECATED
#else
#define CAJITA_DEPRECATED                                                      \
    [[deprecated( "Cajita is now Cabana::Grid. The Cajita namespace will be "  \
                  "removed in a future release." )]]
#endif

//! \endcond

} // namespace Impl
} // namespace Cabana

#endif
