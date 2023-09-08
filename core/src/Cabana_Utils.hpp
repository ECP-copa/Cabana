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

namespace Cabana
{
namespace Impl
{
//! \cond Impl

// Custom warning for switch from device_type to memory_space.
constexpr bool warn( std::false_type ) { return true; }

[[deprecated( "Template parameter should be converted from device type to "
              "memory space." )]] constexpr bool
warn( std::true_type )
{
    return true;
}
//! \endcond

} // namespace Impl
} // namespace Cabana

#endif
