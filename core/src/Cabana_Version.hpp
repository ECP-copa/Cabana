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
  \file Cabana_Version.hpp
  \brief Cabana git version
*/
#ifndef CABANA_VERSION_HPP
#define CABANA_VERSION_HPP

#include <Cabana_Core_Config.hpp>

#include <string>

namespace Cabana
{

//! Cabana version.
inline std::string version() { return Cabana_VERSION_STRING; }

//! Cabana git hash.
inline std::string git_commit_hash() { return Cabana_GIT_COMMIT_HASH; }

} // end namespace Cabana

#endif // end CABANA_VERSION_HPP
