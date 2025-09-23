/****************************************************************************
 * Copyright (c) 2018-2025 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Tags.hpp
  \brief Type tags used in Cabana
*/
#ifndef CABANA_TAGS_HPP
#define CABANA_TAGS_HPP

namespace Cabana
{
//---------------------------------------------------------------------------//
// Communication backend types.
//---------------------------------------------------------------------------//
/*!
    \brief Vanilla MPI backend tag - default.
*/
struct Mpi
{
};

//---------------------------------------------------------------------------//
// Communication driver construction type tags.
//---------------------------------------------------------------------------//
/*!
    \brief Export-based tag - default.
*/
struct Export
{
};

/*!
    \brief Import-based tag.
*/
struct Import
{
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_TAGS_HPP
