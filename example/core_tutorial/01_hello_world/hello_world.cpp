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

#include <Cabana_Core.hpp>

#include <iostream>

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    /* The Kokkos runtime used by Cabana must be initialized and finalized.
       Kokkos::ScopeGuard initializes Kokkos and guarantees it is finalized,
       even if the code returns early.
     */
    Kokkos::ScopeGuard scope_guard( argc, argv );

    // Any code using Cabana should be after the ScopeGuard is constructed
    std::cout << "Hello world from Cabana!" << std::endl;

    return 0;
}

//---------------------------------------------------------------------------//
