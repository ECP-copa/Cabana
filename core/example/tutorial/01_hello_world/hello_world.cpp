/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
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
    // The Kokkos runtime must first be initialized.
    Kokkos::initialize(argc,argv);

    // Any code using Cabana should be inserted between initialize and
    // finalize.
    std::cout << "Hello world from Cabana!" << std::endl;

    /* The Kokkos runtime must also be finalized.
       Kokkos::ScopeGuard ensures that Kokkos::finalize() is called, even
       if the code returns early.
     */
    Kokkos::ScopeGuard();

    return 0;
}

//---------------------------------------------------------------------------//
