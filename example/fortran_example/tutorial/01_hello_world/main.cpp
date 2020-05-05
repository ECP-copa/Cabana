/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
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

/*
  Declare functions that will be mixed with Fortran
 */
extern "C" void print_hello_world();

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char *argv[] )
{
    /* The Kokkos runtime used by Cabana must be initialized and finalized.
         Kokkos::ScopeGuard inializes Kokkos and guarantees it is finalized,
         even if the code returns early.
    */
    Kokkos::ScopeGuard scope_guard( argc, argv );

    // Any code using Cabana should be after the ScopeGuard is constructed

    // Call the Fortran subroutine
    print_hello_world();

    return 0;
}

//---------------------------------------------------------------------------//
