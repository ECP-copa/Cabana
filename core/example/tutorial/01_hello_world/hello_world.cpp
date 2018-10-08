/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
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
    /* The Cabana runtime must be initialized. This is ultimately to ensure
       the initialization of the Kokkos runtime that Cabana uses as an
       implementation detail. Some notes:

           *: If Kokkos has already been initialized by some other part of the
              application then this initialization does nothing

           *: If Kokkos has not yet been initialized then Cabana will do the
              initialization.

       You have two options for initialization:

           1) Call Cabana::initialize() with argc and argv as below. This will
           pass any command line argument to the Kokkos runtime.

           2) Call Cabana::initialize() with no arguments. Defaults and system
           variables will be used to select things like number of threads.
    */
    Cabana::initialize(argc,argv);

    // Any code using Cabana should be inserted between initialize and
    // finalize.
    std::cout << "Hello world from Cabana!" << std::endl;

    /* The Cabana runtime must also be finalized. If Cabana initialized the
     * Kokkos runtime then it will finalize it at this time. If Cabana did not
     * initialize the Kokkos runtime then this function does nothing.
     */
    Cabana::finalize();

    return 0;
}

//---------------------------------------------------------------------------//
