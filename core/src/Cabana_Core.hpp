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

#ifndef CABANA_CORE_HPP
#define CABANA_CORE_HPP

#include <CabanaCore_config.hpp>

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_LinkedCellList.hpp>
#include <Cabana_Macros.hpp>
#include <Cabana_MemberTypes.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Parallel.hpp>
#include <Cabana_Slice.hpp>
#include <Cabana_SoA.hpp>
#include <Cabana_Sort.hpp>
#include <Cabana_Tuple.hpp>
#include <Cabana_Types.hpp>
#include <Cabana_VerletList.hpp>
#include <Cabana_Version.hpp>

#ifdef Cabana_ENABLE_MPI
#include <Cabana_Distributor.hpp>
#include <Cabana_Halo.hpp>
#endif

#include <Kokkos_Core.hpp>

#include <exception>

namespace Cabana {
namespace { // anonymous namespace

//---------------------------------------------------------------------------//
// Whether one of the Cabana::initialize() functions has been called before.
bool is_cabana_initialized = false;

//---------------------------------------------------------------------------//
// Whether Cabana initialized Kokkos. Cabana's finalize() only finalizes
// Kokkos if it initialized Kokkos. Otherwise, something else initialized
// Kokkos and is responsible for finalizing it.
bool cabana_initialized_kokkos = false;

//---------------------------------------------------------------------------//
// Initialize Kokkos, if it needs initialization.
template <typename... Args>
CABANA_DEPRECATED void initKokkos( Args &&... args ) {
    if ( !cabana_initialized_kokkos ) {
        // Kokkos doesn't have a global is_initialized().  However,
        // Kokkos::initialize() always initializes the default execution
        // space, so it suffices to check whether that was initialized.
        const bool kokkosIsInitialized =
            Kokkos::DefaultExecutionSpace::is_initialized();

        if ( !kokkosIsInitialized ) {
            // Kokkos will remove all arguments Kokkos recognizes which start
            // with '--kokkos' (e.g.,--kokkos-threads)
            Kokkos::initialize( std::forward<Args>( args )... );
            cabana_initialized_kokkos = true;
        }
    }

    const bool kokkosIsInitialized =
        Kokkos::DefaultExecutionSpace::is_initialized();

    if ( !kokkosIsInitialized )
        throw std::runtime_error( "At the end of initKokkos, Kokkos"
                                  " is not initialized. Please report"
                                  " this bug to the Cabana developers." );
}

//---------------------------------------------------------------------------//

} // end anonymous namespace

//---------------------------------------------------------------------------//
template <typename... Args>
CABANA_DEPRECATED void initialize( Args &&... args ) {
    if ( !is_cabana_initialized )
        initKokkos( std::forward<Args>( args )... );
    is_cabana_initialized = true;
}

//---------------------------------------------------------------------------//
CABANA_DEPRECATED
bool isInitialized() { return is_cabana_initialized; }

//---------------------------------------------------------------------------//
CABANA_DEPRECATED
void finalize() {
    if ( !is_cabana_initialized )
        return;

    // Cabana should only finalize Kokkos if it initialized it
    if ( cabana_initialized_kokkos )
        Kokkos::finalize();

    is_cabana_initialized = false;
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_CORE_HPP
