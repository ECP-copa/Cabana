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
  \file Cajita_Hypre.hpp
  \brief HYPRE memory space handling
*/
#ifndef CAJITA_HYPRE_HPP
#define CAJITA_HYPRE_HPP

#include <HYPRE_config.h>
#include <HYPRE_struct_ls.h>
#include <HYPRE_struct_mv.h>

#include <Kokkos_Core.hpp>

#include <memory>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Hypre memory space selection. Don't compile if HYPRE wasn't configured to
// use the device.
// ---------------------------------------------------------------------------//

//! Hypre device compatibility check.
template <class MemorySpace>
struct HypreIsCompatibleWithMemorySpace : std::false_type
{
};

// FIXME: This is currently written in this structure because HYPRE only has
// compile-time switches for backends and hence only one can be used at a
// time. Once they have a run-time switch we can use that instead.
#ifdef HYPRE_USING_CUDA
#ifdef KOKKOS_ENABLE_CUDA
#ifdef HYPRE_USING_DEVICE_MEMORY
//! Hypre device compatibility check - CUDA memory.
template <>
struct HypreIsCompatibleWithMemorySpace<Kokkos::CudaSpace> : std::true_type
{
};
#endif // end HYPRE_USING_DEVICE_MEMORY

//! Hypre device compatibility check - CUDA UVM memory.
#ifdef HYPRE_USING_UNIFIED_MEMORY
template <>
struct HypreIsCompatibleWithMemorySpace<Kokkos::CudaUVMSpace> : std::true_type
{
};
#endif // end HYPRE_USING_UNIFIED_MEMORY
#endif // end KOKKOS_ENABLE_CUDA
#endif // end HYPRE_USING_CUDA

#ifdef HYPRE_USING_HIP
#ifdef KOKKOS_ENABLE_HIP
//! Hypre device compatibility check - HIP memory. FIXME - make this true when
//! the HYPRE CMake includes HIP
template <>
struct HypreIsCompatibleWithMemorySpace<Kokkos::ExperimentalHIPSpace>
    : std::true_type
{
};
#endif // end KOKKOS_ENABLE_HIP
#endif // end HYPRE_USING_HIP

#ifndef HYPRE_USING_GPU
//! Hypre device compatibility check - host memory.
template <>
struct HypreIsCompatibleWithMemorySpace<Kokkos::HostSpace> : std::true_type
{
};
#endif // end HYPRE_USING_GPU

} // namespace Cajita

#endif // end HYPRE_HPP
