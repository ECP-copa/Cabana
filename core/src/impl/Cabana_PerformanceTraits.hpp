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
  \file Cabana_PerformanceTraits.hpp
  \brief Default settings for execution spaces.
*/
#ifndef CABANA_PERFORMANCETRAITS_HPP
#define CABANA_PERFORMANCETRAITS_HPP

#include <Cabana_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Cabana
{
namespace Impl
{
//---------------------------------------------------------------------------//
/*!
  \brief Default settings for execution spaces.
*/
template <class ExecutionSpace>
class PerformanceTraits;

//---------------------------------------------------------------------------//
// Serial specialization.
#if defined( KOKKOS_ENABLE_SERIAL )
template <>
class PerformanceTraits<Kokkos::Serial>
{
  public:
    static constexpr int vector_length = 16;
};
#endif

//---------------------------------------------------------------------------//
// Threads specialization.
#if defined( KOKKOS_ENABLE_THREADS )
template <>
class PerformanceTraits<Kokkos::Threads>
{
  public:
    static constexpr int vector_length = 16;
};
#endif

//---------------------------------------------------------------------------//
// OpenMP specialization.
#if defined( KOKKOS_ENABLE_OPENMP )
template <>
class PerformanceTraits<Kokkos::OpenMP>
{
  public:
    static constexpr int vector_length = 16;
};
#endif

//---------------------------------------------------------------------------//
// Cuda specialization. Use the warp traits.
#if defined( KOKKOS_ENABLE_CUDA )
template <>
class PerformanceTraits<Kokkos::Cuda>
{
  public:
    static constexpr int vector_length = 32; // warp size
};
#endif

//---------------------------------------------------------------------------//
#if defined( KOKKOS_ENABLE_HIP )
template <>
class PerformanceTraits<Kokkos::HIP>
{
  public:
    static constexpr int vector_length = 64; // wavefront size
};
#endif

//---------------------------------------------------------------------------//
#if defined( KOKKOS_ENABLE_SYCL )
template <>
class PerformanceTraits<Kokkos::Experimental::SYCL>
{
  public:
    static constexpr int vector_length = 16; // FIXME_SYCL
};
#endif

//---------------------------------------------------------------------------//
#if defined( KOKKOS_ENABLE_OPENMPTARGET )
template <>
class PerformanceTraits<Kokkos::Experimental::OpenMPTarget>
{
  public:
    static constexpr int vector_length = 16; // FIXME_OPENMPTARGET
};
#endif

//---------------------------------------------------------------------------//

} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_PERFORMANCETRAITS_HPP
