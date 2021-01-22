/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_TYPES_HPP
#define CABANA_TYPES_HPP

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Memory access tags.
//---------------------------------------------------------------------------//
template <class>
struct is_memory_access_tag : public std::false_type
{
};

//! Default memory access. Default memory is restricted to prevent aliasing in
//! the larger AoSoA memory block to allow for potential vectorization.
struct DefaultAccessMemory
{
    using memory_access_type = DefaultAccessMemory;
    using kokkos_memory_traits =
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Aligned |
                             Kokkos::Restrict>;
};

template <>
struct is_memory_access_tag<DefaultAccessMemory> : public std::true_type
{
};

//! Random access memory. Read-only and const with limited spatial locality.
struct RandomAccessMemory
{
    using memory_access_type = RandomAccessMemory;
    using kokkos_memory_traits =
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Aligned |
                             Kokkos::RandomAccess>;
};

template <>
struct is_memory_access_tag<RandomAccessMemory> : public std::true_type
{
};

//! Atomic memory access. All reads and writes are atomic.
struct AtomicAccessMemory
{
    using memory_access_type = AtomicAccessMemory;
    using kokkos_memory_traits =
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Aligned |
                             Kokkos::Atomic>;
};

template <>
struct is_memory_access_tag<AtomicAccessMemory> : public std::true_type
{
};

// Checks whether memory space is accessible from execution space.
// This was taken from
// https://github.com/arborx/ArborX/blob/c757ffcc0e7d2da4da2b4b4df8975365480e7bac/src/details/ArborX_DetailsKokkosExt.hpp#L33-L46
template <typename MemorySpace, typename ExecutionSpace, typename = void>
struct is_accessible_from : std::false_type
{
    static_assert( Kokkos::is_memory_space<MemorySpace>::value, "" );
    static_assert( Kokkos::is_execution_space<ExecutionSpace>::value, "" );
};

template <typename MemorySpace, typename ExecutionSpace>
struct is_accessible_from<
    MemorySpace, ExecutionSpace,
    typename std::enable_if<Kokkos::Impl::SpaceAccessibility<
        ExecutionSpace, MemorySpace>::accessible>::type> : std::true_type
{
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_TYPES_HPP
