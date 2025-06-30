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
  \file Cabana_Types.hpp
  \brief Memory access type checking
*/
#ifndef CABANA_TYPES_HPP
#define CABANA_TYPES_HPP

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
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
// Memory access tags.
//---------------------------------------------------------------------------//
//! Memory access type checker.
template <class>
struct is_memory_access_tag : public std::false_type
{
};

//! Default memory access. Default memory is restricted to prevent aliasing in
//! the larger AoSoA memory block to allow for potential vectorization.
struct DefaultAccessMemory
{
    //! Access type.
    using memory_access_type = DefaultAccessMemory;
    //! Kokkos traits.
    using kokkos_memory_traits =
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Aligned |
                             Kokkos::Restrict>;
};

//! Memory access type checker.
template <>
struct is_memory_access_tag<DefaultAccessMemory> : public std::true_type
{
};

//! Random access memory. Read-only and const with limited spatial locality.
struct RandomAccessMemory
{
    //! Access type.
    using memory_access_type = RandomAccessMemory;
    //! Kokkos traits.
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
    //! Access type.
    using memory_access_type = AtomicAccessMemory;
    //! Kokkos traits.
    using kokkos_memory_traits =
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Aligned |
                             Kokkos::Atomic>;
};

template <>
struct is_memory_access_tag<AtomicAccessMemory> : public std::true_type
{
};

// Checks whether memory space is accessible from execution space.
// This was taken from <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
template <typename MemorySpace, typename ExecutionSpace, typename = void>
struct is_accessible_from : std::false_type
{
    static_assert( Kokkos::is_memory_space<MemorySpace>::value, "" );
    static_assert( Kokkos::is_execution_space<ExecutionSpace>::value, "" );
};

template <typename MemorySpace, typename ExecutionSpace>
struct is_accessible_from<MemorySpace, ExecutionSpace,
                          std::enable_if_t<Kokkos::SpaceAccessibility<
                              ExecutionSpace, MemorySpace>::accessible>>
    : std::true_type
{
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_TYPES_HPP
