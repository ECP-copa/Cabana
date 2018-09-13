#ifndef CABANA_TYPES_HPP
#define CABANA_TYPES_HPP

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Memory spaces
//---------------------------------------------------------------------------//
template<class >
struct is_memory_space : public std::false_type {};

//! Host memory space
struct HostSpace
{
    using memory_space_type = HostSpace;
    using kokkos_memory_space = Kokkos::HostSpace;
    using kokkos_execution_space =
        typename kokkos_memory_space::execution_space;
};

template<>
struct is_memory_space<HostSpace> : public std::true_type {};

#if defined( KOKKOS_ENABLE_CUDA )
//! Cuda UVM memory space
struct CudaUVMSpace
{
    using memory_space_type = CudaUVMSpace;
    using kokkos_memory_space = Kokkos::CudaUVMSpace;
    using kokkos_execution_space =
        typename kokkos_memory_space::execution_space;
};

template<>
struct is_memory_space<CudaUVMSpace> : public std::true_type {};
#endif

//---------------------------------------------------------------------------//
// Memory access tags.
//---------------------------------------------------------------------------//
template<class >
struct is_memory_access_tag : public std::false_type {};

//! Default memory access. Default memory is restricted to prevent aliasing in
//! the larger AoSoA memory block to allow for potential vectorization.
struct DefaultAccessMemory
{
    using memory_access_type = DefaultAccessMemory;
    using kokkos_memory_traits = Kokkos::MemoryTraits< Kokkos::Unmanaged |
                                                       Kokkos::Aligned |
                                                       Kokkos::Restrict >;
};

template<>
struct is_memory_access_tag<DefaultAccessMemory> : public std::true_type {};

//! Random access memory. Read-only and const with limited spatial locality.
struct RandomAccessMemory
{
    using memory_access_type = RandomAccessMemory;
    using kokkos_memory_traits = Kokkos::MemoryTraits< Kokkos::Unmanaged |
                                                       Kokkos::Aligned |
                                                       Kokkos::RandomAccess >;
};

template<>
struct is_memory_access_tag<RandomAccessMemory> : public std::true_type {};

//! Atomic memory access. All reads and writes are atomic.
struct AtomicAccessMemory
{
    using memory_access_type = AtomicAccessMemory;
    using kokkos_memory_traits = Kokkos::MemoryTraits< Kokkos::Unmanaged |
                                                       Kokkos::Aligned |
                                                       Kokkos::Atomic >;
};

template<>
struct is_memory_access_tag<AtomicAccessMemory> : public std::true_type {};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_TYPES_HPP
