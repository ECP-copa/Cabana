#ifndef CABANA_INNERARRAYSIZE_HPP
#define CABANA_INNERARRAYSIZE_HPP

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class InnerArraySize
  \brief Static inner array size definition. This is the size of the arrays in
  the struct-of-arrays.
*/
template<std::size_t N>
class InnerArraySize : public std::integral_constant<std::size_t,N> {};

//---------------------------------------------------------------------------//
// Static type checker.
template<class >
struct is_inner_array_size : public std::false_type {};

template<std::size_t N>
struct is_inner_array_size<InnerArraySize<N> > : public std::true_type {};

template<std::size_t N>
struct is_inner_array_size<const InnerArraySize<N> > : public std::true_type {};

//---------------------------------------------------------------------------//
/*!
  \class ExecutionSpaceInnerArraySize
  \brief Inner array sizes specific for execution spaces.

  Specializations will set this specifically for the given space.
*/
template<class ExecutionSpace>
class ExecutionSpaceInnerArraySize;

//---------------------------------------------------------------------------//
// Serial specialization. Inner array size is 1.
#if defined( KOKKOS_ENABLE_SERIAL )
template<>
class ExecutionSpaceInnerArraySize<Kokkos::Serial>
    : public InnerArraySize<1> {};
#endif

//---------------------------------------------------------------------------//
// OpenMP specialization.
#if defined( KOKKOS_ENABLE_OPENMP )
template<>
class ExecutionSpaceInnerArraySize<Kokkos::OpenMP>
    : public InnerArraySize<64> {};
#endif

//---------------------------------------------------------------------------//
// Cuda specialization. Use the warp size.
#if defined( KOKKOS_ENABLE_CUDA )
template<>
class ExecutionSpaceInnerArraySize<Kokkos::Cuda>
    : public InnerArraySize<Kokkos::Impl::CudaTraits::WarpSize> {};
#endif

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_INNERARRAYSIZE_HPP
