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

  Default version has an inner array size of 1. Specializations will set this
  specifically for the given space.
*/
template<class ExecutionSpace>
class ExecutionSpaceInnerArraySize : public InnerArraySize<1> {};

//---------------------------------------------------------------------------//
// Serial specialization.
#if defined( KOKKOS_ENABLE_SERIAL )

#endif

//---------------------------------------------------------------------------//
// OpenMP specialization.
#if defined( KOKKOS_ENABLE_OPENMP )

#endif

//---------------------------------------------------------------------------//
// Cuda specialization.
#if defined( KOKKOS_ENABLE_CUDA )

#endif

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_INNERARRAYSIZE_HPP
