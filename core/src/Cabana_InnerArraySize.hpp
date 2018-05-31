#ifndef CABANA_INNERARRAYSIZE_HPP
#define CABANA_INNERARRAYSIZE_HPP

#include <Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class InnerArraySize
  \brief Static inner array size definition. This is the size of the arrays in
  the struct-of-arrays. The size must be a power of 2 and greater than 0.
*/
template<int N,
         typename std::enable_if<
             (Impl::IsPowerOfTwo<N>::value && N > 0),int>::type = 0>
class InnerArraySize : public std::integral_constant<int,N> {};

//---------------------------------------------------------------------------//
// Static type checker.
template<class >
struct is_inner_array_size : public std::false_type {};

template<int N>
struct is_inner_array_size<InnerArraySize<N> > : public std::true_type {};

template<int N>
struct is_inner_array_size<const InnerArraySize<N> > : public std::true_type {};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_INNERARRAYSIZE_HPP
