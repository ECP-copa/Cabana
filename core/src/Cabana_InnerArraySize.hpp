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

} // end namespace Cabana

#endif // end CABANA_INNERARRAYSIZE_HPP
