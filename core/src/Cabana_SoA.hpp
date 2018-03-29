#ifndef CABANA_SOA_HPP
#define CABANA_SOA_HPP

#include <Cabana_IndexSequence.hpp>
#include <Cabana_MemberDataTypes.hpp>

#include <Kokkos_Macros.hpp>

#include <type_traits>
#include <cstdlib>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \brief Struct member.

  A statically sized array member of the struct. T can be of arbitrary type
  (including multidimensional arrays) as long as the type of T is trivial. A
  struct-of-arrays will be composed of these members of different types.
*/
template<std::size_t I, std::size_t ArraySize, typename T>
struct StructMember
{
    T _data[ArraySize];
};

//---------------------------------------------------------------------------//
// SoA implementation detail to hide the index sequence.
template<std::size_t ArraySize, typename Sequence, typename... Types>
struct SoAImpl;

template<std::size_t ArraySize, std::size_t... Indices, typename... Types>
struct SoAImpl<ArraySize,IndexSequence<Indices...>,Types...>
    : StructMember<Indices,ArraySize,Types>...
{
    static constexpr std::size_t array_size = ArraySize;
};

//---------------------------------------------------------------------------//
/*!
  \brief Struct-of-Arrays

  A struct-of-arrays (SoA) is composed of groups of statically sized arrays
  of size ArraySize. The array element types, which will be composed as
  members of the struct, are indicated through the Types parameter pack. If
  the types of the members are contiguous then the struct itself will be
  contiguous.

  Example - The SoA defined as: SoA<4,double,int,double[3][3]>
  gives the same data structure as:

  struct MySoA
  {
    double _d1[4];
    int    _d2[4];
    double _d3[4][3][3];
  };

  The SoA data is accessed through a get function. In the example above,
  getting the 3x3 array for a given element and assigning values in the
  struct would be written as:

  SoA<4,double,int,double[3][3]> soa;
  auto a3x3 = getStructMember<2>( soa );
  for ( int b = 0; b < 4; ++b )
    for ( int i = 0; i < 3; ++i )
      for ( int j = 0; j < 3; ++j )
        a3x3[b][i][j] = some_value;

*/
template<std::size_t ArraySize, typename... Types>
struct SoA
    : SoAImpl<ArraySize,
              typename MakeIndexSequence<sizeof...(Types)>::type,
              Types...>
{};

//---------------------------------------------------------------------------//
// Member array types.
template<std::size_t I, std::size_t ArraySize, typename... Types>
struct ArrayTypeAtIndex
{
    using member_type = typename MemberDataTypeAtIndex<I,Types...>::type;
    using array_type = member_type[ArraySize];
    using return_type = typename std::decay<array_type>::type;
};

//---------------------------------------------------------------------------//
/*!
  \brief Member accessor.

  Access the member of the struct at index I.

  Example:

  SoA<4,double,int,double[3][3]> soa;
  double*       d0 = getStructMember<0>( soa );
  int*          d1 = getStructMember<1>( soa );
  double*[3][3] d2 = getStructMember<2>( soa );
*/
template<std::size_t I, std::size_t ArraySize, typename... Types>
KOKKOS_INLINE_FUNCTION
typename ArrayTypeAtIndex<I,ArraySize,Types...>::return_type
getStructMember( SoA<ArraySize,Types...>& soa )
{
    StructMember<
        I,
        ArraySize,
        typename ArrayTypeAtIndex<I,ArraySize,Types...>::member_type>& base =
        soa;
    return base._data;
}

//---------------------------------------------------------------------------//
} // end namespace Cabana

#endif // end CABANA_SOA_HPP
