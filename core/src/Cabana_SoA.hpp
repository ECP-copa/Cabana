#ifndef CABANA_SOA_HPP
#define CABANA_SOA_HPP

#include <Cabana_IndexSequence.hpp>
#include <Cabana_MemberDataTypes.hpp>

#include <Kokkos_Core.hpp>

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
    using data_type = T;
    using array_type = T[ArraySize];
    using value_type = typename std::remove_all_extents<T>::type;
    using reference_type = typename std::add_lvalue_reference<value_type>::type;
    using pointer_type = typename std::decay<array_type>::type;

    array_type _data;

    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<U>::value),reference_type>::type
    access( StructMember& m,
            const int i )
    { return m._data[i]; }

    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==std::rank<U>::value),reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0 )
    { return m._data[i][d0]; }

    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==std::rank<U>::value),reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1 )
    { return m._data[i][d0][d1]; }

    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(3==std::rank<U>::value),reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1,
            const int d2 )
    { return m._data[i][d0][d1][d2]; }

    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(4==std::rank<U>::value),reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1,
            const int d2,
            const int d3 )
    { return m._data[i][d0][d1][d2][d3]; }
};

//---------------------------------------------------------------------------//
// SoA implementation detail to hide the index sequence.
template<std::size_t ArraySize, typename Sequence, typename... Types>
struct SoAImpl;

template<std::size_t ArraySize, std::size_t... Indices, typename... Types>
struct SoAImpl<ArraySize,IndexSequence<Indices...>,Types...>
    : StructMember<Indices,ArraySize,Types>...
{};

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
  auto a3x3 = accessStructMember<2>( soa );
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
// Helper traits.
template<std::size_t I, std::size_t ArraySize, typename... Types>
struct SMT
{
    using type = StructMember<
        I,ArraySize,typename MemberDataTypeAtIndex<I,Types...>::type>;
    using data_type = typename type::data_type;
    using reference_type = typename type::reference_type;
    using pointer_type = typename type::pointer_type;
};

//---------------------------------------------------------------------------//
/*!
  \brief Access an individual element of a member.
*/
template<std::size_t I, std::size_t ArraySize, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (0==std::rank<typename SMT<I,ArraySize,Types...>::data_type>::value),
    typename SMT<I,ArraySize,Types...>::reference_type>::type
accessStructMember( SoA<ArraySize,Types...>& s,
                    const int i )
{
    return SMT<I,ArraySize,Types...>::type::access( s, i );
}

template<std::size_t I, std::size_t ArraySize, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (1==std::rank<typename SMT<I,ArraySize,Types...>::data_type>::value),
    typename SMT<I,ArraySize,Types...>::reference_type>::type
accessStructMember( SoA<ArraySize,Types...>& s,
                    const int i,
                    const int d0 )
{
    return SMT<I,ArraySize,Types...>::type::access(s,i,d0);
}

template<std::size_t I, std::size_t ArraySize, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (2==std::rank<typename SMT<I,ArraySize,Types...>::data_type>::value),
    typename SMT<I,ArraySize,Types...>::reference_type>::type
accessStructMember( SoA<ArraySize,Types...>& s,
                    const int i,
                    const int d0,
                    const int d1 )
{
    return SMT<I,ArraySize,Types...>::type::access(s,i,d0,d1);
}

template<std::size_t I, std::size_t ArraySize, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (3==std::rank<typename SMT<I,ArraySize,Types...>::data_type>::value),
    typename SMT<I,ArraySize,Types...>::reference_type>::type
accessStructMember( SoA<ArraySize,Types...>& s,
                    const int i,
                    const int d0,
                    const int d1,
                    const int d2 )
{
    return SMT<I,ArraySize,Types...>::type::access(s,i,d0,d1,d2);
}

template<std::size_t I, std::size_t ArraySize, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (4==std::rank<typename SMT<I,ArraySize,Types...>::data_type>::value),
    typename SMT<I,ArraySize,Types...>::reference_type>::type
accessStructMember( SoA<ArraySize,Types...>& s,
                    const int i,
                    const int d0,
                    const int d1,
                    const int d2,
                    const int d3 )
{
    return SMT<I,ArraySize,Types...>::type::access(s,i,d0,d1,d2,d3);
}

//---------------------------------------------------------------------------//
/*!
  \brief Get a pointer to a member.
*/
template<std::size_t I, std::size_t ArraySize, typename... Types>
KOKKOS_INLINE_FUNCTION
typename SMT<I,ArraySize,Types...>::pointer_type
getStructMember( SoA<ArraySize,Types...>& soa )
{
    typename SMT<I,ArraySize,Types...>::type& base = soa;
    return base._data;
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_SOA_HPP
