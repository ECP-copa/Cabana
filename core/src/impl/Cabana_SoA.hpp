#ifndef CABANA_SOA_HPP
#define CABANA_SOA_HPP

#include <Cabana_Types.hpp>
#include <Cabana_InnerArrayLayout.hpp>
#include <impl/Cabana_IndexSequence.hpp>
#include <Cabana_MemberDataTypes.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <cstdlib>

namespace Cabana
{
namespace Impl
{
//---------------------------------------------------------------------------//
/*!
  \brief Struct member.

  A statically sized array member of the struct. T can be of arbitrary type
  (including multidimensional arrays) as long as the type of T is trivial. A
  struct-of-arrays will be composed of these members of different types.
*/
template<std::size_t I, int VectorLength, typename T>
struct StructMember
{
    // type aliases
    using data_type = T;
    using array_type = typename Impl::InnerArrayType<T,VectorLength>::type;
    using value_type = typename std::remove_all_extents<T>::type;
    using reference_type = typename std::add_lvalue_reference<value_type>::type;
    using pointer_type = typename std::decay<array_type>::type;

    // data
    array_type _data;

    // rank-0
    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<U>::value),reference_type>::type
    access( StructMember& m,
            const int i )
    { return m._data[i]; }

    // rank-1
    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==std::rank<U>::value),reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0 )
    { return m._data[d0][i]; }

    // rank-2
    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==std::rank<U>::value),reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1 )
    { return m._data[d1][d0][i]; }

    // rank-3
    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(3==std::rank<U>::value),reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1,
            const int d2 )
    { return m._data[d2][d1][d0][i]; }

    // rank-4
    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(4==std::rank<U>::value),reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1,
            const int d2,
            const int d3 )
    { return m._data[d3][d2][d1][d0][i]; }
};

//---------------------------------------------------------------------------//
// SoA implementation detail to hide the index sequence.
template<int VectorLength, typename Sequence, typename... Types>
struct SoAImpl;

template<int VectorLength, std::size_t... Indices, typename... Types>
struct SoAImpl<VectorLength,Impl::IndexSequence<Indices...>,Types...>
    : StructMember<Indices,VectorLength,Types>...
{};

//---------------------------------------------------------------------------//
/*!
  \brief Struct-of-Arrays

  A struct-of-arrays (SoA) is composed of groups of statically sized
  arrays. The array element types, which will be composed as members of the
  struct, are indicated through the Types parameter pack. If the types of the
  members are contiguous then the struct itself will be contiguous. The layout
  of the arrays is a function of the layout type. The layout type indicates
  the size of the arrays and, if they have multidimensional data, if they are
  row or column major order.
*/
template<int VectorLength, typename... Types>
struct SoA;

template<int VectorLength, typename... Types>
struct SoA<VectorLength,MemberDataTypes<Types...> >
    : SoAImpl<VectorLength,
              typename Impl::MakeIndexSequence<sizeof...(Types)>::type,
              Types...>
{};

//---------------------------------------------------------------------------//
// Helper traits.
template<std::size_t I, int VectorLength, typename... Types>
struct SMT
{
    using type = StructMember<
        I,VectorLength,typename MemberDataTypeAtIndex<I,MemberDataTypes<Types...> >::type>;
    using data_type = typename type::data_type;
    using reference_type = typename type::reference_type;
    using pointer_type = typename type::pointer_type;
};

//---------------------------------------------------------------------------//
/*!
  \brief Access an individual element of a member.
*/
template<std::size_t I, int VectorLength, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (0==std::rank<typename SMT<I,VectorLength,Types...>::data_type>::value),
    typename SMT<I,VectorLength,Types...>::reference_type>::type
accessStructMember( SoA<VectorLength,MemberDataTypes<Types...> >& s,
                    const int i )
{
    return SMT<I,VectorLength,Types...>::type::access( s, i );
}

template<std::size_t I, int VectorLength, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (1==std::rank<typename SMT<I,VectorLength,Types...>::data_type>::value),
    typename SMT<I,VectorLength,Types...>::reference_type>::type
accessStructMember( SoA<VectorLength,MemberDataTypes<Types...> >& s,
                    const int i,
                    const int d0 )
{
    return SMT<I,VectorLength,Types...>::type::access(s,i,d0);
}

template<std::size_t I, int VectorLength, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (2==std::rank<typename SMT<I,VectorLength,Types...>::data_type>::value),
    typename SMT<I,VectorLength,Types...>::reference_type>::type
accessStructMember( SoA<VectorLength,MemberDataTypes<Types...> >& s,
                    const int i,
                    const int d0,
                    const int d1 )
{
    return SMT<I,VectorLength,Types...>::type::access(s,i,d0,d1);
}

template<std::size_t I, int VectorLength, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (3==std::rank<typename SMT<I,VectorLength,Types...>::data_type>::value),
    typename SMT<I,VectorLength,Types...>::reference_type>::type
accessStructMember( SoA<VectorLength,MemberDataTypes<Types...> >& s,
                    const int i,
                    const int d0,
                    const int d1,
                    const int d2 )
{
    return SMT<I,VectorLength,Types...>::type::access(s,i,d0,d1,d2);
}

template<std::size_t I, int VectorLength, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (4==std::rank<typename SMT<I,VectorLength,Types...>::data_type>::value),
    typename SMT<I,VectorLength,Types...>::reference_type>::type
accessStructMember( SoA<VectorLength,MemberDataTypes<Types...> >& s,
                    const int i,
                    const int d0,
                    const int d1,
                    const int d2,
                    const int d3 )
{
    return SMT<I,VectorLength,Types...>::type::access(s,i,d0,d1,d2,d3);
}

//---------------------------------------------------------------------------//
/*!
  \brief Get a pointer to a member.
*/
template<std::size_t I, int VectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION
typename SMT<I,VectorLength,Types...>::pointer_type
getStructMember( SoA<VectorLength,MemberDataTypes<Types...> >& soa )
{
    return static_cast<typename SMT<I,VectorLength,Types...>::type&>(soa)._data;
}

//---------------------------------------------------------------------------//

} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_SOA_HPP
