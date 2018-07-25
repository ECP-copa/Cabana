#ifndef CABANA_SOA_HPP
#define CABANA_SOA_HPP

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
template<std::size_t I, typename InnerArrayLayout_t, typename T>
struct StructMember
{
    // type aliases
    using data_type = T;
    using array_type = typename Impl::InnerArrayType<T,InnerArrayLayout_t>::type;
    using array_layout = typename InnerArrayLayout_t::layout;
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
    typename std::enable_if<
        (1==std::rank<U>::value &&
         std::is_same<array_layout,Kokkos::LayoutRight>::value),
        reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0 )
    { return m._data[i][d0]; }

    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (1==std::rank<U>::value &&
         std::is_same<array_layout,Kokkos::LayoutLeft>::value),
        reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0 )
    { return m._data[d0][i]; }

    // rank-2
    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (2==std::rank<U>::value &&
         std::is_same<array_layout,Kokkos::LayoutRight>::value),
        reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1 )
    { return m._data[i][d0][d1]; }

    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (2==std::rank<U>::value &&
         std::is_same<array_layout,Kokkos::LayoutLeft>::value),
        reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1 )
    { return m._data[d1][d0][i]; }

    // rank-3
    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (3==std::rank<U>::value &&
         std::is_same<array_layout,Kokkos::LayoutRight>::value),
        reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1,
            const int d2 )
    { return m._data[i][d0][d1][d2]; }

    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (3==std::rank<U>::value &&
         std::is_same<array_layout,Kokkos::LayoutLeft>::value),
        reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1,
            const int d2 )
    { return m._data[d2][d1][d0][i]; }

    // rank-4
    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (4==std::rank<U>::value &&
         std::is_same<array_layout,Kokkos::LayoutRight>::value),
        reference_type>::type
    access( StructMember& m,
            const int i,
            const int d0,
            const int d1,
            const int d2,
            const int d3 )
    { return m._data[i][d0][d1][d2][d3]; }

    template<class U = T>
    static KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (4==std::rank<U>::value &&
         std::is_same<array_layout,Kokkos::LayoutLeft>::value),
        reference_type>::type
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
template<typename InnerArrayLayout_t, typename Sequence, typename... Types>
struct SoAImpl;

template<typename InnerArrayLayout_t, std::size_t... Indices, typename... Types>
struct SoAImpl<InnerArrayLayout_t,Impl::IndexSequence<Indices...>,Types...>
    : StructMember<Indices,InnerArrayLayout_t,Types>...
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
template<typename InnerArrayLayout_t, typename... Types>
struct SoA;

template<typename InnerArrayLayout_t, typename... Types>
struct SoA<InnerArrayLayout_t,MemberDataTypes<Types...> >
    : SoAImpl<InnerArrayLayout_t,
              typename Impl::MakeIndexSequence<sizeof...(Types)>::type,
              Types...>
{};

//---------------------------------------------------------------------------//
// Helper traits.
template<std::size_t I, typename InnerArrayLayout_t, typename... Types>
struct SMT
{
    using type = StructMember<
        I,InnerArrayLayout_t,
        typename MemberDataTypeAtIndex<I,MemberDataTypes<Types...> >::type>;
    using data_type = typename type::data_type;
    using reference_type = typename type::reference_type;
    using pointer_type = typename type::pointer_type;
};

//---------------------------------------------------------------------------//
/*!
  \brief Access an individual element of a member.
*/
template<std::size_t I, typename InnerArrayLayout_t, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (0==std::rank<typename SMT<I,InnerArrayLayout_t,Types...>::data_type>::value),
    typename SMT<I,InnerArrayLayout_t,Types...>::reference_type>::type
accessStructMember( SoA<InnerArrayLayout_t,MemberDataTypes<Types...> >& s,
                    const int i )
{
    return SMT<I,InnerArrayLayout_t,Types...>::type::access( s, i );
}

template<std::size_t I, typename InnerArrayLayout_t, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (1==std::rank<typename SMT<I,InnerArrayLayout_t,Types...>::data_type>::value),
    typename SMT<I,InnerArrayLayout_t,Types...>::reference_type>::type
accessStructMember( SoA<InnerArrayLayout_t,MemberDataTypes<Types...> >& s,
                    const int i,
                    const int d0 )
{
    return SMT<I,InnerArrayLayout_t,Types...>::type::access(s,i,d0);
}

template<std::size_t I, typename InnerArrayLayout_t, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (2==std::rank<typename SMT<I,InnerArrayLayout_t,Types...>::data_type>::value),
    typename SMT<I,InnerArrayLayout_t,Types...>::reference_type>::type
accessStructMember( SoA<InnerArrayLayout_t,MemberDataTypes<Types...> >& s,
                    const int i,
                    const int d0,
                    const int d1 )
{
    return SMT<I,InnerArrayLayout_t,Types...>::type::access(s,i,d0,d1);
}

template<std::size_t I, typename InnerArrayLayout_t, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (3==std::rank<typename SMT<I,InnerArrayLayout_t,Types...>::data_type>::value),
    typename SMT<I,InnerArrayLayout_t,Types...>::reference_type>::type
accessStructMember( SoA<InnerArrayLayout_t,MemberDataTypes<Types...> >& s,
                    const int i,
                    const int d0,
                    const int d1,
                    const int d2 )
{
    return SMT<I,InnerArrayLayout_t,Types...>::type::access(s,i,d0,d1,d2);
}

template<std::size_t I, typename InnerArrayLayout_t, typename... Types>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (4==std::rank<typename SMT<I,InnerArrayLayout_t,Types...>::data_type>::value),
    typename SMT<I,InnerArrayLayout_t,Types...>::reference_type>::type
accessStructMember( SoA<InnerArrayLayout_t,MemberDataTypes<Types...> >& s,
                    const int i,
                    const int d0,
                    const int d1,
                    const int d2,
                    const int d3 )
{
    return SMT<I,InnerArrayLayout_t,Types...>::type::access(s,i,d0,d1,d2,d3);
}

//---------------------------------------------------------------------------//
/*!
  \brief Get a pointer to a member.
*/
template<std::size_t I, typename InnerArrayLayout_t, typename... Types>
KOKKOS_INLINE_FUNCTION
typename SMT<I,InnerArrayLayout_t,Types...>::pointer_type
getStructMember( SoA<InnerArrayLayout_t,MemberDataTypes<Types...> >& soa )
{
    return static_cast<
        typename SMT<I,InnerArrayLayout_t,Types...>::type&>(soa)._data;
}

//---------------------------------------------------------------------------//

} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_SOA_HPP
