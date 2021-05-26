/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_SOA_HPP
#define CABANA_SOA_HPP

#include <Cabana_MemberTypes.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <type_traits>
#include <utility>

namespace Cabana
{
//---------------------------------------------------------------------------//
// SoA forward declaration.
template <typename Types, int VectorLength>
struct SoA;

//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_soa_impl : public std::false_type
{
};

template <class DataTypes, int VectorLength>
struct is_soa_impl<SoA<DataTypes, VectorLength>> : public std::true_type
{
};

template <class T>
struct is_soa : public is_soa_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//

namespace Impl
{
//! \cond Impl

//---------------------------------------------------------------------------//
// Given an array layout and a potentially multi dimensional type T along with
// its rank, compose the inner array type.
template <typename T, std::size_t Rank, int VectorLength>
struct InnerArrayTypeImpl;

// rank-0 specialization.
template <typename T, int VectorLength>
struct InnerArrayTypeImpl<T, 0, VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    using type = value_type[VectorLength];
};

// rank-1 specialization.
template <typename T, int VectorLength>
struct InnerArrayTypeImpl<T, 1, VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T, 0>::value;
    using type = value_type[D0][VectorLength];
};

// rank-2 specialization.
template <typename T, int VectorLength>
struct InnerArrayTypeImpl<T, 2, VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T, 0>::value;
    static constexpr std::size_t D1 = std::extent<T, 1>::value;
    using type = value_type[D0][D1][VectorLength];
};

// rank-3 specialization.
template <typename T, int VectorLength>
struct InnerArrayTypeImpl<T, 3, VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T, 0>::value;
    static constexpr std::size_t D1 = std::extent<T, 1>::value;
    static constexpr std::size_t D2 = std::extent<T, 2>::value;
    using type = value_type[D0][D1][D2][VectorLength];
};

//---------------------------------------------------------------------------//
// Inner array type.
template <typename T, int VectorLength>
struct InnerArrayType
{
    using type =
        typename InnerArrayTypeImpl<T, std::rank<T>::value, VectorLength>::type;
};

//---------------------------------------------------------------------------//
/*!
  \brief Struct member.

  A statically sized array member of the struct. T can be of arbitrary type
  (including multidimensional arrays) as long as the type of T is trivial. A
  struct-of-arrays will be composed of these members of different types.
*/
template <std::size_t M, int VectorLength, typename T>
struct StructMember
{
    using array_type = typename InnerArrayType<T, VectorLength>::type;
    array_type _data;
};

//---------------------------------------------------------------------------//
// SoA implementation detail to hide the index sequence.
template <int VectorLength, typename Sequence, typename... Types>
struct SoAImpl;

template <int VectorLength, std::size_t... Indices, typename... Types>
struct SoAImpl<VectorLength, std::index_sequence<Indices...>, Types...>
    : StructMember<Indices, VectorLength, Types>...
{
};

//---------------------------------------------------------------------------//
// Given an SoA cast it to to one of its member types.
template <std::size_t M, class SoA_t>
KOKKOS_FORCEINLINE_FUNCTION const typename SoA_t::template base<M>&
soaMemberCast( const SoA_t& soa )
{
    static_assert( is_soa<SoA_t>::value, "soaMemberCast only for SoAs" );
    return static_cast<const typename SoA_t::template base<M>&>( soa );
}

template <std::size_t M, class SoA_t>
KOKKOS_FORCEINLINE_FUNCTION typename SoA_t::template base<M>&
soaMemberCast( SoA_t& soa )
{
    static_assert( is_soa<SoA_t>::value, "soaMemberCast only for SoAs" );
    return static_cast<typename SoA_t::template base<M>&>( soa );
}

//---------------------------------------------------------------------------//
// Get a pointer to the first element of a member in a given SoA.
template <std::size_t M, class SoA_t>
typename SoA_t::template member_pointer_type<M> soaMemberPtr( SoA_t* p )
{
    static_assert( is_soa<SoA_t>::value, "soaMemberPtr only for SoAs" );
    void* member = static_cast<typename SoA_t::template base<M>*>( p );
    return static_cast<typename SoA_t::template member_pointer_type<M>>(
        member );
}

//---------------------------------------------------------------------------//

//! \endcond
} // end namespace Impl

//---------------------------------------------------------------------------//
// Get template helper.

// Rank-0 non-const
template <std::size_t M, class SoA_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_soa<SoA_t>::value,
    typename SoA_t::template member_reference_type<M>>::type
get( SoA_t& soa, const std::size_t a )
{
    return Impl::soaMemberCast<M>( soa )._data[a];
}

// Rank-0 const
template <std::size_t M, class SoA_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_soa<SoA_t>::value,
    typename SoA_t::template member_const_reference_type<M>>::type
get( const SoA_t& soa, const std::size_t a )
{
    return Impl::soaMemberCast<M>( soa )._data[a];
}

// Rank-1 non-const
template <std::size_t M, class SoA_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_soa<SoA_t>::value,
    typename SoA_t::template member_reference_type<M>>::type
get( SoA_t& soa, const std::size_t a, const std::size_t d0 )
{
    return Impl::soaMemberCast<M>( soa )._data[d0][a];
}

// Rank-1 const
template <std::size_t M, class SoA_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_soa<SoA_t>::value,
    typename SoA_t::template member_const_reference_type<M>>::type
get( const SoA_t& soa, const std::size_t a, const std::size_t d0 )
{
    return Impl::soaMemberCast<M>( soa )._data[d0][a];
}

// Rank-2 non-const
template <std::size_t M, class SoA_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_soa<SoA_t>::value,
    typename SoA_t::template member_reference_type<M>>::type
get( SoA_t& soa, const std::size_t a, const std::size_t d0,
     const std::size_t d1 )
{
    return Impl::soaMemberCast<M>( soa )._data[d0][d1][a];
}

// Rank-2 const
template <std::size_t M, class SoA_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_soa<SoA_t>::value,
    typename SoA_t::template member_const_reference_type<M>>::type
get( const SoA_t& soa, const std::size_t a, const std::size_t d0,
     const std::size_t d1 )
{
    return Impl::soaMemberCast<M>( soa )._data[d0][d1][a];
}

// Rank-3 non-const
template <std::size_t M, class SoA_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_soa<SoA_t>::value,
    typename SoA_t::template member_reference_type<M>>::type
get( SoA_t& soa, const std::size_t a, const std::size_t d0,
     const std::size_t d1, const std::size_t d2 )
{
    return Impl::soaMemberCast<M>( soa )._data[d0][d1][d2][a];
}

// Rank-3 const
template <std::size_t M, class SoA_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_soa<SoA_t>::value,
    typename SoA_t::template member_const_reference_type<M>>::type
get( const SoA_t& soa, const std::size_t a, const std::size_t d0,
     const std::size_t d1, const std::size_t d2 )
{
    return Impl::soaMemberCast<M>( soa )._data[d0][d1][d2][a];
}

//---------------------------------------------------------------------------//
/*!
  \brief Struct-of-Arrays

  A struct-of-arrays (SoA) is composed of groups of statically sized
  arrays. The array element types, which will be composed as members of the
  struct, are indicated through the Types parameter pack. If the types of the
  members are contiguous then the struct itself will be contiguous. The vector
  length indicates the static length of each array.
*/
template <typename... Types, int VectorLength>
struct SoA<MemberTypes<Types...>, VectorLength>
    : Impl::SoAImpl<VectorLength, std::make_index_sequence<sizeof...( Types )>,
                    Types...>
{
    // Vector length
    static constexpr int vector_length = VectorLength;

    // Member data types.
    using member_types = MemberTypes<Types...>;

    // Number of member types.
    static constexpr std::size_t number_of_members = member_types::size;

    // The maximum rank supported for member types.
    static constexpr std::size_t max_supported_rank = 3;

    // Member data type.
    template <std::size_t M>
    using member_data_type = typename MemberTypeAtIndex<M, member_types>::type;

    // Value type at a given index M.
    template <std::size_t M>
    using member_value_type =
        typename std::remove_all_extents<member_data_type<M>>::type;

    // Reference type at a given index M.
    template <std::size_t M>
    using member_reference_type = member_value_type<M>&;

    // Const reference type at a given index M.
    template <std::size_t M>
    using member_const_reference_type = member_value_type<M> const&;

    // Pointer type at a given index M.
    template <std::size_t M>
    using member_pointer_type =
        typename std::add_pointer<member_value_type<M>>::type;

    // Base type.
    template <std::size_t M>
    using base = Impl::StructMember<M, vector_length, member_data_type<M>>;

    // -------------------------------
    // Member data type properties.

    /*!
      \brief Get the rank of the data for a given member at index M.

      \tparam M The member index to get the rank for.

      \return The rank of the given member index data.
    */
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION constexpr std::size_t rank() const
    {
        return std::rank<member_data_type<M>>::value;
    }

    /*!
      \brief Get the extent of a given member data dimension.

      \tparam M The member index to get the extent for.

      \tparam D The member data dimension to get the extent for.

      \return The extent of the dimension.
    */
    template <std::size_t M, std::size_t D>
    KOKKOS_FORCEINLINE_FUNCTION constexpr std::size_t extent() const
    {
        return std::extent<member_data_type<M>, D>::value;
    }
};

//---------------------------------------------------------------------------//

namespace Impl
{
//---------------------------------------------------------------------------//
// Member element copy operators.
//---------------------------------------------------------------------------//

// Copy a single member from one SoA to another.

// Rank 0
template <std::size_t M, int DstVectorLength, int SrcVectorLength,
          typename... Types>
KOKKOS_INLINE_FUNCTION typename std::enable_if<
    ( 0 == std::rank<typename MemberTypeAtIndex<
               M, MemberTypes<Types...>>::type>::value ),
    void>::type
soaElementMemberCopy( SoA<MemberTypes<Types...>, DstVectorLength>& dst,
                      const std::size_t dst_idx,
                      const SoA<MemberTypes<Types...>, SrcVectorLength>& src,
                      const std::size_t src_idx )
{
    get<M>( dst, dst_idx ) = get<M>( src, src_idx );
}

// Rank 1
template <std::size_t M, int DstVectorLength, int SrcVectorLength,
          typename... Types>
KOKKOS_INLINE_FUNCTION typename std::enable_if<
    ( 1 == std::rank<typename MemberTypeAtIndex<
               M, MemberTypes<Types...>>::type>::value ),
    void>::type
soaElementMemberCopy( SoA<MemberTypes<Types...>, DstVectorLength>& dst,
                      const std::size_t dst_idx,
                      const SoA<MemberTypes<Types...>, SrcVectorLength>& src,
                      const std::size_t src_idx )
{
    for ( std::size_t i0 = 0; i0 < dst.template extent<M, 0>(); ++i0 )
        get<M>( dst, dst_idx, i0 ) = get<M>( src, src_idx, i0 );
}

// Rank 2
template <std::size_t M, int DstVectorLength, int SrcVectorLength,
          typename... Types>
KOKKOS_INLINE_FUNCTION typename std::enable_if<
    ( 2 == std::rank<typename MemberTypeAtIndex<
               M, MemberTypes<Types...>>::type>::value ),
    void>::type
soaElementMemberCopy( SoA<MemberTypes<Types...>, DstVectorLength>& dst,
                      const std::size_t dst_idx,
                      const SoA<MemberTypes<Types...>, SrcVectorLength>& src,
                      const std::size_t src_idx )
{
    for ( std::size_t i0 = 0; i0 < dst.template extent<M, 0>(); ++i0 )
        for ( std::size_t i1 = 0; i1 < dst.template extent<M, 1>(); ++i1 )
            get<M>( dst, dst_idx, i0, i1 ) = get<M>( src, src_idx, i0, i1 );
}

// Rank 3
template <std::size_t M, int DstVectorLength, int SrcVectorLength,
          typename... Types>
KOKKOS_INLINE_FUNCTION typename std::enable_if<
    ( 3 == std::rank<typename MemberTypeAtIndex<
               M, MemberTypes<Types...>>::type>::value ),
    void>::type
soaElementMemberCopy( SoA<MemberTypes<Types...>, DstVectorLength>& dst,
                      const std::size_t dst_idx,
                      const SoA<MemberTypes<Types...>, SrcVectorLength>& src,
                      const std::size_t src_idx )
{
    for ( std::size_t i0 = 0; i0 < dst.template extent<M, 0>(); ++i0 )
        for ( std::size_t i1 = 0; i1 < dst.template extent<M, 1>(); ++i1 )
            for ( std::size_t i2 = 0; i2 < dst.template extent<M, 2>(); ++i2 )
                get<M>( dst, dst_idx, i0, i1, i2 ) =
                    get<M>( src, src_idx, i0, i1, i2 );
}

// Copy the values of all members of an SoA from a source to a destination at
// the given indices.
template <std::size_t M, int DstVectorLength, int SrcVectorLength,
          typename... Types>
KOKKOS_INLINE_FUNCTION void soaElementCopy(
    SoA<MemberTypes<Types...>, DstVectorLength>& dst, const std::size_t dst_idx,
    const SoA<MemberTypes<Types...>, SrcVectorLength>& src,
    const std::size_t src_idx, std::integral_constant<std::size_t, M> )
{
    soaElementMemberCopy<M>( dst, dst_idx, src, src_idx );
    soaElementCopy( dst, dst_idx, src, src_idx,
                    std::integral_constant<std::size_t, M - 1>() );
}

template <int DstVectorLength, int SrcVectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION void soaElementCopy(
    SoA<MemberTypes<Types...>, DstVectorLength>& dst, const std::size_t dst_idx,
    const SoA<MemberTypes<Types...>, SrcVectorLength>& src,
    const std::size_t src_idx, std::integral_constant<std::size_t, 0> )
{
    soaElementMemberCopy<0>( dst, dst_idx, src, src_idx );
}

// Copy the data from one struct at a given index to another.
template <int DstVectorLength, int SrcVectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION void
tupleCopy( SoA<MemberTypes<Types...>, DstVectorLength>& dst,
           const std::size_t dst_idx,
           const SoA<MemberTypes<Types...>, SrcVectorLength>& src,
           const std::size_t src_idx )
{
    soaElementCopy(
        dst, dst_idx, src, src_idx,
        std::integral_constant<std::size_t, sizeof...( Types ) - 1>() );
}

//---------------------------------------------------------------------------//

} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_SOA_HPP
