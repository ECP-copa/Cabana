#ifndef CABANA_SOA_HPP
#define CABANA_SOA_HPP

#include <Cabana_Types.hpp>
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
// Given an array layout and a potentially multi dimensional type T along with
// its rank, compose the inner array type.
template<typename T, std::size_t Rank, int VectorLength>
struct InnerArrayTypeImpl;

// rank-0 specialization.
template<typename T, int VectorLength>
struct InnerArrayTypeImpl<T,0,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    using type = value_type[VectorLength];
};

// rank-1 specialization.
template<typename T, int VectorLength>
struct InnerArrayTypeImpl<T,1,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    using type = value_type[D0][VectorLength];
};

// rank-2 specialization.
template<typename T, int VectorLength>
struct InnerArrayTypeImpl<T,2,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    using type = value_type[D1][D0][VectorLength];
};

// rank-3 specialization.
template<typename T, int VectorLength>
struct InnerArrayTypeImpl<T,3,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    using type = value_type[D2][D1][D0][VectorLength];
};

// rank-4 specialization.
template<typename T, int VectorLength>
struct InnerArrayTypeImpl<T,4,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    static constexpr std::size_t D3 = std::extent<T,3>::value;
    using type = value_type[D3][D2][D1][D0][VectorLength];
};

//---------------------------------------------------------------------------//
// Inner array type.
template<typename T,int VectorLength>
struct InnerArrayType
{
    using type =
        typename InnerArrayTypeImpl<T,std::rank<T>::value,VectorLength>::type;
};

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
    using array_type = typename InnerArrayType<T,VectorLength>::type;
    array_type _data;
};

//---------------------------------------------------------------------------//
// SoA implementation detail to hide the index sequence.
template<int VectorLength, typename Sequence, typename... Types>
struct SoAImpl;

template<int VectorLength, std::size_t... Indices, typename... Types>
struct SoAImpl<VectorLength,IndexSequence<Indices...>,Types...>
    : StructMember<Indices,VectorLength,Types>...
{};

//---------------------------------------------------------------------------//

} // end namespace Impl

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
    : Impl::SoAImpl<VectorLength,
                    typename Impl::MakeIndexSequence<sizeof...(Types)>::type,
                    Types...>
{
    // Vector length
    static constexpr int vector_length = VectorLength;

    // Member data types.
    using member_types = MemberDataTypes<Types...>;

    // Number of member types.
    static constexpr std::size_t number_of_members = member_types::size;

    // The maximum rank supported for member types.
    static constexpr std::size_t max_supported_rank = 4;

    // Member data type.
    template<std::size_t M>
    using member_data_type =
        typename MemberDataTypeAtIndex<M,member_types>::type;

    // Value type at a given index M.
    template<std::size_t M>
    using member_value_type =
        typename std::remove_all_extents<member_data_type<M> >::type;

    // Reference type at a given index M.
    template<std::size_t M>
    using member_reference_type =
        typename std::add_lvalue_reference<member_value_type<M> >::type;

    // Pointer type at a given index M.
    template<std::size_t M>
    using member_pointer_type =
        typename std::add_pointer<member_value_type<M> >::type;


    // -------------------------------
    // Member data type properties.

    /*!
      \brief Get the rank of the data for a given member at index M.

      \tparam M The member index to get the rank for.

      \return The rank of the given member index data.
    */
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    constexpr int rank() const
    {
        return std::rank<member_data_type<M> >::value;
    }

    /*!
      \brief Get the extent of a given member data dimension.

      \tparam M The member index to get the extent for.

      \tparam D The member data dimension to get the extent for.

      \return The extent of the dimension.
    */
    template<std::size_t M, std::size_t D>
    KOKKOS_FORCEINLINE_FUNCTION
    constexpr int extent() const
    {
        return std::extent<member_data_type<M>,D>::value;
    }

    // -------------------------------
    // Access the data value at a given member index.

    // Rank 0
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int i )
    {
        Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return base._data[i];
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int i ) const
    {
        const Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return base._data[i];
    }

    // Rank 1
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get(  const int i,
          const int d0 )
    {
        Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return base._data[d0][i];
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get(  const int i,
          const int d0 ) const
    {
        const Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return base._data[d0][i];
    }

    // Rank 2
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int i,
         const int d0,
         const int d1 )
    {
        Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return base._data[d1][d0][i];
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int i,
         const int d0,
         const int d1 ) const
    {
        const Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return base._data[d1][d0][i];
    }

    // Rank 3
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(3==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int i,
         const int d0,
         const int d1,
         const int d2 )
    {
        Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return base._data[d2][d1][d0][i];
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(3==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int i,
         const int d0,
         const int d1,
         const int d2 ) const
    {
        const Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return base._data[d2][d1][d0][i];
    }

    // Rank 4
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(4==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int i,
         const int d0,
         const int d1,
         const int d2,
         const int d3 )
    {
        Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return base._data[d3][d2][d1][d0][i];
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(4==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int i,
         const int d0,
         const int d1,
         const int d2,
         const int d3 ) const
    {
        const Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return base._data[d3][d2][d1][d0][i];
    }

    // ----------------
    // Raw data access

    // Get a pointer to a member.
    template<std::size_t M>
    void* ptr()
    {
        Impl::StructMember<M,vector_length,member_data_type<M> >& base = *this;
        return &base;
    }
};

//---------------------------------------------------------------------------//
// Member element copy operators.
//---------------------------------------------------------------------------//

namespace Impl
{

// Copy a single member from one SoA to another.

// Rank 0
template<std::size_t M, int DstVectorLength, int SrcVectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    (0==std::rank<typename MemberDataTypeAtIndex<M,MemberDataTypes<Types...> >::type>::value),void>::type
soaElementMemberCopy( SoA<DstVectorLength,MemberDataTypes<Types...> >& dst,
                      const int dst_idx,
                      const SoA<SrcVectorLength,MemberDataTypes<Types...> >& src,
                      const int src_idx )
{
    dst.template get<M>( dst_idx ) = src.template get<M>( src_idx );
}

// Rank 1
template<std::size_t M, int DstVectorLength, int SrcVectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    (1==std::rank<typename MemberDataTypeAtIndex<M,MemberDataTypes<Types...> >::type>::value),void>::type
soaElementMemberCopy( SoA<DstVectorLength,MemberDataTypes<Types...> >& dst,
                      const int dst_idx,
                      const SoA<SrcVectorLength,MemberDataTypes<Types...> >& src,
                      const int src_idx )
{
    for ( int i0 = 0; i0 < dst.template extent<M,0>(); ++i0 )
        dst.template get<M>( dst_idx, i0 ) = src.template get<M>( src_idx, i0 );
}

// Rank 2
template<std::size_t M, int DstVectorLength, int SrcVectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    (2==std::rank<typename MemberDataTypeAtIndex<M,MemberDataTypes<Types...> >::type>::value),void>::type
soaElementMemberCopy( SoA<DstVectorLength,MemberDataTypes<Types...> >& dst,
                      const int dst_idx,
                      const SoA<SrcVectorLength,MemberDataTypes<Types...> >& src,
                      const int src_idx )
{
    for ( int i0 = 0; i0 < dst.template extent<M,0>(); ++i0 )
        for ( int i1 = 0; i1 < dst.template extent<M,1>(); ++i1 )
                dst.template get<M>( dst_idx, i0, i1 ) =
                    src.template get<M>( src_idx, i0, i1 );
}

// Rank 3
template<std::size_t M, int DstVectorLength, int SrcVectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    (3==std::rank<typename MemberDataTypeAtIndex<M,MemberDataTypes<Types...> >::type>::value),void>::type
soaElementMemberCopy( SoA<DstVectorLength,MemberDataTypes<Types...> >& dst,
                      const int dst_idx,
                      const SoA<SrcVectorLength,MemberDataTypes<Types...> >& src,
                      const int src_idx )
{
    for ( int i0 = 0; i0 < dst.template extent<M,0>(); ++i0 )
        for ( int i1 = 0; i1 < dst.template extent<M,1>(); ++i1 )
            for ( int i2 = 0; i2 < dst.template extent<M,2>(); ++i2 )
                dst.template get<M>( dst_idx, i0, i1, i2 ) =
                    src.template get<M>( src_idx, i0, i1, i2 );
}

// Rank r
template<std::size_t M, int DstVectorLength, int SrcVectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION
typename std::enable_if<
    (4==std::rank<typename MemberDataTypeAtIndex<M,MemberDataTypes<Types...> >::type>::value),void>::type
soaElementMemberCopy( SoA<DstVectorLength,MemberDataTypes<Types...> >& dst,
                      const int dst_idx,
                      const SoA<SrcVectorLength,MemberDataTypes<Types...> >& src,
                      const int src_idx )
{
    for ( int i0 = 0; i0 < dst.template extent<M,0>(); ++i0 )
        for ( int i1 = 0; i1 < dst.template extent<M,1>(); ++i1 )
            for ( int i2 = 0; i2 < dst.template extent<M,2>(); ++i2 )
                for ( int i3 = 0; i3 < dst.template extent<M,3>(); ++i3 )
                    dst.template get<M>( dst_idx, i0, i1, i2, i3 ) =
                        src.template get<M>( src_idx, i0, i1, i2, i3 );
}

// Copy the values of all members of an SoA from a source to a destination at
// the given indices.
template<std::size_t M, int DstVectorLength, int SrcVectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION
void soaElementCopy( SoA<DstVectorLength,MemberDataTypes<Types...> >& dst,
                     const int dst_idx,
                     const SoA<SrcVectorLength,MemberDataTypes<Types...> >& src,
                     const int src_idx,
                     std::integral_constant<std::size_t,M> )
{
    soaElementMemberCopy<M>( dst, dst_idx, src, src_idx );
    soaElementCopy( dst, dst_idx, src, src_idx,
                    std::integral_constant<std::size_t,M-1>() );
}

template<int DstVectorLength, int SrcVectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION
void soaElementCopy( SoA<DstVectorLength,MemberDataTypes<Types...> >& dst,
                     const int dst_idx,
                     const SoA<SrcVectorLength,MemberDataTypes<Types...> >& src,
                     const int src_idx,
                     std::integral_constant<std::size_t,0> )
{
    soaElementMemberCopy<0>( dst, dst_idx, src, src_idx );
}

// Copy the data from one struct at a given index to another.
template<int DstVectorLength, int SrcVectorLength, typename... Types>
KOKKOS_INLINE_FUNCTION
void tupleCopy( SoA<DstVectorLength,MemberDataTypes<Types...> >& dst,
                const int dst_idx,
                const SoA<SrcVectorLength,MemberDataTypes<Types...> >& src,
                const int src_idx )
{
    soaElementCopy( dst, dst_idx, src, src_idx,
                    std::integral_constant<std::size_t,sizeof...(Types)-1>() );
}

//---------------------------------------------------------------------------//

} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_SOA_HPP
