/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_DeepCopy.hpp
  \brief AoSoA and slice extensions for Kokkos deep copy and mirrors
*/
#ifndef CABANA_DEEPCOPY_HPP
#define CABANA_DEEPCOPY_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_ParticleList.hpp>
#include <Cabana_Slice.hpp>
#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>

#include <exception>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \brief Allocate a mirror of the given AoSoA in the given space.
  \return AoSoA in the new space.
 */
template <class Space, class SrcAoSoA>
inline AoSoA<typename SrcAoSoA::member_types, Space, SrcAoSoA::vector_length>
create_mirror(
    const Space&, const SrcAoSoA& src,
    typename std::enable_if<
        ( !std::is_same<typename SrcAoSoA::memory_space,
                        typename Space::memory_space>::value )>::type* = 0 )
{
    static_assert( is_aosoa<SrcAoSoA>::value,
                   "create_mirror() requires an AoSoA" );
    return AoSoA<typename SrcAoSoA::member_types, Space,
                 SrcAoSoA::vector_length>(
        std::string( src.label() ).append( "_mirror" ), src.size() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a mirror view of the given AoSoA in the given space. Same
  space specialization returns the input AoSoA.
  \return The original AoSoA.

  \note Memory allocation will only occur if the requested mirror memory space
  is different from that of the input AoSoA. If they are the same, the
  original AoSoA (e.g. a view of that AoSoA) is returned.
 */
template <class Space, class SrcAoSoA>
inline SrcAoSoA create_mirror_view(
    const Space&, const SrcAoSoA& src,
    typename std::enable_if<
        ( std::is_same<typename SrcAoSoA::memory_space,
                       typename Space::memory_space>::value )>::type* = 0 )
{
    static_assert( is_aosoa<SrcAoSoA>::value,
                   "create_mirror_view() requires an AoSoA" );
    return src;
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a mirror view of the given AoSoA in the given memory space.
  Different space specialization allocates a new AoSoA.
  \return AoSoA in the new space.

  \note Memory allocation will only occur if the requested mirror
  memory space is different from that of the input AoSoA. If they are the
  same, the original AoSoA (e.g. a view of that AoSoA) is returned.
 */
template <class Space, class SrcAoSoA>
inline AoSoA<typename SrcAoSoA::member_types, Space, SrcAoSoA::vector_length>
create_mirror_view(
    const Space& space, const SrcAoSoA& src,
    typename std::enable_if<
        ( !std::is_same<typename SrcAoSoA::memory_space,
                        typename Space::memory_space>::value )>::type* = 0 )
{
    static_assert( is_aosoa<SrcAoSoA>::value,
                   "create_mirror_view() requires an AoSoA" );
    return create_mirror( space, src );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a mirror view of the given AoSoA in the given memory space and
  copy the contents of the input AoSoA. Same space specialization returns the
  input AoSoA.
  \return The original AoSoA.

  \note Memory allocation will only occur if the requested mirror memory space
  is different from that of the input AoSoA. If they are the same, the
  original AoSoA (e.g. a view of that AoSoA) is returned.
 */
template <class Space, class SrcAoSoA>
inline SrcAoSoA create_mirror_view_and_copy(
    const Space&, const SrcAoSoA& src,
    typename std::enable_if<
        ( std::is_same<typename SrcAoSoA::memory_space,
                       typename Space::memory_space>::value &&
          is_aosoa<SrcAoSoA>::value )>::type* = 0 )
{
    return src;
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a mirror of the given AoSoA in the given memory space and deep
  copy the AoSoA into the mirror. Different space specialization allocates a
  new AoSoA and performs the deep copy.
  \return The new AoSoA.

  \note Memory allocation will only occur if the requested mirror
  memory space is different from that of the input AoSoA. If they are the
  same, the original AoSoA (e.g. a view of that AoSoA) is returned.
 */
template <class Space, class SrcAoSoA>
inline AoSoA<typename SrcAoSoA::member_types, Space, SrcAoSoA::vector_length>
create_mirror_view_and_copy(
    const Space& space, const SrcAoSoA& src,
    typename std::enable_if<
        ( !std::is_same<typename SrcAoSoA::memory_space,
                        typename Space::memory_space>::value &&
          is_aosoa<SrcAoSoA>::value )>::type* = 0 )
{
    auto dst = create_mirror( space, src );

    Kokkos::deep_copy(
        typename decltype( dst )::soa_view( dst.data(), dst.numSoA() ),
        typename SrcAoSoA::soa_view( src.data(), src.numSoA() ) );

    return dst;
}

//---------------------------------------------------------------------------//
/*!
  \brief Deep copy data between compatible AoSoA objects.
  \param dst The destination for the copied data.
  \param src The source of the copied data.

  Only AoSoA objects with the same set of member data types and size may be
  copied.
*/
template <class DstAoSoA, class SrcAoSoA>
inline void
deep_copy( DstAoSoA& dst, const SrcAoSoA& src,
           typename std::enable_if<( is_aosoa<DstAoSoA>::value &&
                                     is_aosoa<SrcAoSoA>::value )>::type* = 0 )
{
    using dst_type = DstAoSoA;
    using src_type = SrcAoSoA;
    using dst_memory_space = typename dst_type::memory_space;
    using src_memory_space = typename src_type::memory_space;
    using dst_soa_type = typename dst_type::soa_type;
    using src_soa_type = typename src_type::soa_type;

    // Check that the data types are the same.
    static_assert(
        std::is_same<typename dst_type::member_types,
                     typename src_type::member_types>::value,
        "Attempted to deep copy AoSoA objects of different member types" );

    // Check for the same number of values.
    if ( dst.size() != src.size() )
    {
        throw std::runtime_error(
            "Attempted to deep copy AoSoA objects of different sizes" );
    }

    // Get the pointers to the beginning of the data blocks.
    void* dst_data = dst.data();
    const void* src_data = src.data();

    // Return if both pointers are null.
    if ( dst_data == nullptr && src_data == nullptr )
    {
        return;
    }

    // Get the number of SoA's in each object.
    auto dst_num_soa = dst.numSoA();
    auto src_num_soa = src.numSoA();

    // Return if the AoSoA memory occupies the same space.
    if ( ( dst_data == src_data ) && ( dst_num_soa * sizeof( dst_soa_type ) ==
                                       src_num_soa * sizeof( src_soa_type ) ) )
    {
        return;
    }

    // If the inner array size is the same and both AoSoAs have the same number
    // of values then we can do a byte-wise copy directly.
    if ( std::is_same<dst_soa_type, src_soa_type>::value )
    {
        Kokkos::deep_copy( Kokkos::View<char*, dst_memory_space>(
                               reinterpret_cast<char*>( dst.data() ),
                               dst.numSoA() * sizeof( dst_soa_type ) ),
                           Kokkos::View<char*, src_memory_space>(
                               reinterpret_cast<char*>( src.data() ),
                               src.numSoA() * sizeof( src_soa_type ) ) );
    }

    // Otherwise copy the data element-by-element because the data layout is
    // different.
    else
    {
        // Create an AoSoA in the destination space with the same data layout
        // as the source.
        auto src_copy_on_dst = create_mirror_view_and_copy(
            typename dst_type::memory_space(), src );

        // Copy via tuples.
        auto copy_func = KOKKOS_LAMBDA( const std::size_t i )
        {
            dst.setTuple( i, src_copy_on_dst.getTuple( i ) );
        };
        Kokkos::RangePolicy<typename dst_type::execution_space> exec_policy(
            0, dst.size() );
        Kokkos::parallel_for( "Cabana::deep_copy", exec_policy, copy_func );
        Kokkos::fence();
    }
}

//---------------------------------------------------------------------------//
/*!
  \brief Deep copy data between compatible ParticleList objects.
  \param dst The destination for the copied data.
  \param src The source of the copied data.
s*/
template <class DstMemorySpace, class SrcMemorySpace, class... FieldTags>
inline void deep_copy( ParticleList<DstMemorySpace, FieldTags...>& dst,
                       const ParticleList<SrcMemorySpace, FieldTags...>& src )
{
    // Copy particle data to new memory space.
    auto aosoa_src = src.aosoa();
    auto& aosoa_dst = dst.aosoa();

    // Set the new data.
    Cabana::deep_copy( aosoa_dst, aosoa_src );
}

//---------------------------------------------------------------------------//
/*!
  \brief Fill an AoSoA with a tuple.
  \param aosoa The AoSoA to fill.
  \param tuple The tuple to assign. All AoSoA elements will be assigned this
  value.
*/
template <class AoSoA_t>
inline void deep_copy( AoSoA_t& aosoa,
                       const typename AoSoA_t::tuple_type& tuple )
{
    static_assert( is_aosoa<AoSoA_t>::value,
                   "Only AoSoAs can be assigned tuples" );
    auto assign_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        aosoa.setTuple( i, tuple );
    };
    Kokkos::RangePolicy<typename AoSoA_t::execution_space> exec_policy(
        0, aosoa.size() );
    Kokkos::parallel_for( "Cabana::deep_copy", exec_policy, assign_func );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
/*!
  \brief Deep copy data between compatible Slice objects.
  \param dst The destination for the copied data.
  \param src The source of the copied data.

  Only Slice objects with the same set of member data types and size may be
  copied.
*/
template <class DstSlice, class SrcSlice>
inline void
deep_copy( DstSlice& dst, const SrcSlice& src,
           typename std::enable_if<( is_slice<DstSlice>::value &&
                                     is_slice<SrcSlice>::value )>::type* = 0 )
{
    using dst_type = DstSlice;
    using src_type = SrcSlice;

    // Check that the data types are the same.
    static_assert(
        std::is_same<typename dst_type::value_type,
                     typename src_type::value_type>::value,
        "Attempted to deep copy Slice objects of different value types" );

    // Check that the element dimensions are the same.
    static_assert( SrcSlice::view_layout::D0 == SrcSlice::view_layout::D0,
                   "Slice dimension 0 is different" );
    static_assert( SrcSlice::view_layout::D1 == SrcSlice::view_layout::D1,
                   "Slice dimension 1 is different" );
    static_assert( SrcSlice::view_layout::D2 == SrcSlice::view_layout::D2,
                   "Slice dimension 2 is different" );
    static_assert( SrcSlice::view_layout::D3 == SrcSlice::view_layout::D3,
                   "Slice dimension 3 is different" );
    static_assert( SrcSlice::view_layout::D4 == SrcSlice::view_layout::D4,
                   "Slice dimension 4 is different" );
    static_assert( SrcSlice::view_layout::D5 == SrcSlice::view_layout::D5,
                   "Slice dimension 5 is different" );

    // Check for the same number of elements.
    if ( dst.size() != src.size() )
    {
        throw std::runtime_error(
            "Attempted to deep copy Slice objects of different sizes" );
    }

    // Get the pointers to the beginning of the data blocks.
    auto dst_data = dst.data();
    const auto src_data = src.data();

    // Return if both pointers are null.
    if ( dst_data == nullptr && src_data == nullptr )
    {
        return;
    }

    // Get the number of SoA's in each object.
    auto dst_num_soa = dst.numSoA();
    auto src_num_soa = src.numSoA();

    // Return if the slice memory occupies the same space.
    if ( ( dst_data == src_data ) &&
         ( dst_num_soa * dst.stride( 0 ) == src_num_soa * src.stride( 0 ) ) )
    {
        return;
    }

    // Get the number of components in each slice element.
    std::size_t num_comp = 1;
    for ( std::size_t d = 2; d < dst.viewRank(); ++d )
        num_comp *= dst.extent( d );

    // Gather the slice data in a flat view in the source space and copy it to
    // the destination space.
    Kokkos::View<typename dst_type::value_type*,
                 typename dst_type::memory_space>
        gather_dst( "gather_dst", num_comp * dst.size() );
    {
        Kokkos::View<typename src_type::value_type*,
                     typename src_type::memory_space>
            gather_src( "gather_src", num_comp * src.size() );
        auto gather_func = KOKKOS_LAMBDA( const std::size_t i )
        {
            auto src_offset = SrcSlice::index_type::s( i ) * src.stride( 0 ) +
                              SrcSlice::index_type::a( i );
            for ( std::size_t n = 0; n < num_comp; ++n )
                gather_src( i * num_comp + n ) =
                    src_data[src_offset + SrcSlice::vector_length * n];
        };
        Kokkos::RangePolicy<typename src_type::execution_space> gather_policy(
            0, src.size() );
        Kokkos::parallel_for( "Cabana::deep_copy::gather", gather_policy,
                              gather_func );
        Kokkos::fence();
        Kokkos::deep_copy( gather_dst, gather_src );
    }

    // Scatter back into the destination slice from the gathered slice.
    auto scatter_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        auto dst_offset = DstSlice::index_type::s( i ) * dst.stride( 0 ) +
                          DstSlice::index_type::a( i );
        for ( std::size_t n = 0; n < num_comp; ++n )
            dst_data[dst_offset + DstSlice::vector_length * n] =
                gather_dst( i * num_comp + n );
    };
    Kokkos::RangePolicy<typename dst_type::execution_space> scatter_policy(
        0, dst.size() );
    Kokkos::parallel_for( "Cabana::deep_copy::scatter", scatter_policy,
                          scatter_func );
    Kokkos::fence();
}

//---------------------------------------------------------------------------//
/*!
  \brief Fill a slice with a scalar.
  \param slice The slice to fill.
  \param scalar The scalar to assign. All slice elements will be assigned this
  value.
*/
template <class Slice_t>
inline void deep_copy( Slice_t& slice,
                       const typename Slice_t::value_type scalar )
{
    static_assert( is_slice<Slice_t>::value,
                   "Only slices can be assigned scalars" );
    Kokkos::deep_copy( slice.view(), scalar );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a mirror of the given ParticleList in the given memory space.

  \note Memory allocation will only occur if the requested mirror
  memory space is different from that of the input AoSoA. If they are the
  same, the original ParticleList is returned.
 */
template <class DstMemorySpace, class SrcMemorySpace, class... FieldTags>
auto create_mirror_view_and_copy(
    DstMemorySpace, ParticleList<SrcMemorySpace, FieldTags...> plist_src,
    typename std::enable_if<
        std::is_same<SrcMemorySpace, DstMemorySpace>::value>::type* = 0 )
{
    return plist_src;
}

/*!
  \brief Create a mirror of the given ParticleList in the given memory space.

  \note Memory allocation will only occur if the requested mirror
  memory space is different from that of the input AoSoA. If they are the
  same, the original ParticleList is returned.
 */
template <class DstMemorySpace, class SrcMemorySpace, class... FieldTags>
auto create_mirror_view_and_copy(
    DstMemorySpace, ParticleList<SrcMemorySpace, FieldTags...> plist_src,
    typename std::enable_if<
        !std::is_same<SrcMemorySpace, DstMemorySpace>::value>::type* = 0 )
{
    // Extract the original AoSoA.
    auto aosoa_src = plist_src.aosoa();

    // Create an AoSoA in the new memory space.
    using src_plist_type = ParticleList<SrcMemorySpace, FieldTags...>;
    using member_types = typename src_plist_type::member_types;
    AoSoA<member_types, DstMemorySpace> aosoa_dst( aosoa_src.label(),
                                                   aosoa_src.size() );

    // Copy data to new AoAoA.
    deep_copy( aosoa_dst, aosoa_src );

    // Create new list with the copied data.
    return ParticleList<DstMemorySpace, FieldTags...>( aosoa_dst );
}

} // end namespace Cabana

#endif // end CABANA_DEEPCOPY_HPP
