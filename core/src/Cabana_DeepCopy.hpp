/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_DEEPCOPY_HPP
#define CABANA_DEEPCOPY_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_Slice.hpp>
#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ExecPolicy.hpp>

#include <exception>
#include <type_traits>

#include <cassert>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \brief Allocate a mirror of the given AoSoA in the given space.
 */
template <class Space, class SrcAoSoA>
inline AoSoA<typename SrcAoSoA::member_types, Space, SrcAoSoA::vector_length>
create_mirror(
    const Space &, const SrcAoSoA &src,
    typename std::enable_if<
        ( !std::is_same<typename SrcAoSoA::memory_space,
                        typename Space::memory_space>::value )>::type * = 0 )
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

  \note Memory allocation will only occur if the requested mirror memory space
  is different from that of the input AoSoA. If they are the same, the
  original AoSoA (e.g. a view of that AoSoA) is returned.
 */
template <class Space, class SrcAoSoA>
inline SrcAoSoA create_mirror_view(
    const Space &, const SrcAoSoA &src,
    typename std::enable_if<
        ( std::is_same<typename SrcAoSoA::memory_space,
                       typename Space::memory_space>::value )>::type * = 0 )
{
    static_assert( is_aosoa<SrcAoSoA>::value,
                   "create_mirror_view() requires an AoSoA" );
    return src;
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a mirror view of the given AoSoA in the given memory space.
  Different space specialization allocates a new AoSoA.

  \note Memory allocation will only occur if the requested mirror
  memory space is different from that of the input AoSoA. If they are the
  same, the original AoSoA (e.g. a view of that AoSoA) is returned.
 */
template <class Space, class SrcAoSoA>
inline AoSoA<typename SrcAoSoA::member_types, Space, SrcAoSoA::vector_length>
create_mirror_view(
    const Space &space, const SrcAoSoA &src,
    typename std::enable_if<
        ( !std::is_same<typename SrcAoSoA::memory_space,
                        typename Space::memory_space>::value )>::type * = 0 )
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

  \note Memory allocation will only occur if the requested mirror memory space
  is different from that of the input AoSoA. If they are the same, the
  original AoSoA (e.g. a view of that AoSoA) is returned.
 */
template <class Space, class SrcAoSoA>
inline SrcAoSoA create_mirror_view_and_copy(
    const Space &, const SrcAoSoA &src,
    typename std::enable_if<
        ( std::is_same<typename SrcAoSoA::memory_space,
                       typename Space::memory_space>::value )>::type * = 0 )
{
    static_assert( is_aosoa<SrcAoSoA>::value,
                   "create_mirror_view_and_copy() requires an AoSoA" );
    return src;
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a mirror of the given AoSoA in the given memory space and deep
  copy the AoSoA into the mirror. Different space specialization allocates a
  new AoSoA and performs the deep copy.

  \note Memory allocation will only occur if the requested mirror
  memory space is different from that of the input AoSoA. If they are the
  same, the original AoSoA (e.g. a view of that AoSoA) is returned.
 */
template <class Space, class SrcAoSoA>
inline AoSoA<typename SrcAoSoA::member_types, Space, SrcAoSoA::vector_length>
create_mirror_view_and_copy(
    const Space &space, const SrcAoSoA &src,
    typename std::enable_if<
        ( !std::is_same<typename SrcAoSoA::memory_space,
                        typename Space::memory_space>::value )>::type * = 0 )
{
    static_assert( is_aosoa<SrcAoSoA>::value,
                   "create_mirror_view_and_copy() requires an AoSoA" );

    auto dst = create_mirror( space, src );

    Kokkos::Impl::DeepCopy<typename Space::memory_space,
                           typename SrcAoSoA::memory_space>(
        dst.data(), src.data(),
        src.numSoA() * sizeof( typename SrcAoSoA::soa_type ) );

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
deep_copy( DstAoSoA &dst, const SrcAoSoA &src, bool async = false,
           typename std::enable_if<( is_aosoa<DstAoSoA>::value &&
                                     is_aosoa<SrcAoSoA>::value )>::type * = 0 )
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
    void *dst_data = dst.data();
    const void *src_data = src.data();

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
        if ( async )
        {
            std::cout << __FILE__ << ":" << __LINE__ << " => Async copy "
                      << std::endl;
            Kokkos::Impl::DeepCopyAsyncCuda(
                dst_data, src_data, dst_num_soa * sizeof( dst_soa_type ) );
        }
        else
        {
            Kokkos::Impl::DeepCopy<dst_memory_space, src_memory_space>(
                dst_data, src_data, dst_num_soa * sizeof( dst_soa_type ) );
        }
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
  \brief Fill an AoSoA with a tuple.

  \param aosoa The AoSoA to fill.

  \param tuple The tuple to assign. All AoSoA elements will be assigned this
  value.
*/
template <class AoSoA_t>
inline void deep_copy( AoSoA_t &aosoa,
                       const typename AoSoA_t::tuple_type &tuple )
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
deep_copy( DstSlice &dst, const SrcSlice &src,
           typename std::enable_if<( is_slice<DstSlice>::value &&
                                     is_slice<SrcSlice>::value )>::type * = 0 )
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
    for ( std::size_t d = 2; d < dst.rank(); ++d )
        num_comp *= dst.extent( d );

    // Gather the slice data in a flat view in the source space and copy it to
    // the destination space.
    Kokkos::View<typename dst_type::value_type *,
                 typename dst_type::memory_space>
        gather_dst( "gather_dst", num_comp * dst.size() );
    {
        Kokkos::View<typename src_type::value_type *,
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
inline void deep_copy( Slice_t &slice,
                       const typename Slice_t::value_type scalar )
{
    static_assert( is_slice<Slice_t>::value,
                   "Only slices can be assigned scalars" );
    Kokkos::deep_copy( slice.view(), scalar );
}

//---------------------------------------------------------------------------//

/**
 * @brief Copy a partial chunk from src into the destination array
 *
 * @param dst
 * @param src
 * @param start_from
 * @param end_from
 * @param start_to
 */
template <class DstAoSoA, class SrcAoSoA>
inline void deep_copy_partial_src(
    DstAoSoA &dst, const SrcAoSoA &src,
    // const int to_index,
    // TODO: the order of these params is questionable
    const int from_index, const int count,
    typename std::enable_if<( is_aosoa<DstAoSoA>::value &&
                              is_aosoa<SrcAoSoA>::value )>::type * = 0 )
{
    // TODO: this assumes you're trying to cross exec spaces (i.e partial copy
    // from CPU to GPU). You can likely do this faster and avoid data
    // duplication if that is not true
    // TODO: it might make sense to quick path this if start=0 and end=n

    // When trying to copy partial data across execution spaces using Kokkos,
    // you have to invoke a (potentially overly) complex pattern. You are not
    // allowed to cross execution spaces by copying subviews, so you have to
    // take great care (and pain!) to martial you data correctly. This
    // typically means you have to make local copies of the data of the correct
    // size, and copy those around. In this function, that will mean making an
    // AoSoA of type SrcAoSoA that is of reduced size, populating it, and then
    // invoking a Cabana::depp_copy (defined above). In the future there may be
    // a more elegant way to do this, but for now this is "safe". Typically
    // this creates extra memory overhead as we duplicate the copyable chunk in
    // the src memory space, which can be painful if we're using with src=GPU,
    // but luckily that is not the most common use-case.

    // Make AoSoA in src space to copy over
    SrcAoSoA src_partial( "deep_copy_partial src", count );

    assert( ( size_t )( from_index + count ) <= src.size() );

    std::cout << "Looping copy from 0 to " << src_partial.size()
              << " where src size is " << src.size() << std::endl;
    std::cout << "From index is " << from_index << " so pull from src is "
              << from_index << ".." << src_partial.size() + from_index
              << std::endl;

    // Populate it with data using a parallel for
    // TODO: this copy_func is borrow from above, so we could DRY
    auto copy_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        src_partial.setTuple( i, src.getTuple( i + from_index ) );
    };

    Kokkos::RangePolicy<typename SrcAoSoA::execution_space> exec_policy(
        0, src_partial.size() );

    Kokkos::parallel_for( "Cabana::deep_copy", exec_policy, copy_func );
    Kokkos::fence();

    std::cout << "src particle size " << src_partial.size() << " dst size "
              << dst.size() << std::endl;

    assert( src_partial.size() == dst.size() );

    // It should now be safe to rely on existing deep copy, assuming dst is
    // sized the same as src_partial. If that is not true we need to apply the
    // same pattern on the other end

    bool async = 1; // TODO: this could be a template?
    Cabana::deep_copy( dst, src_partial, async );
    // Cabana::deep_copy( dst, src_partial );
}

// TODO: this can be DRYd with deep_copy_partial, but we need a
// way to denote if src or dst is the partial
/**
 * @brief Copy the full src array into a partial place in destination
 *
 * @param dst TODO
 * @param src
 * @param to_index
 * @param count
 * @param
 */
template <class DstAoSoA, class SrcAoSoA>
inline void deep_copy_partial_dst(
    DstAoSoA dst, const SrcAoSoA src,
    const int to_index, // TODO: the order of these params is questionable
    // const int from_index, // TODO: not honored
    const int count,
    typename std::enable_if<( is_aosoa<DstAoSoA>::value &&
                              is_aosoa<SrcAoSoA>::value )>::type * = 0 )
{
    std::cout << "About to do deep copy back to " << to_index << " .. "
              << count + to_index << std::endl;

    // Make AoSoA in dst space to copy over
    DstAoSoA dst_partial( "deep_copy_partial dst", count );

    std::cout << "dst partial size " << dst_partial.size() << std::endl;
    std::cout << "src size " << src.size() << std::endl;

    Cabana::deep_copy( dst_partial, src );
    Kokkos::fence();

    // assert( count <= src.size() );
    assert( to_index + count <= dst.size() );

    auto d_0 = Cabana::slice<0>( dst );
    auto dp_0 = Cabana::slice<0>( dst_partial );
    auto s_0 = Cabana::slice<0>( src );

    // Populate it with data using a parallel for
    auto copy_func = KOKKOS_LAMBDA( const std::size_t i )
    {
        dst.setTuple( i + to_index, dst_partial.getTuple( i ) );
    };

    Kokkos::RangePolicy<typename DstAoSoA::execution_space> exec_policy(
        0, dst_partial.size() );

    Kokkos::parallel_for( "Cabana::deep_copy", exec_policy, copy_func );
    Kokkos::fence();

    assert( dst_partial.size() == src.size() );
}

} // end namespace Cabana

#endif // end CABANA_DEEPCOPY_HPP
