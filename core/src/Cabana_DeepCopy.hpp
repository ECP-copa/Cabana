/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
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
#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ExecPolicy.hpp>

#include <type_traits>
#include <exception>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \brief Deep copy data between compatible AoSoA objects.

  \param dst The destination for the copied data.

  \param src The source of the copied data.

  Only AoSoA objects with the same set of member data types and size may be
  copied.
*/
template<class DstAoSoA, class SrcAoSoA>
inline void deep_copy(
    DstAoSoA& dst,
    const SrcAoSoA& src,
    typename std::enable_if<(is_aosoa<DstAoSoA>::value &&
                             is_aosoa<SrcAoSoA>::value)>::type *  = 0 )
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
    void* dst_data = dst.ptr();
    const void* src_data = src.ptr();

    // Return if both pointers are null.
    if ( dst_data == nullptr && src_data == nullptr )
    {
        return;
    }

    // Get the number of SoA's in each object.
    auto dst_num_soa = dst.numSoA();
    auto src_num_soa = src.numSoA();

    // Return if the AoSoA memory occupies the same space.
    if ( (dst_data == src_data) &&
         (dst_num_soa * sizeof(dst_soa_type) ==
          src_num_soa * sizeof(src_soa_type)) )
    {
        return;
    }

    // If the inner array size is the same and both AoSoAs have the same number
    // of values then we can do a byte-wise copy directly.
    if ( std::is_same<dst_soa_type,src_soa_type>::value &&
         ( dst_type::vector_length == src_type::vector_length ) )
    {
        Kokkos::Impl::DeepCopy<dst_memory_space,src_memory_space>(
            dst_data, src_data, dst_num_soa * sizeof(dst_soa_type) );
    }

    // Otherwise copy the data element-by-element because the data layout is
    // different.
    else
    {
        // Define a AoSoA type in the destination space with the same data
        // layout as the source.
        using src_mirror_type = AoSoA<typename src_type::member_types,
                                      typename dst_type::memory_space,
                                      src_type::vector_length>;
        static_assert(
            std::is_same<src_soa_type,typename src_mirror_type::soa_type>::value,
            "Incompatible source mirror type in destination space" );

        // Create an AoSoA in the destination space with the same data layout
        // as the source.
        src_mirror_type src_copy_on_dst( src.size() );

        // Copy the source to the destination space.
        Kokkos::Impl::DeepCopy<dst_memory_space,src_memory_space>(
            src_copy_on_dst.ptr(),
            src_data,
            src_num_soa * sizeof(src_soa_type) );

        // Copy via tuples.
        auto copy_func =
            KOKKOS_LAMBDA( const std::size_t i )
            { dst.setTuple( i, src_copy_on_dst.getTuple(i) ); };
        Kokkos::RangePolicy<typename dst_memory_space::execution_space>
            exec_policy( 0, dst.size() );
        Kokkos::parallel_for( "Cabana::deep_copy", exec_policy, copy_func );
        Kokkos::fence();
    }
}

//---------------------------------------------------------------------------//
namespace Experimental
{
//---------------------------------------------------------------------------//
/*!
  \brief Create a mirror view of the given AoSoA in the given memory
  space. Same space specialization returns the input AoSoA.

  \note Memory allocation will only occur if the requested mirror memory space
  is different from that of the input AoSoA. If they are the same, the
  original AoSoA (e.g. a view of that AoSoA) is returned.
 */
template<class Space, class SrcAoSoA>
inline
SrcAoSoA
create_mirror_view_and_copy(
    const Space&,
    const SrcAoSoA& src,
    typename std::enable_if<(is_aosoa<SrcAoSoA>::value &&
                             std::is_same<typename SrcAoSoA::memory_space,
                             typename Space::memory_space>::value)>::type* = 0 )
{
    return src;
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a mirror on the host of the given AoSoA in the given memory
  space and deep copy the AoSoA into the mirror. Different space
  specialization allocates a new AoSoA and performs the deep copy.

  \note Memory allocation will only occur if the requested mirror
  memory space is different from that of the input AoSoA. If they are the
  same, the original AoSoA (e.g. a view of that AoSoA) is returned.
 */
template<class Space, class SrcAoSoA>
inline
AoSoA<typename SrcAoSoA::member_types,
      typename Space::memory_space,
      SrcAoSoA::vector_length>
create_mirror_view_and_copy(
    const Space&,
    const SrcAoSoA& src,
    typename std::enable_if<(is_aosoa<SrcAoSoA>::value &&
                             !std::is_same<typename SrcAoSoA::memory_space,
                             typename Space::memory_space>::value)>::type* = 0 )
{
    auto dst = AoSoA<typename SrcAoSoA::member_types,
                     typename Space::memory_space,
                     SrcAoSoA::vector_length>( src.size() );
    deep_copy( dst, src );
    return dst;
}

//---------------------------------------------------------------------------//

} // end namespace Experimental

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_DEEPCOPY_HPP
