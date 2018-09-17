/****************************************************************************
 * Copyright (c) 2018 by the Cabana authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_INDEX_HPP
#define CABANA_INDEX_HPP

#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
namespace Impl
{

//---------------------------------------------------------------------------//
/*!
  \class Index

  \brief Class for converting between 1d and 2d aosoa indices.

  \tparam VectorLength The inner array size of the AoSoA.
*/
template<int VectorLength,
         typename std::enable_if<
             (Impl::IsVectorLengthValid<VectorLength>::value),
             int>::type = 0>
class Index
{
  public:

    // Inner array size.
    static constexpr int vector_length = VectorLength;

    // Array size offset.
    static constexpr int vector_length_offset = (vector_length - 1);

    // Number of binary bits needed to hold the array size.
    static constexpr int vector_length_binary_bits =
        Impl::LogBase2<vector_length>::value;

    /*!
      \brief Given a tuple index get the AoSoA struct index.

      \param i The tuple index.

      \return The index of the struct in which the tuple is located.
    */
    template<typename I>
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr
    typename std::enable_if<std::is_integral<I>::value,std::size_t>::type
    s( const I& i )
    {
        return (i - (i & vector_length_offset)) >>
            vector_length_binary_bits;
    }

    /*!
      \brief Given a tuple index get the AoSoA array index.

      \param i The tuple index.

      \return The index of the array index in the struct in which the tuple
      is located.
    */
    template<typename I>
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr
    typename std::enable_if<std::is_integral<I>::value,int>::type
    a( const I& i )
    {
        return i & vector_length_offset;
    }

    /*!
      \brief Given a struct index and array index in an AoSoA get the tuple
      index.

      \param struct_index The struct index.

      \param array_index The array index.

      \return The tuple index.
    */
    template<typename S, typename A>
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr
    typename std::enable_if<(std::is_integral<S>::value &&
                             std::is_integral<A>::value),std::size_t>::type
    i( const S& s, const A& a )
    {
        return (s << vector_length_binary_bits) + a;
    }
};

//---------------------------------------------------------------------------//

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_INDEX_HPP
