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
  \file Cabana_Index.hpp
  \brief AoSoA indexing
*/
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
  \brief Class for converting between 1d and 2d aosoa indices.

  \tparam VectorLength The inner array size of the AoSoA.
*/
template <int VectorLength>
class Index
{
  public:
    // Validate the inner array size.
    static_assert( Impl::IsVectorLengthValid<VectorLength>::value,
                   "Invalid vector length" );

    //! Inner array size.
    static constexpr int vector_length = VectorLength;

    //! Array size offset.
    static constexpr int vector_length_offset = ( vector_length - 1 );

    //! Number of binary bits needed to hold the array size.
    static constexpr int vector_length_binary_bits =
        Impl::LogBase2<vector_length>::value;

    /*!
      \brief Given a tuple index get the AoSoA struct index.

      \param i The tuple index.

      \return The index of the struct in which the tuple is located.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr std::size_t s( const std::size_t i )
    {
        return ( i - ( i & vector_length_offset ) ) >>
               vector_length_binary_bits;
    }

    /*!
      \brief Given a tuple index get the AoSoA array index.

      \param i The tuple index.

      \return The index of the array index in the struct in which the tuple
      is located.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr std::size_t a( const std::size_t i )
    {
        return i & vector_length_offset;
    }

    /*!
      \brief Given a struct index and array index in an AoSoA get the tuple
      index.

      \param s The struct index.

      \param a The array index.

      \return The tuple index.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr std::size_t i( const std::size_t s, const std::size_t a )
    {
        return ( s << vector_length_binary_bits ) + a;
    }
};

//---------------------------------------------------------------------------//

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_INDEX_HPP
