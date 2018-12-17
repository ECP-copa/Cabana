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

#ifndef CABANA_EXECUTIONPOLICY_HPP
#define CABANA_EXECUTIONPOLICY_HPP

#include <Cabana_Macros.hpp>
#include <impl/Cabana_Index.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
namespace Impl
{
/*!
  \class

  \brief 2D loop outer index range giving struct index range bounds based on a
  1D range input.

  \tparam VectorLength The inner array size of the AoSoA.
*/
template<int VectorLength>
class StructRange
{
  public:

    KOKKOS_INLINE_FUNCTION
    static constexpr std::size_t
    structBegin( const std::size_t& begin )
    {
        return Index<VectorLength>::s(begin);
    }

    KOKKOS_INLINE_FUNCTION
    static constexpr std::size_t
    structEnd( const std::size_t& end )
    {
        // If the end is also at the front of an array that means the struct
        // index of end is also the ending struct index. If not, we are not
        // iterating all the way through the arrays of the last struct. In
        // this case we add 1 to ensure that the loop over structs loops
        // through all structs with data.
        return (0 == Index<VectorLength>::a(end))
            ? Index<VectorLength>::s(end) : Index<VectorLength>::s(end) + 1;
    }

    KOKKOS_INLINE_FUNCTION
    static constexpr std::size_t
    size( const std::size_t& begin, const std::size_t end )
    { return structEnd(end) - structBegin(begin); }
};

} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \class SimdPolicy
  \brief Execution policy over a range of 2d indices.

  Gives 2D range of indices for executing a vectorized functor over the inner
  array index.
*/
template<class ExecutionSpace, int VectorLength>
class SimdPolicy : public Kokkos::TeamPolicy<ExecutionSpace,
                                             Kokkos::IndexType<int>,
                                             Kokkos::Schedule<Kokkos::Dynamic> >
{
  public:

    using base_type = Kokkos::TeamPolicy<ExecutionSpace,
                                         Kokkos::IndexType<int>,
                                         Kokkos::Schedule<Kokkos::Dynamic> >;

    using execution_policy = base_type;

    using execution_space = ExecutionSpace;

    /*!
      \brief Range constructor.
      \param begin The begininning of the 1D range. This will be decomposed
      into 2D indices.
      \param begin The ending of the 1D range. This will be decomposed
      into 2D indices.
    */
    SimdPolicy( const std::size_t begin, const std::size_t end )
        : base_type( Impl::StructRange<VectorLength>::size(begin,end),
                     1, VectorLength )
        , _struct_begin( Impl::StructRange<VectorLength>::structBegin(begin) )
        , _struct_end( Impl::StructRange<VectorLength>::structEnd(end) )
        , _array_begin( Impl::Index<VectorLength>::a(begin) )
        , _array_end( Impl::Index<VectorLength>::a(end) )
    {}

    //! Get the starting struct index.
    KOKKOS_INLINE_FUNCTION std::size_t structBegin() const
    { return _struct_begin; }

    //! Get the ending struct index.
    KOKKOS_INLINE_FUNCTION std::size_t structEnd() const
    { return _struct_end; }

    //! Given a struct id get the beginning array index.
    KOKKOS_INLINE_FUNCTION std::size_t arrayBegin( const std::size_t s ) const
    {
        // If the given struct index is also the index of the struct index in
        // begin, use the starting array index. If not, that means we have
        // passed the first struct and all subsequent structs start at array
        // index 0.
        return ( s == _struct_begin ) ? _array_begin : 0;
    }

    // Given a struct id get the ending array index.
    KOKKOS_INLINE_FUNCTION std::size_t arrayEnd( const std::size_t s ) const
    {
        // If we are in the last unfilled struct then use the array
        // index of end. If not, we are looping through the current array all
        // the way to the end so use the vector length.
        return ( (s == _struct_end - 1) && (_array_end != 0) )
            ? _array_end : VectorLength;
    }

  private:

    std::size_t _struct_begin;
    std::size_t _struct_end;
    std::size_t _array_begin;
    std::size_t _array_end;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_EXECUTIONPOLICY_HPP
