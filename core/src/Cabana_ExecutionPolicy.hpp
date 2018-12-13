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

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class RangePolicy1d
  \brief Execution policy over a range of 1d indices.

  Gives a linear range of indices for executing a functor with concurrency
  over all indices.
*/
template<class ExecutionSpace>
class RangePolicy1d
{
  public:

    using execution_space = ExecutionSpace;

    /*!
      \brief Range constructor.
      \param begin The begininning of the 1D range.
      \param begin The ending of the 1D range.
    */
    RangePolicy1d( const std::size_t begin, const std::size_t end )
        : _begin( begin )
        , _end( end )
    {}

    /*!
      \brief Size constructor. Loop over n elements starting at 0.
      \param n The number of elements in the range.
    */
    RangePolicy1d( const std::size_t n )
        : _begin( 0 )
        , _end( n )
    {}

    /*!
      \brief Container constructor.

      The container must have a size() function that returns an int. C++
      concepts would be really nice here. Valid containers include the AoSoA.

      \param container The container over which to build the range policy.
    */
    template<class Container>
    RangePolicy1d( Container container )
        : _begin( 0 )
        , _end( container.size() )
    {}

    //! First linear index in the range.
    CABANA_INLINE_FUNCTION std::size_t begin() const { return _begin; }

    //! Second linear index in the range.
    CABANA_INLINE_FUNCTION std::size_t end() const { return _end; }

  private:

    std::size_t _begin;
    std::size_t _end;
};

//---------------------------------------------------------------------------//
/*!
  \class RangePolicy2d
  \brief Execution policy over a range of 2d indices.

  Gives 2D range of indices for executing a vectorized functor over the inner
  array index.
*/
template<class ExecutionSpace, int VectorLength>
class RangePolicy2d
{
  public:

    using execution_space = ExecutionSpace;

    static constexpr int vector_length = VectorLength;

    /*!
      \brief Range constructor.
      \param begin The begininning of the 1D range. This will be decomposed
      into 2D indices.
      \param begin The ending of the 1D range. This will be decomposed
      into 2D indices.
    */
    RangePolicy2d( const std::size_t begin, const std::size_t end )
        : _struct_begin( Impl::Index<vector_length>::s(begin) )
        , _struct_end( Impl::Index<vector_length>::s(end) )
        , _array_begin( Impl::Index<vector_length>::a(begin) )
        , _array_end( Impl::Index<vector_length>::a(end) )
    {}

    /*!
      \brief Size constructor. Loop over n elements starting at 0.
      \param n The number of elements in the range.
    */
    RangePolicy2d( const std::size_t n )
        : _struct_begin( 0 )
        , _struct_end( Impl::Index<vector_length>::s(n) )
        , _array_begin( 0 )
        , _array_end( Impl::Index<vector_length>::a(n) )
    {}

    /*!
      \brief Container constructor.

      The container must have a size() function that returns an int. C++
      concepts would be really nice here. Valid containers include the AoSoA.

      \param container The container over which to build the range policy.
    */
    template<class Container>
    RangePolicy2d( Container container )
        : _struct_begin( 0 )
        , _struct_end( Impl::Index<vector_length>::s(container.size()) )
        , _array_begin( 0 )
        , _array_end( Impl::Index<vector_length>::a(container.size()) )
    {}

    //! Get the starting struct index.
    CABANA_INLINE_FUNCTION std::size_t structBegin() const
    {
        return _struct_begin;
    }

    //! Get the ending struct index.
    CABANA_INLINE_FUNCTION std::size_t structEnd() const
    {
        // If the end is also at the front of an array that means the struct
        // index of end is also the ending struct index. If not, we are not
        // iterating all the way through the arrays of the last struct. In
        // this case we add 1 to ensure that the loop over structs loops
        // through all structs with data.
        return (0 == _array_end) ? _struct_end : _struct_end + 1;
    }

    //! Get the number of structs.
    CABANA_INLINE_FUNCTION std::size_t numStruct() const
    {
        return structEnd() - structBegin();
    }

    //! Given a struct id get the beginning array index.
    CABANA_INLINE_FUNCTION std::size_t arrayBegin( const std::size_t s ) const
    {
        // If the given struct index is also the index of the struct index in
        // begin, use the starting array index. If not, that means we have
        // passed the first struct and all subsequent structs start at array
        // index 0.
        return ( s == structBegin() ) ? _array_begin : 0;
    }

    // Given a struct id get the ending array index.
    CABANA_INLINE_FUNCTION std::size_t arrayEnd( const std::size_t s ) const
    {
        // If we are in the last unfilled struct then use the array
        // index of end. If not, we are looping through the current array all
        // the way to the end so use the vector length.
        return ( (s == structEnd() - 1) && (_array_end != 0) )
            ? _array_end : vector_length;
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
