/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cajita_IndexSpace.hpp
  \brief Logical grid indexing
*/
#ifndef CAJITA_INDEXSPACE_HPP
#define CAJITA_INDEXSPACE_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <string>

namespace Cajita
{
//---------------------------------------------------------------------------//
/*!
  \brief Structured index space.
 */
template <long N>
class IndexSpace
{
  public:
    //! Number of dimensions.
    static constexpr long Rank = N;

    //! Default constructor.
    IndexSpace()
    {
        std::fill( _min.data(), _min.data() + Rank, -1 );
        std::fill( _max.data(), _max.data() + Rank, -1 );
    }

    /*!
      \brief Initializer list size constructor.
    */
    IndexSpace( const std::initializer_list<long>& size )
    {
        std::fill( _min.data(), _min.data() + Rank, 0 );
        std::copy( size.begin(), size.end(), _max.data() );
    }

    /*!
      \brief Initializer list range constructor.
    */
    IndexSpace( const std::initializer_list<long>& min,
                const std::initializer_list<long>& max )
    {
        std::copy( min.begin(), min.end(), _min.data() );
        std::copy( max.begin(), max.end(), _max.data() );
    }

    /*!
      \brief Vector size constructor.
    */
    IndexSpace( const std::array<long, N>& size )
    {
        std::fill( _min.data(), _min.data() + Rank, 0 );
        std::copy( size.begin(), size.end(), _max.data() );
    }

    /*!
      \brief Vector range constructor.
    */
    IndexSpace( const std::array<long, N>& min, const std::array<long, N>& max )
    {
        std::copy( min.begin(), min.end(), _min.data() );
        std::copy( max.begin(), max.end(), _max.data() );
    }

    //! Comparison operator.
    KOKKOS_INLINE_FUNCTION
    bool operator==( const IndexSpace<N>& rhs ) const
    {
        for ( long i = 0; i < N; ++i )
        {
            if ( min( i ) != rhs.min( i ) || max( i ) != rhs.max( i ) )
                return false;
        }
        return true;
    }

    //! Comparison operator.
    KOKKOS_INLINE_FUNCTION
    bool operator!=( const IndexSpace<N>& rhs ) const
    {
        return !( operator==( rhs ) );
    }

    //! Get the minimum index in a given dimension.
    KOKKOS_INLINE_FUNCTION
    long min( const long dim ) const { return _min[dim]; }

    //! Get the minimum indices in all dimensions.
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<long, Rank> min() const { return _min; }

    //! Get the maximum index in a given dimension.
    KOKKOS_INLINE_FUNCTION
    long max( const long dim ) const { return _max[dim]; }

    //! Get the maximum indices in all dimensions.
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<long, Rank> max() const { return _max; }

    //! Get the range of a given dimension.
    KOKKOS_INLINE_FUNCTION
    Kokkos::pair<long, long> range( const long dim ) const
    {
        return Kokkos::tie( _min[dim], _max[dim] );
    }

    //! Get the number of dimensions.
    KOKKOS_INLINE_FUNCTION
    long rank() const { return Rank; }

    //! Get the extent of a given dimension.
    KOKKOS_INLINE_FUNCTION
    long extent( const long dim ) const { return _max[dim] - _min[dim]; }

    //! Get the total size of the index space.
    KOKKOS_INLINE_FUNCTION
    long size() const
    {
        long size = 1;
        for ( long d = 0; d < Rank; ++d )
            size *= extent( d );
        return size;
    }

    //! Determine if a set of indices is within the range of the index space.
    KOKKOS_INLINE_FUNCTION
    bool inRange( const long index[N] ) const
    {
        bool result = true;
        for ( long i = 0; i < N; ++i )
            result =
                result && ( _min[i] <= index[i] ) && ( index[i] < _max[i] );
        return result;
    }

  protected:
    //! Minimum index bounds.
    Kokkos::Array<long, Rank> _min;

    //! Maximum index bounds.
    Kokkos::Array<long, Rank> _max;
};

//---------------------------------------------------------------------------//
/*!
  \brief Create a multi-dimensional execution policy over an index space.

  Rank-1 specialization.
*/
template <class ExecutionSpace>
Kokkos::RangePolicy<ExecutionSpace>
createExecutionPolicy( const IndexSpace<1>& index_space, const ExecutionSpace& )
{
    return Kokkos::RangePolicy<ExecutionSpace>( index_space.min( 0 ),
                                                index_space.max( 0 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a multi-dimensional execution policy over an index space.

  Rank-1 specialization with a work tag.
*/
template <class ExecutionSpace, class WorkTag>
Kokkos::RangePolicy<ExecutionSpace, WorkTag>
createExecutionPolicy( const IndexSpace<1>& index_space, const ExecutionSpace&,
                       const WorkTag& )
{
    return Kokkos::RangePolicy<ExecutionSpace, WorkTag>( index_space.min( 0 ),
                                                         index_space.max( 0 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a multi-dimensional execution policy over an index space.
*/
template <class IndexSpace_t, class ExecutionSpace>
Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<IndexSpace_t::Rank>>
createExecutionPolicy( const IndexSpace_t& index_space, const ExecutionSpace& )
{
    return Kokkos::MDRangePolicy<ExecutionSpace,
                                 Kokkos::Rank<IndexSpace_t::Rank>>(
        index_space.min(), index_space.max() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a multi-dimensional execution policy over an index space with
  a work tag.
*/
template <class IndexSpace_t, class ExecutionSpace, class WorkTag>
Kokkos::MDRangePolicy<ExecutionSpace, WorkTag, Kokkos::Rank<IndexSpace_t::Rank>>
createExecutionPolicy( const IndexSpace_t& index_space, const ExecutionSpace&,
                       const WorkTag& )
{
    return Kokkos::MDRangePolicy<ExecutionSpace, WorkTag,
                                 Kokkos::Rank<IndexSpace_t::Rank>>(
        index_space.min(), index_space.max() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given an index space create a view over the extent of that index
  space.

  Rank-1 specialization.
*/
template <class Scalar, class... Params>
Kokkos::View<Scalar*, Params...> createView( const std::string& label,
                                             const IndexSpace<1>& index_space )
{
    return Kokkos::View<Scalar*, Params...>(
        Kokkos::ViewAllocateWithoutInitializing( label ),
        index_space.extent( 0 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given an index space and a data pointer create an unmanaged view over
  the extent of that index space.

  Rank-1 specialization.
*/
template <class Scalar, class... Params>
Kokkos::View<Scalar*, Params..., Kokkos::MemoryUnmanaged>
createView( const IndexSpace<1>& index_space, Scalar* data )
{
    return Kokkos::View<Scalar*, Params..., Kokkos::MemoryUnmanaged>(
        data, index_space.extent( 0 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given an index space create a view over the extent of that index
  space.

  Rank-2 specialization.
*/
template <class Scalar, class... Params>
Kokkos::View<Scalar**, Params...> createView( const std::string& label,
                                              const IndexSpace<2>& index_space )
{
    return Kokkos::View<Scalar**, Params...>(
        Kokkos::ViewAllocateWithoutInitializing( label ),
        index_space.extent( 0 ), index_space.extent( 1 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given an index space and a data pointer create an unmanaged view over
  the extent of that index space.

  Rank-2 specialization.
*/
template <class Scalar, class... Params>
Kokkos::View<Scalar**, Params..., Kokkos::MemoryUnmanaged>
createView( const IndexSpace<2>& index_space, Scalar* data )
{
    return Kokkos::View<Scalar**, Params..., Kokkos::MemoryUnmanaged>(
        data, index_space.extent( 0 ), index_space.extent( 1 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given an index space create a view over the extent of that index
  space.

  Rank-3 specialization.
*/
template <class Scalar, class... Params>
Kokkos::View<Scalar***, Params...>
createView( const std::string& label, const IndexSpace<3>& index_space )
{
    return Kokkos::View<Scalar***, Params...>(
        Kokkos::ViewAllocateWithoutInitializing( label ),
        index_space.extent( 0 ), index_space.extent( 1 ),
        index_space.extent( 2 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given an index space and a data pointer create an unmanaged view over
  the extent of that index space.

  Rank-3 specialization.
*/
template <class Scalar, class... Params>
Kokkos::View<Scalar***, Params..., Kokkos::MemoryUnmanaged>
createView( const IndexSpace<3>& index_space, Scalar* data )
{
    return Kokkos::View<Scalar***, Params..., Kokkos::MemoryUnmanaged>(
        data, index_space.extent( 0 ), index_space.extent( 1 ),
        index_space.extent( 2 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given an index space create a view over the extent of that index
  space.

  Rank-4 specialization.
*/
template <class Scalar, class... Params>
Kokkos::View<Scalar****, Params...>
createView( const std::string& label, const IndexSpace<4>& index_space )
{
    return Kokkos::View<Scalar****, Params...>(
        Kokkos::ViewAllocateWithoutInitializing( label ),
        index_space.extent( 0 ), index_space.extent( 1 ),
        index_space.extent( 2 ), index_space.extent( 3 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given an index space and a data pointer create an unmanaged view over
  the extent of that index space.

  Rank-4 specialization.
*/
template <class Scalar, class... Params>
Kokkos::View<Scalar****, Params..., Kokkos::MemoryUnmanaged>
createView( const IndexSpace<4>& index_space, Scalar* data )
{
    return Kokkos::View<Scalar****, Params..., Kokkos::MemoryUnmanaged>(
        data, index_space.extent( 0 ), index_space.extent( 1 ),
        index_space.extent( 2 ), index_space.extent( 3 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a view create a subview over the given index space.

  Rank-1 specialization.
*/
template <class ViewType>
KOKKOS_INLINE_FUNCTION auto createSubview( const ViewType& view,
                                           const IndexSpace<1>& index_space )
    -> decltype( Kokkos::subview( view, index_space.range( 0 ) ) )
{
    static_assert( 1 == ViewType::Rank, "Incorrect view rank" );
    return Kokkos::subview( view, index_space.range( 0 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a view create a subview over the given index space.

  Rank-2 specialization.
*/
template <class ViewType>
KOKKOS_INLINE_FUNCTION auto createSubview( const ViewType& view,
                                           const IndexSpace<2>& index_space )
    -> decltype( Kokkos::subview( view, index_space.range( 0 ),
                                  index_space.range( 1 ) ) )
{
    static_assert( 2 == ViewType::Rank, "Incorrect view rank" );
    return Kokkos::subview( view, index_space.range( 0 ),
                            index_space.range( 1 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a view create a subview over the given index space.

  Rank-3 specialization.
*/
template <class ViewType>
KOKKOS_INLINE_FUNCTION auto createSubview( const ViewType& view,
                                           const IndexSpace<3>& index_space )
    -> decltype( Kokkos::subview( view, index_space.range( 0 ),
                                  index_space.range( 1 ),
                                  index_space.range( 2 ) ) )
{
    static_assert( 3 == ViewType::Rank, "Incorrect view rank" );
    return Kokkos::subview( view, index_space.range( 0 ),
                            index_space.range( 1 ), index_space.range( 2 ) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a view create a subview over the given index space.

  Rank-4 specialization.
*/
template <class ViewType>
KOKKOS_INLINE_FUNCTION auto createSubview( const ViewType& view,
                                           const IndexSpace<4>& index_space )
    -> decltype( Kokkos::subview( view, index_space.range( 0 ),
                                  index_space.range( 1 ),
                                  index_space.range( 2 ),
                                  index_space.range( 3 ) ) )
{
    static_assert( 4 == ViewType::Rank, "Incorrect view rank" );
    return Kokkos::subview( view, index_space.range( 0 ),
                            index_space.range( 1 ), index_space.range( 2 ),
                            index_space.range( 3 ) );
}

//---------------------------------------------------------------------------//
/*!
  Given an N-dimensional index space append an additional dimension with the
  given size.
*/
template <long N>
IndexSpace<N + 1> appendDimension( const IndexSpace<N>& index_space,
                                   const long size )
{
    std::array<long, N + 1> min;
    for ( int d = 0; d < N; ++d )
        min[d] = index_space.min( d );
    min[N] = 0;

    std::array<long, N + 1> max;
    for ( int d = 0; d < N; ++d )
        max[d] = index_space.max( d );
    max[N] = size;

    return IndexSpace<N + 1>( min, max );
}

//---------------------------------------------------------------------------//
/*!
  Given an N-dimensional index space append an additional dimension with the
  given range.
*/
template <long N>
IndexSpace<N + 1> appendDimension( const IndexSpace<N>& index_space,
                                   const long min, const long max )
{
    std::array<long, N + 1> range_min;
    for ( int d = 0; d < N; ++d )
        range_min[d] = index_space.min( d );
    range_min[N] = min;

    std::array<long, N + 1> range_max;
    for ( int d = 0; d < N; ++d )
        range_max[d] = index_space.max( d );
    range_max[N] = max;

    return IndexSpace<N + 1>( range_min, range_max );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_INDEXSPACE_HPP
