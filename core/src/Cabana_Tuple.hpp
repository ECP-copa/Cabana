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

#ifndef CABANA_TUPLE_HPP
#define CABANA_TUPLE_HPP

#include <Cabana_Macros.hpp>
#include <Cabana_MemberTypes.hpp>
#include <Cabana_SoA.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Forward declaration of tuple.
template <typename DataTypes>
struct Tuple;

//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_tuple : public std::false_type
{
};

template <class DataTypes>
struct is_tuple<Tuple<DataTypes>> : public std::true_type
{
};

template <class DataTypes>
struct is_tuple<const Tuple<DataTypes>> : public std::true_type
{
};

//---------------------------------------------------------------------------//
// Get template helper.

// Rank-0 non-const
template <std::size_t M, class Tuple_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_tuple<Tuple_t>::value,
    typename Tuple_t::template member_reference_type<M>>::type
get( Tuple_t &tp )
{
    return get<M>( static_cast<typename Tuple_t::base &>( tp ), 0 );
}

// Rank-0 const
template <std::size_t M, class Tuple_t>
KOKKOS_FORCEINLINE_FUNCTION typename Tuple_t::template member_value_type<M>
get( const Tuple_t &tp )
{
    return get<M>( static_cast<const typename Tuple_t::base &>( tp ), 0 );
}

// Rank-1 non-const
template <std::size_t M, class Tuple_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_tuple<Tuple_t>::value,
    typename Tuple_t::template member_reference_type<M>>::type
get( Tuple_t &tp, const std::size_t d0 )
{
    return get<M>( static_cast<typename Tuple_t::base &>( tp ), 0, d0 );
}

// Rank-1 const
template <std::size_t M, class Tuple_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_tuple<Tuple_t>::value,
    typename Tuple_t::template member_value_type<M>>::type
get( const Tuple_t &tp, const std::size_t d0 )
{
    return get<M>( static_cast<const typename Tuple_t::base &>( tp ), 0, d0 );
}

// Rank-2 non-const
template <std::size_t M, class Tuple_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_tuple<Tuple_t>::value,
    typename Tuple_t::template member_reference_type<M>>::type
get( Tuple_t &tp, const std::size_t d0, const std::size_t d1 )
{
    return get<M>( static_cast<typename Tuple_t::base &>( tp ), 0, d0, d1 );
}

// Rank-2 const
template <std::size_t M, class Tuple_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_tuple<Tuple_t>::value,
    typename Tuple_t::template member_value_type<M>>::type
get( const Tuple_t &tp, const std::size_t d0, const std::size_t d1 )
{
    return get<M>( static_cast<const typename Tuple_t::base &>( tp ), 0, d0,
                   d1 );
}

// Rank-3 non-const
template <std::size_t M, class Tuple_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_tuple<Tuple_t>::value,
    typename Tuple_t::template member_reference_type<M>>::type
get( Tuple_t &tp, const std::size_t d0, const std::size_t d1,
     const std::size_t d2 )
{
    return get<M>( static_cast<typename Tuple_t::base &>( tp ), 0, d0, d1, d2 );
}

// Rank-3 const
template <std::size_t M, class Tuple_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_tuple<Tuple_t>::value,
    typename Tuple_t::template member_value_type<M>>::type
get( const Tuple_t &tp, const std::size_t d0, const std::size_t d1,
     const std::size_t d2 )
{
    return get<M>( static_cast<const typename Tuple_t::base &>( tp ), 0, d0, d1,
                   d2 );
}

//---------------------------------------------------------------------------//
/*!
  \brief Tuple

  A tuple is a single element of a struct-of-arrays (SoA) (i.e. the struct)
  and is composed of groups of statically sized arrays. The array element
  types, which will be composed as members of the tuple, are indicated
  through the Types parameter pack. If the types of the members are contiguous
  then the tuple itself will be contiguous.
*/
template <typename... Types>
struct Tuple<MemberTypes<Types...>> : SoA<MemberTypes<Types...>, 1>
{
    // Base class.
    using base = SoA<MemberTypes<Types...>, 1>;

    // -------------------------------
    // Access the data value at a given member index. These accessors are
    // deprecated.

    // Rank 0
    CABANA_DEPRECATED
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
        0 == std::rank<typename base::template member_data_type<M>>::value,
        typename base::template member_reference_type<M>>::type
    get()
    {
        return Cabana::get<M>( *this );
    }

    CABANA_DEPRECATED
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
        0 == std::rank<typename base::template member_data_type<M>>::value,
        typename base::template member_value_type<M>>::type
    get() const
    {
        return Cabana::get<M>( *this );
    }

    // Rank 1
    CABANA_DEPRECATED
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
        1 == std::rank<typename base::template member_data_type<M>>::value,
        typename base::template member_reference_type<M>>::type
    get( const std::size_t d0 )
    {
        return Cabana::get<M>( *this, d0 );
    }

    CABANA_DEPRECATED
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
        1 == std::rank<typename base::template member_data_type<M>>::value,
        typename base::template member_value_type<M>>::type
    get( const std::size_t d0 ) const
    {
        return Cabana::get<M>( *this, d0 );
    }

    // Rank 2
    CABANA_DEPRECATED
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
        2 == std::rank<typename base::template member_data_type<M>>::value,
        typename base::template member_reference_type<M>>::type
    get( const std::size_t d0, const std::size_t d1 )
    {
        return Cabana::get<M>( *this, d0, d1 );
    }

    CABANA_DEPRECATED
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
        2 == std::rank<typename base::template member_data_type<M>>::value,
        typename base::template member_value_type<M>>::type
    get( const std::size_t d0, const std::size_t d1 ) const
    {
        return Cabana::get<M>( *this, d0, d1 );
    }

    // Rank 3
    CABANA_DEPRECATED
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
        3 == std::rank<typename base::template member_data_type<M>>::value,
        typename base::template member_reference_type<M>>::type
    get( const std::size_t d0, const std::size_t d1, const std::size_t d2 )
    {
        return Cabana::get<M>( *this, d0, d1, d2 );
    }

    CABANA_DEPRECATED
    template <std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
        3 == std::rank<typename base::template member_data_type<M>>::value,
        typename base::template member_value_type<M>>::type
    get( const std::size_t d0, const std::size_t d1,
         const std::size_t d2 ) const
    {
        return Cabana::get<M>( *this, d0, d1, d2 );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_TUPLE_HPP
