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

#ifndef CABANA_MEMBERTYPES_HPP
#define CABANA_MEMBERTYPES_HPP

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
 \class MemberTypes
 \brief General sequence of types for SoA and AoSoA member data.
*/
template <typename... Types>
struct MemberTypes
{
    static constexpr std::size_t size = sizeof...( Types );
};

//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_member_types_impl : public std::false_type
{
};

template <typename... Types>
struct is_member_types_impl<MemberTypes<Types...>> : public std::true_type
{
};

template <class T>
struct is_member_types
    : public is_member_types_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
/*!
  \class MemberTypeAtIndex
  \brief Get the type of the member at a given index.
*/
template <std::size_t M, typename T, typename... Types>
struct MemberTypeAtIndexImpl;

template <typename T, typename... Types>
struct MemberTypeAtIndexImpl<0, T, Types...>
{
    using type = T;
};

template <std::size_t M, typename T, typename... Types>
struct MemberTypeAtIndexImpl
{
    using type = typename MemberTypeAtIndexImpl<M - 1, Types...>::type;
};

template <std::size_t M, typename... Types>
struct MemberTypeAtIndex;

template <std::size_t M, typename... Types>
struct MemberTypeAtIndex<M, MemberTypes<Types...>>
{
    using type = typename MemberTypeAtIndexImpl<M, Types...>::type;
};

//---------------------------------------------------------------------------//
/*!
  \class CheckMemberTypes
  \brief Check that member types are valied.
*/
template <std::size_t M, typename T, typename... Types>
struct CheckMemberTypesImpl;

template <typename T, typename... Types>
struct CheckMemberTypesImpl<0, T, Types...>
{
    using type = T;
    static_assert( std::is_trivial<type>::value,
                   "Member types must be trivial" );

    using value_type = typename std::remove_all_extents<type>::type;
    static_assert( std::is_arithmetic<value_type>::value,
                   "Member value types must be arithmetic" );

    // Return true so we get the whole stack to evaluate all the assertions.
    static constexpr bool value = true;
};

template <std::size_t M, typename T, typename... Types>
struct CheckMemberTypesImpl
{
    using type = T;
    static_assert( std::is_trivial<type>::value,
                   "Member types must be trivial" );

    using value_type = typename std::remove_all_extents<type>::type;
    static_assert( std::is_arithmetic<value_type>::value,
                   "Member value types must be arithmetic" );

    static constexpr bool value = CheckMemberTypesImpl<M - 1, Types...>::value;
};

template <typename... Types>
struct CheckMemberTypes;

template <typename... Types>
struct CheckMemberTypes<MemberTypes<Types...>>
{
    static constexpr int size = MemberTypes<Types...>::size;
    static constexpr bool value =
        CheckMemberTypesImpl<size - 1, Types...>::value;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_MEMBERTYPES_HPP
