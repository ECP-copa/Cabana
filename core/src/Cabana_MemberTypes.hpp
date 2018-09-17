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
template<typename... Types>
struct MemberTypes
{
    static constexpr std::size_t size = sizeof...(Types);
};

//---------------------------------------------------------------------------//
// Static type checker.
template<class >
struct is_member_types : public std::false_type {};

template<typename... Types>
struct is_member_types<MemberTypes<Types...> >
    : public std::true_type {};

template<typename... Types>
struct is_member_types<const MemberTypes<Types...> >
    : public std::true_type {};

//---------------------------------------------------------------------------//
/*!
  \class MemberTypeAtIndex
  \brief Get the type of the member at a given index.
*/
template<std::size_t I, typename T, typename... Types>
struct MemberTypeAtIndexImpl;

template<typename T, typename... Types>
struct MemberTypeAtIndexImpl<0,T,Types...>
{
    using type = T;
};

template<std::size_t I, typename T, typename... Types>
struct MemberTypeAtIndexImpl
{
    using type = typename MemberTypeAtIndexImpl<I-1,Types...>::type;
};

template<std::size_t I, typename... Types>
struct MemberTypeAtIndex;

template<std::size_t I, typename... Types>
struct MemberTypeAtIndex<I,MemberTypes<Types...> >
{
    using type =
        typename MemberTypeAtIndexImpl<I,Types...>::type;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_MEMBERTYPES_HPP
