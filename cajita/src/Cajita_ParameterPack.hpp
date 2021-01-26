/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_PARAMETERPACK_HPP
#define CAJITA_PARAMETERPACK_HPP

#include <Cabana_Utils.hpp>

namespace Cajita
{
// Static type checker.
template <class T>
struct is_parameter_pack : public Cabana::is_parameter_pack_impl<
                               typename std::remove_cv<T>::type>::type
{
};

// Create a parameter pack.
template <typename... Types>
Cabana::ParameterPack<Types...> makeParameterPack( const Types&... ts )
{
    auto pp = Cabana::makeParameterPack( ts... );
    return pp;
}

// Get an element from a parameter pack.
template <std::size_t N, class ParameterPack_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    Cabana::is_parameter_pack<ParameterPack_t>::value,
    typename ParameterPack_t::template value_type<N>&>::type
get( ParameterPack_t& pp )
{
    return Cabana::get<N, ParameterPack_t>( pp );
}

// Get an element from a parameter pack.
template <std::size_t N, class ParameterPack_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    Cabana::is_parameter_pack<ParameterPack_t>::value,
    const typename ParameterPack_t::template value_type<N>&>::type
get( const ParameterPack_t& pp )
{
    return Cabana::get<N, ParameterPack_t>( pp );
}

} // namespace Cajita

#endif
