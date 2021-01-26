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

#ifndef CABANA_PARAMETERPACK_HPP
#define CABANA_PARAMETERPACK_HPP

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <utility>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Parameter pack device capture.
//
// NOTE: In general this would not be needed but NVCC cannot capture parameter
// packs in lambda functions hence we need to wrap them in something that can
// be captured.
//---------------------------------------------------------------------------//
// Get the type at the given index of a paremeter pack.
template <std::size_t N, typename T, typename... Types>
struct PackTypeAtIndexImpl;

template <typename T, typename... Types>
struct PackTypeAtIndexImpl<0, T, Types...>
{
    using type = T;
};

template <std::size_t N, typename T, typename... Types>
struct PackTypeAtIndexImpl
{
    using type = typename PackTypeAtIndexImpl<N - 1, Types...>::type;
};

template <std::size_t N, typename... Types>
struct PackTypeAtIndex
{
    using type = typename PackTypeAtIndexImpl<N, Types...>::type;
    static_assert( N < sizeof...( Types ), "Type index out of bounds" );
};

//---------------------------------------------------------------------------//
// Parameter pack element.
template <std::size_t N, typename T>
struct ParameterPackElement
{
    T _m;
};

//---------------------------------------------------------------------------//
// Capture a parameter pack. All parameter pack elements must be copyable to
// device.
template <typename Sequence, typename... Types>
struct ParameterPackImpl;

template <std::size_t... Indices, typename... Types>
struct ParameterPackImpl<std::index_sequence<Indices...>, Types...>
    : ParameterPackElement<Indices, Types>...
{
};

template <typename... Types>
struct ParameterPack
    : ParameterPackImpl<std::make_index_sequence<sizeof...( Types )>, Types...>
{
    template <std::size_t N>
    using value_type = typename PackTypeAtIndex<N, Types...>::type;

    template <std::size_t N>
    using const_value_type = typename std::add_const<value_type<N>>::type;

    template <std::size_t N>
    using element_type = ParameterPackElement<N, value_type<N>>;

    static constexpr std::size_t size = sizeof...( Types );
};

//---------------------------------------------------------------------------//
// Static type checker.
template <class>
struct is_parameter_pack_impl : public std::false_type
{
};

template <typename... Types>
struct is_parameter_pack_impl<ParameterPack<Types...>> : public std::true_type
{
};

template <class T>
struct is_parameter_pack
    : public is_parameter_pack_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Get an element from a parameter pack.
template <std::size_t N, class ParameterPack_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_parameter_pack<ParameterPack_t>::value,
    typename ParameterPack_t::template value_type<N>&>::type
get( ParameterPack_t& pp )
{
    return static_cast<typename ParameterPack_t::template element_type<N>&>(
               pp )
        ._m;
}

template <std::size_t N, class ParameterPack_t>
KOKKOS_FORCEINLINE_FUNCTION typename std::enable_if<
    is_parameter_pack<ParameterPack_t>::value,
    const typename ParameterPack_t::template value_type<N>&>::type
get( const ParameterPack_t& pp )
{
    return static_cast<
               const typename ParameterPack_t::template element_type<N>&>( pp )
        ._m;
}

//---------------------------------------------------------------------------//
// Fill a parameter pack. Note the indexing is such that the Nth element of a
// parameter pack is the Nth element of the tuple.
template <typename ParameterPack_t, typename T, typename... Types>
void fillParameterPackImpl( ParameterPack_t& pp,
                            const std::integral_constant<std::size_t, 0>,
                            const T& t, const Types&... )
{
    get<ParameterPack_t::size - 1>( pp ) = t;
}

template <typename ParameterPack_t, std::size_t N, typename T,
          typename... Types>
void fillParameterPackImpl( ParameterPack_t& pp,
                            const std::integral_constant<std::size_t, N>,
                            const T& t, const Types&... ts )
{
    get<ParameterPack_t::size - 1 - N>( pp ) = t;
    fillParameterPackImpl( pp, std::integral_constant<std::size_t, N - 1>(),
                           ts... );
}

template <typename ParameterPack_t, typename... Types>
void fillParameterPack( ParameterPack_t& pp, const Types&... ts )
{
    fillParameterPackImpl(
        pp, std::integral_constant<std::size_t, ParameterPack_t::size - 1>(),
        ts... );
}

// Empty case.
template <typename ParameterPack_t>
void fillParameterPack( ParameterPack_t& )
{
}

//---------------------------------------------------------------------------//
// Create a parameter pack.
template <typename... Types>
ParameterPack<Types...> makeParameterPack( const Types&... ts )
{
    ParameterPack<Types...> pp;
    fillParameterPack( pp, ts... );
    return pp;
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_PARAMETERPACK_HPP
