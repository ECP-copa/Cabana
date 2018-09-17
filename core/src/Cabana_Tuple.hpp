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

#ifndef CABANA_TUPLE_HPP
#define CABANA_TUPLE_HPP

#include <Cabana_SoA.hpp>
#include <Cabana_MemberTypes.hpp>
#include <Cabana_Macros.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <cstdlib>

namespace Cabana
{
//---------------------------------------------------------------------------//
template<typename DataTypes>
struct Tuple;

template<typename... Types>
struct Tuple<MemberTypes<Types...> >
    : SoA<1,MemberTypes<Types...> >
{
    // Base class.
    using base = SoA<1,MemberTypes<Types...> >;

    // -------------------------------
    // Access the data value at a given member index.

    // Rank 0
    template<std::size_t M>
    CABANA_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (0==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_reference_type<M> >::type
    get()
    {
        base& b = *this;
        return b.template get<M>( 0 );
    }

    template<std::size_t M>
    CABANA_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (0==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_value_type<M> >::type
    get() const
    {
        const base& b = *this;
        return b.template get<M>( 0 );
    }

    // Rank 1
    template<std::size_t M,
             typename D0>
    CABANA_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (1==std::rank<typename base::template member_data_type<M> >::value &&
         std::is_integral<D0>::value),
        typename base::template member_reference_type<M> >::type
    get( const D0& d0 )
    {
        base& b = *this;
        return b.template get<M>( 0, d0 );
    }

    template<std::size_t M,
             typename D0>
    CABANA_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (1==std::rank<typename base::template member_data_type<M> >::value &&
         std::is_integral<D0>::value),
        typename base::template member_value_type<M> >::type
    get( const D0& d0 ) const
    {
        const base& b = *this;
        return b.template get<M>( 0, d0 );
    }

    // Rank 2
    template<std::size_t M,
             typename D0,
             typename D1>
    CABANA_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (2==std::rank<typename base::template member_data_type<M> >::value &&
         std::is_integral<D0>::value &&
         std::is_integral<D1>::value),
        typename base::template member_reference_type<M> >::type
    get( const D0& d0,
         const D1& d1 )
    {
        base& b = *this;
        return b.template get<M>( 0, d0, d1 );
    }

    template<std::size_t M,
             typename D0,
             typename D1>
    CABANA_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (2==std::rank<typename base::template member_data_type<M> >::value &&
         std::is_integral<D0>::value &&
         std::is_integral<D1>::value),
        typename base::template member_value_type<M> >::type
    get( const D0& d0,
         const D1& d1 ) const
    {
        const base& b = *this;
        return b.template get<M>( 0, d0, d1 );
    }

    // Rank 3
    template<std::size_t M,
             typename D0,
             typename D1,
             typename D2>
    CABANA_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (3==std::rank<typename base::template member_data_type<M> >::value &&
         std::is_integral<D0>::value &&
         std::is_integral<D1>::value &&
         std::is_integral<D2>::value),
        typename base::template member_reference_type<M> >::type
    get( const D0& d0,
         const D1& d1,
         const D2& d2 )
    {
        base& b = *this;
        return b.template get<M>( 0, d0, d1, d2 );
    }

    template<std::size_t M,
             typename D0,
             typename D1,
             typename D2>
    CABANA_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (3==std::rank<typename base::template member_data_type<M> >::value &&
         std::is_integral<D0>::value &&
         std::is_integral<D1>::value &&
         std::is_integral<D2>::value),
        typename base::template member_value_type<M> >::type
    get( const D0& d0,
         const D1& d1,
         const D2& d2 ) const
    {
        const base& b = *this;
        return b.template get<M>( 0, d0, d1, d2 );
    }

    // Rank 4
    template<std::size_t M,
             typename D0,
             typename D1,
             typename D2,
             typename D3>
    CABANA_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (4==std::rank<typename base::template member_data_type<M> >::value &&
         std::is_integral<D0>::value &&
         std::is_integral<D1>::value &&
         std::is_integral<D2>::value &&
         std::is_integral<D3>::value),
        typename base::template member_reference_type<M> >::type
    get( const D0& d0,
         const D1& d1,
         const D2& d2,
         const D3& d3 )
    {
        base& b = *this;
        return b.template get<M>( 0, d0, d1, d2, d3 );
    }

    template<std::size_t M,
             typename D0,
             typename D1,
             typename D2,
             typename D3>
    CABANA_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (4==std::rank<typename base::template member_data_type<M> >::value &&
         std::is_integral<D0>::value &&
         std::is_integral<D1>::value &&
         std::is_integral<D2>::value &&
         std::is_integral<D3>::value),
        typename base::template member_value_type<M> >::type
    get( const D0& d0,
         const D1& d1,
         const D2& d2,
         const D3& d3 ) const
    {
        const base& b = *this;
        return b.template get<M>( 0, d0, d1, d2, d3 );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_TUPLE_HPP
