#ifndef CABANA_PARTICLE_HPP
#define CABANA_PARTICLE_HPP

#include <Cabana_SoA.hpp>
#include <Cabana_MemberDataTypes.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <cstdlib>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Particle declaration
template<typename DataTypes>
struct Particle;

//---------------------------------------------------------------------------//
// Particle implementation.
template<typename... Types>
struct Particle<MemberDataTypes<Types...> >
    : SoA<1,MemberDataTypes<Types...> >
{
    // Base class.
    using base = SoA<1,MemberDataTypes<Types...> >;

    // -------------------------------
    // Access the data value at a given member index.

    // Rank 0
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (0==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_reference_type<M> >::type
    get()
    {
        base& b = *this;
        return b.template get<M>( 0 );
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (0==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_value_type<M> >::type
    get() const
    {
        const base& b = *this;
        return b.template get<M>( 0 );
    }

    // Rank 1
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (1==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_reference_type<M> >::type
    get( const int d0 )
    {
        base& b = *this;
        return b.template get<M>( 0, d0 );
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (1==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_value_type<M> >::type
    get( const int d0 ) const
    {
        const base& b = *this;
        return b.template get<M>( 0, d0 );
    }

    // Rank 2
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (2==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_reference_type<M> >::type
    get( const int d0,
         const int d1 )
    {
        base& b = *this;
        return b.template get<M>( 0, d0, d1 );
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (2==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_value_type<M> >::type
    get( const int d0,
         const int d1 ) const
    {
        const base& b = *this;
        return b.template get<M>( 0, d0, d1 );
    }

    // Rank 3
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (3==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_reference_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2 )
    {
        base& b = *this;
        return b.template get<M>( 0, d0, d1, d2 );
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (3==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_value_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2 ) const
    {
        const base& b = *this;
        return b.template get<M>( 0, d0, d1, d2 );
    }

    // Rank 4
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (4==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_reference_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2,
         const int d3 )
    {
        base& b = *this;
        return b.template get<M>( 0, d0, d1, d2, d3 );
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<
        (4==std::rank<typename base::template member_data_type<M> >::value),
        typename base::template member_value_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2,
         const int d3 ) const
    {
        const base& b = *this;
        return b.template get<M>( 0, d0, d1, d2, d3 );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_PARTICLE_HPP
