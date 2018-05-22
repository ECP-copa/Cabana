#ifndef CABANA_PARTICLE_HPP
#define CABANA_PARTICLE_HPP

#include <Cabana_IndexSequence.hpp>
#include <Cabana_MemberDataTypes.hpp>
#include <Cabana_Index.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <cstdlib>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \brief Particle member.

  A particle data member. T can be of arbitrary type (including
  multidimensional arrays) as long as the type of T is trivial. A particle
  will be composed of these members of different types.
*/
template<std::size_t M, typename T>
struct ParticleMember
{
    T _data;
};

//---------------------------------------------------------------------------//
// Particle implementation detail to hide the index sequence.
template<typename Sequence, typename... Types>
struct ParticleImpl;

template<std::size_t... Indices, typename... Types>
struct ParticleImpl<IndexSequence<Indices...>,Types...>
    : ParticleMember<Indices,Types>...
{};

//---------------------------------------------------------------------------//
/*!
  \class Particle

  \brief A single particle composed of general data types. A particle is
  trivially copyable.
*/
template<class DataTypes>
struct Particle;

// Static type checking.
template<class >
struct is_particle : public std::false_type {};

template<class DataTypes>
struct is_particle<Particle<DataTypes> > : public std::true_type {};

template<class DataTypes>
struct is_particle<const Particle<DataTypes> > : public std::true_type {};

//---------------------------------------------------------------------------//
namespace Impl
{

//---------------------------------------------------------------------------//
/*!
  \brief Particle member accessor.
*/

// Rank 0.
template<std::size_t M, class Particle_t>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (0==std::rank<typename Particle_t::member_data_type<M> >::value),
    typename Particle_t::member_reference_type<M> >::type
getParticleMember( Particle_t& particle )
{
    ParticleMember<M,typename Particle_t::member_data_type<M> >&
        base = particle;
    return base._data;
}

template<std::size_t M, class Particle_t>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (0==std::rank<typename Particle_t::member_data_type<M> >::value),
    typename Particle_t::member_value_type<M> >::type
getParticleMember( const Particle_t& particle )
{
    const ParticleMember<M,typename Particle_t::member_data_type<M> >&
        base = particle;
    return base._data;
}

// Rank 1.
template<std::size_t M, class Particle_t>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (1==std::rank<typename Particle_t::member_data_type<M> >::value),
    typename Particle_t::member_reference_type<M> >::type
getParticleMember( Particle_t& particle,
                   const int d0 )
{
    ParticleMember<M,typename Particle_t::member_data_type<M> >&
        base = particle;
    return base._data[d0];
}

template<std::size_t M, class Particle_t>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (1==std::rank<typename Particle_t::member_data_type<M> >::value),
    typename Particle_t::member_value_type<M> >::type
getParticleMember( const Particle_t& particle,
                   const int d0 )
{
    const ParticleMember<M,typename Particle_t::member_data_type<M> >&
        base = particle;
    return base._data[d0];
}

// Rank 2.
template<std::size_t M, class Particle_t>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (2==std::rank<typename Particle_t::member_data_type<M> >::value),
    typename Particle_t::member_reference_type<M> >::type
getParticleMember( Particle_t& particle,
                   const int d0,
                   const int d1 )
{
    ParticleMember<M,typename Particle_t::member_data_type<M> >&
        base = particle;
    return base._data[d0][d1];
}

template<std::size_t M, class Particle_t>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (2==std::rank<typename Particle_t::member_data_type<M> >::value),
    typename Particle_t::member_value_type<M> >::type
getParticleMember( const Particle_t& particle,
                   const int d0,
                   const int d1 )
{
    const ParticleMember<M,typename Particle_t::member_data_type<M> >&
        base = particle;
    return base._data[d0][d1];
}

// Rank 3.
template<std::size_t M, class Particle_t>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (3==std::rank<typename Particle_t::member_data_type<M> >::value),
    typename Particle_t::member_reference_type<M> >::type
getParticleMember( Particle_t& particle,
                   const int d0,
                   const int d1,
                   const int d2 )
{
    ParticleMember<M,typename Particle_t::member_data_type<M> >&
        base = particle;
    return base._data[d0][d1][d2];
}

template<std::size_t M, class Particle_t>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (3==std::rank<typename Particle_t::member_data_type<M> >::value),
    typename Particle_t::member_value_type<M> >::type
getParticleMember( const Particle_t& particle,
                   const int d0,
                   const int d1,
                   const int d2 )
{
    const ParticleMember<M,typename Particle_t::member_data_type<M> >&
        base = particle;
    return base._data[d0][d1][d2];
}

// Rank 4.
template<std::size_t M, class Particle_t>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (4==std::rank<typename Particle_t::member_data_type<M> >::value),
    typename Particle_t::member_reference_type<M> >::type
getParticleMember( Particle_t& particle,
                   const int d0,
                   const int d1,
                   const int d2,
                   const int d3 )
{
    ParticleMember<M,typename Particle_t::member_data_type<M> >&
        base = particle;
    return base._data[d0][d1][d2][d3];
}

template<std::size_t M, class Particle_t>
KOKKOS_FORCEINLINE_FUNCTION
typename std::enable_if<
    (4==std::rank<typename Particle_t::member_data_type<M> >::value),
    typename Particle_t::member_value_type<M> >::type
getParticleMember( const Particle_t& particle,
                   const int d0,
                   const int d1,
                   const int d2,
                   const int d3 )
{
    const ParticleMember<M,typename Particle_t::member_data_type<M> >&
        base = particle;
    return base._data[d0][d1][d2][d3];
}

//---------------------------------------------------------------------------//

} // end namespace Impl

//---------------------------------------------------------------------------//
// Particle implementation.
template<typename... Types>
struct Particle<MemberDataTypes<Types...> >
    : ParticleImpl<typename MakeIndexSequence<sizeof...(Types)>::type,Types...>
{
    // Particle type.
    using particle_type = Particle<MemberDataTypes<Types...> >;

    // Member data types.
    using member_types = MemberDataTypes<Types...>;

    // Number of member types.
    static constexpr std::size_t number_of_members = member_types::size;

    // The maximum rank supported for member types.
    static constexpr std::size_t max_supported_rank = 4;

    // Member data type.
    template<std::size_t M>
    using member_data_type = typename MemberDataTypeAtIndex<M,Types...>::type;

    // Value type at a given index M.
    template<std::size_t M>
    using member_value_type =
        typename std::remove_all_extents<member_data_type<M> >::type;

    // Reference type at a given index M.
    template<std::size_t M>
    using member_reference_type =
        typename std::add_lvalue_reference<member_value_type<M> >::type;

    // -------------------------------
    // Member data type properties.

    /*!
      \brief Get the rank of the data for a given member at index M.

      \tparam M The member index to get the rank for.

      \return The rank of the given member index data.
    */
    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    constexpr std::size_t rank() const
    {
        return std::rank<member_data_type<M> >::value;
    }

    /*!
      \brief Get the extent of a given member data dimension.

      \tparam M The member index to get the extent for.

      \tparam D The member data dimension to get the extent for.

      \return The extent of the dimension.
    */
    template<std::size_t M, std::size_t D>
    KOKKOS_INLINE_FUNCTION
    constexpr std::size_t extent() const
    {
        return std::extent<member_data_type<M>,D>::value;
    }

    // -------------------------------
    // Access the data value at a given member index.

    // Rank 0
    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(0==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get()
    {
        return Impl::getParticleMember<M>( *this );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(0==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get() const
    {
        return Impl::getParticleMember<M>( *this );
    }

    // Rank 1
    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(1==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int d0 )
    {
        return Impl::getParticleMember<M>( *this, d0 );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(1==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int d0 ) const
    {
        return Impl::getParticleMember<M>( *this, d0 );
    }

    // Rank 2
    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(2==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int d0,
         const int d1 )
    {
        return Impl::getParticleMember<M>( *this, d0, d1 );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(2==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int d0,
         const int d1 ) const
    {
        return Impl::getParticleMember<M>( *this, d0, d1 );
    }

    // Rank 3
    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(3==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2 )
    {
        return Impl::getParticleMember<M>( *this, d0, d1, d2 );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(3==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2 ) const
    {
        return Impl::getParticleMember<M>( *this, d0, d1, d2 );
    }

    // Rank 4
    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(4==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2,
         const int d3 )
    {
        return Impl::getParticleMember<M>( *this, d0, d1, d2, d3 );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(4==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2,
         const int d3 ) const
    {
        return Impl::getParticleMember<M>( *this, d0, d1, d2, d3 );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_PARTICLE_HPP
