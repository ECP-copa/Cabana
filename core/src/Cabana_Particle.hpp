#ifndef CABANA_PARTICLE_HPP
#define CABANA_PARTICLE_HPP

#include <impl/Cabana_IndexSequence.hpp>
#include <Cabana_MemberDataTypes.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <cstdlib>

namespace Cabana
{
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

} // end namespace Impl

//---------------------------------------------------------------------------//
// Particle implementation.
template<typename... Types>
struct Particle<MemberDataTypes<Types...> >
    : Impl::ParticleImpl<
    typename Impl::MakeIndexSequence<sizeof...(Types)>::type,Types...>
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
    using member_data_type =
        typename MemberDataTypeAtIndex<M,member_types>::type;

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
    KOKKOS_FORCEINLINE_FUNCTION
    constexpr int rank() const
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
    KOKKOS_FORCEINLINE_FUNCTION
    constexpr int extent() const
    {
        return std::extent<member_data_type<M>,D>::value;
    }

    // -------------------------------
    // Access the data value at a given member index.

    // Rank 0
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get()
    {
        Impl::ParticleMember<M,member_data_type<M> >& base = *this;
        return base._data;
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get() const
    {
        const Impl::ParticleMember<M,member_data_type<M> >& base = *this;
        return base._data;
    }

    // Rank 1
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int d0 )
    {
        Impl::ParticleMember<M,member_data_type<M> >& base = *this;
        return base._data[d0];
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int d0 ) const
    {
        const Impl::ParticleMember<M,member_data_type<M> >& base = *this;
        return base._data[d0];
    }

    // Rank 2
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int d0,
         const int d1 )
    {
        Impl::ParticleMember<M,member_data_type<M> >& base = *this;
        return base._data[d0][d1];
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int d0,
         const int d1 ) const
    {
        const Impl::ParticleMember<M,member_data_type<M> >& base = *this;
        return base._data[d0][d1];
    }

    // Rank 3
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(3==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2 )
    {
        Impl::ParticleMember<M,member_data_type<M> >& base = *this;
        return base._data[d0][d1][d2];
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(3==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2 ) const
    {
        const Impl::ParticleMember<M,member_data_type<M> >& base = *this;
        return base._data[d0][d1][d2];
    }

    // Rank 4
    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(4==std::rank<member_data_type<M> >::value),
                            member_reference_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2,
         const int d3 )
    {
        Impl::ParticleMember<M,member_data_type<M> >& base = *this;
        return base._data[d0][d1][d2][d3];
    }

    template<std::size_t M>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(4==std::rank<member_data_type<M> >::value),
                            member_value_type<M> >::type
    get( const int d0,
         const int d1,
         const int d2,
         const int d3 ) const
    {
        const Impl::ParticleMember<M,member_data_type<M> >& base = *this;
        return base._data[d0][d1][d2][d3];
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_PARTICLE_HPP
