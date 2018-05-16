#ifndef CABANA_BUFFERPARTICLE_HPP
#define CABANA_BUFFERPARTICLE_HPP

#include <Cabana_MemberDataTypes.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Index.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
namespace Impl
{
//---------------------------------------------------------------------------//
// Get the offset into the particle data buffer of a member at a given index.
template<std::size_t M, typename T, typename... Types>
struct MemberByteOffset
{
    static constexpr std::size_t value =
        sizeof(T) + MemberByteOffset<M-1,Types...>::value;
};

template<typename T, typename... Types>
struct MemberByteOffset<0,T,Types...>
{
    static constexpr std::size_t value = 0;
};

//---------------------------------------------------------------------------//
// Get the total byte size of a type list excluding any padding.
template<typename T, typename ... Types>
struct TotalByteSize
{
    static constexpr std::size_t value =
        sizeof(T) + TotalByteSize<Types...>::value;
};

template<typename T>
struct TotalByteSize<T>
{
    static constexpr std::size_t value = sizeof(T);
};

//---------------------------------------------------------------------------//
// Member size and type traits for packing/unpacking with a buffer particle.
template<std::size_t M, typename ... Types>
struct BufferParticleMemberTraits;

template<std::size_t M, typename ... Types>
struct BufferParticleMemberTraits<M,MemberDataTypes<Types...> >
{
    // Member data types.
    using member_types = MemberDataTypes<Types...>;

    // Get the member type at a given index.
    using type = typename MemberDataTypeAtIndex<M,Types...>::type;

    // Get the number of bytes for a given member.
    static constexpr std::size_t size = sizeof(type);

    // Get the offset into the buffer element data for a given member.
    static constexpr std::size_t offset = MemberByteOffset<M,Types...>::value;
};

//---------------------------------------------------------------------------//
/*
  \class BufferParticle

  \brief A single element of a communication buffer representing the data of
  an entire particle in a single byte stream. Effectively a single serialized
  particle. Particle data members are packed without byte-boundary padding. In
  many cases this could reduce overall memory usage and allows individual
  particles to be packed/unpacked from a single byte stream.
*/
template<typename ... Types>
struct BufferParticle;

//---------------------------------------------------------------------------//
// Static type checker.
template<class >
struct is_buffer_particle : public std::false_type {};

template<class DataTypes>
struct is_buffer_particle<BufferParticle<DataTypes> >
    : public std::true_type {};

template<class DataTypes>
struct is_buffer_particle<const BufferParticle<DataTypes> >
    : public std::true_type {};

//---------------------------------------------------------------------------//

template<typename ... Types>
struct BufferParticle<MemberDataTypes<Types...> >
{
    // Member data types.
    using member_types = MemberDataTypes<Types...>;

    // Total number of bytes in the buffer element.
    static constexpr std::size_t total_bytes = TotalByteSize<Types...>::value;

    // Buffer element data.
    char data[ total_bytes ];
};

//---------------------------------------------------------------------------//
// Free function for packing a member into a buffer particle.
template<std::size_t M, typename AoSoA_t, typename BufferParticle_t>
KOKKOS_INLINE_FUNCTION
void packMember(
    const Index& index,
    const AoSoA_t& aosoa,
    BufferParticle_t& particle,
    typename std::enable_if<
    ( is_aosoa<AoSoA_t>::value &&
      is_buffer_particle<BufferParticle_t>::value &&
      std::is_same<typename AoSoA_t::member_types,
      typename BufferParticle_t::member_types>::value
        )>::type * = 0 )
{
    // Member buffer traits
    using buffer_traits =
        BufferParticleMemberTraits<M,typename AoSoA_t::member_types>;

    // Get the pointer to the first element of the member in the struct we are
    // working wth.
    auto aosoa_struct_ptr =
        static_cast<typename AoSoA_t::struct_member_pointer_type<M> >(
            const_cast<void*>(aosoa.data(M))) +
        index.s() * aosoa.stride(M);

    // Get the pointer to the front of the element in the struct we are
    // working with and interpret it as the front of a byte stream.
    auto aosoa_element_ptr =
        reinterpret_cast<char*>(aosoa_struct_ptr) +
        index.i() * buffer_traits::size;

    // Get the pointer to the location in the buffer particle we will write
    // to.
    auto particle_ptr = particle.data + buffer_traits::offset;

    // Copy the AoSoA element into the proper location in the buffer particle.
    for ( std::size_t b = 0; b < buffer_traits::size; ++b )
        particle_ptr[b] = aosoa_element_ptr[b];
}

//---------------------------------------------------------------------------//
// Free function for unpacking a member element.
template<std::size_t M, typename AoSoA_t, typename BufferParticle_t>
KOKKOS_INLINE_FUNCTION
void unpackMember(
    const Index& index,
    const BufferParticle_t& particle,
    AoSoA_t& aosoa,
    typename std::enable_if<
    ( is_aosoa<AoSoA_t>::value &&
      is_buffer_particle<BufferParticle_t>::value &&
      std::is_same<typename AoSoA_t::member_types,
      typename BufferParticle_t::member_types>::value
        )>::type * = 0 )
{
    // Member buffer traits
    using buffer_traits =
        BufferParticleMemberTraits<M,typename AoSoA_t::member_types>;

    // Get the pointer to the location in the buffer particle we will write
    // to.
    auto particle_ptr = particle.data + buffer_traits::offset;

    // Get the pointer to the first element of the member in the struct we are
    // working wth.
    auto aosoa_struct_ptr =
        static_cast<typename AoSoA_t::struct_member_pointer_type<M> >(
            const_cast<void*>(aosoa.data(M))) +
        index.s() * aosoa.stride(M);

    // Get the pointer to the front of the element in the struct we are
    // working with and interpret it as the front of a byte stream.
    auto aosoa_element_ptr =
        reinterpret_cast<char*>(aosoa_struct_ptr) +
        index.i() * buffer_traits::size;

    // Copy the particle buffer data into the AoSoA element.
    for ( std::size_t b = 0; b < buffer_traits::size; ++b )
        aosoa_element_ptr[b] = particle_ptr[b];
}

//---------------------------------------------------------------------------//
// Particle member serializer. Allows for read/write of a single AoSoA element
// to and from buffer particles for communication. Recursively serializes
// all data members of an element.
template<std::size_t M>
struct ParticleSerializer;

template<>
struct ParticleSerializer<0>
{
    template<typename AoSoA_t, typename BufferParticle_t>
    KOKKOS_INLINE_FUNCTION
    static void pack( const Index& index,
                      const AoSoA_t& aosoa,
                      BufferParticle_t& particle )
    {
        packMember<0,AoSoA_t,BufferParticle_t>(
            index, aosoa, particle );
    }

    template<typename AoSoA_t, typename BufferParticle_t>
    KOKKOS_INLINE_FUNCTION
    static void unpack( const Index& index,
                        const BufferParticle_t& particle,
                        AoSoA_t& aosoa )
    {
        unpackMember<0,AoSoA_t,BufferParticle_t>(
            index, particle, aosoa );
    }
};

template<std::size_t M>
struct ParticleSerializer
{
    template< typename AoSoA_t, typename BufferParticle_t>
    KOKKOS_INLINE_FUNCTION
    static void pack( const Index& index,
                      const AoSoA_t& aosoa,
                      BufferParticle_t& particle )
    {
        packMember<M,AoSoA_t,BufferParticle_t>(
            index, aosoa, particle );
        ParticleSerializer<M-1>::pack( index, aosoa, particle );
    }

    template< typename AoSoA_t, typename BufferParticle_t>
    KOKKOS_INLINE_FUNCTION
    static void unpack( const Index& index,
                        const BufferParticle_t& particle,
                        AoSoA_t& aosoa )
    {
        unpackMember<M,AoSoA_t,BufferParticle_t>(
            index, particle, aosoa );
        ParticleSerializer<M-1>::unpack( index, particle, aosoa );
    }
};

//---------------------------------------------------------------------------//
/*!
  \brief Free function for packing an AoSoA element into a buffer particle in
  the same memory space.

  \param index The index of the element in the AoSoA to pack.

  \param aosoa The AoSoA to get the data from.

  \param particle The buffer particle to pack the data into.
*/
template<typename AoSoA_t, typename BufferParticle_t>
KOKKOS_INLINE_FUNCTION
void pack( const Index& index,
           const AoSoA_t& aosoa,
           BufferParticle_t& particle,
           typename std::enable_if<(
               is_aosoa<AoSoA_t>::value &&
               is_buffer_particle<BufferParticle_t>::value &&
               std::is_same<typename AoSoA_t::member_types,
               typename BufferParticle_t::member_types>::value
               )>::type * = 0 )

{
    ParticleSerializer<AoSoA_t::number_of_members-1>::pack(
        index, aosoa, particle );
};

//---------------------------------------------------------------------------//
/*!
  \brief Free function for unpacking an AoSoA element from a buffer particle
  in the same memory space.

  \param index The index of the AoSoA element to unpack the data into.

  \param particle The buffer particle to unpack the data from.

  \param aosoa The AoSoA to unpack the data into.
*/
template<typename AoSoA_t, typename BufferParticle_t>
KOKKOS_INLINE_FUNCTION
void unpack( const Index& index,
             const BufferParticle_t& particle,
             AoSoA_t& aosoa,
             typename std::enable_if<(
                 is_aosoa<AoSoA_t>::value &&
                 is_buffer_particle<BufferParticle_t>::value &&
                 std::is_same<typename AoSoA_t::member_types,
                 typename BufferParticle_t::member_types>::value
                 )>::type * = 0 )
{
    ParticleSerializer<AoSoA_t::number_of_members-1>::unpack(
        index, particle, aosoa );
};

//---------------------------------------------------------------------------//

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_BUFFERPARTICLE_HPP
