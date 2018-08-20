#ifndef CABANA_AOSOA_HPP
#define CABANA_AOSOA_HPP

#include <Cabana_MemberDataTypes.hpp>
#include <Cabana_MemberSlice.hpp>
#include <Cabana_InnerArrayLayout.hpp>
#include <Cabana_Particle.hpp>
#include <Cabana_Types.hpp>
#include <impl/Cabana_SoA.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_PerformanceTraits.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <cmath>
#include <cstdlib>
#include <string>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class AoSoA

  \brief Array-of-Struct-of-Arrays

  A AoSoA represents particles and their data via an
  array-of-structs-of-arrays.

  This class has both required and optional template parameters.  The
  \c DataType parameter must always be provided, and must always be
  first. The parameters \c Arg1Type, \c Arg2Type, and \c Arg3Type are
  placeholders for different template parameters.  The default value
  of the fifth template parameter \c Specialize suffices for most use
  cases.  When explaining the template parameters, we won't refer to
  \c Arg1Type, \c Arg2Type, and \c Arg3Type; instead, we will refer
  to the valid categories of template parameters, in whatever order
  they may occur.

  Valid ways in which template arguments may be specified:
  - AoSoA< DataType >
  - AoSoA< DataType , StaticInnerArrayLayout >
  - AoSoA< DataType , StaticInnerArrayLayout , Space >
  - AoSoA< DataType , Space >

  \tparam DataType (required) Specifically this must be an instance of
  \c MemberDataTypes with the data layout of the structs. For example:
  \code
  using DataType = MemberDataTypes<double[3][3],double[3],int>;
  \endcode
  would define an AoSoA where each particle had a 3x3 matrix of doubles, a
  3-vector of doubles, and an integer. The AoSoA is then templated on this
  sequence of types. In general, put larger datatypes first in the
  MemberDataType parameter pack (i.e. matrices and vectors) and group members
  of the same type together to achieve the smallest possible memory footprint
  based on compiler-generated padding.

  \tparam Space (required) The memory space.

  \tparam StaticInnerArrayLayout (optional) The layout of the inner array in
  the AoSoA. If not specified, this defaults to the preferred layout for the
  <tt>Space</tt>.
 */
template<class DataTypes,
         class MemorySpace,
         class DataLayout = typename Impl::PerformanceTraits<
             typename MemorySpace::kokkos_execution_space>::inner_array_layout>
class AoSoA
{
  public:

    // AoSoA type.
    using aosoa_type = AoSoA<DataTypes,MemorySpace,DataLayout>;

    // Member data types.
    using member_types = DataTypes;

    // Memory space.
    using memory_space = MemorySpace;

    // Inner array layout.
    using inner_array_layout = DataLayout;

    // Inner array size (size of the arrays held by the structs).
    static constexpr int array_size = inner_array_layout::size;

    // SoA type.
    using soa_type = Impl::SoA<inner_array_layout,member_types>;

    // Managed data view.
    using soa_view = Kokkos::View<soa_type*,typename memory_space::kokkos_memory_space>;

    // Number of member types.
    static constexpr std::size_t number_of_members = member_types::size;

    // The maximum rank supported for member types.
    static constexpr int max_supported_rank = 4;

    // Index type.
    using index_type = Impl::Index<array_size>;

    // Particle type.
    using particle_type = Particle<member_types>;

    // Member data type at a given index M. Note this is the user-defined
    // member data type - not the potentially transformed type actually stored
    // by the structs (SoAs) to achieve a given layout.
    template<std::size_t Field>
    using member_data_type =
        typename MemberDataTypeAtIndex<Field,member_types>::type;

    // Struct member array element value type at a given index M.
    template<std::size_t Field>
    using member_value_type =
        typename std::remove_all_extents<member_data_type<Field> >::type;

    // Struct member array element pointer type at a given index M.
    template<std::size_t Field>
    using member_pointer_type =
        typename std::add_pointer<member_value_type<Field> >::type;

  public:

    /*!
      \brief Default constructor.

      The container size is zero and no memory is allocated.
    */
    AoSoA()
        : _size( 0 )
        , _capacity( 0 )
        , _num_soa( 0 )
    {}

    /*!
      \brief Allocate a container with n particles.

      \param n The number of particles in the container.
    */
    explicit AoSoA( const int n )
        : _size( n )
        , _capacity( 0 )
        , _num_soa( 0 )
    {
        resize( _size );
    }

    /*!
      \brief Returns the number of particles in the container.

      \return The number of particles in the container.

      This is the number of actual objects held in the container, which is not
      necessarily equal to its storage capacity.
    */
    KOKKOS_FUNCTION
    int size() const { return _size; }

    /*!
      \brief Returns the size of the storage space currently allocated for the
      container, expressed in terms of particles.

      \return The capacity of the container.

      This capacity is not necessarily equal to the container size. It can be
      equal or greater, with the extra space allowing to accommodate for
      growth without the need to reallocate on each insertion.

      Notice that this capacity does not suppose a limit on the size of the
      container. When this capacity is exhausted and more is needed, it is
      automatically expanded by the container (reallocating it storage space).

      The capacity of a container can be explicitly altered by calling member
      reserve.
    */
    KOKKOS_FUNCTION
    int capacity() const { return _capacity; }

    /*!
      \brief Resizes the container so that it contains n particles.

      If n is smaller than the current container size, the content is reduced
      to its first n particles.

      If n is greater than the current container size, the content is expanded
      by inserting at the end as many particles as needed to reach a size of n.

      If n is also greater than the current container capacity, an automatic
      reallocation of the allocated storage space takes place.

      Notice that this function changes the actual content of the container by
      inserting or erasing particles from it.
    */
    void resize( const int n )
    {
        reserve( n );
        _size = n;
        _num_soa = std::floor( n / array_size );
        if ( 0 < n % array_size ) ++_num_soa;
    }

    /*!
      \brief Requests that the container capacity be at least enough to contain n
      particles.

      If n is greater than the current container capacity, the function causes
      the container to reallocate its storage increasing its capacity to n (or
      greater).

      In all other cases, the function call does not cause a reallocation and
      the container capacity is not affected.

      This function has no effect on the container size and cannot alter its
      particles.
    */
    void reserve( const int n )
    {
        // If we aren't asking for more memory then we have nothing to do.
        if ( n <= _capacity ) return;

        // Figure out the new capacity.
        int num_soa_alloc = std::floor( n / array_size );
        if ( 0 < n % array_size ) ++num_soa_alloc;
        _capacity = num_soa_alloc * array_size;

        // Allocate a new block of memory.
        soa_view data_block( "aosoa_data", num_soa_alloc );

        // If we have already allocated memory, copy the old memory into the
        // new memory. Fence when we are done to ensure copy is complete
        // before continuing.
        if ( _data.size() > 0 )
            Kokkos::deep_copy( data_block, _data );

        // Swap blocks. The old block will be destroyed when this function
        // exits.
        _data = data_block;

        // Get new pointers and strides for the members.
        storePointersAndStrides(
            std::integral_constant<std::size_t,number_of_members-1>() );
    }

    /*!
      \brief Get the number of structs-of-arrays in the container.

      \return The number of structs-of-arrays in the container.
    */
    KOKKOS_FUNCTION
    int numSoA() const { return _num_soa; }

    /*!
      \brief Get the size of the data array at a given struct member index.

      \param s The struct index to get the array size for.

      \return The size of the array at the given struct index.
    */
    KOKKOS_FUNCTION
    int arraySize( const int s ) const
    {
        return
            ( s < _num_soa - 1 ) ? array_size : ( _size % array_size );
    }

    /*!
      \brief Get a particle at a given index.

      \param idx The index to get the particle from.

      \return A particle containing a copy of the data at the given index.
    */
    KOKKOS_INLINE_FUNCTION
    particle_type getParticle( const int particle_index ) const
    {
        particle_type particle;
        copyToParticle(
            particle_index,
            particle,
            std::integral_constant<std::size_t,number_of_members-1>() );
        return particle;
    }

    /*!
      \brief Set a particle at a given index.

      \param particle_index The index to set the particle at.

      \param particle The particle to get the data from.
    */
    KOKKOS_INLINE_FUNCTION
    void setParticle( const int particle_index,
                      const particle_type& particle ) const
    {
        copyFromParticle(
            particle_index,
            particle,
            std::integral_constant<std::size_t,number_of_members-1>() );
    }

    /*!
      \brief Get an unmanaged view of a particle field with the given tag and
      memory traits.
      \param The tag identifying which field to get a view of.
      \param The tag identifying which type of memory access traits the view
      should have.
      \return The field view.
    */
    template<std::size_t Field, typename MemoryAccessType>
    MemberSlice<member_data_type<Field>,
                typename inner_array_layout::layout,
                memory_space,
                MemoryAccessType,
                array_size>
    view( MemberTag<Field>, MemoryAccessType ) const
    {
        return
            MemberSlice<member_data_type<Field>,
                        typename inner_array_layout::layout,
                        memory_space,
                        MemoryAccessType,
                        array_size>(
                            (member_pointer_type<Field>) _pointers[Field],
                            _size, _strides[Field], _num_soa );
    }

    /*!
      \brief Get an unmanaged view of a particle field with default memory
      access.
      \param The tag identifying which field to get a view of.
      \return The field view.
    */
    template<std::size_t Field>
    MemberSlice<member_data_type<Field>,
                typename inner_array_layout::layout,
                memory_space,
                DefaultAccessMemory,
                array_size>
    view( MemberTag<Field> ) const
    {
        return
            MemberSlice<member_data_type<Field>,
                        typename inner_array_layout::layout,
                        memory_space,
                        DefaultAccessMemory,
                        array_size>(
                            (member_pointer_type<Field>) _pointers[Field],
                            _size, _strides[Field], _num_soa );
    }

    /*!
      \brief Get an un-typed raw pointer to the entire data block.
      \return An un-typed raw-pointer to the entire data block.
    */
    void* ptr() const
    { return _data.data(); }

  private:

    // Store the pointers and strides for each member element.
    template<std::size_t N>
    void assignPointersAndStrides()
    {
        static_assert( 0 <= N && N < number_of_members,
                       "Static loop out of bounds!" );
        _pointers[N] =
            static_cast<void*>( Impl::getStructMember<N>(_data(0)) );
        static_assert( 0 ==
                       sizeof(soa_type) % sizeof(member_value_type<N>),
                       "Stride cannot be calculated for misaligned memory!" );
        _strides[N] = sizeof(soa_type) / sizeof(member_value_type<N>);
    }

    // Static loop through each member element to extract pointers and strides.
    template<std::size_t N>
    void storePointersAndStrides( std::integral_constant<std::size_t,N> )
    {
        assignPointersAndStrides<N>();
        storePointersAndStrides( std::integral_constant<std::size_t,N-1>() );
    }

    void storePointersAndStrides( std::integral_constant<std::size_t,0> )
    {
        assignPointersAndStrides<0>();
    }

    // Copy a member to a particle.
    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<
        (0==std::rank<member_data_type<M> >::value),void>::type
    copyMemberToParticle( const int particle_index,
                          particle_type& particle ) const
    {
        particle.template get<M>() =
            Impl::accessStructMember<M>(
                _data(index_type::s(particle_index)),
                index_type::i(particle_index) );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<
        (1==std::rank<member_data_type<M> >::value),void>::type
    copyMemberToParticle( const int particle_index,
                          particle_type& particle ) const
    {
        for ( int i0 = 0; i0 < particle.template extent<M,0>(); ++i0 )
            particle.template get<M>( i0 ) =
                Impl::accessStructMember<M>(
                    _data(index_type::s(particle_index)),
                    index_type::i(particle_index),
                    i0 );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<
        (2==std::rank<member_data_type<M> >::value),void>::type
    copyMemberToParticle( const int particle_index,
                          particle_type& particle ) const
    {
        for ( int i0 = 0; i0 < particle.template extent<M,0>(); ++i0 )
            for ( int i1 = 0; i1 < particle.template extent<M,1>(); ++i1 )
                particle.template get<M>( i0, i1 ) =
                    Impl::accessStructMember<M>(
                        _data(index_type::s(particle_index)),
                        index_type::i(particle_index),
                        i0, i1 );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<
        (3==std::rank<member_data_type<M> >::value),void>::type
    copyMemberToParticle( const int particle_index,
                          particle_type& particle ) const
    {
        for ( int i0 = 0; i0 < particle.template extent<M,0>(); ++i0 )
            for ( int i1 = 0; i1 < particle.template extent<M,1>(); ++i1 )
                for ( int i2 = 0; i2 < particle.template extent<M,2>(); ++i2 )
                    particle.template get<M>( i0, i1, i2 ) =
                        Impl::accessStructMember<M>(
                            _data(index_type::s(particle_index)),
                            index_type::i(particle_index),
                            i0, i1, i2 );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<
        (4==std::rank<member_data_type<M> >::value),void>::type
    copyMemberToParticle( const int particle_index,
                          particle_type& particle ) const
    {
        for ( int i0 = 0; i0 < particle.template extent<M,0>(); ++i0 )
            for ( int i1 = 0; i1 < particle.template extent<M,1>(); ++i1 )
                for ( int i2 = 0; i2 < particle.template extent<M,2>(); ++i2 )
                    for ( int i3 = 0; i3 < particle.template extent<M,3>(); ++i3 )
                        particle.template get<M>( i0, i1, i2, i3 ) =
                            Impl::accessStructMember<M>(
                                _data(index_type::s(particle_index)),
                                index_type::i(particle_index),
                                i0, i1, i2, i3 );
    }

    // Copy to a particle a given index.
    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    void copyToParticle( const int particle_index,
                         particle_type& particle,
                         std::integral_constant<std::size_t,M> ) const
    {
        copyMemberToParticle<M>( particle_index, particle );
        copyToParticle(
            particle_index, particle,
            std::integral_constant<std::size_t,M-1>() );
    }

    KOKKOS_INLINE_FUNCTION
    void copyToParticle( const int particle_index,
                         particle_type& particle,
                         std::integral_constant<std::size_t,0> ) const
    {
        copyMemberToParticle<0>( particle_index, particle );
    }

    // Copy a particle to a member.
    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<
        (0==std::rank<member_data_type<M> >::value),void>::type
    copyParticleToMember( const int particle_index,
                          const particle_type& particle ) const
    {
        Impl::accessStructMember<M>( _data(index_type::s(particle_index)),
                                     index_type::i(particle_index) )
            = particle.template get<M>();
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<
        (1==std::rank<member_data_type<M> >::value),void>::type
    copyParticleToMember( const int particle_index,
                          const particle_type& particle ) const
    {
        for ( int i0 = 0; i0 < particle.template extent<M,0>(); ++i0 )
            Impl::accessStructMember<M>(
                _data(index_type::s(particle_index)),
                index_type::i(particle_index), i0 )
                = particle.template get<M>( i0 );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<
        (2==std::rank<member_data_type<M> >::value),void>::type
    copyParticleToMember( const int particle_index,
                          const particle_type& particle ) const
    {
        for ( int i0 = 0; i0 < particle.template extent<M,0>(); ++i0 )
            for ( int i1 = 0; i1 < particle.template extent<M,1>(); ++i1 )
                Impl::accessStructMember<M>(
                    _data(index_type::s(particle_index)),
                    index_type::i(particle_index), i0, i1 )
                    = particle.template get<M>( i0, i1 );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<
        (3==std::rank<member_data_type<M> >::value),void>::type
    copyParticleToMember( const int particle_index, const
                          particle_type& particle ) const
    {
        for ( int i0 = 0; i0 < particle.template extent<M,0>(); ++i0 )
            for ( int i1 = 0; i1 < particle.template extent<M,1>(); ++i1 )
                for ( int i2 = 0; i2 < particle.template extent<M,2>(); ++i2 )
                    Impl::accessStructMember<M>(
                        _data(index_type::s(particle_index)),
                        index_type::i(particle_index), i0, i1, i2 )
                        = particle.template get<M>( i0, i1, i2 );
    }

    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<
        (4==std::rank<member_data_type<M> >::value),void>::type
    copyParticleToMember( const int particle_index,
                          const particle_type& particle ) const
    {
        for ( int i0 = 0; i0 < particle.template extent<M,0>(); ++i0 )
            for ( int i1 = 0; i1 < particle.template extent<M,1>(); ++i1 )
                for ( int i2 = 0; i2 < particle.template extent<M,2>(); ++i2 )
                    for ( int i3 = 0; i3 < particle.template extent<M,3>(); ++i3 )
                        Impl::accessStructMember<M>(
                            _data(index_type::s(particle_index)),
                            index_type::i(particle_index), i0, i1, i2, i3 )
                            = particle.template get<M>( i0, i1, i2, i3 );
    }

    // Copy to a given index from a particle.
    template<std::size_t M>
    KOKKOS_INLINE_FUNCTION
    void copyFromParticle( const int particle_index,
                           const particle_type& particle,
                           std::integral_constant<std::size_t,M> ) const
    {
        copyParticleToMember<M>( particle_index, particle );
        copyFromParticle(
            particle_index, particle,
            std::integral_constant<std::size_t,M-1>() );
    }

    KOKKOS_INLINE_FUNCTION
    void copyFromParticle( const int particle_index,
                           const particle_type& particle,
                           std::integral_constant<std::size_t,0> ) const
    {
        copyParticleToMember<0>( particle_index, particle );
    }

  private:

    // Total number of particles in the container.
    int _size;

    // Allocated number of particles in all arrays in all structs.
    int _capacity;

    // Number of structs-of-arrays in the array.
    int _num_soa;

    // Structs-of-Arrays managed data. This Kokkos View manages the block of
    // memory owned by this class such that the copy constructor and
    // assignment operator for this class perform a shallow and reference
    // counted copy of the data.
    soa_view _data;

    // Pointers to the first element of each member.
    void* _pointers[number_of_members];

    // Strides for each member. Note that these strides are computed in the
    // context of the *value_type* of each member.
    int _strides[number_of_members];
};

//---------------------------------------------------------------------------//
// Static type checker.
template<class >
struct is_aosoa : public std::false_type {};

template<class DataTypes, class MemorySpace, class DataLayout>
struct is_aosoa<AoSoA<DataTypes,MemorySpace,DataLayout> >
    : public std::true_type {};

template<class DataTypes, class MemorySpace, class DataLayout>
struct is_aosoa<const AoSoA<DataTypes,MemorySpace,DataLayout> >
    : public std::true_type {};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // CABANA_AOSOA_HPP
