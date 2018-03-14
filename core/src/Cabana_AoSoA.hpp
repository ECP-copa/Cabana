#ifndef CABANA_AOSOA_HPP
#define CABANA_AOSOA_HPP

#include <Cabana_MemberDataTypes.hpp>
#include <Cabana_SoA.hpp>
#include <Cabana_Index.hpp>
#include <Cabana_InnerArraySize.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_ExecPolicy.hpp>

#include <type_traits>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <string>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*! \class AoSoATraits
  \brief Traits class for accessing attributes of a AoSoA.

  This is an implementation detail of AoSoA.  It is only of interest
  to developers implementing a new specialization of AoSoA.

  Template argument options:
  - AoSoA< DataTypes >
  - AoSoA< DataTypes , Space >
  - AoSoA< DataTypes , Space , MemoryTraits >
  - AoSoA< DataTypes , ArraySize >
  - AoSoA< DataTypes , ArraySize , Space >
  - AoSoA< DataTypes , ArraySize , MemoryTraits >
  - AoSoA< DataTypes , ArraySize , Space , MemoryTraits >
  - AoSoA< DataTypes , MemoryTraits >

  Note that this is effectively a reimplementation of Kokkos::ViewTraits for
  the AoSoA with ArrayLayout replaced by ArraySize.
*/
template<class DataTypes , class ... Properties>
class AoSoATraits ;

// Void specialization.
template<>
class AoSoATraits<void>
{
  public:
    using execution_space = void;
    using memory_space = void;
    using host_mirror_space = void;
    using array_size = void;
    using memory_traits = void;
};

// Extract the array size.
template<class ArraySize, class ... Properties>
class AoSoATraits<
    typename std::enable_if<is_inner_array_size<ArraySize>::value>::type,
    ArraySize, Properties...>
{
  public:
    using execution_space = typename AoSoATraits<void,Properties...>::execution_space;
    using memory_space = typename AoSoATraits<void,Properties...>::memory_space;
    using host_mirror_space = typename AoSoATraits<void,Properties...>::host_mirror_space;
    using array_size = ArraySize;
    using memory_traits = typename AoSoATraits<void,Properties...>::memory_traits;
};

// Extract the space - either a Kokkos memory space or execution space. Can be
// on or the other but not both.
template<class Space, class ... Properties>
class AoSoATraits<
    typename std::enable_if<Kokkos::Impl::is_space<Space>::value>::type,
    Space, Properties ...>
{
  public:
    static_assert(
        std::is_same<typename AoSoATraits<void,Properties...>::execution_space,void>::value &&
        std::is_same<typename AoSoATraits<void,Properties...>::memory_space,void>::value &&
        std::is_same<typename AoSoATraits<void,Properties...>::host_mirror_space,void>::value &&
        std::is_same<typename AoSoATraits<void,Properties...>::array_size,void>::value
        , "Only one AoSoA Execution or Memory Space template argument" );

    using execution_space = typename Space::execution_space;
    using memory_space = typename Space::memory_space;
    using host_mirror_space = typename Kokkos::Impl::HostMirror<Space>::Space;
    using array_size = ExecutionSpaceInnerArraySize<execution_space>;
    using memory_traits = typename AoSoATraits<void,Properties...>::memory_traits;
};

// Extract the memory traits - this must be the last template parameter in the pack.
template<class MemoryTraits, class ... Properties>
class AoSoATraits<
    typename std::enable_if<Kokkos::Impl::is_memory_traits<MemoryTraits>::value>::type,
    MemoryTraits, Properties...>
{
  public:
    static_assert( std::is_same<typename AoSoATraits<void,Properties...>::execution_space,void>::value &&
                   std::is_same<typename AoSoATraits<void,Properties...>::memory_space,void>::value &&
                   std::is_same<typename AoSoATraits<void,Properties...>::array_size,void>::value &&
                   std::is_same<typename AoSoATraits<void,Properties...>::memory_traits,void>::value
                   , "MemoryTrait is the final optional template argument for a AoSoA" );

    using execution_space = void;
    using memory_space = void;
    using host_mirror_space = void;
    using array_size = void;
    using memory_traits = MemoryTraits;
};

// Set the traits for a given set of properties.
template<class DataTypes, class ... Properties>
class AoSoATraits
{
  private:

    typedef AoSoATraits<void,Properties...>  properties;

    using ExecutionSpace =
        typename
        std::conditional<
        !std::is_same<typename properties::execution_space,void>::value,
        typename properties::execution_space,
        Kokkos::DefaultExecutionSpace
        >::type;

    using MemorySpace =
        typename std::conditional<
        !std::is_same<typename properties::memory_space,void>::value,
        typename properties::memory_space,
        typename ExecutionSpace::memory_space
        >::type;

    using ArraySize =
        typename std::conditional<
        !std::is_same<typename properties::array_size,void>::value,
        typename properties::array_size,
        ExecutionSpaceInnerArraySize<ExecutionSpace>
        >::type;

    using HostMirrorSpace =
        typename std::conditional<
        !std::is_same<typename properties::host_mirror_space,void>::value,
        typename properties::host_mirror_space,
        typename Kokkos::Impl::HostMirror<ExecutionSpace>::Space
        >::type;

    using MemoryTraits =
        typename std::conditional<
        !std::is_same<typename properties::memory_traits,void>::value,
        typename properties::memory_traits,
        typename Kokkos::MemoryManaged
        >::type;

  public:

    using data_types = DataTypes;
    using execution_space = ExecutionSpace;
    using memory_space = MemorySpace;
    using device_type = Kokkos::Device<ExecutionSpace,MemorySpace>;
    using memory_traits = MemoryTraits;
    using host_mirror_space = HostMirrorSpace;
    using size_type = typename memory_space::size_type;

    static constexpr std::size_t array_size = ArraySize::value;
};

//---------------------------------------------------------------------------//
// Forward declaration.
template<typename DataTypes, typename ... Properties>
class AoSoA;

//---------------------------------------------------------------------------//
// Static type checker.
template<class >
struct is_aosoa : public std::false_type {};

template<class DataTypes, class ... Properties>
struct is_aosoa<AoSoA<DataTypes,Properties...> >
    : public std::true_type {};

template<class DataTypes, class ... Properties>
struct is_aosoa<const AoSoA<DataTypes,Properties...> >
    : public std::true_type {};

//---------------------------------------------------------------------------//
/*!
  \class AoSoA
  \brief Array-of-Structs-of-Arrays
*/
template<class ... Types, class ... Properties>
class AoSoA<MemberDataTypes<Types...>,Properties...>
{
  public:

    // Traits.
    using traits = AoSoATraits<MemberDataTypes<Types...>,Properties...>;

    // AoSoA type.
    using aosoa_type = AoSoA<MemberDataTypes<Types...>,Properties...>;

    // Array size.
    static constexpr std::size_t array_size = traits::array_size;

    // SoA type.
    using soa_type = SoA<array_size,Types...>;

    // Member data types.
    using member_types = MemberDataTypes<Types...>;

    // Number of member types.
    static constexpr std::size_t number_of_members = member_types::size;

    // The maximum rank supported for member types.
    static constexpr std::size_t max_supported_rank = 4;

    // Struct member array return type at a given index I.
    template<std::size_t I>
    using struct_member_array_type =
        typename ArrayTypeAtIndex<I,array_size,Types...>::return_type;

    // Struct member array const return type at a given index I.
    template<std::size_t I>
    using struct_member_const_array_type =
        typename std::add_const<struct_member_array_type<I> >::type;

    // Struct member array data type at a given index I.
    template<std::size_t I>
    using struct_member_data_type =
        typename std::remove_pointer<struct_member_array_type<I> >::type;

    // Struct member array element value type at a given index I.
    template<std::size_t I>
    using struct_member_value_type =
        typename std::remove_all_extents<struct_member_data_type<I> >::type;

    // Struct member array element reference type at a given index I.
    template<std::size_t I>
    using struct_member_reference_type =
        typename std::add_lvalue_reference<struct_member_value_type<I> >::type;

    // Struct member array element const reference type at a given index I.
    template<std::size_t I>
    using struct_member_const_reference_type =
        typename std::add_const<struct_member_reference_type<I> >::type;

    // Struct member array element pointer type at a given index I.
    template<std::size_t I>
    using struct_member_pointer_type =
        typename std::add_pointer<struct_member_value_type<I> >::type;

    // Struct member array element const pointer type at a given index I.
    template<std::size_t I>
    using struct_member_const_pointer_type =
        typename std::add_const<struct_member_pointer_type<I> >::type;

  public:

    // Default constructor.
    AoSoA()
        : _size( 0 )
        , _capacity( 0 )
        , _num_soa( 0 )
        , _managed_data( nullptr )
    {
        storeRanksAndExtents(
            std::integral_constant<std::size_t,number_of_members-1>() );
    }

    // Construct a container with n elements.
    AoSoA( const std::size_t n )
        : _size( n )
        , _capacity( 0 )
        , _num_soa( 0 )
        , _managed_data( nullptr )
    {
        resize( _size );
        storeRanksAndExtents(
            std::integral_constant<std::size_t,number_of_members-1>() );
    }

    // Returns the number of elements in the container.
    //
    // This is the number of actual objects held in the container, which is
    // not necessarily equal to its storage capacity.
    KOKKOS_FUNCTION
    std::size_t size() const { return _size; }

    // Returns the size of the storage space currently allocated for the
    // container, expressed in terms of elements.
    //
    // This capacity is not necessarily equal to the container size. It can be
    // equal or greater, with the extra space allowing to accommodate for
    // growth without the need to reallocate on each insertion.
    //
    // Notice that this capacity does not suppose a limit on the size of the
    // container. When this capacity is exhausted and more is needed, it is
    // automatically expanded by the container (reallocating it storage
    // space).
    //
    // The capacity of a container can be explicitly altered by calling member
    // reserve.
    KOKKOS_FUNCTION
    std::size_t capacity() const { return _capacity; }

    // Resizes the container so that it contains n elements.
    //
    // If n is smaller than the current container size, the content is reduced
    // to its first n elements.
    //
    // If n is greater than the current container size, the content is
    // expanded by inserting at the end as many elements as needed to reach a
    // size of n.
    //
    // If n is also greater than the current container capacity, an automatic
    // reallocation of the allocated storage space takes place.
    //
    // Notice that this function changes the actual content of the container
    // by inserting or erasing elements from it.
    void resize( const std::size_t n )
    {
        reserve( n );
        _size = n;
        _num_soa = std::floor( n / array_size );
        if ( 0 < n % array_size ) ++_num_soa;
    }

    // Requests that the container capacity be at least enough to contain n
    // elements.
    //
    // If n is greater than the current container capacity, the function
    // causes the container to reallocate its storage increasing its capacity
    // to n (or greater).
    //
    // In all other cases, the function call does not cause a reallocation and
    // the container capacity is not affected.
    //
    // This function has no effect on the container size and cannot alter its
    // elements.
    void reserve( const std::size_t n )
    {
        // If we aren't asking for more memory then we have nothing to do.
        if ( n <= _capacity ) return;

        // Figure out the new capacity.
        std::size_t num_soa_alloc = std::floor( n / array_size );
        if ( 0 < n % array_size ) ++num_soa_alloc;
        _capacity = num_soa_alloc * array_size;

        // Allocate a new block of memory.
        std::shared_ptr<void> sp(
            Kokkos::kokkos_malloc(num_soa_alloc * sizeof(soa_type)),
            Kokkos::kokkos_free<typename traits::memory_space> );

        // Fence before continuing to ensure the allocation is completed.
        Kokkos::fence();

        // If we have already allocated memory, copy the old memory into the
        // new memory. Fence when we are done to ensure copy is complete
        // before continuing.
        if ( _managed_data != nullptr )
        {
            Kokkos::Impl::DeepCopy<
                typename traits::memory_space,
                typename traits::memory_space,
                typename traits::execution_space>(
                    sp.get(), _managed_data.get(), _num_soa * sizeof(soa_type) );
            Kokkos::fence();
        }

        // Swap blocks. The old block will be destroyed when this function exits.
        std::swap( _managed_data, sp );

        // Get new pointers and strides for the members.
        storePointersAndStrides(
            std::integral_constant<std::size_t,number_of_members-1>() );
    }

    // Get the number of structs-of-arrays in the array.
    KOKKOS_FUNCTION
    std::size_t numSoA() const { return _num_soa; }

    // Get the size of the data array at a given struct member index.
    KOKKOS_FUNCTION
    std::size_t arraySize( const std::size_t s ) const
    {
        return
            ( s < _num_soa - 1 ) ? array_size : ( _size % array_size );
    }

    // -------------------------------
    // Member data type properties.

    // Get the rank of the data for a given member at index I.
    KOKKOS_INLINE_FUNCTION
    std::size_t rank( const std::size_t I ) const
    {
        return _ranks[I];
    }

    // Get the extent of a given member data dimension.
    KOKKOS_INLINE_FUNCTION
    std::size_t extent( const std::size_t I, const std::size_t D ) const
    {
        return _extents[I][D];
    }

    // -----------------------------
    // Array range

    // Get the index at the beginning of the entire AoSoA.
    KOKKOS_FUNCTION
    Index begin() const
    {
        return Index( array_size, 0, 0 );
    }

    // Get the index at end of the entire AoSoA.
    KOKKOS_FUNCTION
    Index end() const
    {
        std::size_t remainder = _size % array_size;
        std::size_t s = ( 0 == remainder ) ? _num_soa : _num_soa - 1;
        std::size_t i = ( 0 == remainder ) ? 0 : remainder;
        return Index( array_size, s, i );
    }

    // -------------------------------
    // Access the data value at a given member index, struct index, and array
    // index

    // Rank 0
    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(0==std::rank<struct_member_data_type<I> >::value),
                            struct_member_reference_type<I> >::type
    get( const Index& idx )
    {
        return array<I>(idx.s())[idx.i()];
    }

    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(0==std::rank<struct_member_data_type<I> >::value),
                            struct_member_const_reference_type<I> >::type
    get( const Index& idx ) const
    {
        return array<I>(idx.s())[idx.i()];
    }

    // Rank 1
    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(1==std::rank<struct_member_data_type<I> >::value),
                            struct_member_reference_type<I> >::type
    get( const Index& idx,
         const int d0 )
    {
        return array<I>(idx.s())[idx.i()][d0];
    }

    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(1==std::rank<struct_member_data_type<I> >::value),
                            struct_member_const_reference_type<I> >::type
    get( const Index& idx,
         const int d0 ) const
    {
        return array<I>(idx.s())[idx.i()][d0];
    }

    // Rank 2
    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(2==std::rank<struct_member_data_type<I> >::value),
                            struct_member_reference_type<I> >::type
    get( const Index& idx,
         const int d0,
         const int d1 )
    {
        return array<I>(idx.s())[idx.i()][d0][d1];
    }

    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(2==std::rank<struct_member_data_type<I> >::value),
                            struct_member_const_reference_type<I> >::type
    get( const Index& idx,
         const int d0,
         const int d1 ) const
    {
        return array<I>(idx.s())[idx.i()][d0][d1];
    }

    // Rank 3
    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(3==std::rank<struct_member_data_type<I> >::value),
                            struct_member_reference_type<I> >::type
    get( const Index& idx,
         const int d0,
         const int d1,
         const int d2 )
    {
        return array<I>(idx.s())[idx.i()][d0][d1][d2];
    }

    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(3==std::rank<struct_member_data_type<I> >::value),
                            struct_member_const_reference_type<I> >::type
    get( const Index& idx,
         const int d0,
         const int d1,
         const int d2 ) const
    {
        return array<I>(idx.s())[idx.i()][d0][d1][d2];
    }

    // Rank 4
    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(4==std::rank<struct_member_data_type<I> >::value),
                            struct_member_reference_type<I> >::type
    get( const Index& idx,
         const int d0,
         const int d1,
         const int d2,
         const int d3 )
    {
        return array<I>(idx.s())[idx.i()][d0][d1][d2][d3];
    }

    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    typename std::enable_if<(4==std::rank<struct_member_data_type<I> >::value),
                            struct_member_const_reference_type<I> >::type
    get( const Index& idx,
         const int d0,
         const int d1,
         const int d2,
         const int d3 ) const
    {
        return array<I>(idx.s())[idx.i()][d0][d1][d2][d3];
    }

    // -------------------------------
    // Raw data access.

    // Get the stride between SoA data for a given member at index I. Note
    // that this strides are computed in the context of the *value_type* for
    // each member.
    KOKKOS_INLINE_FUNCTION
    std::size_t stride( const std::size_t I ) const
    {
        return _strides[I];
    }

    // Get an un-typed raw pointer to the data for a given member at index
    // I. Users will need to cast this pointer to the appropriate type for the
    // stride associated with this member to mean anything.
    KOKKOS_INLINE_FUNCTION
    void* data( const std::size_t I )
    {
        return _pointers[I];
    }

    KOKKOS_INLINE_FUNCTION
    const void* data( const std::size_t I ) const
    {
        return _pointers[I];
    }

  private:

    // Get a typed pointer to the data for a given member at index I.
    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    struct_member_pointer_type<I> typedPointer()
    {
        return static_cast<struct_member_pointer_type<I> >( _pointers[I] );
    }

    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    struct_member_const_pointer_type<I> typedPointer() const
    {
        return static_cast<struct_member_pointer_type<I> >( _pointers[I] );
    }

    // Get the array at the given struct index.
    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    struct_member_array_type<I> array( const std::size_t s )
    {
        return reinterpret_cast<struct_member_array_type<I> >(
            typedPointer<I>() + s * _strides[I] );
    }

    template<std::size_t I>
    KOKKOS_INLINE_FUNCTION
    struct_member_const_array_type<I> array( const std::size_t s ) const
    {
        return reinterpret_cast<struct_member_array_type<I> >(
            typedPointer<I>() + s * _strides[I] );
    }

    // Store the pointers and strides for each member element.
    template<std::size_t N>
    void assignPointersAndStrides()
    {
        static_assert( 0 <= N && N < number_of_members,
                       "Static loop out of bounds!" );
        soa_type* data_block =
            std::static_pointer_cast<soa_type>(_managed_data).get();
        _pointers[N] =
            static_cast<void*>( getStructMember<N>(data_block[0]) );
        static_assert( 0 ==
                       sizeof(soa_type) % sizeof(struct_member_value_type<N>),
                       "Stride cannont be calculated for misaligned memory!" );
        _strides[N] = sizeof(soa_type) / sizeof(struct_member_value_type<N>);
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

    // Store the extents of each of the member types.
    template<std::size_t I, std::size_t N>
    void assignExtents()
    {
        static_assert( 0 <= N && N < max_supported_rank,
                       "Static loop out of bounds!" );
        _extents[I][N] = ( N < std::rank<struct_member_data_type<I> >::value )
                         ? std::extent<struct_member_data_type<I>,N>::value
                         : 0;
    }

    // Static loop over extents for each member element.
    template<std::size_t I, std::size_t N>
    void storeExtents( std::integral_constant<std::size_t,I>,
                       std::integral_constant<std::size_t,N> )
    {
        assignExtents<I,N>();
        storeExtents( std::integral_constant<std::size_t,I>(),
                      std::integral_constant<std::size_t,N-1>() );
    }

    template<std::size_t I>
    void storeExtents( std::integral_constant<std::size_t,I>,
                       std::integral_constant<std::size_t,0> )
    {
        assignExtents<I,0>();
    }

    // Store the rank for each member element type.
    template<std::size_t N>
    void assignRanks()
    {
        static_assert( std::rank<struct_member_data_type<N> >::value <=
                       max_supported_rank,
                       "Member type rank larger than max supported rank" );
        static_assert( 0 <= N && N < number_of_members, "Static loop out of bounds!" );
        _ranks[N] = std::rank<struct_member_data_type<N> >::value;
    }

    // Static loop over ranks and extents for each element.
    template<std::size_t N>
    void storeRanksAndExtents( std::integral_constant<std::size_t,N> )
    {
        assignRanks<N>();
        storeExtents(
            std::integral_constant<std::size_t,N>(),
            std::integral_constant<std::size_t,max_supported_rank-1>() );
        storeRanksAndExtents( std::integral_constant<std::size_t,N-1>() );
    }

    void storeRanksAndExtents( std::integral_constant<std::size_t,0> )
    {
        storeExtents(
            std::integral_constant<std::size_t,0>(),
            std::integral_constant<std::size_t,max_supported_rank-1>() );
        assignRanks<0>();
    }

  private:

    // Total number of elements in all arrays in all structs.
    std::size_t _size;

    // Allocated number of elements in all arrays in all structs.
    std::size_t _capacity;

    // Number of structs-of-arrays in the array.
    std::size_t _num_soa;

    // Structs-of-Arrays managed data. This shared pointer manages the block
    // of memory owned by this class such that the copy constructor and
    // assignment operator for this class perform a shallow and reference
    // counted copy of the data. The underlying pointer is to an array of
    // soa_type objects.
    std::shared_ptr<void> _managed_data;

    // Pointers to the first element of each member.
    void* _pointers[number_of_members];

    // Strides for each member. Note that these strides are computed in the
    // context of the *value_type* of each member.
    std::size_t _strides[number_of_members];

    // The ranks of each of the data member types.
    std::size_t _ranks[number_of_members];

    // The extents of each of the data member type dimensions.
    std::size_t _extents[number_of_members][max_supported_rank];
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // CABANA_AOSOA_HPP
