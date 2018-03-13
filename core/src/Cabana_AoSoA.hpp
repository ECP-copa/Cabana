#ifndef CABANA_AOSOA_HPP
#define CABANA_AOSOA_HPP

#include <Cabana_MemberDataTypes.hpp>
#include <Cabana_MemoryPolicy.hpp>
#include <Cabana_Macros.hpp>
#include <Cabana_SoA.hpp>
#include <Cabana_Index.hpp>

#include <type_traits>
#include <memory>
#include <cmath>
#include <cstdlib>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Forward declaration.
template<typename DataTypes, typename Device, std::size_t ArraySize>
class AoSoA;

//---------------------------------------------------------------------------//
// Static type checker.
template<typename >
struct is_aosoa
    : public std::false_type {};

template<typename DataTypes, typename Device, std::size_t ArraySize>
struct is_aosoa<AoSoA<DataTypes,Device,ArraySize> >
    : public std::true_type {};

template<typename DataTypes, typename Device, std::size_t ArraySize>
struct is_aosoa<const AoSoA<DataTypes,Device,ArraySize> >
    : public std::true_type {};

//---------------------------------------------------------------------------//
/*!
  \class AoSoA
  \brief Array-of-Structs-of-Arrays
*/
template<typename Device, std::size_t ArraySize, typename... Types>
class AoSoA<MemberDataTypes<Types...>,Device,ArraySize>
{
  public:

    // AoSoA type.
    using aosoa_type = AoSoA<MemberDataTypes<Types...>,Device,ArraySize>;

    // Device type.
    using device_type = Device;

    // Memory policy.
    using memory_policy = MemoryPolicy<device_type>;

    // Inner array size (size of the arrays held by the structs).
    static constexpr std::size_t array_size = ArraySize;

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
    CABANA_FUNCTION
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
    CABANA_FUNCTION
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
        _size = n;
        _num_soa = std::floor( n / array_size );
        if ( 0 < n % array_size ) ++_num_soa;
        reserve( _size );
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
        if ( n <= _capacity ) return;

        std::size_t num_soa_alloc = std::floor( n / array_size );
        if ( 0 < n % array_size ) ++num_soa_alloc;
        _capacity = num_soa_alloc * array_size;

        soa_type* data_block;
        memory_policy::allocate( data_block, num_soa_alloc );
        std::shared_ptr<soa_type> sp(
            data_block, memory_policy::template deallocate<soa_type> );

        if ( _managed_data != nullptr )
            memory_policy::copy( data_block, _managed_data.get(), _num_soa );

        std::swap( _managed_data, sp );

        storePointersAndStrides(
            std::integral_constant<std::size_t,number_of_members-1>() );
    }

    // Get the number of structs-of-arrays in the array.
    CABANA_FUNCTION
    std::size_t numSoA() const { return _num_soa; }

    // Get the size of the data array at a given struct member index.
    CABANA_FUNCTION
    std::size_t arraySize( const std::size_t s ) const
    {
        return
            ( s < _num_soa - 1 ) ? array_size : ( _size % array_size );
    }

    // -------------------------------
    // Member data type properties.

    // Get the rank of the data for a given member at index I.
    CABANA_INLINE_FUNCTION
    std::size_t rank( const std::size_t I ) const
    {
        return _ranks[I];
    }

    // Get the extent of a given member data dimension.
    CABANA_INLINE_FUNCTION
    std::size_t extent( const std::size_t I, const std::size_t D ) const
    {
        return _extents[I][D];
    }

    // -----------------------------
    // Array range

    // Get the index at the beginning of the entire AoSoA.
    CABANA_FUNCTION
    Index begin() const
    {
        return Index( array_size, 0, 0 );
    }

    // Get the index at end of the entire AoSoA.
    CABANA_FUNCTION
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
    CABANA_INLINE_FUNCTION
    typename std::enable_if<(0==std::rank<struct_member_data_type<I> >::value),
                            struct_member_reference_type<I> >::type
    get( const Index& idx )
    {
        return array<I>(idx.s())[idx.i()];
    }

    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<(0==std::rank<struct_member_data_type<I> >::value),
                            struct_member_const_reference_type<I> >::type
    get( const Index& idx ) const
    {
        return array<I>(idx.s())[idx.i()];
    }

    // Rank 1
    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<(1==std::rank<struct_member_data_type<I> >::value),
                            struct_member_reference_type<I> >::type
    get( const Index& idx,
         const int d0 )
    {
        return array<I>(idx.s())[idx.i()][d0];
    }

    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<(1==std::rank<struct_member_data_type<I> >::value),
                            struct_member_const_reference_type<I> >::type
    get( const Index& idx,
         const int d0 ) const
    {
        return array<I>(idx.s())[idx.i()][d0];
    }

    // Rank 2
    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<(2==std::rank<struct_member_data_type<I> >::value),
                            struct_member_reference_type<I> >::type
    get( const Index& idx,
         const int d0,
         const int d1 )
    {
        return array<I>(idx.s())[idx.i()][d0][d1];
    }

    template<std::size_t I>
    CABANA_INLINE_FUNCTION
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
    CABANA_INLINE_FUNCTION
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
    CABANA_INLINE_FUNCTION
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
    CABANA_INLINE_FUNCTION
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
    CABANA_INLINE_FUNCTION
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
    CABANA_INLINE_FUNCTION
    std::size_t stride( const std::size_t I ) const
    {
        return _strides[I];
    }

    // Get an un-typed raw pointer to the data for a given member at index
    // I. Users will need to cast this pointer to the appropriate type for the
    // stride associated with this member to mean anything.
    CABANA_INLINE_FUNCTION
    void* data( const std::size_t I )
    {
        return _pointers[I];
    }

    CABANA_INLINE_FUNCTION
    const void* data( const std::size_t I ) const
    {
        return _pointers[I];
    }

  private:

    // Get a typed pointer to the data for a given member at index I.
    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    struct_member_pointer_type<I> typedPointer()
    {
        return static_cast<struct_member_pointer_type<I> >( _pointers[I] );
    }

    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    struct_member_const_pointer_type<I> typedPointer() const
    {
        return static_cast<struct_member_pointer_type<I> >( _pointers[I] );
    }

    // Get the array at the given struct index.
    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    struct_member_array_type<I> array( const std::size_t s )
    {
        return reinterpret_cast<struct_member_array_type<I> >(
            typedPointer<I>() + s * _strides[I] );
    }

    template<std::size_t I>
    CABANA_INLINE_FUNCTION
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
        soa_type* data_block = _managed_data.get();
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
    // counted copy of the data.
    std::shared_ptr<soa_type> _managed_data;

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
