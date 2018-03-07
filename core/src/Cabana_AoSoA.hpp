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

    // Member data types.
    using member_types = MemberDataTypes<Types...>;

    // Number of member types.
    static constexpr std::size_t number_of_members = member_types::size;

    // Device type.
    using device_type = Device;

    // Memory policy.
    using memory_policy = MemoryPolicy<device_type>;

    // Inner array size (size of the arrays held by the structs).
    static constexpr std::size_t array_size = ArraySize;

    // SoA type.
    using soa_type = SoA<array_size,Types...>;

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
        , _raw_data( nullptr )
    {}

    // Construct a container with n elements.
    AoSoA( const std::size_t n )
        : _size( n )
        , _capacity( 0 )
        , _num_soa( 0 )
        , _managed_data( nullptr )
        , _raw_data( nullptr )
    {
        resize( _size );
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

        soa_type* rp;
        memory_policy::allocate( rp, num_soa_alloc );
        std::shared_ptr<soa_type> sp(
            rp, memory_policy::template deallocate<soa_type> );

        if ( _raw_data != nullptr )
            memory_policy::copy( rp, _raw_data, _num_soa );

        std::swap( _managed_data, sp );
        std::swap( _raw_data, rp );
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
    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    constexpr std::size_t rank() const
    {
        return std::rank<struct_member_data_type<I> >::value;
    }

    // Get the extent of a given member data dimension.
    template<std::size_t I, std::size_t DIM>
    CABANA_INLINE_FUNCTION
    constexpr std::size_t extent() const
    {
        return std::extent<struct_member_data_type<I>,DIM>::value;
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
        return Index( array_size, _num_soa - 1, _size % array_size );
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

    // Get the stride between SoA data for a given member at index I.
    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    constexpr std::size_t stride() const
    {
        static_assert( 0 ==
                       sizeof(soa_type) % sizeof(struct_member_value_type<I>),
                       "Stride cannont be calculated for misaligned memory!" );
        return sizeof(soa_type) / sizeof(struct_member_value_type<I>);
    }

    // Get a pointer to the data for a given member at index I.
    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    struct_member_pointer_type<I> pointer()
    {
        return static_cast<struct_member_pointer_type<I> >( array<I>(0) );
    }

    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    struct_member_const_pointer_type<I> pointer() const
    {
        return static_cast<struct_member_pointer_type<I> >( array<I>(0) );
    }

  private:

    // -------------------------------
    // Direct array data access within a struct

    // Access the data array at a given struct member index.
    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    struct_member_array_type<I> array( const std::size_t s )
    {

        return getStructMember<I>( _raw_data[s] );
    }

    template<std::size_t I>
    CABANA_INLINE_FUNCTION
    struct_member_const_array_type<I> array( const std::size_t s ) const
    {
        return getStructMember<I>( _raw_data[s] );
    }

  private:

    // Total number of elements in all arrays in all structs.
    std::size_t _size;

    // Allocated number of elements in all arrays in all structs.
    std::size_t _capacity;

    // Number of structs-of-arrays in the array.
    std::size_t _num_soa;

    // Structs-of-Arrays managed data. This shared pointer manages the memory
    // pointed to by _raw_data such that the copy constructor and assignment
    // operator for this class perform a shallow and reference counted copy of
    // the data.
    std::shared_ptr<soa_type> _managed_data;

    // Structs-of-Arrays raw data. This data will be allocated per the
    // MemoryPolicy of the given device type on which the class is
    // templated. This pointer is managed by _managed_data and will be
    // deallocated per the MemoryPolicy when the last copy of this class
    // instance is destroyed.
    soa_type* _raw_data;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // CABANA_AOSOA_HPP
