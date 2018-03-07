#ifndef CABANA_MEMBERSLICE_HPP
#define CABANA_MEMBERSLICE_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_Index.hpp>
#include <Cabana_Macros.hpp>

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Forward declaration.
template<std::size_t I, typename AoSoA_t>
class MemberSlice;

//---------------------------------------------------------------------------//
// Static type checker.
template<typename >
struct is_member_slice
    : public std::false_type {};

// True only if the type is a member slice *AND* the member slice is templated
// on an AoSoA type.
template<std::size_t I, typename AoSoA_t>
struct is_member_slice<MemberSlice<I,AoSoA_t> >
    : public is_aosoa<AoSoA_t>::type {};

template<std::size_t I, typename AoSoA_t>
struct is_member_slice<const MemberSlice<I,AoSoA_t> >
    : public is_aosoa<AoSoA_t>::type {};

//---------------------------------------------------------------------------//
/*!
  \class MemberSlice

  \brief A slice of an array-of-structs-of-arrays with data access to a single
  member.
*/
template<std::size_t I, typename AoSoA_t>
class MemberSlice
{
  public:

    // Slice type.
    using slice_type = MemberSlice<I,AoSoA_t>;

    // AoSoA type this slice wraps.
    using aosoa_type = AoSoA_t;

    // Device type.
    using device_type = typename aosoa_type::device_type;

    // Inner array size.
    static constexpr std::size_t array_size = aosoa_type::array_size;

    // Struct member array return type at a given index I.
    using array_type =
        typename aosoa_type::template struct_member_array_type<I>;

    // Struct member array const return type at a given index I.
    using const_array_type =
        typename aosoa_type::template struct_member_const_array_type<I>;

    // Struct member array data type at a given index I.
    using data_type =
        typename aosoa_type::template struct_member_data_type<I>;

    // Struct member array element value type at a given index I.
    using value_type =
        typename aosoa_type::template struct_member_value_type<I>;

    // Struct member array element reference type at a given index I.
    using reference_type =
        typename aosoa_type::template struct_member_reference_type<I>;

    // Struct member array element const reference type at a given index I.
    using const_reference_type =
        typename aosoa_type::template struct_member_const_reference_type<I>;

    // Struct member array element pointer type at a given index I.
    using pointer_type =
        typename aosoa_type::template struct_member_pointer_type<I>;

    // Struct member array element const pointer type at a given index I.
    using const_pointer_type =
        typename aosoa_type::template struct_member_const_pointer_type<I>;

  public:

    // Constructor.
    MemberSlice( aosoa_type aosoa )
        : _aosoa( aosoa )
    {
        static_assert( is_aosoa<aosoa_type>::value,
                       "MemberSlice can only wrap AoSoA objects!" );
        static_assert( I < aosoa_type::number_of_members,
                       "MemberSlice index is out of range!" );
    }

    // Returns the number of elements in the container.
    CABANA_FUNCTION
    std::size_t size() const { return _aosoa.size(); }

    // Get the number of structs-of-arrays in the array.
    CABANA_FUNCTION
    std::size_t numSoA() const { return _aosoa.numSoA(); }

    // Get the size of the data array at a given struct member index.
    CABANA_FUNCTION
    std::size_t arraySize( const std::size_t s ) const
    { return _aosoa.arraySize(s); }

    // -------------------------------
    // Member data type properties.

    // Get the rank of the data for this member.
    CABANA_INLINE_FUNCTION
    constexpr std::size_t rank() const
    { return _aosoa.template rank<I>(); }

    // Get the extent of a given member data dimension.
    template<std::size_t DIM>
    CABANA_INLINE_FUNCTION
    constexpr std::size_t extent() const
    { return _aosoa.template extent<I,DIM>(); }

    // -----------------------------
    // Array range

    // Get the index at the beginning of the entire AoSoA.
    CABANA_FUNCTION
    Index begin() const { return _aosoa.begin(); }

    // Get the index at end of the entire AoSoA.
    CABANA_FUNCTION
    Index end() const { return _aosoa.end(); }

    // -------------------------------
    // Access the data value at a given struct index and array index

    // Rank 0
    template<std::size_t J = I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<
        (0==std::rank<
         typename aosoa_type::template struct_member_data_type<J> >::value),
        reference_type>::type
    operator()( const Index& idx )
    {
        static_assert( J == I,
                       "Do not call with different template arguments!" );
        return _aosoa.template get<J>(idx);
    }

    template<std::size_t J = I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<
        (0==std::rank<
         typename aosoa_type::template struct_member_data_type<J> >::value),
        const_reference_type>::type
    operator()( const Index& idx ) const
    {
        static_assert( J == I,
                       "Do not call with different template arguments!" );
        return _aosoa.template get<J>(idx);
    }

    // Rank 1
    template<std::size_t J = I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<
        (1==std::rank<
         typename aosoa_type::template struct_member_data_type<J> >::value),
        reference_type>::type
    operator()( const Index& idx,
                const int d0 )
    {
        static_assert( J == I,
                       "Do not call with different template arguments!" );
        return _aosoa.template get<J>(idx,d0);
    }

    template<std::size_t J = I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<
        (1==std::rank<
         typename aosoa_type::template struct_member_data_type<J> >::value),
        const_reference_type>::type
    operator()( const Index& idx,
                const int d0 ) const
    {
        static_assert( J == I,
                       "Do not call with different template arguments!" );
        return _aosoa.template get<J>(idx,d0);
    }

    // Rank 2
    template<std::size_t J = I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<
        (2==std::rank<typename aosoa_type::template struct_member_data_type<J> >::value),
        reference_type>::type
    operator()( const Index& idx,
                const int d0,
                const int d1 )
    {
        static_assert( J == I, "Do not call with different template arguments!" );
        return _aosoa.template get<J>(idx,d0,d1);
    }

    template<std::size_t J = I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<
        (2==std::rank<
         typename aosoa_type::template struct_member_data_type<J> >::value),
        const_reference_type>::type
    operator()( const Index& idx,
                const int d0,
                const int d1 ) const
    {
        static_assert( J == I,
                       "Do not call with different template arguments!" );
        return _aosoa.template get<J>(idx,d0,d1);
    }

    // Rank 3
    template<std::size_t J = I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<
        (3==std::rank<
         typename aosoa_type::template struct_member_data_type<J> >::value),
        reference_type>::type
    operator()( const Index& idx,
                const int d0,
                const int d1,
                const int d2 )
    {
        static_assert( J == I,
                       "Do not call with different template arguments!" );
        return _aosoa.template get<J>(idx,d0,d1,d2);
    }

    template<std::size_t J = I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<
        (3==std::rank<
         typename aosoa_type::template struct_member_data_type<J> >::value),
        const_reference_type>::type
    operator()( const Index& idx,
                const int d0,
                const int d1,
                const int d2 ) const
    {
        static_assert( J == I,
                       "Do not call with different template arguments!" );
        return _aosoa.template get<J>(idx,d0,d1,d2);
    }

    // Rank 4
    template<std::size_t J = I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<
        (4==std::rank<
         typename aosoa_type::template struct_member_data_type<J> >::value),
        reference_type>::type
    operator()( const Index& idx,
                const int d0,
                const int d1,
                const int d2,
                const int d3 )
    {
        static_assert( J == I,
                       "Do not call with different template arguments!" );
        return _aosoa.template get<J>(idx,d0,d1,d2,d3);
    }

    template<std::size_t J = I>
    CABANA_INLINE_FUNCTION
    typename std::enable_if<
        (4==std::rank<
         typename aosoa_type::template struct_member_data_type<J> >::value),
        const_reference_type>::type
    operator()( const Index& idx,
                const int d0,
                const int d1,
                const int d2,
                const int d3 ) const
    {
        static_assert( J == I,
                       "Do not call with different template arguments!" );
        return _aosoa.template get<J>(idx,d0,d1,d2,d3);
    }

    // -------------------------------
    // Raw data access.

    // Get the stride between SoA data for a given member at index I.
    CABANA_INLINE_FUNCTION
    constexpr std::size_t stride() const
    { return _aosoa.template stride<I>(); }

    // Get a pointer to the data for a given member at index I.
    CABANA_INLINE_FUNCTION
    pointer_type pointer()
    { return _aosoa.template pointer<I>(); }

    CABANA_INLINE_FUNCTION
    const_pointer_type pointer() const
    { return _aosoa.template pointer<I>(); }

  private:

    // The array-of-structs-of-arrays this slice wraps. We hold this by value
    // because the underlying block of data in the AoSoA is reference counted.
    aosoa_type _aosoa;
};

//---------------------------------------------------------------------------//
// Free function so slice an AoSoA over a given member.
template<std::size_t I, typename AoSoA_t>
MemberSlice<I,AoSoA_t> slice( AoSoA_t aosoa )
{ return MemberSlice<I,AoSoA_t>(aosoa); }

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_MEMBERSLICE_HPP
