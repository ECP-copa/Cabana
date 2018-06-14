#ifndef CABANA_MEMBERSLICE_HPP
#define CABANA_MEMBERSLICE_HPP

#include <Cabana_InnerArrayLayout.hpp>
#include <impl/Cabana_Index.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \class MemberSlice

  \brief A slice of an array-of-structs-of-arrays with data access to a single
  member.

  A slice provides a simple wrapper around a single data member of an
  AoSoA. This does a few convenient things. First, it decouples access of the
  member from the AoSoA meaning that functionality can be implemented using
  multiple slices from potentially multiple AoSoA containers. Second, it
  eliminates the member index template parameter from the AoSoA get function,
  instead giving an operator() syntax for accessing the member data.

  Construction of slices is recommended using the free helper function as:

  \code
  auto slice = Cabana::slice<M>(aosoa);
  \endcode
*/
//---------------------------------------------------------------------------//
template<typename DataType, typename ArrayLayout>
class MemberSlice
{
  public:

    // DataType
    using data_type = DataType;

    // Ordered array type.
    using ordered_array_type =
        typename Impl::InnerArrayType<data_type,ArrayLayout>::type;

    // Pointer to an array.
    using pointer_to_array_type = typename std::decay<ordered_array_type>::type;

    // Value type.
    using value_type = typename std::remove_all_extents<data_type>::type;

    // Reference type
    using reference_type = typename std::add_lvalue_reference<value_type>::type;

    // Poiner type.
    using pointer_type = typename std::add_pointer<value_type>::type;

    // Array size.
    static constexpr int array_size = ArrayLayout::size;

    // Index type.
    using index_type = Impl::Index<array_size>;

    // Maximum supported rank.
    static constexpr int max_supported_rank = 4;

    // Array layout.
    using array_layout = typename ArrayLayout::layout;

  private:

    enum {
        is_layout_left = std::is_same<array_layout,Kokkos::LayoutLeft>::value,
        is_layout_right = std::is_same<array_layout,Kokkos::LayoutRight>::value
    };

  public:

    /*!
      \brief Constructor.
    */
    MemberSlice( const pointer_type data,
                 const int size,
                 const int stride,
                 const int num_soa )
        : _data( data )
        , _size( size )
        , _stride( stride )
        , _num_soa( num_soa )
    {
        storeExtents(
            std::integral_constant<std::size_t,max_supported_rank-1>() );
    }

    /*!
      \brief Copy constructor.
    */
    MemberSlice( const MemberSlice& slice )
    {
        _data = slice._data;
        _size = slice._size;
        _stride = slice._stride;
        _num_soa = slice._num_soa;
        storeExtents(
            std::integral_constant<std::size_t,max_supported_rank-1>() );
    }

    /*!
     * \brief Assignment operator.
     */
    MemberSlice& operator=( const MemberSlice& slice )
    {
        _data = slice._data;
        _size = slice._size;
        _stride = slice._stride;
        _num_soa = slice._num_soa;
        storeExtents(
            std::integral_constant<std::size_t,max_supported_rank-1>() );
    }

    /*!
     * \brief Move operators.
     */
    MemberSlice( MemberSlice && ) = default ;
    MemberSlice & operator = ( MemberSlice && ) = default ;

    /*!
      \brief Returns the number of elements in the container.

      \return The number of elements in the container.
    */
    KOKKOS_FUNCTION
    int size() const { return _size; }

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

    // -------------------------------
    // Member data type properties.

    /*!
      \brief Get the rank of the data for this member.

      \return The rank of the data for this member.
    */
    KOKKOS_INLINE_FUNCTION
    constexpr int rank() const
    { return std::rank<data_type>::value; }

    /*!
      \brief Get the extent of a given member data dimension.

      \param D The member data dimension to get the extent for.

      \return The extent of the given member data dimension.
    */
    KOKKOS_INLINE_FUNCTION
    int extent( const std::size_t D ) const
    { return _extents[D]; }

    // -------------------------------
    // Access the data value at a given particle index.

    // Rank 0
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<U>::value),
                            reference_type>::type
    operator()( const int p ) const
    {
        return array(index_type::s(p))[index_type::i(p)];
    }

    // Rank 1
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((1==std::rank<U>::value) && is_layout_right),
                            reference_type>::type
    operator()( const int p,
                const int d0 ) const
    {
        return array(index_type::s(p))[index_type::i(p)][d0];
    }

    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((1==std::rank<U>::value) && is_layout_left),
                            reference_type>::type
    operator()( const int p,
                const int d0 ) const
    {
        return array(index_type::s(p))[d0][index_type::i(p)];
    }

    // Rank 2
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((2==std::rank<U>::value) && is_layout_right),
                            reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1 ) const
    {
        return array(index_type::s(p))[index_type::i(p)][d0][d1];
    }

    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((2==std::rank<U>::value) && is_layout_left),
                            reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1 ) const
    {
        return array(index_type::s(p))[d1][d0][index_type::i(p)];
    }

    // Rank 3
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((3==std::rank<U>::value) && is_layout_right),
                            reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1,
                const int d2 ) const
    {
        return array(index_type::s(p))[index_type::i(p)][d0][d1][d2];
    }

    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((3==std::rank<U>::value) && is_layout_left),
                            reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1,
                const int d2 ) const
    {
        return array(index_type::s(p))[d2][d1][d0][index_type::i(p)];
    }

    // Rank 4
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((4==std::rank<U>::value) && is_layout_right),
                            reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1,
                const int d2,
                const int d3 ) const
    {
        return array(index_type::s(p))[index_type::i(p)][d0][d1][d2][d3];
    }

    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((4==std::rank<U>::value) && is_layout_left),
                            reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1,
                const int d2,
                const int d3 ) const
    {
        return array(index_type::s(p))[d3][d2][d1][d0][index_type::i(p)];
    }

    // -------------------------------
    // Raw data access.

    /*!
      \brief Get the stride between SoA data for this member.

      \return The stride at the given member index.

      Note that these strides are computed in the context of the *value_type*
      for each member.
    */
    KOKKOS_INLINE_FUNCTION
    int stride() const
    {
        return _stride;
    }

    /*!
      \brief Get a raw pointer to the data for this member

      \return A raw pointer to the member data at the given index.
    */
    KOKKOS_INLINE_FUNCTION
    pointer_type data() const
    {
        return _data;
    }

  private:

    // Get the array corresponding to the given struct index.
    KOKKOS_FORCEINLINE_FUNCTION
    pointer_to_array_type array( const int s ) const
    {
        return reinterpret_cast<pointer_to_array_type>( _data + s*_stride );
    }

    // Store the extents of each of the member types.
    template<std::size_t D>
    void assignExtents()
    {
        static_assert( 0 <= D && D < max_supported_rank,
                       "Static loop out of bounds!" );
        _extents[D] = ( D < std::rank<data_type>::value )
                      ? std::extent<data_type,D>::value
                      : 0;
    }

    // Static loop over extents for each member element.
    template<std::size_t D>
    void storeExtents( std::integral_constant<std::size_t,D> )
    {
        assignExtents<D>();
        storeExtents( std::integral_constant<std::size_t,D-1>() );
    }

    void storeExtents( std::integral_constant<std::size_t,0> )
    {
        assignExtents<0>();
    }

  private:

    // The data this slice wraps. We restrict to convince the compiler we are
    // not aliasing.
    pointer_type __restrict__ _data;

    // Total number of elements.
    int _size;

    // Stride between tiles.
    int _stride;

    // Total number of tiles.
    int _num_soa;

    // Extents
    int _extents[max_supported_rank];
};

//---------------------------------------------------------------------------//
// Static type checker.
template<typename >
struct is_member_slice : public std::false_type {};

// True only if the type is a member slice *AND* the member slice is templated
// on an AoSoA type.
template<typename T, typename Layout>
struct is_member_slice<MemberSlice<T,Layout> > : public std::true_type {};

template<typename T, typename Layout>
struct is_member_slice<const MemberSlice<T,Layout> > : public std::true_type {};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // end CABANA_MEMBERSLICE_HPP
