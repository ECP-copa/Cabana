#ifndef CABANA_MEMBERSLICE_HPP
#define CABANA_MEMBERSLICE_HPP

#include <Cabana_InnerArrayLayout.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
// Memory access tags for particle fields.
//---------------------------------------------------------------------------//

//! Default memory access. Default memory (and all memory) is unmanaged in
//! Kokkos and restricted to prevent aliasing in the larger AoSoA memory
//! block.
struct DefaultAccessMemory
{
    using memory_access_type = DefaultAccessMemory;
    using kokkos_memory_traits = Kokkos::MemoryTraits< Kokkos::Unmanaged |
                                                       Kokkos::Restrict >;
};

//! Random access memory. Read-only and const.
struct RandomAccessMemory
{
    using memory_access_type = RandomAccessMemory;
    using kokkos_memory_traits = Kokkos::MemoryTraits< Kokkos::Unmanaged |
                                                       Kokkos::Restrict |
                                                       Kokkos::RandomAccess >;
};

//! Atomic memory access. All reads and writes are atomic.
struct AtomicAccessMemory
{
    using memory_access_type = AtomicAccessMemory;
    using kokkos_memory_traits = Kokkos::MemoryTraits< Kokkos::Unmanaged |
                                                       Kokkos::Restrict |
                                                       Kokkos::Atomic >;
};

//---------------------------------------------------------------------------//

namespace Impl
{
//---------------------------------------------------------------------------//
// Given a particle field type T of the given rank get the Kokkos view
// data layout parameters. The particle index effectively introduces 2 new
// dimensions to the problem on top of the field dimensions - one for the
// struct index and one for the vector index.
template<typename T, typename DataLayout, std::size_t Rank>
struct KokkosDataLayoutImpl;

// Rank-0
template<typename T, typename DataLayout>
struct KokkosDataLayoutImpl<T,DataLayout,0>
{
    using value_type = typename std::remove_all_extents<T>::type;
    using data_type = value_type**;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride,
                                                     const int vector_length )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     vector_length, 1 );
    }
};

// Rank-1
template<typename T>
struct KokkosDataLayoutImpl<T,Kokkos::LayoutRight,1>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    using data_type = value_type***;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride,
                                                     const int vector_length )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     vector_length, 1,
                                     D0, 1 );
    }
};

template<typename T>
struct KokkosDataLayoutImpl<T,Kokkos::LayoutLeft,1>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    using data_type = value_type***;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride,
                                                     const int vector_length )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     D0, 1,
                                     vector_length, 1 );
    }
};

// Rank-2
template<typename T>
struct KokkosDataLayoutImpl<T,Kokkos::LayoutRight,2>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    using data_type = value_type***;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride,
                                                     const int vector_length )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     vector_length, 1,
                                     D0, 1,
                                     D1, 1 );
    }
};

template<typename T>
struct KokkosDataLayoutImpl<T,Kokkos::LayoutLeft,2>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    using data_type = value_type***;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride,
                                                     const int vector_length )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     D1, 1,
                                     D0, 1,
                                     vector_length, 1 );
    }
};

// Rank-3
template<typename T>
struct KokkosDataLayoutImpl<T,Kokkos::LayoutRight,3>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    using data_type = value_type*****;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride,
                                                     const int vector_length )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     vector_length, 1,
                                     D0, 1,
                                     D1, 1,
                                     D2, 1 );
    }
};

template<typename T>
struct KokkosDataLayoutImpl<T,Kokkos::LayoutLeft,3>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    using data_type = value_type*****;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride,
                                                     const int vector_length )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     D2, 1,
                                     D1, 1,
                                     D0, 1,
                                     vector_length, 1 );
    }
};

// Rank-4
template<typename T>
struct KokkosDataLayoutImpl<T,Kokkos::LayoutRight,4>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    static constexpr std::size_t D3 = std::extent<T,3>::value;
    using data_type = value_type******;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride,
                                                     const int vector_length )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     vector_length, 1,
                                     D0, 1,
                                     D1, 1,
                                     D2, 1,
                                     D3, 1 );
    }
};

template<typename T>
struct KokkosDataLayoutImpl<T,Kokkos::LayoutLeft,4>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    static constexpr std::size_t D3 = std::extent<T,3>::value;
    using data_type = value_type******;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride,
                                                     const int vector_length )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     D3, 1,
                                     D2, 1,
                                     D1, 1,
                                     D0, 1,
                                     vector_length, 1 );
    }
};

// Data type specialization.
template<typename T,typename DataLayout>
struct KokkosDataType
{
    using data_type =
        typename KokkosDataTypeImpl<T,DataLayout,std::rank<T>::value>::data_type;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride,
                                                     const int vector_length )
    {
        return
            KokkosDataTypeImpl<T,DataLayout,std::rank<T>::value>::createLayout(
                num_soa, stride, vector_length );
    }
};

//---------------------------------------------------------------------------//
// Kokkos view wrapper for particle fields
template<typename T,
         typename DataLayout,
         typename MemoryAccessType,
         int VectorLength,
         typename std::enable_if<
             (IsPowerOfTwo<N>::value && N > 0),int>::type = 0>
struct KokkosViewWrapper
{
    using data_type = typename KokkosDataType<T,DataLayout>::data_type;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride )
    {
        KokkosDataType<vector_type,array_layout>::createLayout(
            num_soa, stride, VectorLength );
    }
};

//---------------------------------------------------------------------------//
// Kokkos View data type selector.
template<ViewWrapper,MemoryAccessType>
struct KokkosTypeSelector
{
    using type = typename ViewWrapper::data_type;
};

// Random access memory specialization.
template<ViewWrapper>
struct KokkosTypeSelector<ViewWrapper,RandomAccessMemory>
{
    using type = typename std::add_const<typename ViewWrapper::data_type>::type;
};

//---------------------------------------------------------------------------//

} // end namespace Impl

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
  instead giving an operator() syntax for accessing the member data. Third, it
  allows for the prescription of data with a given set of access traits.
*/
//---------------------------------------------------------------------------//
template<typename DataType,
         typename DataLayout,
         typename MemorySpace,
         typename MemoryAccessType,
         int VectorLength>
class MemberSlice
{
  public:

    // Data layout.
    using data_layout = DataLayout;

    // Vector length.
    static constexpr int vector_length = VectorLength;

    // Index type.
    using index_type = Impl::Index<vector_length>;

    // Maximum supported rank.
    static constexpr int max_supported_rank = 4;

    // Kokkos view wrapper.
    using view_wrapper =
        Impl::KokkosViewWrapper<DataType,data_layout,vector_length>;

    // Kokkos view data type.
    using kokkos_data_type =
        typename KokkosTypeSelector<view_wrapper,MemoryAccessType>::type;

    // Kokkos view type.
    using kokkos_view =
        Kokkos::View<kokkos_data_type,
                     Kokkos::LayoutStride,
                     MemorySpace,
                     typename MemoryAccessType::kokkos_memory_traits>;

    // View type aliases.
    using reference_type = typename kokkos_view::reference_type;
    using value_type = typename kokkos_view::value_type;
    using pointer_type = typename kokkos_view::pointer_type;
    using memory_space = typename kokkos_view::memory_space;
    using device_type = typename kokkos_view::device_type;
    using execution_space = typename kokkos_view::execution_space;
    using data_type = typename kokkos_view::data_type;
    using const_data_type = typename kokkos_view::const_data_type;

  public:

    /*!
      \brief Constructor.
    */
    MemberSlice( const pointer_type data,
                 const int size,
                 const int stride,
                 const int num_soa )
        : _view( data, view_wrapper::createLayout(num_soa,stride) )
        , _size( size )
        , _num_soa( num_soa )
    {}

    /*!
      \brief Returns the total number particles in the field.

      \return The number of particles in the field.
    */
    KOKKOS_FUNCTION
    int size() const
    {
        return _size;
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
            ( s < _num_soa - 1 ) ? vector_length : ( _size % vector_length );
    }

    // -------------------------------
    // Member data type properties.

    /*!
      \brief Get the rank of the particle data for this field.
      \return The rank of the data for this field.
    */
    KOKKOS_INLINE_FUNCTION
    constexpr int rank() const
    { return std::rank<DataType>::value; }

    /*!
      \brief Get the extent of a given particle field dimension.

      \param D The member data dimension to get the extent for.

      \return The extent of the given member data dimension.
    */
    KOKKOS_INLINE_FUNCTION
    int extent( const std::size_t D ) const
    { return _view.extent(D+2); }

    // -------------------------------
    // Access the data value at a given struct and array index.

    // Rank 0
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<U>::value),
                            reference_type>::type
    access( const int s, const int i ) const
    {
        return _view( s, i );
    }

    // Rank 1
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((1==std::rank<U>::value) && is_layout_right),
                            reference_type>::type
    access( const int s, const int i,
            const int d0 ) const
    {
        return _view( s, i, d0 );
    }

    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((1==std::rank<U>::value) && is_layout_left),
                            reference_type>::type
    access( const int s, const int i,
            const int d0 ) const
    {
        return _view( s, d0, i);
    }

    // Rank 2
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((2==std::rank<U>::value) && is_layout_right),
                            reference_type>::type
    access( const int s, const int i,
            const int d0,
            const int d1 ) const
    {
        return _view( s, i, d0, d1);
    }

    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((2==std::rank<U>::value) && is_layout_left),
                            reference_type>::type
    access( const int s, const int i,
            const int d0,
            const int d1 ) const
    {
        return _view( s, d1, d0, i);
    }

    // Rank 3
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((3==std::rank<U>::value) && is_layout_right),
                            reference_type>::type
    access( const int s, const int i,
            const int d0,
            const int d1,
            const int d2 ) const
    {
        return _view( s, i, d0, d1, d2);
    }

    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((3==std::rank<U>::value) && is_layout_left),
                            reference_type>::type
    access( const int s, const int i,
            const int d0,
            const int d1,
            const int d2 ) const
    {
        return _view( s, d2, d1, d0, i);
    }

    // Rank 4
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((4==std::rank<U>::value) && is_layout_right),
                            reference_type>::type
    access( const int s, const int i,
            const int d0,
            const int d1,
            const int d2,
            const int d3 ) const
    {
        return _view( s, i, d0, d1, d2, d3);
    }

    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<((4==std::rank<U>::value) && is_layout_left),
                            reference_type>::type
    access( const int s, const int i,
            const int d0,
            const int d1,
            const int d2,
            const int d3 ) const
    {
        return _view( s, d3, d2, d1, d0, i);
    }

    // -------------------------------
    // Access the data value at a given particle index.

    // Rank 0
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<U>::value),reference_type>::type
    operator()( const int p ) const
    {
        return access( index_type::s(p), index_type::i(p) );
    }

    // Rank 1
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==std::rank<U>::value),reference_type>::type
    operator()( const int p,
                const int d0 ) const
    {
        return access( index_type::s(p), index_type::i(p), d0 );
    }

    // Rank 2
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==std::rank<U>::value),reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1 ) const
    {
        return access( index_type::s(p), index_type::i(p), d0, d1 );
    }

    // Rank 3
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(3==std::rank<U>::value),reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1,
                const int d2 ) const
    {
        return access( index_type::s(p), index_type::i(p), d0, d1, d2 );
    }

    // Rank 4
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(4==std::rank<U>::value),reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1,
                const int d2,
                const int d3 ) const
    {
        return access( index_type::s(p), index_type::i(p), d0, d1, d2, d3 );
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
        return _view.stride(0);
    }

    /*!
      \brief Get a raw pointer to the data for this member

      \return A raw pointer to the member data at the given index.
    */
    KOKKOS_INLINE_FUNCTION
    pointer_type data() const
    {
        return _view.data();
    }

  private:

    // The data view. This view is unmanaged an has access traits specified by
    // the template parameters of this class.
    kokkos_view _view;

    // Total number of elements.
    int _size;

    // Total number of tiles.
    int _num_soa;
};

//---------------------------------------------------------------------------//
// Static type checker.
template<typename >
struct is_member_slice : public std::false_type {};

// True only if the type is a member slice *AND* the member slice is templated
// on an AoSoA type.
template<typename ... Params>
struct is_member_slice<MemberSlice<Params...> > : public std::true_type {};

template<typename ... Params>
struct is_member_slice<const MemberSlice<Params...> > : public std::true_type {};

//---------------------------------------------------------------------------//

} // end namespace Cabana

//---------------------------------------------------------------------------//

#endif // end CABANA_MEMBERSLICE_HPP
