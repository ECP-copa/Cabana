#ifndef CABANA_SLICE_HPP
#define CABANA_SLICE_HPP

#include <Cabana_Types.hpp>
#include <Cabana_Types.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <type_traits>

namespace Cabana
{
namespace Impl
{
//---------------------------------------------------------------------------//
// Given a tuple member type T of the given rank get the Kokkos view
// data layout parameters. The tuple index effectively introduces 2 new
// dimensions to the problem on top of the member dimensions - one for the
// struct index and one for the vector index.
template<typename T, std::size_t Rank, int VectorLength>
struct KokkosDataTypeImpl;

// Rank-0
template<typename T, int VectorLength>
struct KokkosDataTypeImpl<T,0,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    using data_type = value_type*[VectorLength];

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     VectorLength, 1 );
    }
};

// Rank-1
template<typename T, int VectorLength>
struct KokkosDataTypeImpl<T,1,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    using data_type = value_type*[VectorLength][D0];

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     VectorLength, 1,
                                     D0, VectorLength );
    }
};

// Rank-2
template<typename T, int VectorLength>
struct KokkosDataTypeImpl<T,2,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    using data_type = value_type*[VectorLength][D0][D1];

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     VectorLength, 1,
                                     D0, VectorLength,
                                     D1, D0*VectorLength );
    }
};

// Rank-3
template<typename T, int VectorLength>
struct KokkosDataTypeImpl<T,3,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    using data_type = value_type*[VectorLength][D0][D1][D2];

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     VectorLength, 1,
                                     D0, VectorLength,
                                     D1, D0*VectorLength,
                                     D2, D1*D0*VectorLength );
    }
};

// Rank-4
template<typename T, int VectorLength>
struct KokkosDataTypeImpl<T,4,VectorLength>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T,0>::value;
    static constexpr std::size_t D1 = std::extent<T,1>::value;
    static constexpr std::size_t D2 = std::extent<T,2>::value;
    static constexpr std::size_t D3 = std::extent<T,3>::value;
    using data_type = value_type*[VectorLength][D0][D1][D2][D3];

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride )
    {
        return Kokkos::LayoutStride( num_soa, stride,
                                     VectorLength, 1,
                                     D0, VectorLength,
                                     D1, D0*VectorLength,
                                     D2, D1*D0*VectorLength,
                                     D3, D2*D1*D0*VectorLength );
    }
};

// Data type specialization.
template<typename T, int VectorLength>
struct KokkosDataType
{
    using kokkos_data_type =
        KokkosDataTypeImpl<T,std::rank<T>::value,VectorLength>;
    using data_type = typename kokkos_data_type::data_type;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride )
    {
        return kokkos_data_type::createLayout( num_soa, stride );
    }
};

//---------------------------------------------------------------------------//
// Kokkos view wrapper for tuple members
template<typename T,
         int VectorLength,
         typename std::enable_if<
             Impl::IsVectorLengthValid<VectorLength>::value,int>::type = 0>
struct KokkosViewWrapper
{
    using data_type =
        typename KokkosDataType<T,VectorLength>::data_type;

    inline static Kokkos::LayoutStride createLayout( const int num_soa,
                                                     const int stride )
    {
        return KokkosDataType<T,VectorLength>::createLayout( num_soa, stride );
    }
};

//---------------------------------------------------------------------------//

} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \class Slice

  \brief A slice of an array-of-structs-of-arrays with data access to a single
  member.

  A slice provides a simple wrapper around a single data member of an
  AoSoA. This does a few convenient things. First, it decouples access of the
  member from the AoSoA meaning that functionality can be implemented using
  multiple slices from potentially multiple AoSoA containers. Second, it
  eliminates the member index template parameter from the AoSoA get function,
  instead giving an operator() syntax for accessing the member data. Third, it
  allows for the prescription of a given set of access traits for the data.
*/
//---------------------------------------------------------------------------//
template<typename DataType,
         typename MemorySpace,
         typename MemoryAccessType,
         int VectorLength>
class Slice
{
  public:

    // Slice type.
    using slice_type =
        Slice<DataType,MemorySpace,MemoryAccessType,VectorLength>;

    // Vector length.
    static constexpr int vector_length = VectorLength;

    // Index type.
    using index_type = Impl::Index<vector_length>;

    // Maximum supported rank.
    static constexpr int max_supported_rank = 4;

    // Kokkos view wrapper.
    using view_wrapper = Impl::KokkosViewWrapper<DataType,vector_length>;

    // Kokkos view type.
    using kokkos_view =
        Kokkos::View<typename view_wrapper::data_type,
                     Kokkos::LayoutStride,
                     typename MemorySpace::kokkos_memory_space,
                     typename MemoryAccessType::kokkos_memory_traits>;

    // View type aliases.
    using reference_type = typename kokkos_view::reference_type;
    using value_type = typename kokkos_view::value_type;
    using pointer_type = typename kokkos_view::pointer_type;
    using kokkos_memory_space = typename kokkos_view::memory_space;
    using kokkos_execution_space = typename kokkos_view::execution_space;
    using kokkos_device_type = typename kokkos_view::device_type;

    // Compatible memory access slice types.
    using default_access_slice =
        Slice<DataType,MemorySpace,DefaultAccessMemory,VectorLength>;
    using atomic_access_slice =
        Slice<DataType,MemorySpace,AtomicAccessMemory,VectorLength>;
    using random_access_slice =
        Slice<DataType,MemorySpace,RandomAccessMemory,VectorLength>;

    // Declare slices of different memory access types to be friends.
    friend class Slice<DataType,MemorySpace,DefaultAccessMemory,VectorLength>;
    friend class Slice<DataType,MemorySpace,AtomicAccessMemory,VectorLength>;
    friend class Slice<DataType,MemorySpace,RandomAccessMemory,VectorLength>;

  public:

    /*!
      \brief Default constructor.
    */
    Slice()
        : _size( 0 )
    {}

    /*!
      \brief Constructor.
      \param data Pointer to the first member of the slice.
      \param The number of tuples in the slice.
      \param The number of elements in the slice's value type between starting
      elements of a struct.
      \param The number of structs in the slice.
    */
    Slice( const pointer_type data,
           const int size,
           const int soa_stride,
           const int num_soa )
        : _view( data, view_wrapper::createLayout(num_soa,soa_stride) )
        , _size( size )
    {}

    /*!
      \brief Shallow copy constructor for different memory spaces for
      assigning new memory access traits to the view.
      \param rhs The slice to shallow copy with a potentially different memory
      space.
     */
    template<class MAT>
    Slice( const Slice<DataType,MemorySpace,MAT,VectorLength>& rhs )
        : _view( rhs._view )
        , _size( rhs._size )
    {}

    /*!
      \brief Assignement operator for different memory spaces for assigning
      new memory access traits to the view.
      \param rhs The slice to shallow copy with a potentially different memory
      space.
     */
    template<class MAT>
    Slice& operator=( const Slice<DataType,MemorySpace,MAT,VectorLength>& rhs )
    {
        _view = rhs._view;
        _size = rhs._size;
        return *this;
    }

    /*!
      \brief Returns the total number tuples in the slice.
      \return The number of tuples in the slice.
    */
    KOKKOS_INLINE_FUNCTION
    int size() const
    { return _size; }

    /*!
      \brief Get the number of structs-of-arrays in the container.
      \return The number of structs-of-arrays in the container.
    */
    KOKKOS_INLINE_FUNCTION
    int numSoA() const { return _view.extent(0); }

    /*!
      \brief Get the size of the data array at a given struct member index.
      \param s The struct index to get the array size for.
      \return The size of the array at the given struct index.
    */
    KOKKOS_INLINE_FUNCTION
    int arraySize( const int s ) const
    {
        return ( s < (int) _view.extent(0) - 1 )
            ? vector_length : ( _size % vector_length );
    }

    // ------------
    // 2-D accessor

    // Rank 0
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<U>::value),reference_type>::type
    access( const int s, const int i ) const
    { return _view( s, i ); }

    // Rank 1
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==std::rank<U>::value),reference_type>::type
    access( const int s, const int i,
            const int d0 ) const
    { return _view( s, i, d0 ); }

    // Rank 2
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==std::rank<U>::value),reference_type>::type
    access( const int s, const int i,
            const int d0,
            const int d1 ) const
    { return _view( s, i, d0, d1); }

    // Rank 3
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(3==std::rank<U>::value),reference_type>::type
    access( const int s, const int i,
            const int d0,
            const int d1,
            const int d2 ) const
    { return _view( s, i, d0, d1, d2); }

    // Rank 4
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(4==std::rank<U>::value),reference_type>::type
    access( const int s, const int i,
            const int d0,
            const int d1,
            const int d2,
            const int d3 ) const
    { return _view( s, i, d0, d1, d2, d3); }

    // ------------
    // 1-D accessor

    // Rank 0
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==std::rank<U>::value),reference_type>::type
    operator()( const int p ) const
    { return access( index_type::s(p), index_type::i(p) ); }

    // Rank 1
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==std::rank<U>::value),reference_type>::type
    operator()( const int p,
                const int d0 ) const
    { return access( index_type::s(p), index_type::i(p), d0 ); }

    // Rank 2
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==std::rank<U>::value),reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1 ) const
    { return access( index_type::s(p), index_type::i(p), d0, d1 ); }

    // Rank 3
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(3==std::rank<U>::value),reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1,
                const int d2 ) const
    { return access( index_type::s(p), index_type::i(p), d0, d1, d2 ); }

    // Rank 4
    template<typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(4==std::rank<U>::value),reference_type>::type
    operator()( const int p,
                const int d0,
                const int d1,
                const int d2,
                const int d3 ) const
    { return access( index_type::s(p), index_type::i(p), d0, d1, d2, d3 ); }

    // -------------------------------
    // Raw data access.

    /*!
      \brief Get a raw pointer to the data for this member
      \return A raw pointer to the data for this slice.
    */
    KOKKOS_INLINE_FUNCTION
    pointer_type data() const
    { return _view.data(); }

    /*!
      \brief Get the rank of the raw data for this slice. This includes
      the struct dimension, array dimension, and all tuple slice
      dimensions.
      \return The rank of the data for this slice.
    */
    KOKKOS_INLINE_FUNCTION
    constexpr int rank() const
    { return _view.Rank; }

    /*!
      \brief Get the extent of a given raw slice data dimension. This includes
      the struct dimension, array dimension, and all tuple slice
      dimensions.
      \param d The member data dimension to get the extent for.
      \return The extent of the given member data dimension.
    */
    KOKKOS_INLINE_FUNCTION
    int extent( const std::size_t d ) const
    { return _view.extent(d); }

    /*!
      \brief Get the stride of a given raw slice dimension. This includes the
      struct dimension, array dimension, and all tuple slice dimensions.
      \param d The member data dimension to get the stride for.
      \return The stride of the given member data dimension.
    */
    KOKKOS_INLINE_FUNCTION
    int stride( const std::size_t d ) const
    { return _view.stride(d); }

  private:

    // The data view. This view is unmanaged and has access traits specified
    // by the template parameters of this class.
    kokkos_view _view;

    // Number of tuples in the slice.
    int _size;
};

//---------------------------------------------------------------------------//
// Static type checker.
template<typename >
struct is_slice : public std::false_type {};

// True only if the type is a member slice *AND* the member slice is templated
// on an AoSoA type.
template<typename DataType,
         typename MemorySpace,
         typename MemoryAccessType,
         int VectorLength>
struct is_slice<Slice<DataType,
                      MemorySpace,
                      MemoryAccessType,
                      VectorLength> >
    : public std::true_type {};

template<typename DataType,
         typename MemorySpace,
         typename MemoryAccessType,
         int VectorLength>
struct is_slice<const Slice<DataType,
                            MemorySpace,
                            MemoryAccessType,
                            VectorLength> >
    : public std::true_type {};

//---------------------------------------------------------------------------//

} // end namespace Cabana

//---------------------------------------------------------------------------//

#endif // end CABANA_SLICE_HPP
