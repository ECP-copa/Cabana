/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Slice.hpp
  \brief Slice a single particle property from an AoSoA
*/
#ifndef CABANA_SLICE_HPP
#define CABANA_SLICE_HPP

#include <Cabana_Types.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_TypeTraits.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <string>
#include <type_traits>

//---------------------------------------------------------------------------//
namespace Kokkos
{
//---------------------------------------------------------------------------//
//! Cabana Slice layout.
template <int SOASTRIDE, int VLEN, int DIM0 = 0, int DIM1 = 0, int DIM2 = 0,
          int DIM3 = 0, int DIM4 = 0, int DIM5 = 0>
struct LayoutCabanaSlice
{
    //! Slice array layout.
    typedef LayoutCabanaSlice array_layout;
    //! Slice is extent constructible.
    enum
    {
        is_extent_constructible = true
    };

    //! Slice SoA stride.
    static constexpr int Stride = SOASTRIDE;
    //! Slice vectorlength.
    static constexpr int VectorLength = VLEN;
    //! Slice zeroth dimension size.
    static constexpr int D0 = DIM0;
    //! Slice first dimension size.
    static constexpr int D1 = DIM1;
    //! Slice second dimension size.
    static constexpr int D2 = DIM2;
    //! Slice third dimension size.
    static constexpr int D3 = DIM3;
    //! Slice fourth dimension size.
    static constexpr int D4 = DIM4;
    //! Slice fifth dimension size.
    static constexpr int D5 = DIM5;

    //! Slice dimension.
    size_t dimension[ARRAY_LAYOUT_MAX_RANK];

    //! Const copy constructor.
    LayoutCabanaSlice( LayoutCabanaSlice const& ) = default;
    //! Copy constructor.
    LayoutCabanaSlice( LayoutCabanaSlice&& ) = default;
    //! Const assignment operator.
    LayoutCabanaSlice& operator=( LayoutCabanaSlice const& ) = default;
    //! Assignment operator.
    LayoutCabanaSlice& operator=( LayoutCabanaSlice&& ) = default;

    //! Constructor.
    KOKKOS_INLINE_FUNCTION
    explicit constexpr LayoutCabanaSlice( size_t num_soa = 0,
                                          size_t vector_length = VectorLength,
                                          size_t d0 = D0, size_t d1 = D1,
                                          size_t d2 = D2, size_t d3 = D3,
                                          size_t d4 = D4, size_t d5 = D5 )
        : dimension{ num_soa, vector_length, d0, d1, d2, d3, d4, d5 }
    {
    }
};

//---------------------------------------------------------------------------//
namespace Impl
{
//! \cond Impl

//---------------------------------------------------------------------------//
// View offset of LayoutCabanaSlice.
template <class Dimension, int... LayoutDims>
struct ViewOffset<Dimension, Kokkos::LayoutCabanaSlice<LayoutDims...>, void>
{
  public:
    using is_mapping_plugin = std::true_type;
    using is_regular = std::true_type;

    typedef size_t size_type;
    typedef Dimension dimension_type;
    typedef Kokkos::LayoutCabanaSlice<LayoutDims...> array_layout;

    static constexpr int Stride = array_layout::Stride;
    static constexpr int VectorLength = array_layout::VectorLength;
    static constexpr int D0 = array_layout::D0;
    static constexpr int D1 = array_layout::D1;
    static constexpr int D2 = array_layout::D2;
    static constexpr int D3 = array_layout::D3;
    static constexpr int D4 = array_layout::D4;
    static constexpr int D5 = array_layout::D5;

    dimension_type m_dim;

    //----------------------------------------

    // rank 1
    template <typename S>
    KOKKOS_INLINE_FUNCTION constexpr size_type operator()( S const& s ) const
    {
        return Stride * s;
    }

    // rank 2
    template <typename S, typename A>
    KOKKOS_INLINE_FUNCTION constexpr size_type operator()( S const& s,
                                                           A const& a ) const
    {
        return Stride * s + a;
    }

    // rank 3
    template <typename S, typename A, typename I0>
    KOKKOS_INLINE_FUNCTION constexpr size_type
    operator()( S const& s, A const& a, I0 const& i0 ) const
    {
        return Stride * s + a + VectorLength * i0;
    }

    // rank 4
    template <typename S, typename A, typename I0, typename I1>
    KOKKOS_INLINE_FUNCTION constexpr size_type
    operator()( S const& s, A const& a, I0 const& i0, I1 const& i1 ) const
    {
        return Stride * s + a + VectorLength * ( i1 + D1 * i0 );
    }

    // rank 5
    template <typename S, typename A, typename I0, typename I1, typename I2>
    KOKKOS_INLINE_FUNCTION constexpr size_type
    operator()( S const& s, A const& a, I0 const& i0, I1 const& i1,
                I2 const& i2 ) const
    {
        return Stride * s + a + VectorLength * ( i2 + D2 * ( i1 + D1 * i0 ) );
    }

    // rank 6
    template <typename S, typename A, typename I0, typename I1, typename I2,
              typename I3>
    KOKKOS_INLINE_FUNCTION constexpr size_type
    operator()( S const& s, A const& a, I0 const& i0, I1 const& i1,
                I2 const& i2, I3 const& i3 ) const
    {
        return Stride * s + a +
               VectorLength * ( i3 + D3 * i2 + D2 * ( i1 + D1 * i0 ) );
    }

    // rank 7
    template <typename S, typename A, typename I0, typename I1, typename I2,
              typename I3, typename I4>
    KOKKOS_INLINE_FUNCTION constexpr size_type
    operator()( S const& s, A const& a, I0 const& i0, I1 const& i1,
                I2 const& i2, I3 const& i3, I4 const& i4 ) const
    {
        return Stride * s + a +
               VectorLength *
                   ( i4 + D4 * ( i3 + D3 * i2 + D2 * ( i1 + D1 * i0 ) ) );
    }

    // rank 8
    template <typename S, typename A, typename I0, typename I1, typename I2,
              typename I3, typename I4, typename I5>
    KOKKOS_INLINE_FUNCTION constexpr size_type
    operator()( S const& s, A const& a, I0 const& i0, I1 const& i1,
                I2 const& i2, I3 const& i3, I4 const& i4, I5 const& i5 ) const
    {
        return Stride * s + a +
               VectorLength *
                   ( i5 + D5 * ( i4 + D4 * ( i3 + D3 * i2 +
                                             D2 * ( i1 + D1 * i0 ) ) ) );
    }

    //----------------------------------------

    KOKKOS_INLINE_FUNCTION
    constexpr array_layout layout() const { return array_layout( m_dim.N0 ); }

    KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const
    {
        return m_dim.N0;
    }
    KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const
    {
        return m_dim.N1;
    }
    KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const
    {
        return m_dim.N2;
    }
    KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const
    {
        return m_dim.N3;
    }
    KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const
    {
        return m_dim.N4;
    }
    KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const
    {
        return m_dim.N5;
    }
    KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const
    {
        return m_dim.N6;
    }
    KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const
    {
        return m_dim.N7;
    }

    /* Cardinality of the domain index space */
    KOKKOS_INLINE_FUNCTION
    constexpr size_type size() const
    {
        return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
               m_dim.N6 * m_dim.N7;
    }

    /* Span of the range space, largest stride * dimension */
    KOKKOS_INLINE_FUNCTION
    constexpr size_type span() const { return m_dim.N0 * Stride; }

    KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const
    {
        return span() == size();
    }

    /* Strides of dimensions */
    KOKKOS_INLINE_FUNCTION constexpr size_type stride_0() const
    {
        return Stride;
    }

    KOKKOS_INLINE_FUNCTION constexpr size_type stride_1() const { return 1; }

    KOKKOS_INLINE_FUNCTION constexpr size_type stride_2() const
    {
        return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * m_dim.N3 *
               VectorLength;
    }

    KOKKOS_INLINE_FUNCTION constexpr size_type stride_3() const
    {
        return m_dim.N7 * m_dim.N6 * m_dim.N5 * m_dim.N4 * VectorLength;
    }

    KOKKOS_INLINE_FUNCTION constexpr size_type stride_4() const
    {
        return m_dim.N7 * m_dim.N6 * m_dim.N5 * VectorLength;
    }

    KOKKOS_INLINE_FUNCTION constexpr size_type stride_5() const
    {
        return m_dim.N7 * m_dim.N6 * VectorLength;
    }

    KOKKOS_INLINE_FUNCTION constexpr size_type stride_6() const
    {
        return m_dim.N7 * VectorLength;
    }

    KOKKOS_INLINE_FUNCTION constexpr size_type stride_7() const
    {
        return VectorLength;
    }

    // Stride with [ rank ] value is the total length
    template <typename iType>
    KOKKOS_INLINE_FUNCTION void stride( iType* const s ) const
    {
        if ( 0 < dimension_type::rank )
        {
            s[0] = stride_0();
        }
        if ( 1 < dimension_type::rank )
        {
            s[1] = stride_1();
        }
        if ( 2 < dimension_type::rank )
        {
            s[2] = stride_2();
        }
        if ( 3 < dimension_type::rank )
        {
            s[3] = stride_3();
        }
        if ( 4 < dimension_type::rank )
        {
            s[4] = stride_4();
        }
        if ( 5 < dimension_type::rank )
        {
            s[5] = stride_5();
        }
        if ( 6 < dimension_type::rank )
        {
            s[6] = stride_6();
        }
        if ( 7 < dimension_type::rank )
        {
            s[7] = stride_7();
        }
        s[dimension_type::rank] = span();
    }

    //----------------------------------------

    ViewOffset() = default;
    ViewOffset( const ViewOffset& ) = default;
    ViewOffset& operator=( const ViewOffset& ) = default;

    KOKKOS_INLINE_FUNCTION
    constexpr ViewOffset( std::integral_constant<unsigned, 0> const&,
                          Kokkos::LayoutCabanaSlice<LayoutDims...> const& rhs )
        : m_dim( rhs.dimension[0], rhs.dimension[1], rhs.dimension[2],
                 rhs.dimension[3], rhs.dimension[4], rhs.dimension[5],
                 rhs.dimension[6], rhs.dimension[7] )
    {
    }

    template <class DimRHS, class LayoutRHS>
    KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
        const ViewOffset<DimRHS, LayoutRHS, void>& rhs )
        : m_dim( rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3,
                 rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7 )
    {
        static_assert( int( DimRHS::rank ) == int( dimension_type::rank ),
                       "ViewOffset assignment requires equal rank" );
    }

    //----------------------------------------
    // Subview construction

    template <class DimRHS, class LayoutRHS>
    KOKKOS_INLINE_FUNCTION constexpr ViewOffset(
        const ViewOffset<DimRHS, LayoutRHS, void>&,
        const SubviewExtents<DimRHS::rank, dimension_type::rank>& sub )
        : m_dim( sub.range_extent( 0 ), sub.range_extent( 1 ),
                 sub.range_extent( 2 ), sub.range_extent( 3 ),
                 sub.range_extent( 4 ), sub.range_extent( 5 ),
                 sub.range_extent( 6 ), sub.range_extent( 7 ) )
    {
    }
};

//---------------------------------------------------------------------------//

//! \endcond
} // namespace Impl

} // end namespace Kokkos

//---------------------------------------------------------------------------//
namespace Cabana
{
namespace Impl
{
//! \cond Impl

//---------------------------------------------------------------------------//
// Given a tuple member type T of the given rank get the Kokkos view
// data layout parameters. The tuple index effectively introduces 2 new
// dimensions to the problem on top of the member dimensions - one for the
// struct index and one for the vector index.
template <typename T, std::size_t Rank, int VectorLength, int Stride>
struct KokkosDataTypeImpl;

// Rank-0
template <typename T, int VectorLength, int Stride>
struct KokkosDataTypeImpl<T, 0, VectorLength, Stride>
{
    using value_type = typename std::remove_all_extents<T>::type;
    using data_type = value_type* [VectorLength];
    using cabana_layout = Kokkos::LayoutCabanaSlice<Stride, VectorLength>;

    inline static cabana_layout createLayout( const std::size_t num_soa )
    {
        return cabana_layout( num_soa );
    }
};

// Rank-1
template <typename T, int VectorLength, int Stride>
struct KokkosDataTypeImpl<T, 1, VectorLength, Stride>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T, 0>::value;
    using data_type = value_type* [VectorLength][D0];
    using cabana_layout = Kokkos::LayoutCabanaSlice<Stride, VectorLength, D0>;

    inline static cabana_layout createLayout( const std::size_t num_soa )
    {
        return cabana_layout( num_soa );
    }
};

// Rank-2
template <typename T, int VectorLength, int Stride>
struct KokkosDataTypeImpl<T, 2, VectorLength, Stride>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T, 0>::value;
    static constexpr std::size_t D1 = std::extent<T, 1>::value;
    using data_type = value_type* [VectorLength][D0][D1];
    using cabana_layout =
        Kokkos::LayoutCabanaSlice<Stride, VectorLength, D0, D1>;

    inline static cabana_layout createLayout( const std::size_t num_soa )
    {
        return cabana_layout( num_soa );
    }
};

// Rank-3
template <typename T, int VectorLength, int Stride>
struct KokkosDataTypeImpl<T, 3, VectorLength, Stride>
{
    using value_type = typename std::remove_all_extents<T>::type;
    static constexpr std::size_t D0 = std::extent<T, 0>::value;
    static constexpr std::size_t D1 = std::extent<T, 1>::value;
    static constexpr std::size_t D2 = std::extent<T, 2>::value;
    using data_type = value_type* [VectorLength][D0][D1][D2];
    using cabana_layout =
        Kokkos::LayoutCabanaSlice<Stride, VectorLength, D0, D1, D2>;

    inline static cabana_layout createLayout( const std::size_t num_soa )
    {
        return cabana_layout( num_soa );
    }
};

// Data type specialization.
template <typename T, int VectorLength, int Stride>
struct KokkosDataType
{
    using kokkos_data_type =
        KokkosDataTypeImpl<T, std::rank<T>::value, VectorLength, Stride>;
    using data_type = typename kokkos_data_type::data_type;
    using cabana_layout = typename kokkos_data_type::cabana_layout;

    inline static cabana_layout createLayout( const std::size_t num_soa )
    {
        return kokkos_data_type::createLayout( num_soa );
    }
};

//---------------------------------------------------------------------------//
// Kokkos view wrapper for tuple members
template <typename T, int VectorLength, int Stride,
          typename std::enable_if<
              Impl::IsVectorLengthValid<VectorLength>::value, int>::type = 0>
struct KokkosViewWrapper
{
    using data_type =
        typename KokkosDataType<T, VectorLength, Stride>::data_type;

    using cabana_layout =
        typename KokkosDataType<T, VectorLength, Stride>::cabana_layout;

    inline static cabana_layout createLayout( const std::size_t num_soa )
    {
        return KokkosDataType<T, VectorLength, Stride>::createLayout( num_soa );
    }
};

//---------------------------------------------------------------------------//

//! \endcond
} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief A slice of an array-of-structs-of-arrays with data access to a single
  multidimensional member.
*/
//---------------------------------------------------------------------------//
template <typename DataType, typename DeviceType, typename MemoryAccessType,
          int VectorLength, int Stride>
class Slice
{
  public:
    // Ensure the vector length is valid.
    static_assert( Impl::IsVectorLengthValid<VectorLength>::value,
                   "Vector length must be valid" );

    //! Slice type.
    using slice_type =
        Slice<DataType, DeviceType, MemoryAccessType, VectorLength, Stride>;

    //! Device type
    using device_type = DeviceType;

    //! Memory space.
    using memory_space = typename device_type::memory_space;

    //! Execution space.
    using execution_space = typename device_type::execution_space;

    static_assert( is_memory_access_tag<MemoryAccessType>::value,
                   "Slice memory access type must be a Cabana access type" );
    //! Memory access type.
    using memory_access_type = MemoryAccessType;

    //! Vector length.
    static constexpr int vector_length = VectorLength;

    //! SoA stride.
    static constexpr int soa_stride = Stride;

    //! Size type.
    using size_type = typename memory_space::size_type;

    //! Index type.
    using index_type = Impl::Index<vector_length>;

    //! Maximum supported rank.
    static constexpr std::size_t max_supported_rank = 3;

    //! Maximum label length.
    static constexpr std::size_t max_label_length = 128;

    //! Kokkos view wrapper.
    using view_wrapper =
        Impl::KokkosViewWrapper<DataType, vector_length, soa_stride>;

    //! Kokkos view type.
    using kokkos_view =
        Kokkos::View<typename view_wrapper::data_type,
                     typename view_wrapper::cabana_layout, DeviceType,
                     typename MemoryAccessType::kokkos_memory_traits>;

    //! View reference type alias.
    using reference_type = typename kokkos_view::reference_type;
    //! View value type alias.
    using value_type = typename kokkos_view::value_type;
    //! View pointer type alias.
    using pointer_type = typename kokkos_view::pointer_type;
    //! View array layout type alias.
    using view_layout = typename kokkos_view::array_layout;

    //! Default memory access slice type.
    using default_access_slice =
        Slice<DataType, DeviceType, DefaultAccessMemory, VectorLength, Stride>;
    //! Atomic memory access slice type.
    using atomic_access_slice =
        Slice<DataType, DeviceType, AtomicAccessMemory, VectorLength, Stride>;
    //! Random memory access slice type.
    using random_access_slice =
        Slice<DataType, DeviceType, RandomAccessMemory, VectorLength, Stride>;

    // Declare slices of different memory access types to be friends.
    friend class Slice<DataType, DeviceType, DefaultAccessMemory, VectorLength,
                       Stride>;
    friend class Slice<DataType, DeviceType, AtomicAccessMemory, VectorLength,
                       Stride>;
    friend class Slice<DataType, DeviceType, RandomAccessMemory, VectorLength,
                       Stride>;

    // Equivalent Kokkos view rank. This rank assumes that the struct and
    // array dimension are merged. For the true rank of the raw AoSoA data use
    // the rank() function below which will account for the extra dimension
    // from separate struct and array indices. This enumeration is for
    // compatibility with Kokkos views.
    enum
    {
        rank = std::rank<DataType>::value + 1
    };

  public:
    /*!
      \brief Default constructor.
    */
    Slice()
        : _size( 0 )
    {
    }

    /*!
      \brief Constructor.
      \param data Pointer to the first member of the slice.
      \param size The number of tuples in the slice.
      \param num_soa The number of structs in the slice.
      \param label An optional label for the slice.
    */
    Slice( const pointer_type data, const size_type size,
           const size_type num_soa, const std::string& label = "" )
        : _view( data, view_wrapper::createLayout( num_soa ) )
        , _size( size )
    {
        std::strcpy( _label, label.c_str() );
    }

    /*!
      \brief Shallow copy constructor for different memory spaces for
      assigning new memory access traits to the view.
      \tparam MAT Memory access type.
      \param rhs The slice to shallow copy with a potentially different memory
      space.
    */
    template <class MAT>
    Slice( const Slice<DataType, DeviceType, MAT, VectorLength, Stride>& rhs )
        : _view( rhs._view )
        , _size( rhs._size )
    {
        std::strcpy( _label, rhs._label );
    }

    /*!
      \brief Assignement operator for different memory spaces for assigning
      new memory access traits to the view.
      \tparam MAT Memory access type
      \param rhs The slice to shallow copy with a potentially different memory
      space.
      \return A reference to a new slice with a potentially different memory
      space.
     */
    template <class MAT>
    Slice& operator=(
        const Slice<DataType, DeviceType, MAT, VectorLength, Stride>& rhs )
    {
        _view = rhs._view;
        _size = rhs._size;
        std::strcpy( _label, rhs._label );
        return *this;
    }

    /*!
      \brief Returns the data structure label.

      \return A string identifying the data structure.
    */
    std::string label() const { return std::string( _label ); }

    /*!
      \brief Returns the total number tuples in the slice.
      \return The number of tuples in the slice.
    */
    KOKKOS_INLINE_FUNCTION
    size_type size() const { return _size; }

    /*!
      \brief Get the number of structs-of-arrays in the container.
      \return The number of structs-of-arrays in the container.
    */
    KOKKOS_INLINE_FUNCTION
    size_type numSoA() const { return _view.extent( 0 ); }

    /*!
      \brief Get the size of the data array at a given struct member index.
      \param s The struct index to get the array size for.
      \return The size of the array at the given struct index.
    */
    KOKKOS_INLINE_FUNCTION
    size_type arraySize( const size_type s ) const
    {
        return ( static_cast<size_type>( s ) < _view.extent( 0 ) - 1 )
                   ? vector_length
                   : ( _size % vector_length );
    }

    // ------------
    // 2-D accessor

    //! 2d access for  Rank 0
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 0 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        access( const size_type s, const size_type a ) const
    {
        return _view( s, a );
    }

    //! 2d access for  Rank 1
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 1 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        access( const size_type s, const size_type a, const size_type d0 ) const
    {
        return _view( s, a, d0 );
    }

    //! 2d access for  Rank 2
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 2 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        access( const size_type s, const size_type a, const size_type d0,
                const size_type d1 ) const
    {
        return _view( s, a, d0, d1 );
    }

    //! 2d access for  Rank 3
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 3 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        access( const size_type s, const size_type a, const size_type d0,
                const size_type d1, const size_type d2 ) const
    {
        return _view( s, a, d0, d1, d2 );
    }

    // ------------
    // 1-D accessor

    //! 1d access for  Rank 0
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 0 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        operator()( const size_type i ) const
    {
        return access( index_type::s( i ), index_type::a( i ) );
    }

    //! 1d access for  Rank 1
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 1 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        operator()( const size_type i, const size_type d0 ) const
    {
        return access( index_type::s( i ), index_type::a( i ), d0 );
    }

    //! 1d access for  Rank 2
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 2 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        operator()( const size_type i, const size_type d0,
                    const size_type d1 ) const
    {
        return access( index_type::s( i ), index_type::a( i ), d0, d1 );
    }

    //! 1d access for  Rank 3
    template <typename U = DataType>
    KOKKOS_FORCEINLINE_FUNCTION
        typename std::enable_if<( 3 == std::rank<U>::value &&
                                  std::is_same<U, DataType>::value ),
                                reference_type>::type
        operator()( const size_type i, const size_type d0, const size_type d1,
                    const size_type d2 ) const
    {
        return access( index_type::s( i ), index_type::a( i ), d0, d1, d2 );
    }

    // -------------------------------
    // Raw data access.

    /*!
      \brief Get a raw pointer to the data for this member
      \return A raw pointer to the data for this slice.
    */
    KOKKOS_INLINE_FUNCTION
    pointer_type data() const { return _view.data(); }

    /*!
      \brief Get the rank of the raw data for this slice. This includes
      the struct dimension, array dimension, and all tuple slice
      dimensions.
      \return The rank of the data for this slice.
    */
    KOKKOS_INLINE_FUNCTION
    constexpr size_type viewRank() const { return _view.rank; }

    /*!
      \brief Get the extent of a given raw slice data dimension. This includes
      the struct dimension, array dimension, and all tuple slice
      dimensions.
      \param d The member data dimension to get the extent for.
      \return The extent of the given member data dimension.
    */
    KOKKOS_INLINE_FUNCTION
    size_type extent( const size_type d ) const { return _view.extent( d ); }

    /*!
      \brief Get the stride of a given raw slice dimension. This includes the
      struct dimension, array dimension, and all tuple slice dimensions.
      \param d The member data dimension to get the stride for.
      \return The stride of the given member data dimension.
    */
    KOKKOS_INLINE_FUNCTION
    size_type stride( const size_type d ) const { return _view.stride( d ); }

    /*!
      \brief Get the underlying Kokkos View managing the slice data.
    */
    KOKKOS_INLINE_FUNCTION
    kokkos_view view() const { return _view; }

  private:
    // The data view. This view is unmanaged and has access traits specified
    // by the template parameters of this class.
    kokkos_view _view;

    // Number of tuples in the slice.
    size_type _size;

    // Slice label.
    char _label[max_label_length];
};

//---------------------------------------------------------------------------//
//! \cond Impl
template <typename>
struct is_slice_impl : public std::false_type
{
};

// True only if the type is a member slice *AND* the member slice is templated
// on an AoSoA type.
template <typename DataType, typename DeviceType, typename MemoryAccessType,
          int VectorLength, int Stride>
struct is_slice_impl<
    Slice<DataType, DeviceType, MemoryAccessType, VectorLength, Stride>>
    : public std::true_type
{
};
//! \endcond

//! Slice static type checker.
template <class T>
struct is_slice : public is_slice_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Copy to View.
//---------------------------------------------------------------------------//

//! Copy from slice to View. Rank-0
template <class ExecutionSpace, class ViewType, class SliceType>
void copySliceToView(
    ExecutionSpace exec_space, ViewType& view, const SliceType& slice,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        2 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copySliceToView::Rank0",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) { view( i - begin ) = slice( i ); } );
}

//! Copy from slice to View. Rank-1
template <class ExecutionSpace, class ViewType, class SliceType>
void copySliceToView(
    ExecutionSpace exec_space, ViewType& view, const SliceType& slice,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        3 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copySliceToView::FieldRank1",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < slice.extent( 2 ); ++d0 )
                view( i - begin, d0 ) = slice( i, d0 );
        } );
}

//! Copy from slice to View. Rank-2
template <class ExecutionSpace, class ViewType, class SliceType>
void copySliceToView(
    ExecutionSpace exec_space, ViewType& view, const SliceType& slice,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        4 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copySliceToView::writeFieldRank2",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < slice.extent( 2 ); ++d0 )
                for ( std::size_t d1 = 0; d1 < slice.extent( 3 ); ++d1 )
                    view( i - begin, d0, d1 ) = slice( i, d0, d1 );
        } );
}

//! Copy from slice to View with default execution space.
template <class ViewType, class SliceType>
void copySliceToView( ViewType& view, const SliceType& slice,
                      const std::size_t begin, const std::size_t end )
{
    using exec_space = typename SliceType::execution_space;
    copySliceToView( exec_space{}, view, slice, begin, end );
}

//---------------------------------------------------------------------------//
// Copy from View.
//---------------------------------------------------------------------------//

//! Copy from slice to View. Rank-0
template <class ExecutionSpace, class SliceType, class ViewType>
void copyViewToSlice(
    ExecutionSpace exec_space, SliceType& slice, const ViewType& view,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        2 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copyViewToSlice::Rank0",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) { slice( i - begin ) = view( i ); } );
}

//! Copy from slice to View. Rank-1
template <class ExecutionSpace, class SliceType, class ViewType>
void copyViewToSlice(
    ExecutionSpace exec_space, SliceType& slice, const ViewType& view,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        3 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copySliceToView::FieldRank1",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < slice.extent( 2 ); ++d0 )
                slice( i - begin, d0 ) = view( i, d0 );
        } );
}

//! Copy from slice to View. Rank-2
template <class ExecutionSpace, class SliceType, class ViewType>
void copyViewToSlice(
    ExecutionSpace exec_space, SliceType& slice, const ViewType& view,
    const std::size_t begin, const std::size_t end,
    typename std::enable_if<
        4 == SliceType::kokkos_view::traits::dimension::rank, int*>::type = 0 )
{
    Kokkos::parallel_for(
        "Cabana::copySliceToView::writeFieldRank2",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, begin, end ),
        KOKKOS_LAMBDA( const int i ) {
            for ( std::size_t d0 = 0; d0 < slice.extent( 2 ); ++d0 )
                for ( std::size_t d1 = 0; d1 < slice.extent( 3 ); ++d1 )
                    slice( i - begin, d0, d1 ) = view( i, d0, d1 );
        } );
}

//! Copy from slice to View with default execution space.
template <class ViewType, class SliceType>
void copyViewToSlice( ViewType& view, const SliceType& slice,
                      const std::size_t begin, const std::size_t end )
{
    using exec_space = typename SliceType::execution_space;
    copyViewToSlice( exec_space{}, view, slice, begin, end );
}

//! Check slice size (differs from Kokkos View).
template <class SliceType>
void checkSize(
    [[maybe_unused]] SliceType slice, [[maybe_unused]] const std::size_t size,
    typename std::enable_if<is_slice<SliceType>::value, int>::type* = 0 )
{
    // Allow unused for release builds.
    assert( slice.size() == size );
}

//! Check View size (differs from Slice).
template <class ViewType>
void checkSize(
    [[maybe_unused]] ViewType view, [[maybe_unused]] const std::size_t size,
    typename std::enable_if<Kokkos::is_view<ViewType>::value, int>::type* = 0 )
{
    // Allow unused for release builds.
    assert( view.extent( 0 ) == size );
}

//---------------------------------------------------------------------------//

} // end namespace Cabana

//---------------------------------------------------------------------------//

#endif // end CABANA_SLICE_HPP
