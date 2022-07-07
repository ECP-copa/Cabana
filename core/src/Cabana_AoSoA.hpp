/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_AoSoA.hpp
  \brief Array-of-Struct-of-Arrays particle data structure
*/
#ifndef CABANA_AOSOA_HPP
#define CABANA_AOSOA_HPP

#include <Cabana_MemberTypes.hpp>
#include <Cabana_Slice.hpp>
#include <Cabana_SoA.hpp>
#include <Cabana_Tuple.hpp>
#include <Cabana_Types.hpp>
#include <impl/Cabana_Index.hpp>
#include <impl/Cabana_PerformanceTraits.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace Cabana
{
//---------------------------------------------------------------------------//
// AoSoA forward declaration.
template <class DataTypes, class DeviceType, int VectorLength,
          class MemoryTraits>
class AoSoA;

//---------------------------------------------------------------------------//
//! \cond Impl
template <class>
struct is_aosoa_impl : public std::false_type
{
};

template <class DataTypes, class DeviceType, int VectorLength,
          class MemoryTraits>
struct is_aosoa_impl<AoSoA<DataTypes, DeviceType, VectorLength, MemoryTraits>>
    : public std::true_type
{
};
//! \endcond

//! AoSoA static type checker.
template <class T>
struct is_aosoa : public is_aosoa_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
/*!
  \brief Create a slice from an AoSoA.

  \tparam M Slice index.
  \tparam AoSoA_t AoSoA type.

  \param aosoa AoSoA to slice from.
  \param slice_label Optional slice label.
*/
template <std::size_t M, class AoSoA_t>
typename AoSoA_t::template member_slice_type<M>
slice( const AoSoA_t& aosoa, const std::string& slice_label = "" )
{
    static_assert(
        0 == sizeof( typename AoSoA_t::soa_type ) %
                 sizeof( typename AoSoA_t::template member_value_type<M> ),
        "Slice stride cannot be calculated for misaligned memory!" );

    return typename AoSoA_t::template member_slice_type<M>(
        Impl::soaMemberPtr<M>( aosoa.data() ), aosoa.size(), aosoa.numSoA(),
        slice_label );
}

//---------------------------------------------------------------------------//
/*!
  \brief Array-of-Struct-of-Arrays

  Represents tuples and their data via an array-of-structs-of-arrays.

  \tparam DataType (required) Specifically this must be an instance of
  \c MemberTypes with the data layout of the structs. For example:
  \code
  using DataType = MemberTypes<double[3][3],double[3],int>;
  \endcode
  would define an AoSoA where each tuple had a 3x3 matrix of doubles, a
  3-vector of doubles, and an integer. The AoSoA is then templated on this
  sequence of types. In general, put larger datatypes first in the
  MemberType parameter pack (i.e. matrices and vectors) and group members
  of the same type together to achieve the smallest possible memory footprint
  based on compiler-generated padding.

  \tparam DeviceType (required) The device type.

  \tparam VectorLength (optional) The vector length within the structs of
  the AoSoA. If not specified, this defaults to the preferred layout for the
  <tt>DeviceType</tt>.

  \tparam MemoryTraits (optional) Memory traits for the AoSoA data. Can be
  used to indicate managed memory, unmanaged memory, etc.
 */
template <class DataTypes, class DeviceType,
          int VectorLength = Impl::PerformanceTraits<
              typename DeviceType::execution_space>::vector_length,
          class MemoryTraits = Kokkos::MemoryManaged>
class AoSoA
{
  public:
    //! AoSoA type.
    using aosoa_type = AoSoA<DataTypes, DeviceType, VectorLength, MemoryTraits>;

    //! Host mirror type.
    using host_mirror_type = AoSoA<DataTypes, Kokkos::HostSpace, VectorLength>;

    static_assert( is_member_types<DataTypes>::value,
                   "AoSoA data types must be member types" );
    static_assert( CheckMemberTypes<DataTypes>::value,
                   "AoSoA data type failure" );
    //! Member data types.
    using member_types = DataTypes;

    //! Device type.
    using device_type = DeviceType;

    //! Memory space.
    using memory_space = typename device_type::memory_space;

    //! Execution space.
    using execution_space = typename device_type::execution_space;

    static_assert( Impl::IsVectorLengthValid<VectorLength>::value,
                   "Vector length must be valid" );
    //! Vector length (size of the arrays held by the structs).
    static constexpr int vector_length = VectorLength;

    //! Memory traits type.
    using memory_traits = MemoryTraits;

    //! Size type.
    using size_type = typename memory_space::size_type;

    //! SoA type.
    using soa_type = SoA<member_types, vector_length>;

    //! Managed data view.
    using soa_view = Kokkos::View<soa_type*, device_type, memory_traits>;

    //! Number of member types.
    static constexpr std::size_t number_of_members = member_types::size;

    //! The maximum rank supported for member types.
    static constexpr std::size_t max_supported_rank = 3;

    //! Index type.
    using index_type = Impl::Index<vector_length>;

    //! Tuple type.
    using tuple_type = Tuple<member_types>;

    //! Member data type at a given index M. Note this is the user-defined
    //! member data type - not the potentially transformed type actually stored
    //! by the structs (SoAs) to achieve a given layout.
    template <std::size_t M>
    using member_data_type = typename MemberTypeAtIndex<M, member_types>::type;

    //! Struct member array element value type at a given index M.
    template <std::size_t M>
    using member_value_type =
        typename std::remove_all_extents<member_data_type<M>>::type;

    //! Struct member array element pointer type at a given index M.
    template <std::size_t M>
    using member_pointer_type =
        typename std::add_pointer<member_value_type<M>>::type;

    //! Member slice type at a given member index M.
    template <std::size_t M>
    using member_slice_type =
        Slice<member_data_type<M>, device_type, DefaultAccessMemory,
              vector_length,
              sizeof( soa_type ) / sizeof( member_value_type<M> )>;

  public:
    /*!
      \brief Default constructor.

      \param label An optional label for the data structure.

      The container size is zero and no memory is allocated.
    */
    AoSoA( const std::string& label = "" )
        : _size( 0 )
        , _capacity( 0 )
        , _num_soa( 0 )
        , _data( Kokkos::ViewAllocateWithoutInitializing( label ), 0 )
    {
        static_assert(
            !memory_traits::is_unmanaged,
            "Construction by allocation cannot use unmanaged memory" );
    }

    /*!
      \brief Allocate a container with n tuples.

      \param label A label for the data structure.

      \param n The number of tuples in the container.
    */
    AoSoA( const std::string label, const size_type n )
        : _size( n )
        , _capacity( 0 )
        , _num_soa( 0 )
        , _data( Kokkos::ViewAllocateWithoutInitializing( label ), 0 )
    {
        static_assert(
            !memory_traits::is_unmanaged,
            "Construction by allocation cannot use unmanaged memory" );
        resize( _size );
    }

    /*!
      \brief Create an unmanaged AoSoA with user-provided memory.

      \param ptr Pointer to user-allocated AoSoA data.

      \param num_soa The number of SoAs the user has allocated.

      \param n The number of tuples in the container.
    */
    AoSoA( soa_type* ptr, const size_type num_soa, const size_type n )
        : _size( n )
        , _capacity( num_soa * vector_length )
        , _num_soa( num_soa )
        , _data( ptr, num_soa )
    {
        static_assert( memory_traits::is_unmanaged,
                       "Pointer construction requires unmanaged memory" );
    }

    /*!
      \brief Returns the data structure label.

      \return A string identifying the data structure.

      This label will be assigned to the underlying Kokkos view managing the
      data of this class and can be used for debugging and profiling purposes.
    */
    std::string label() const { return _data.label(); }

    /*!
      \brief Returns the number of tuples in the container.

      \return The number of tuples in the container.

      This is the number of actual objects held in the container, which is not
      necessarily equal to its storage capacity.
    */
    KOKKOS_FUNCTION
    size_type size() const { return _size; }

    /*!
      \brief Returns if the container is empty or not.

      \return True if the number of tuples in the container is zero.

      This is the number of actual objects held in the container, which is not
      necessarily equal to its storage capacity.
    */
    KOKKOS_FUNCTION
    bool empty() const { return ( size() == 0 ); }

    /*!
      \brief Returns the size of the storage space currently allocated for the
      container, expressed in terms of tuples.

      \return The capacity of the container.

      This capacity is not necessarily equal to the container size. It can be
      equal or greater, with the extra space allowing to accommodate for
      growth without the need to reallocate on each insertion.

      Notice that this capacity does not suppose a limit on the size of the
      container. When this capacity is exhausted and more is needed, it is
      automatically expanded by the container (reallocating the storage
      space).

      The capacity of a container can be explicitly altered by calling member
      reserve.
    */
    KOKKOS_FUNCTION
    size_type capacity() const { return _capacity; }

    /*!
      \brief Resizes the container so that it contains n tuples.

      If n is smaller than the current container size, the content is reduced
      to its first n tuples.

      If n is greater than the current container size, the content is expanded
      by inserting at the end as many tuples as needed to reach a size of n.

      If n is also greater than the current container capacity, an automatic
      reallocation of the allocated storage space takes place.

      Notice that this function changes the actual content of the container by
      inserting or erasing tuples from it. If reallocation occurs, all slices
      and all references to the elements are invalidated. If no reallocation
      takes place, no slices or references are invalidated.
    */
    void resize( const size_type n )
    {
        static_assert( !memory_traits::is_unmanaged,
                       "Cannot resize unmanaged memory" );

        // Reserve memory if needed.
        reserve( n );

        // Update the sizes of the data. This is potentially different than
        // the amount of allocated data.
        _size = n;
        _num_soa = std::ceil( static_cast<double>( n ) / vector_length );
    }

    /*!
      \brief Requests that the container capacity be at least enough to contain
      n tuples.

      If n is greater than the current container capacity, the function causes
      the container to reallocate its storage increasing its capacity to n (or
      greater).

      In all other cases, the function call does not cause a reallocation and
      the container capacity is not affected.

      If reallocation occurs, all slices and all references to the elements
      are invalidated. If no reallocation takes place, no slices or references
      are invalidated.

      This function has no effect on the container size and cannot alter its
      tuples.
    */
    void reserve( const size_type n )
    {
        static_assert( !memory_traits::is_unmanaged,
                       "Cannot reserve unmanaged memory" );

        // If we aren't asking for more memory then we have nothing to do.
        if ( n <= _capacity )
            return;

        // Figure out the new capacity.
        size_type num_soa_alloc = std::floor( n / vector_length );
        if ( 0 < n % vector_length )
            ++num_soa_alloc;

        // If we aren't asking for any more SoA objects then we still have
        // nothing to do.
        if ( num_soa_alloc <= _num_soa )
            return;

        // Assign the new capacity.
        _capacity = num_soa_alloc * vector_length;

        // We need more SoA objects so allocate a new view and copy the
        // existing data.
        soa_view resized_data(
            Kokkos::ViewAllocateWithoutInitializing( _data.label() ),
            num_soa_alloc );
        if ( _num_soa > 0 )
            Kokkos::deep_copy(
                Kokkos::subview(
                    resized_data,
                    Kokkos::pair<size_type, size_type>( 0, _num_soa ) ),
                Kokkos::subview( _data, Kokkos::pair<size_type, size_type>(
                                            0, _num_soa ) ) );
        _data = resized_data;
    }

    /*!
      \brief Remove unused capacity.

      Will reduce the capacity to be the smallest number of SoAs needed to
      hold size() tuples. If reallocation occurs, all slices and all
      references to the elements are invalidated. If no reallocation takes
      place, no slices or references are invalidated.
    */
    void shrinkToFit()
    {
        static_assert( !memory_traits::is_unmanaged,
                       "Cannot shrink unmanaged memory" );

        // If we aren't asking for any fewer SoA objects then we have nothing
        // to do. The amount of allocated data has to be at least as big as
        // _num_soa so we just need to check here that they are equivalent. If
        // they are equivalent, the container is already as small as it can be.
        if ( _data.size() == _num_soa )
            return;

        // Assign the new capacity.
        _capacity = _num_soa * vector_length;

        // We need fewer SoA objects so allocate a new view and copy the
        // existing data.
        soa_view resized_data(
            Kokkos::ViewAllocateWithoutInitializing( _data.label() ),
            _num_soa );
        if ( _num_soa > 0 )
            Kokkos::deep_copy(
                resized_data,
                Kokkos::subview( _data, Kokkos::pair<size_type, size_type>(
                                            0, _num_soa ) ) );
        _data = resized_data;
    }

    /*!
      \brief Get the number of structs-of-arrays in the container.

      \return The number of structs-of-arrays in the container.
    */
    KOKKOS_INLINE_FUNCTION
    size_type numSoA() const { return _num_soa; }

    /*!
      \brief Get the size of the data array at a given struct member index.

      \param s The struct index to get the array size for.

      \return The size of the array at the given struct index.
    */
    KOKKOS_INLINE_FUNCTION
    size_type arraySize( const size_type s ) const
    {
        // the SoA struct size should be full size, i.e. vector_length if:
        // 1) s is not the last SoA struct
        // or 2) if _size = _num_soa * vector_length
        return ( ( (size_type)s < _num_soa - 1 ) ||
                 ( _size % vector_length == 0 ) )
                   ? vector_length              // if s is a full SoA struct
                   : ( _size % vector_length ); // if s is the last SoA struct
    }

    /*!
      \brief Get a reference to the SoA at a given index.

      \param s The SoA index.

      \return The SoA reference at the given index.
    */
    KOKKOS_FORCEINLINE_FUNCTION
    soa_type& access( const size_type s ) const { return _data( s ); }

    /*!
      \brief Get a tuple at a given index via a deep copy.

      \param i The index to get the tuple from.

      \return A tuple containing a deep copy of the data at the given index.
    */
    KOKKOS_INLINE_FUNCTION
    tuple_type getTuple( const size_type i ) const
    {
        tuple_type tpl;
        Impl::tupleCopy( tpl, 0, _data( index_type::s( i ) ),
                         index_type::a( i ) );
        return tpl;
    }

    /*!
      \brief Set a tuple at a given index via a deep copy.

      \param i The index to set the tuple at.

      \param tpl The tuple to get the data from.
    */
    KOKKOS_INLINE_FUNCTION
    void setTuple( const size_type i, const tuple_type& tpl ) const
    {
        Impl::tupleCopy( _data( index_type::s( i ) ), index_type::a( i ), tpl,
                         0 );
    }

    /*!
      \brief Get a typed raw pointer to the entire data block.
      \return A typed raw-pointer to the entire data block.
    */
    soa_type* data() const { return _data.data(); }

  private:
    // Total number of tuples in the container.
    size_type _size;

    // Allocated number of tuples in all arrays in all structs.
    size_type _capacity;

    // Number of structs-of-arrays in the array.
    size_type _num_soa;

    // Structs-of-Arrays managed data. This Kokkos View manages the block of
    // memory owned by this class such that the copy constructor and
    // assignment operator for this class perform a shallow and reference
    // counted copy of the data.
    soa_view _data;
};

//---------------------------------------------------------------------------//

} // end namespace Cabana

#endif // CABANA_AOSOA_HPP
