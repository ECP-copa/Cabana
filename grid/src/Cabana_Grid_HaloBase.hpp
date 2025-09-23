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
  \file Cabana_Grid_HaloBase.hpp
  \brief Multi-node grid scatter/gather
*/
#ifndef CABANA_GRID_HALO_HPP
#define CABANA_GRID_HALO_HPP

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Tags.hpp>

#include <Cabana_ParameterPack.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

namespace Cabana
{
namespace Grid
{

//---------------------------------------------------------------------------//
// Halo exchange patterns.
//---------------------------------------------------------------------------//
//! Base halo exchange pattern class.
template <std::size_t NumSpaceDim>
class HaloPattern
{
  public:
    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    // Default constructor.
    HaloPattern() {}

    // Destructor
    virtual ~HaloPattern() = default;

    //! Assign the neighbors that are in the halo pattern.
    void
    setNeighbors( const std::vector<std::array<int, num_space_dim>>& neighbors )
    {
        _neighbors = neighbors;
    }

    //! Get the neighbors that are in the halo pattern.
    std::vector<std::array<int, num_space_dim>> getNeighbors() const
    {
        return _neighbors;
    }

  private:
    std::vector<std::array<int, num_space_dim>> _neighbors;
};

//! %Halo with node connectivity. I.e. communicate with all neighbor ranks with
//! which I share a node.
template <std::size_t NumSpaceDim>
class NodeHaloPattern;

//! 3d halo with node connectivity. I.e. communicate with all neighbor ranks
//! with which I share a node.
template <>
class NodeHaloPattern<3> : public HaloPattern<3>
{
  public:
    NodeHaloPattern()
        : HaloPattern<3>()
    {
        std::vector<std::array<int, 3>> neighbors;
        neighbors.reserve( 26 );
        for ( int i = -1; i < 2; ++i )
            for ( int j = -1; j < 2; ++j )
                for ( int k = -1; k < 2; ++k )
                    if ( !( i == 0 && j == 0 && k == 0 ) )
                        neighbors.push_back( { i, j, k } );
        this->setNeighbors( neighbors );
    }
};

//! 2d halo with node connectivity. I.e. communicate with all neighbor ranks
//! with which I share a node.
template <>
class NodeHaloPattern<2> : public HaloPattern<2>
{
  public:
    NodeHaloPattern()
        : HaloPattern<2>()
    {
        std::vector<std::array<int, 2>> neighbors;
        neighbors.reserve( 8 );
        for ( int i = -1; i < 2; ++i )
            for ( int j = -1; j < 2; ++j )
                if ( !( i == 0 && j == 0 ) )
                    neighbors.push_back( { i, j } );
        this->setNeighbors( neighbors );
    }
};

//! %Halo with face connectivity. I.e. communicate with all neighbor ranks with
//! which I share a face.
template <std::size_t NumSpaceDim>
class FaceHaloPattern;

//! 3d halo with face connectivity. I.e. communicate with all neighbor ranks
//! with which I share a face.
template <>
class FaceHaloPattern<3> : public HaloPattern<3>
{
  public:
    FaceHaloPattern()
        : HaloPattern<3>()
    {
        std::vector<std::array<int, 3>> neighbors = {
            { -1, 0, 0 }, { 1, 0, 0 },  { 0, -1, 0 },
            { 0, 1, 0 },  { 0, 0, -1 }, { 0, 0, 1 } };
        this->setNeighbors( neighbors );
    }
};

//! 2d halo with face connectivity. I.e. communicate with all neighbor ranks
//! with which I share a face.
template <>
class FaceHaloPattern<2> : public HaloPattern<2>
{
  public:
    FaceHaloPattern()
        : HaloPattern<2>()
    {
        std::vector<std::array<int, 2>> neighbors = {
            { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        this->setNeighbors( neighbors );
    }
};

//---------------------------------------------------------------------------//
// Scatter reduction.
//---------------------------------------------------------------------------//
namespace ScatterReduce
{

//! Sum values from neighboring ranks into this rank's data.
struct Sum
{
};

//! Assign this rank's data to be the minimum of it and its neighbor ranks'
//! values.
struct Min
{
};

//! Assign this rank's data to be the maximum of it and its neighbor ranks'
//! values.
struct Max
{
};

//! Replace this rank's data with its neighbor ranks' values. Note that if
//! multiple ranks scatter back to the same grid locations then the value
//! assigned will be from one of the neighbors but it is undetermined from
//! which neighbor that value will come.
struct Replace
{
};

} // end namespace ScatterReduce

//---------------------------------------------------------------------------//
// Halo
// ---------------------------------------------------------------------------//
/*!
  General multiple array halo communication plan for migrating shared data
  between blocks. Arrays may be defined on different entity types and have
  different data types.

  The halo operates on an arbitrary set of arrays. Each of these arrays must
  be defined on the same local grid meaning they that share the same
  communicator and halo size. The arrays must also reside in the same memory
  space. These requirements are checked at construction.
*/
template <class MemorySpace>
class HaloBase
{
  public:
    //! Memory space.
    using memory_space = MemorySpace;

  protected:
    /*!
      \brief Constructor.
      \tparam The arrays types to construct the halo for.
      \param pattern The halo pattern to use for halo communication.
      \param width Halo cell width. Must be less than or equal to the halo
      width of the block.
      \param arrays The arrays to build the halo for. These arrays must be
      provided in the same order
    */
    template <class Pattern, class... ArrayTypes>
    HaloBase( const Pattern& pattern, const int width,
              const ArrayTypes&... arrays )
    {
        // Spatial dimension.
        const std::size_t num_space_dim = Pattern::num_space_dim;

        // Get the local grid.
        auto local_grid = getLocalGrid( arrays... );

        // Function to get the local id of the neighbor.
        auto neighbor_id = []( const std::array<int, num_space_dim>& ijk )
        {
            int id = ijk[0];
            for ( std::size_t d = 1; d < num_space_dim; ++d )
                id += num_space_dim * id + ijk[d];
            return id;
        };

        // Neighbor id flip function. This lets us compute what neighbor we
        // are relative to a given neighbor.
        auto flip_id = [=]( const std::array<int, num_space_dim>& ijk )
        {
            std::array<int, num_space_dim> flip_ijk;
            for ( std::size_t d = 0; d < num_space_dim; ++d )
                flip_ijk[d] = -ijk[d];
            return flip_ijk;
        };

        // Get the neighbor ranks we will exchange with in the halo and
        // allocate buffers. If any of the exchanges are self sends mark these
        // so we know which send buffers correspond to which receive buffers.
        auto neighbors = pattern.getNeighbors();
        for ( const auto& n : neighbors )
        {
            // Get the rank of the neighbor.
            int rank = local_grid->neighborRank( n );

            // If this is a valid rank add it as a neighbor.
            if ( rank >= 0 )
            {
                // Add the rank.
                _neighbor_ranks.push_back( rank );

                // Set the tag we will use to send data to this neighbor. The
                // receiving rank should have a matching tag.
                _send_tags.push_back( neighbor_id( n ) );

                // Set the tag we will use to receive data from this
                // neighbor. The sending rank should have a matching tag.
                _receive_tags.push_back( neighbor_id( flip_id( n ) ) );

                // Create communication data for owned entities.
                buildCommData( Own(), width, n, _owned_buffers, _owned_steering,
                               arrays... );

                // Create communication data for ghosted entities.
                buildCommData( Ghost(), width, n, _ghosted_buffers,
                               _ghosted_steering, arrays... );
            }
        }
    }

  public:
    //! Get the communicator.
    template <class Array_t>
    MPI_Comm getComm( const Array_t& array ) const
    {
        return array.layout()->localGrid()->globalGrid().comm();
    }

    //! Get the communicator and check to make sure all are the same.
    template <class Array_t, class... ArrayTypes>
    MPI_Comm getComm( const Array_t& array, const ArrayTypes&... arrays ) const
    {
        auto comm = getComm( array );

        // Check that the communicator of this array is the same as the other
        // arrays.
        int result;
        MPI_Comm_compare( comm, getComm( arrays... ), &result );
        if ( result != MPI_IDENT && result != MPI_CONGRUENT )
            throw std::runtime_error( "Cabana::Grid::Halo::getComm: Arrays "
                                      "have different communicators" );

        return comm;
    }

    //! Get the local grid from the arrays. Check that the grids have the same
    //! halo size.
    template <class Array_t>
    auto getLocalGrid( const Array_t& array )
    {
        return array.layout()->localGrid();
    }

    //! Get the local grid from the arrays. Check that the grids have the same
    //! halo size.
    template <class Array_t, class... ArrayTypes>
    auto getLocalGrid( const Array_t& array, const ArrayTypes&... arrays )
    {
        // Recurse.
        auto local_grid = getLocalGrid( arrays... );

        // Check that the halo sizes same.
        if ( local_grid->haloCellWidth() !=
             array.layout()->localGrid()->haloCellWidth() )
        {
            throw std::runtime_error( "Cabana::Grid::Halo::getlocalGrid: "
                                      "Arrays have different halo widths" );
        }

        return local_grid;
    }

    //! Build communication data.
    template <class DecompositionTag, std::size_t NumSpaceDim,
              class... ArrayTypes>
    void
    buildCommData( DecompositionTag decomposition_tag, const int width,
                   const std::array<int, NumSpaceDim>& nid,
                   std::vector<Kokkos::View<char*, memory_space>>& buffers,
                   std::vector<Kokkos::View<int**, memory_space>>& steering,
                   const ArrayTypes&... arrays )
    {
        // Number of arrays.
        const std::size_t num_array = sizeof...( ArrayTypes );

        // Get the byte sizes of array value types.
        std::array<std::size_t, num_array> value_byte_sizes = {
            sizeof( typename ArrayTypes::value_type )... };

        // Get the index spaces we share with this neighbor. We
        // get a shared index space for each array.
        std::array<IndexSpace<NumSpaceDim + 1>, num_array> spaces = {
            ( arrays.layout()->sharedIndexSpace( decomposition_tag, nid,
                                                 width ) )... };

        // Compute the buffer size of this neighbor and the
        // number of elements in the buffer.
        int buffer_bytes = 0;
        int buffer_num_element = 0;
        for ( std::size_t a = 0; a < num_array; ++a )
        {
            buffer_bytes += value_byte_sizes[a] * spaces[a].size();
            buffer_num_element += spaces[a].size();
        }

        // Allocate the buffer of data that we share with this neighbor. All
        // arrays will be packed into a single buffer.
        buffers.push_back(
            Kokkos::View<char*, memory_space>( "halo_buffer", buffer_bytes ) );

        // Allocate the steering vector for building the buffer.
        steering.push_back( Kokkos::View<int**, memory_space>(
            "steering", buffer_num_element, 3 + NumSpaceDim ) );

        // Build steering vector.
        buildSteeringVector( spaces, value_byte_sizes, buffer_bytes,
                             buffer_num_element, steering );
    }

    //! Build 3d steering vector.
    template <std::size_t NumArray>
    void buildSteeringVector(
        const std::array<IndexSpace<4>, NumArray>& spaces,
        const std::array<std::size_t, NumArray>& value_byte_sizes,
        const int buffer_bytes, const int buffer_num_element,
        std::vector<Kokkos::View<int**, memory_space>>& steering )
    {
        // Create the steering vector. For each element in the buffer it gives
        // the starting byte location of the element, the array the element is
        // in, and the ijkl structured index in the array of the element.
        auto host_steering =
            Kokkos::create_mirror_view( Kokkos::HostSpace(), steering.back() );
        int elem_counter = 0;
        int byte_counter = 0;
        for ( std::size_t a = 0; a < NumArray; ++a )
        {
            for ( int i = spaces[a].min( 0 ); i < spaces[a].max( 0 ); ++i )
            {
                for ( int j = spaces[a].min( 1 ); j < spaces[a].max( 1 ); ++j )
                {
                    for ( int k = spaces[a].min( 2 ); k < spaces[a].max( 2 );
                          ++k )
                    {
                        for ( int l = spaces[a].min( 3 );
                              l < spaces[a].max( 3 ); ++l )
                        {
                            // Byte starting location in buffer.
                            host_steering( elem_counter, 0 ) = byte_counter;

                            // Array location of element.
                            host_steering( elem_counter, 1 ) = a;

                            // Structured index in array of element.
                            host_steering( elem_counter, 2 ) = i;
                            host_steering( elem_counter, 3 ) = j;
                            host_steering( elem_counter, 4 ) = k;
                            host_steering( elem_counter, 5 ) = l;

                            // Update element id.
                            ++elem_counter;

                            // Update buffer position.
                            byte_counter += value_byte_sizes[a];
                        }
                    }
                }
            }
        }

        // Check that all elements and bytes are accounted for.
        if ( byte_counter != buffer_bytes )
            throw std::logic_error( "Cabana::Grid::Halo::buildSteeringVector: "
                                    "Steering vector contains different number "
                                    "of bytes than buffer" );
        if ( elem_counter != buffer_num_element )
            throw std::logic_error( "Cabana::Grid::Halo::buildSteeringVector: "
                                    "Steering vector contains different number "
                                    "of elements than buffer" );

        // Copy steering vector to device.
        Kokkos::deep_copy( steering.back(), host_steering );
    }

    //! Build 2d steering vector.
    template <std::size_t NumArray>
    void buildSteeringVector(
        const std::array<IndexSpace<3>, NumArray>& spaces,
        const std::array<std::size_t, NumArray>& value_byte_sizes,
        const int buffer_bytes, const int buffer_num_element,
        std::vector<Kokkos::View<int**, memory_space>>& steering )
    {
        // Create the steering vector. For each element in the buffer it gives
        // the starting byte location of the element, the array the element is
        // in, and the ijkl structured index in the array of the element.
        auto host_steering =
            Kokkos::create_mirror_view( Kokkos::HostSpace(), steering.back() );
        int elem_counter = 0;
        int byte_counter = 0;
        for ( std::size_t a = 0; a < NumArray; ++a )
        {
            for ( int i = spaces[a].min( 0 ); i < spaces[a].max( 0 ); ++i )
            {
                for ( int j = spaces[a].min( 1 ); j < spaces[a].max( 1 ); ++j )
                {
                    for ( int l = spaces[a].min( 2 ); l < spaces[a].max( 2 );
                          ++l )
                    {
                        // Byte starting location in buffer.
                        host_steering( elem_counter, 0 ) = byte_counter;

                        // Array location of element.
                        host_steering( elem_counter, 1 ) = a;

                        // Structured index in array of element.
                        host_steering( elem_counter, 2 ) = i;
                        host_steering( elem_counter, 3 ) = j;
                        host_steering( elem_counter, 4 ) = l;

                        // Update element id.
                        ++elem_counter;

                        // Update buffer position.
                        byte_counter += value_byte_sizes[a];
                    }
                }
            }
        }

        // Check that all elements and bytes are accounted for.
        if ( byte_counter != buffer_bytes )
            throw std::logic_error( "Cabana::Grid::Halo::buildSteeringVector: "
                                    "Steering vector contains different number "
                                    "of bytes than buffer" );
        if ( elem_counter != buffer_num_element )
            throw std::logic_error( "Cabana::Grid::Halo::buildSteeringVector: "
                                    "Steering vector contains different number "
                                    "of elements than buffer" );

        // Copy steering vector to device.
        Kokkos::deep_copy( steering.back(), host_steering );
    }

    //! Pack an element into the buffer. Pack by bytes to avoid casting across
    //! alignment boundaries.
    template <class ArrayView>
    KOKKOS_INLINE_FUNCTION static std::enable_if_t<4 == ArrayView::rank, void>
    packElement( const Kokkos::View<char*, memory_space>& buffer,
                 const Kokkos::View<int**, memory_space>& steering,
                 const int element_idx, const ArrayView& array_view )
    {
        const char* elem_ptr = reinterpret_cast<const char*>( &array_view(
            steering( element_idx, 2 ), steering( element_idx, 3 ),
            steering( element_idx, 4 ), steering( element_idx, 5 ) ) );
        for ( std::size_t b = 0; b < sizeof( typename ArrayView::value_type );
              ++b )
        {
            buffer( steering( element_idx, 0 ) + b ) = *( elem_ptr + b );
        }
    }

    //! Pack an element into the buffer. Pack by bytes to avoid casting across
    //! alignment boundaries.
    template <class ArrayView>
    KOKKOS_INLINE_FUNCTION static std::enable_if_t<3 == ArrayView::rank, void>
    packElement( const Kokkos::View<char*, memory_space>& buffer,
                 const Kokkos::View<int**, memory_space>& steering,
                 const int element_idx, const ArrayView& array_view )
    {
        const char* elem_ptr = reinterpret_cast<const char*>(
            &array_view( steering( element_idx, 2 ), steering( element_idx, 3 ),
                         steering( element_idx, 4 ) ) );
        for ( std::size_t b = 0; b < sizeof( typename ArrayView::value_type );
              ++b )
        {
            buffer( steering( element_idx, 0 ) + b ) = *( elem_ptr + b );
        }
    }

    //! Pack an array into a buffer.
    template <class... ArrayViews>
    KOKKOS_INLINE_FUNCTION static void
    packArray( const Kokkos::View<char*, memory_space>& buffer,
               const Kokkos::View<int**, memory_space>& steering,
               const int element_idx,
               const std::integral_constant<std::size_t, 0>,
               const Cabana::ParameterPack<ArrayViews...>& array_views )
    {
        // If the pack element_idx is in the current array, pack it.
        if ( 0 == steering( element_idx, 1 ) )
            packElement( buffer, steering, element_idx,
                         Cabana::get<0>( array_views ) );
    }

    //! Pack an array into a buffer.
    template <std::size_t N, class... ArrayViews>
    KOKKOS_INLINE_FUNCTION static void
    packArray( const Kokkos::View<char*, memory_space>& buffer,
               const Kokkos::View<int**, memory_space>& steering,
               const int element_idx,
               const std::integral_constant<std::size_t, N>,
               const Cabana::ParameterPack<ArrayViews...>& array_views )
    {
        // If the pack element_idx is in the current array, pack it.
        if ( N == steering( element_idx, 1 ) )
        {
            packElement( buffer, steering, element_idx,
                         Cabana::get<N>( array_views ) );
            return;
        }

        // Recurse.
        packArray( buffer, steering, element_idx,
                   std::integral_constant<std::size_t, N - 1>(), array_views );
    }

    //! Pack arrays into a buffer.
    template <class ExecutionSpace, class... ArrayViews>
    void packBuffer( const ExecutionSpace& exec_space,
                     const Kokkos::View<char*, memory_space>& buffer,
                     const Kokkos::View<int**, memory_space>& steering,
                     ArrayViews... array_views ) const
    {
        auto pp = Cabana::makeParameterPack( array_views... );
        Kokkos::parallel_for(
            "Cabana::Grid::Halo::pack_buffer",
            Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0,
                                                 steering.extent( 0 ) ),
            KOKKOS_LAMBDA( const int i ) {
                packArray(
                    buffer, steering, i,
                    std::integral_constant<std::size_t,
                                           sizeof...( ArrayViews ) - 1>(),
                    pp );
            } );
        exec_space.fence();
    }

    //! Reduce an element into the buffer. Sum reduction.
    template <class T>
    KOKKOS_INLINE_FUNCTION static void
    unpackOp( ScatterReduce::Sum, const T& buffer_val, T& array_val )
    {
        array_val += buffer_val;
    }

    //! Reduce an element into the buffer. Min reduction.
    template <class T>
    KOKKOS_INLINE_FUNCTION static void
    unpackOp( ScatterReduce::Min, const T& buffer_val, T& array_val )
    {
        if ( buffer_val < array_val )
            array_val = buffer_val;
    }

    //! Reduce an element into the buffer. Max reduction.
    template <class T>
    KOKKOS_INLINE_FUNCTION static void
    unpackOp( ScatterReduce::Max, const T& buffer_val, T& array_val )
    {
        if ( buffer_val > array_val )
            array_val = buffer_val;
    }

    //! Reduce an element into the buffer. Replace reduction.
    template <class T>
    KOKKOS_INLINE_FUNCTION static void
    unpackOp( ScatterReduce::Replace, const T& buffer_val, T& array_val )
    {
        array_val = buffer_val;
    }

    //! Unpack an element from the buffer. Unpack by bytes to avoid casting
    //! across alignment boundaries.
    template <class ReduceOp, class ArrayView>
    KOKKOS_INLINE_FUNCTION static std::enable_if_t<4 == ArrayView::rank, void>
    unpackElement( const ReduceOp& reduce_op,
                   const Kokkos::View<char*, memory_space>& buffer,
                   const Kokkos::View<int**, memory_space>& steering,
                   const int element_idx, const ArrayView& array_view )
    {
        typename ArrayView::value_type elem;
        char* elem_ptr = reinterpret_cast<char*>( &elem );
        for ( std::size_t b = 0; b < sizeof( typename ArrayView::value_type );
              ++b )
        {
            *( elem_ptr + b ) = buffer( steering( element_idx, 0 ) + b );
        }
        unpackOp( reduce_op, elem,
                  array_view( steering( element_idx, 2 ),
                              steering( element_idx, 3 ),
                              steering( element_idx, 4 ),
                              steering( element_idx, 5 ) ) );
    }

    //! Unpack an element from the buffer. Unpack by bytes to avoid casting
    //! across alignment boundaries.
    template <class ReduceOp, class ArrayView>
    KOKKOS_INLINE_FUNCTION static std::enable_if_t<3 == ArrayView::rank, void>
    unpackElement( const ReduceOp& reduce_op,
                   const Kokkos::View<char*, memory_space>& buffer,
                   const Kokkos::View<int**, memory_space>& steering,
                   const int element_idx, const ArrayView& array_view )
    {
        typename ArrayView::value_type elem;
        char* elem_ptr = reinterpret_cast<char*>( &elem );
        for ( std::size_t b = 0; b < sizeof( typename ArrayView::value_type );
              ++b )
        {
            *( elem_ptr + b ) = buffer( steering( element_idx, 0 ) + b );
        }
        unpackOp( reduce_op, elem,
                  array_view( steering( element_idx, 2 ),
                              steering( element_idx, 3 ),
                              steering( element_idx, 4 ) ) );
    }

    //! Unpack an array from a buffer.
    template <class ReduceOp, class... ArrayViews>
    KOKKOS_INLINE_FUNCTION static void
    unpackArray( const ReduceOp& reduce_op,
                 const Kokkos::View<char*, memory_space>& buffer,
                 const Kokkos::View<int**, memory_space>& steering,
                 const int element_idx,
                 const std::integral_constant<std::size_t, 0>,
                 const Cabana::ParameterPack<ArrayViews...>& array_views )
    {
        // If the unpack element_idx is in the current array, unpack it.
        if ( 0 == steering( element_idx, 1 ) )
            unpackElement( reduce_op, buffer, steering, element_idx,
                           Cabana::get<0>( array_views ) );
    }

    //! Unpack an array from a buffer.
    template <class ReduceOp, std::size_t N, class... ArrayViews>
    KOKKOS_INLINE_FUNCTION static void
    unpackArray( const ReduceOp reduce_op,
                 const Kokkos::View<char*, memory_space>& buffer,
                 const Kokkos::View<int**, memory_space>& steering,
                 const int element_idx,
                 const std::integral_constant<std::size_t, N>,
                 const Cabana::ParameterPack<ArrayViews...>& array_views )
    {
        // If the unpack element_idx is in the current array, unpack it.
        if ( N == steering( element_idx, 1 ) )
        {
            unpackElement( reduce_op, buffer, steering, element_idx,
                           Cabana::get<N>( array_views ) );
            return;
        }

        // Recurse.
        unpackArray( reduce_op, buffer, steering, element_idx,
                     std::integral_constant<std::size_t, N - 1>(),
                     array_views );
    }

    //! Unpack arrays from a buffer.
    template <class ExecutionSpace, class ReduceOp, class... ArrayViews>
    void unpackBuffer( const ReduceOp& reduce_op,
                       const ExecutionSpace& exec_space,
                       const Kokkos::View<char*, memory_space>& buffer,
                       const Kokkos::View<int**, memory_space>& steering,
                       ArrayViews... array_views ) const
    {
        auto pp = Cabana::makeParameterPack( array_views... );
        Kokkos::parallel_for(
            "Cabana::Grid::Halo::unpack_buffer",
            Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0,
                                                 steering.extent( 0 ) ),
            KOKKOS_LAMBDA( const int i ) {
                unpackArray(
                    reduce_op, buffer, steering, i,
                    std::integral_constant<std::size_t,
                                           sizeof...( ArrayViews ) - 1>(),
                    pp );
            } );
    }

  protected:
    //! The ranks we will send/receive from.
    std::vector<int> _neighbor_ranks;

    //! The tag we use for sending to each neighbor.
    std::vector<int> _send_tags;

    //! The tag we use for receiving from each neighbor.
    std::vector<int> _receive_tags;

    //! For each neighbor, send/receive buffers for data we own.
    std::vector<Kokkos::View<char*, memory_space>> _owned_buffers;

    //! For each neighbor, send/receive buffers for data we ghost.
    std::vector<Kokkos::View<char*, memory_space>> _ghosted_buffers;

    //! For each neighbor, steering vector for the owned buffer.
    std::vector<Kokkos::View<int**, memory_space>> _owned_steering;

    //! For each neighbor, steering vector for the ghosted buffer.
    std::vector<Kokkos::View<int**, memory_space>> _ghosted_steering;
};

//---------------------------------------------------------------------------//
// Creation function.
//---------------------------------------------------------------------------//
//! Infer array memory space.
template <class ArrayT, class... Types>
struct ArrayPackMemorySpace
{
    //! Memory space.
    using type = typename ArrayT::memory_space;
};

// Forward declaration of the primary grid Halo template.
template <class MemorySpace, class CommSpaceType = Mpi>
class Halo;

} // end namespace Grid
} // end namespace Cabana

// Include communication backends from what is enabled in CMake.
#ifdef Cabana_ENABLE_MPI
#include <impl/Cabana_Grid_Halo_Mpi.hpp>
#endif // Enable MPI

namespace Cabana
{
namespace Grid
{

//---------------------------------------------------------------------------//
/*!
  \brief Halo creation function.
  \param pattern The pattern to build the halo from.
  \param width Must be less than or equal to the width of the array halo.
  \param arrays The arrays over which to build the halo.
  \return Shared pointer to a Halo.
*/
template <class CommSpaceType = Mpi, class Pattern, class... ArrayTypes>
auto createHalo( const Pattern& pattern, const int width,
                 const ArrayTypes&... arrays )
{
    using memory_space = typename ArrayPackMemorySpace<ArrayTypes...>::type;
    return std::make_shared<Halo<memory_space, CommSpaceType>>( pattern, width,
                                                                arrays... );
}

} // end namespace Grid
} // end namespace Cabana

//---------------------------------------------------------------------------//

#endif // end CABANA_GRID_HALOBASE_HPP
