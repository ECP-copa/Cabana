/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CABANA_BUFFEREDAOSOA_HPP
#define CABANA_BUFFEREDAOSOA_HPP

#include <tuple>
#include <vector>

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_ParameterPack.hpp>

namespace Cabana
{
namespace Experimental
{
//! \cond Impl

template <class AoSoA, std::size_t... Indicies>
auto makeSliceParameterPackImpl( const AoSoA& aosoa,
                                 std::index_sequence<Indicies...> )
{
    return Cabana::makeParameterPack( Cabana::slice<Indicies>( aosoa )... );
}

template <class AoSoA>
auto makeSliceParameterPack( const AoSoA& aosoa )
{
    return makeSliceParameterPackImpl(
        aosoa, std::make_index_sequence<AoSoA::number_of_members>() );
}

/**
 * @brief Target_Memory_Space The memory space to buffer into
 * AoSoA_t The type of the AoSoA we're buffering from
 */
// TODO: requested_buffer_count should be a runy time param, but for now I have
// it as compile time, as it impliues a memory allocation. This currently causes
// a call to host functions / malloc on the GPU when the BufgferedAoSoA is
// passed, which would need to be fixed
template <int requested_buffer_count, class Target_Memory_Space, class AoSoA_t>
class BufferedAoSoA
{
    // TODO: make things private
  public:
    using from_AoSoA_type = AoSoA_t;

    // Figure out the compatible AoSoA_t for the Target_Memory_Space
    using target_AoSoA_t = Cabana::AoSoA<
        typename from_AoSoA_type::member_types, Target_Memory_Space,
        // Doing this put the burden on the user to get it right...
        // Without it you can get into trouble with copy sizes not matching
        // because the sizeof(soa_type) doesn't match
        AoSoA_t::vector_length>;

    // Cabana::AoSoA<DataTypes, TEST_MEMSPACE, vector_length>;

    using cajita_slice = decltype( makeSliceParameterPack( target_AoSoA_t() ) );

    // Holds a collection of slices for a single buffer. Currently means we can
    // only hold slices to a single buffer at a time
    cajita_slice cajita_tuple;

    // (We could make slice_tuple an array to hold multiple buffers worth of
    // slices, but the user should not have to know which buffer they are in.)

    /**
     * @brief Getter to access the slices this class generates for the
     * user
     *
     * @return Valid slice for the current buffer at position i
     */
    template <std::size_t i>
    KOKKOS_INLINE_FUNCTION const typename target_AoSoA_t::
        template member_slice_type<i>&
        get_slice()
    {
        return Cabana::get<i>( cajita_tuple );
    }

    // Internal buffers which we use to buffer data back and forth, and also
    // where we perform the compute
    target_AoSoA_t internal_buffers[requested_buffer_count];
    // std::vector<target_AoSoA_t> internal_buffers;

    /**
     * @brief constructor for BufferedAoSoA
     *
     * @param original_view_in The view to buffer
     * @param requested_buffer_count The number of buffered (3=triple buffered)
     * @param max_buffered_tuples The max size of a single buffer in tuples
     * (the memory requirement is this class is roughly
     * requested_buffer_count*max_buffered_tuples*sizeof(particle)
     */
    BufferedAoSoA( AoSoA_t original_view_in, int max_buffered_tuples )
        : original_view( original_view_in )
        , num_buffers( requested_buffer_count )
        , buffer_size( max_buffered_tuples )
    {
        // TODO: We may we want to override the user on their requested
        // values if they don't make sense? Both num_buffers and buffer_size

        // TODO: add asserts to check the balance between the size (num_data)
        // and the max_buffered_tuples

        // Resize the buffers so we know they can hold enough
        for ( int i = 0; i < num_buffers; i++ )
        {
            std::cout << "Making buffer of size " << buffer_size << std::endl;

            std::string internal_buff_name =
                "internal_buff " + std::to_string( i );
            internal_buffers[i] =
                target_AoSoA_t( internal_buff_name, buffer_size );
        }
    }

    /**
     * @brief Populate the slice_tuple with the correct slices for a given
     * buffer
     *
     * @param buffer_index The index of the buffer, in the internal_buffers, to
     * use when populatating slice_buffer
     */
    void slice_buffer( int buffer_index )
    {
        // TODO: using this global variable is asking for trouble
        // slice_tuple = slice_tuple_t(
        // internal_buffers[buffer_index % get_buffer_count()] );

        cajita_tuple = makeSliceParameterPack(
            internal_buffers[buffer_index % get_buffer_count()] );
        std::cout << "Slicing buffer " << buffer_index
                  << " to give slice<0> = " << &get_slice<0>() << std::endl;
    }

    /** @brief Helper to access the number of buffers which exist.
     * Makes no comment about the state or content of the buffers
     *
     * @return The number of total buffers available
     */
    int get_buffer_count() { return num_buffers; }

    /**
     * @brief Copy the data from the last buffer to the "real data", at
     * location start_index
     *
     * @param start_index Index to place buffer data into in original
     * aosoa
     */
    void copy_buffer_back( int buffer_index, int to_index )
    {
        std::cout << "original view size " << original_view.size() << std::endl;
        Cabana::deep_copy_partial_dst( original_view,
                                       internal_buffers[buffer_index], to_index,
                                       // 0, // from index
                                       buffer_size );
    }

    /**
     * @brief Load a given buffer, populating it with data from the original
     * AoSoA
     *
     * @param buffer_number The index of the (internal) buffer to populate
     */
    void load_buffer( int buffer_number )
    {
        // TODO: does this imply the need for a subview so the sizes
        // match?

        int normalized_buffer_number = buffer_number % get_buffer_count();

        int start_index = normalized_buffer_number * buffer_size;
        // Copy from the main memory store into the "current" buffer
        Cabana::deep_copy_partial_src(
            internal_buffers[normalized_buffer_number], original_view,
            // 0, // to_index,
            start_index, buffer_size );

        // TODO: is this likely to cause a problem at runtime?
        // Update the slice tuple to have slices based on the current buffer
        // slice_tuple = slice_tuple_t( internal_buffers[buffer_number] );
    }

    AoSoA_t original_view;

    /**
     * @brief Number of buffers we decided to use (possibly distinct
     * from how many the user asked for)
     *
     * I think it's important to make a distinction between how
     * many buffers the user thinks they need, and how many we decide
     * to use.
     */
    int num_buffers;

    /**
     * @brief Number of particles in each temporary buffer
     */
    int buffer_size;

  private:
};

//! \endcond
} // end namespace Experimental
} // end namespace Cabana

#endif // CABANA_BUFFEREDAOSOA_HPP
