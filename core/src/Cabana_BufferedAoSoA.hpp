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

namespace Cabana
{
// TODO: this tuple implementation could possibly be a std::index_sequence
// over Cabana::member_slice_type and be simplified?
// TODO: Use the same tuple pattern as in Cabana::SoA using
// std::make_index_sequence

/**
 * @brief Contains the value for one item in the SliceAtIndex tuple.
 *
 * Here this is specialized to be a slice, where it is initialized based on
 * the given aosoa. The extension to remove this and make it general is
 * trivial
 */
template <typename AoSoA_t, std::size_t i, typename Item>
struct SliceAtIndexLeaf
{
    using value_t = typename AoSoA_t::template member_slice_type<i>;
    value_t value;

    SliceAtIndexLeaf( AoSoA_t &aosoa ) { value = Cabana::slice<i>( aosoa ); }
    SliceAtIndexLeaf() {}
};

template <typename AoSoA_t, std::size_t i, typename... Items>
struct SliceAtIndexImpl;

/**
 * @brief Base case for an empty SliceAtIndex tuple
 */
template <typename AoSoA_t, std::size_t i>
struct SliceAtIndexImpl<AoSoA_t, i>
{
    // TODO: see how everyone wants to handle this unused param, or if it can
    // be removed
    SliceAtIndexImpl( __attribute__( ( unused ) ) AoSoA_t &aosoa ) {}
    SliceAtIndexImpl() {}
};

/**
 * @brief Recursive specialization, unrolls the contents and creates the
 * leaf nodes
 */
template <typename AoSoA_t, std::size_t i, typename HeadItem,
          typename... TailItems>
struct SliceAtIndexImpl<AoSoA_t, i, HeadItem, TailItems...>
    : public SliceAtIndexLeaf<AoSoA_t, i,
                              HeadItem>, // This adds a `value` member of type
                                         // HeadItem
      public SliceAtIndexImpl<AoSoA_t, i + 1, TailItems...> // This recurses
{
    SliceAtIndexImpl( AoSoA_t &aosoa )
        : SliceAtIndexLeaf<AoSoA_t, i, HeadItem>( aosoa )
        , SliceAtIndexImpl<AoSoA_t, i + 1, TailItems...>( aosoa )
    {
    }
    SliceAtIndexImpl() {}
};

/**
 * @brief High level class for generic slice holding SliceAtIndex tuple
 *
 * We pass the AoSoA down through the hierarchy so the leaf can initialize
 * the slice value. This is important, as it lets us unroll the types at
 * compile time. Looping over the contents of the tuple is tricky, as we
 * need to use the index as a type. If we don't like this tight c
 */
template <typename AoSoA_t, typename... Items>
struct _SliceAtIndex : SliceAtIndexImpl<AoSoA_t, 0, Items...>
{
    _SliceAtIndex( AoSoA_t &aosoa )
        : SliceAtIndexImpl<AoSoA_t, 0, Items...>( aosoa )
    {
    }
    _SliceAtIndex() {}
};

/**
 * @brief Helper class to support the use of a specialized SliceAtIndex tuple
 * which is able to unpack Cabana::MemberTypes from an AoSoA
 */
template <typename AoSoA_t, typename... Itmes>
struct UnpackSliceAtIndex;

template <typename AoSoA_t, typename... Items>
struct UnpackSliceAtIndex<AoSoA_t, MemberTypes<Items...>>
    : _SliceAtIndex<AoSoA_t, Items...>
{
  public:
    static constexpr auto length = sizeof...( Items );

    UnpackSliceAtIndex( AoSoA_t &aosoa )
        : _SliceAtIndex<AoSoA_t, Items...>( aosoa )
    {
    }
    UnpackSliceAtIndex() {}
};

// TODO: where is the right place for this function to live?
/**
 * @brief Underlying generic style getter to obtain a reference to
 * i-th item in the slice tuple
 *
 * @param tuple The tuple to access from
 *
 * @return The i-th value in the tuple
 */
template <typename AoSoA_t, std::size_t i, typename HeadItem,
          typename... TailItems>
KOKKOS_INLINE_FUNCTION
    typename SliceAtIndexLeaf<AoSoA_t, i, HeadItem>::value_t &
    get( SliceAtIndexImpl<AoSoA_t, i, HeadItem, TailItems...> &tuple )
{
    return tuple.SliceAtIndexLeaf<AoSoA_t, i, HeadItem>::value;
}

/**
 * @brief
 *
 * Target_Memory_Space The memory space to buffer into
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
    using target_AoSoA_t = Cabana::AoSoA<typename from_AoSoA_type::member_types,
                                         Target_Memory_Space>;
    // Cabana::AoSoA<DataTypes, TEST_MEMSPACE, vector_length>;

    using slice_tuple_t =
        UnpackSliceAtIndex<target_AoSoA_t,
                           typename target_AoSoA_t::member_types>;

    slice_tuple_t slice_tuple;

    /**
     * @brief Getter to access the slices this class generates for the
     * user
     *
     * @return Valid slice for the current buffer at position i
     */
    template <std::size_t i>
    KOKKOS_INLINE_FUNCTION
        typename target_AoSoA_t::template member_slice_type<i> &
        get_slice()
    {
        return Cabana::get<target_AoSoA_t, i>( slice_tuple );
    }

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
    BufferedAoSoA( AoSoA_t &original_view_in, int max_buffered_tuples )
        : // I need to init slice_tuple avoid the default constructor, the type
          // may need changing when we respect their passed in space and this
          // can likely be done more cleanly if we simplify the tuple
          // implementation
        // slice_tuple( target_AoSoA_t()  )
        original_view( original_view_in )
        , num_buffers( requested_buffer_count )
        , buffer_size( max_buffered_tuples )
    {
        // TODO: this is only used internally now, and can likely be a
        // non-pointers
        last_filled_buffer = -1;

        // TODO: We may we want to override the user on their requested
        // values if they don't make sense? Both num_buffers and buffer_size

        // internal_buffers.resize( num_buffers );

        std::cout << "The size of the passed view is "
                  << original_view_in.size() << std::endl;
        std::cout << "The size of our view is " << original_view.size()
                  << std::endl;

        // Resize the buffers so we know they can hold enough
        for ( int i = 0; i < num_buffers; i++ )
        {
            // internal_buffers[i].resize(buffer_size);
            std::cout << "Making buffer of size " << buffer_size << std::endl;

            std::string internal_buff_name =
                "internal_buff " + std::to_string( i );
            internal_buffers[i] =
                target_AoSoA_t( internal_buff_name, buffer_size );

            std::cout << "Buff " << i << " has size "
                      << internal_buffers[i].size() << std::endl;
        }
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
    void copy_buffer_back( int to_index )
    {
        Cabana::deep_copy_partial_dst(
            original_view, internal_buffers[last_filled_buffer], to_index,
            // 0, // from index
            buffer_size );
    }

    // Start thinking about how to handle data movement...
    void load_next_buffer( int start_index )
    {
        // TODO: once this is asynchronous we have to be careful here
        last_filled_buffer++;
        if ( last_filled_buffer >= num_buffers )
        {
            last_filled_buffer = 0;
        }

        // TODO: does this imply the need for a subview so the sizes
        // match?

        // Copy from the main memory store into the "current" buffer
        Cabana::deep_copy_partial_src( internal_buffers[last_filled_buffer],
                                       original_view,
                                       // 0, // to_index,
                                       start_index, buffer_size );

        // Update the slice tuple to have slices based on the current buffer
        slice_tuple = slice_tuple_t( internal_buffers[last_filled_buffer] );
    }

    AoSoA_t &original_view;

    /**
     * @brief Track which buffer we "filled" last, so we know where we
     * are in the round robin
     */
    int last_filled_buffer;

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

} // end namespace Cabana

#endif // CABANA_BUFFEREDAOSOA_HPP
