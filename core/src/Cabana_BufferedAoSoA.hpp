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

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>

namespace Cabana
{
    // TODO: this tuple implementation could possibly be a std::index_sequence
    // over Cabana::member_slice_type and be simplified?

    /**
     * @brief Contains the value for one item in the tuple.
     *
     * Here this is specialized to be a slice, where it is initialized based on
     * the given aosoa. The extension to remove this and make it general is
     * trivial
     */
    template<typename AoSoA_t, std::size_t i, typename Item>
        struct TupleLeaf {
            using value_t = typename AoSoA_t::template member_slice_type<i>;
            value_t value;

            TupleLeaf(AoSoA_t& aosoa) {
                value = Cabana::slice<i>(aosoa);
            }
        };

    template<typename AoSoA_t, std::size_t i, typename... Items> struct TupleImpl;

    /**
     * @brief Base case for an empty tuple
     */
    template<typename AoSoA_t, std::size_t i> struct TupleImpl<AoSoA_t, i>{
        TupleImpl(AoSoA_t& aosoa) { }
    };

    /**
     * @brief Recursive specialization, unrolls the contents and creates the
     * leaf nodes
     */
    template<typename AoSoA_t, std::size_t i, typename HeadItem, typename... TailItems>
        struct TupleImpl<AoSoA_t, i, HeadItem, TailItems...> :
        public TupleLeaf<AoSoA_t, i, HeadItem>, // This adds a `value` member of type HeadItem
               public TupleImpl<AoSoA_t, i + 1, TailItems...> // This recurses
    {
        TupleImpl(AoSoA_t& aosoa) :
            TupleLeaf<AoSoA_t, i, HeadItem>( aosoa ),
            TupleImpl<AoSoA_t, i+1, TailItems...>( aosoa ) { }
    };

    /**
     * @brief High level class for generic slice holding tuple
     *
     * We pass the AoSoA down through the hierarchy so the leaf can initialize
     * the slice value. This is important, as it lets us unroll the types at
     * compile time. Looping over the contents of the tuple is tricky, as we
     * need to use the index as a type. If we don't like this tight c
     */
    template<typename AoSoA_t, typename... Items>
        struct _Tuple : TupleImpl<AoSoA_t, 0, Items...>
    {
        _Tuple(AoSoA_t& aosoa) : TupleImpl<AoSoA_t, 0, Items...>(aosoa) { }
    };

    /**
     * @brief Helper class to support the use of a specialized tuples which is
     * able to unpack Cabana::MemberTypes from an AoSoA
     */
    template<typename AoSoA_t, typename... Itmes> struct UnpackTuple;
    template<typename AoSoA_t, typename... Items> struct UnpackTuple<AoSoA_t, MemberTypes<Items...>>
        : _Tuple<AoSoA_t, Items...>
    {
        public:
          static constexpr auto length = sizeof...(Items);
          UnpackTuple(AoSoA_t& aosoa) : _Tuple<AoSoA_t, Items...>(aosoa) { }
    };

    // Requirements:
    // 1) User must be able to specify the memory space the data will be "buffered" into
        // a) we can presumably detect the existing allocated space, and determine if they are the same
    // 2) User must be able to specify the execution space the data will be "buffered" into
        // This will let us detect situations where data can be automatically buffered
    // 3) User must specify the maximum number of tuples that we are allowed to auto allocated without running OOM
    // 4) User can provide a hint for ideal number of buffers to chunk the data into while buffering
    template<
        int max_buffered_tuples,
        int requested_buffer_count, // TODO: should this be an optional hint, and not be prescriptive
        class Exec_Space, // TODO: we can infer this?
        class AoSoA_t
    >
    class BufferedAoSoA {
        // TODO: make things private
        public:

            using AoSoA_type = AoSoA_t;

            using slice_tuple_t = UnpackTuple<AoSoA_t, typename AoSoA_t::member_types>;
            slice_tuple_t slice_tuple;

            // TODO: this has nothing to do with the class, only the
            // specialized get_slice does
            /**
             * @brief Underlying generic style getter to obtain a reference to
             * i-th item in the slice tuple
             *
             * @param tuple The tuple to access from
             *
             * @return The i-th value in the tuple
             */
            template<typename _AoSoA_t, std::size_t i, typename HeadItem, typename... TailItems>
                KOKKOS_INLINE_FUNCTION
                typename TupleLeaf<_AoSoA_t, i, HeadItem>::value_t&
                _Get(TupleImpl<_AoSoA_t, i, HeadItem, TailItems...>& tuple) {
                    return tuple.TupleLeaf<AoSoA_t, i, HeadItem>::value;
                }

            /**
             * @brief Getter to access the slices this class generates for the
             * user
             *
             * @return Valid slice for the current buffer at position i
             */
            template<std::size_t i>
                KOKKOS_INLINE_FUNCTION
                typename AoSoA_t::template member_slice_type<i>&
                get_slice() {
                    return _Get<AoSoA_t, i>(slice_tuple);
                }

            // TODO: should this be a more heap based / dynamic allocation?
            // TODO: should we place these in the "target" memory space, not
            // the default?
            AoSoA_t internal_buffers[requested_buffer_count];

            BufferedAoSoA(AoSoA_t original_view_in) :
                original_view(original_view_in),
                buffer_size(max_buffered_tuples),
                slice_tuple(internal_buffers[0])
            {
                // TODO: this is only used internally now, and can likely be a
                // non-pointers
                last_filled_buffer = new int(1);
                *last_filled_buffer = -1;

                // TODO: We may we want to override the user on their requested
                // values if they don't make sense? Both num_buffers and buffer_size
                num_buffers = requested_buffer_count;

                // Resize the buffers so we know they can hold enough
                for (int i = 0; i < num_buffers; i++)
                {
                    internal_buffers[i].resize(buffer_size);
                }
            }

            /** @brief Helper to access the number of buffers which exist.
             * Makes no comment about the state or content of the buffers
             *
             * @return The number of total buffers available
             */
            int get_buffer_count()
            {
                return num_buffers;
            }

            /**
             * @brief Copy the data from the last buffer to the "real data", at
             * location start_index
             *
             * @param start_index Index to place buffer data into in original
             * aosoa
             */
            void copy_buffer_back(int to_index)
            {
                Cabana::deep_copy_partial_dst(
                    original_view,
                    internal_buffers[(*last_filled_buffer)],
                    to_index,
                    0, //from index
                    buffer_size
                );
            }

            // Start thinking about how to handle data movement...
            void load_next_buffer(int start_index)
            {
                (*last_filled_buffer)++;
                if ((*last_filled_buffer) >= num_buffers)
                {
                    (*last_filled_buffer) = 0;
                }

                // TODO: does this imply the need for a subview so the sizes
                // match?

                // Copy from the main memory store into the "current" buffer
                Cabana::deep_copy_partial_src(
                        internal_buffers[(*last_filled_buffer)],
                        original_view,
                        0, //to_index,
                        start_index,
                        buffer_size
                        );

                // TODO: delete debug print
                printf("Current buffer = %p \n", &internal_buffers[(*last_filled_buffer)] );

                // Update the slice tuple to have slices based on the current buffer
                slice_tuple = slice_tuple_t(internal_buffers[(*last_filled_buffer)]);
            }

            AoSoA_t& original_view;

            /**
             * @brief Track which buffer we "filled" last, so we know where we
             * are in the round robin
             */
            int* last_filled_buffer;

            /**
             * @brief Number of buffers we decided to use (possibly distinct
             * from how many the user asked for)
             *
             * I think it's important to make a distinction between how
             * many buffers the user thinks they need, and how many we decide
             * to use. This could get messy when passing around the templates
             * though, so likely this needs to become constexpr in nature
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
