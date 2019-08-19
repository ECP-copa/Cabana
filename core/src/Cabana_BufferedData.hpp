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

#ifndef CABANA_DATABUFFER_HPP
#define CABANA_DATABUFFER_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>

namespace Cabana
{
    template<
        int max_buffered_tuples,
        int buffer_count,
        class Exec_Space,
        class AoSoA_t
    > class DataBuffer;

    /**
     * @brief Data buffer accessor that gives access to the current buffer
     */
    template<
        class AoSoA_t
    >
    class DataBufferAccessCurrent {
        // TODO: populate
        // TODO: this needs to know which buffer we're in to do correct offsetting

        public:
            // TODO: how complex can the return type be? Likely it's just a pointer part way into an AoSoA?
            using AoSoA_type = AoSoA_t;
            using value_type = typename AoSoA_type::soa_type;

            DataBufferAccessCurrent(const AoSoA_type& aosoa_in) : aosoa(aosoa_in)
            {
                // Intentionally empty
            }

            // TODO: Kokkos rebuilds this from the underlying params, eg:
            //  typedef ScatterView<DataType, Layout, ExecSpace, Op, ScatterNonDuplicated, contribution> view_type;
            template <typename ... Args>
            KOKKOS_FORCEINLINE_FUNCTION
            value_type operator()(Args ... args) {
                // TODO: this doesn't handle the offset
                return aosoa.at(args...);
            }
        private:
            AoSoA_type& aosoa;
    };

    // TODO: this can probably be deleted
    /**
     * @brief Data buffer accessor that gives access to some wrapped version of the DataBuffer
     */
    template<
        int max_buffered_tuples,
        int buffer_count,
        class Exec_Space,
        class AoSoA_t
    >
    class DataBufferAccess {
        // TODO: populate
        // TODO: this needs to know which buffer we're in to do correct offsetting

        public:
            // TODO: how complex can the return type be? Likely it's just a pointer part way into an AoSoA?
            using DataBuffer_t = DataBuffer<max_buffered_tuples, buffer_count, Exec_Space, AoSoA_t>;
            using AoSoA_type = AoSoA_t;
            using value_type = typename AoSoA_type::soa_type;

            DataBufferAccess(const DataBuffer_t& data_buffer_in) :
                data_buffer(data_buffer_in),
                aosoa(data_buffer_in.original_view) // TODO: we can likely remove this
            {
                // Intentionally empty
            }

            // TODO: Kokkos rebuilds this from the underlying params, eg:
            //  typedef ScatterView<DataType, Layout, ExecSpace, Op, ScatterNonDuplicated, contribution> view_type;
            template <typename ... Args>
            KOKKOS_FORCEINLINE_FUNCTION
            value_type operator()(Args ... args) {
                // TODO: this doesn't handle the offset
                return aosoa.at(args...);
            }
        private:
            const DataBuffer_t& data_buffer;
            AoSoA_type& aosoa;
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
        int requested_buffer_count, // TODO: this should probably be an optional hint, and not be prescriptive
        class Exec_Space, // TODO: we can possibly infer this somehow
        class AoSoA_t
    >
    class DataBuffer { // TODO: fix this backwards naming between class and file
        // const Kokkos::RangePolicy<ExecParameters...>& exec_policy,
        public:

            using AoSoA_type = AoSoA_t;

            // We may we want to override the user on this

            // TODO: private?
            // TODO: non-pointer may be preferable to pointer here but implies some extra copy construction? It's not too bad as we can just resize()
            // TODO: should this be a more heap based / dynamic allocation?
            // TODO: we should place these in the "target" memory space
            AoSoA_t internal_buffers[requested_buffer_count];

            // TODO: can we make this a nice auto detected unpack on the aosoa type?
            template<typename T1, typename T2>
            DataBuffer(Cabana::AoSoA<T1, T2> original_view_in) : original_view(original_view_in)
            {
                // TODO: we could probably do an implicit conversion here like // ScatterView does
                // return original_view

                int buffer_size = max_buffered_tuples;

                int num_buffers = requested_buffer_count;

                // Resize the buffers so we know they can hold enough
                for (int i = 0; i < num_buffers; i++)
                {
                    internal_buffers[i].resize(buffer_size);
                }
            }

            int get_buffer_count()
            {
                return num_buffers;
            }

            // Start thinking about how to handle data movement...
            void load_next_buffer()
            {
                last_filled_buffer++;
                if (last_filled_buffer >= num_buffers)
                {
                    last_filled_buffer = 0;
                }

                // Copy from the main memory store into the "current" buffer
                // TODO: does this imply the need for a subview so the sizes match?
                Cabana::deep_copy( internal_buffers[last_filled_buffer] , original_view );
            }

            // TODO: should access give you the "current" buffer?
            KOKKOS_FORCEINLINE_FUNCTION // TODO: check this is the right thing to do?
            DataBufferAccess<max_buffered_tuples, requested_buffer_count, Exec_Space, AoSoA_type> access() const
            {
                return DataBufferAccess<max_buffered_tuples, requested_buffer_count, Exec_Space, AoSoA_type>(*this);
            }

            // TODO: this or the above should probably be deleted
            KOKKOS_FORCEINLINE_FUNCTION // TODO: check this is the right thing to do?
            DataBufferAccess<max_buffered_tuples, requested_buffer_count, Exec_Space, AoSoA_type> access_buffer() const
            {
                return DataBufferAccessCurrent<AoSoA_type>(internal_buffers[last_filled_buffer] );
            }


            // TODO: make private once we figured the above DataBufferAccess
            // `operator()` out
            AoSoA_t& original_view;

            /**
             * @brief Track which buffer we "filled" last, so we know where we
             * are in the round robin
             */
            int last_filled_buffer = -1;

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

        private:
    };


    // Requirements:
    // 1) This must be user callable and seamlessly handle the data buffering
    // 2) It must be able to accept 2D (simd) and 1D (Kokkos) loops/ range policies
        // a) Safely peel/remainder loops if needed for the 2D case?
    // 3) It should be able to warn in debug if you do something that we can detect as affecting performance, eg:
        // a) warn if vector length does not make sense for target execution space
    class DataBufferDispatcher {
    };

} // end namespace Cabana

#endif // CABANA_DATABUFFER_HPP
