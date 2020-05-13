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

#ifndef CABANA_BUFFEREDFOR_HPP
#define CABANA_BUFFEREDFOR_HPP

#include <Kokkos_Core.hpp>
#include <Cabana_Parallel.hpp> // simd_for and SimdPolicy

#include <cassert>

namespace Cabana
{

    // Requirements:
    // 1) This must be user callable and seamlessly handle the data buffering
    // 2) It must be able to accept 2D (simd) and 1D (Kokkos) loops/ range policies
        // a) Safely peel/remainder loops if needed for the 2D case?
    // 3) It should be able to warn in debug if you do something that we can detect as affecting performance, eg:
        // a) warn if vector length does not make sense for target execution space
    template<
        class BufferedAoSoA_t,
        class Slice_list_t,
        class FunctorType,
        class ... ExecParameters
    >
    inline void buffered_parallel_for(
            const Kokkos::RangePolicy<ExecParameters...>& exec_policy, // TODO: global or local range? => global?
            BufferedAoSoA_t buffered_aosoa, // TODO: passing this to a for is a bit odd? // TODO: does it need to be const?
            Slice_list_t slice_list,
            const FunctorType& functor,
            const std::string& str = "" )
    {
        // TODO: passing a kokkos range policy and then building a simd policy
        // doesn't make a whole lot of sense

        // TODO: stop mixing _type and _t like this
        constexpr int VectorLength = BufferedAoSoA_t::AoSoA_type::vector_length;
        using simd_policy = SimdPolicy<VectorLength,ExecParameters...>;
        using work_tag = typename simd_policy::work_tag;
        using team_policy = typename simd_policy::base_type;

        using index_type = typename team_policy::index_type;

        //Cabana::SimdPolicy<AoSoA_t::vector_length,TEST_EXECSPACE>
        int global_begin = exec_policy.begin();
        int global_end = exec_policy.end();
        int nelem = global_end-global_begin;

        // not ready for complex situations yet..
        assert(global_begin == 0);
        assert(global_end == buffered_aosoa.original_view.size() );

        // Calculate number buffer iterations needed to fit the data size
        int buffer_size =  buffered_aosoa.buffer_size;
        int niter = nelem / buffer_size;

        // TODO: delete
        std::cout << "running for " << niter << " buffered iterations " << std::endl;

        int begin = 0;
        int end = begin+buffer_size;

        for (int i = 0; i < niter; i++)
        {
            std::cout << "Looping from " << begin << " to " << end
                << " which is " << i*buffer_size << " in global space " <<
                std::endl;

            buffered_aosoa.load_next_buffer(begin);
            simd_policy policy(begin, end);
            Cabana::simd_parallel_for( policy, functor, str );

            Kokkos::fence();

            // copy all data back from localbuffer into the correct location in
            // global
            buffered_aosoa.copy_buffer_back(i*buffer_size);
        }
    }
} // namespace
#endif // CABANA_BUFFEREDFOR_HPP
