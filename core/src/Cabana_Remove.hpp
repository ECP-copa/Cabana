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
  \file Cabana_Remove.hpp
  \brief Remove particles (without using MPI)
*/
#ifndef CABANA_REMOVE_HPP
#define CABANA_REMOVE_HPP

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_Slice.hpp>

#include <Kokkos_Core.hpp>
namespace Cabana
{
//---------------------------------------------------------------------------//
/*!
  \brief Filter out empty/unneeded particles.
  \param exec_space Kokkos execution space.
  \param num_keep The total number of particles in the compaction section to
  keep.
  \param num_particles_ignore The number of particles to ignore (which precede
  those which may be kept/removed).
  \param remove_particle Boolean Kokkos View of particles to remove (true) or
  keep (false). \param particles The AoSoA containing particles. \param
  shrink_to_fit Whether to remove additional AoSoA capacity or not.
*/
template <class ExecutionSpace, class KeepView, class ParticleAoSoA>
void remove( const ExecutionSpace& exec_space, const int num_keep,
             const KeepView& remove_particle, ParticleAoSoA& particles,
             const int num_particles_ignore = 0,
             const bool shrink_to_fit = true )
{
    using memory_space = typename KeepView::memory_space;

    int num_particles = particles.size();
    int new_num_particles = num_particles_ignore + num_keep;

    // Determine the keep particle positions in the compaction zone.
    Kokkos::View<int*, memory_space> indices(
        Kokkos::ViewAllocateWithoutInitializing( "empty_or_filled" ),
        num_keep );

    // parallel_scan will break if not keeping any particles.
    if ( num_keep > 0 )
    {
        Kokkos::parallel_scan(
            "Cabana::remove::FindEmpty",
            Kokkos::RangePolicy<ExecutionSpace>(
                exec_space, num_particles_ignore, num_particles ),
            KOKKOS_LAMBDA( const int i, int& count, const bool final_pass ) {
                if ( !remove_particle( i ) )
                {
                    if ( final_pass )
                    {
                        indices( count ) = i;
                    }
                    ++count;
                }
            } );
        Kokkos::fence();

        // Compact the list so the it only has real particles.
        Kokkos::parallel_scan(
            "Cabana::remove::RemoveEmpty",
            Kokkos::RangePolicy<ExecutionSpace>(
                exec_space, num_particles_ignore, num_particles ),
            KOKKOS_LAMBDA( const int i, int& count, const bool final_pass ) {
                if ( !remove_particle( i ) )
                {
                    if ( final_pass )
                    {
                        particles.setTuple( indices( count ),
                                            particles.getTuple( i ) );
                    }
                    ++count;
                }
            } );
    }

    particles.resize( new_num_particles );
    if ( shrink_to_fit )
        particles.shrinkToFit();
}

} // end namespace Cabana

#endif // end CABANA_REMOVE_HPP
