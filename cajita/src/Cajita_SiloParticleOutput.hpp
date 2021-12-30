/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_SILOPARTICLEOUTPUT_HPP
#define CAJITA_SILOPARTICLEOUTPUT_HPP

#include <Cajita_GlobalGrid.hpp>

#include <Cabana_SiloParticleOutput.hpp>

#include <Kokkos_Core.hpp>

#include <silo.h>

#include <mpi.h>

#include <pmpio.h>

namespace Cajita
{
namespace SiloParticleOutput
{
// Write a time step.
template <class GlobalGridType, class CoordSliceType, class... FieldSliceTypes>
void writeTimeStep( const GlobalGridType& global_grid,
                    const int time_step_index, const double time,
                    const CoordSliceType& coords, FieldSliceTypes&&... fields )
{
    // Pick a number of groups. We want to write approximately the N^3 blocks
    // to N^2 groups. Pick the block dimension with the largest number of
    // ranks as the number of groups. We may want to tweak this as an optional
    // input later with this behavior as the default.
    int num_group = 0;
    for ( int d = 0; d < 3; ++d )
        if ( global_grid.dimNumBlock( d ) > num_group )
            num_group = global_grid.dimNumBlock( d );

    Cabana::SiloParticleOutput::writeTimeStep( global_grid.comm(), num_group,
                                               time_step_index, time, coords,
                                               fields... );
}
} // namespace SiloParticleOutput
} // namespace Cajita

#endif // CAJITA_SILOPARTICLEOUTPUT_HPP
