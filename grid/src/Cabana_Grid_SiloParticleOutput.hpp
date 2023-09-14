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
  \file Cabana_Grid_SiloParticleOutput.hpp
  \brief Write particle output using the Silo format.
*/

#ifndef CABANA_GRID_SILOPARTICLEOUTPUT_HPP
#define CABANA_GRID_SILOPARTICLEOUTPUT_HPP

#include <Cabana_Grid_GlobalGrid.hpp>

#include <Cabana_SiloParticleOutput.hpp>
#include <Cabana_Slice.hpp>
#include <Cabana_Utils.hpp> // FIXME: remove after next release.

#include <Kokkos_Core.hpp>

#include <silo.h>

#include <mpi.h>

#include <pmpio.h>

namespace Cabana
{
namespace Grid
{
namespace Experimental
{
namespace SiloParticleOutput
{
/*!
  \brief Write particle output in Silo format using mesh information.
  \param prefix Filename prefix.
  \param global_grid Cajita global grid.
  \param time_step_index Current simulation step index.
  \param time Current simulation time.
  \param begin The first particle index to output.
  \param end The final particle index to output.
  \param coords Particle coordinates.
  \param fields Variadic list of particle property fields.
*/
template <class GlobalGridType, class CoordSliceType, class... FieldSliceTypes>
void writePartialRangeTimeStep( const std::string& prefix,
                                const GlobalGridType& global_grid,
                                const int time_step_index, const double time,
                                const std::size_t begin, const std::size_t end,
                                const CoordSliceType& coords,
                                FieldSliceTypes&&... fields )
{
    // Pick a number of groups. We want to write approximately the N^3 blocks
    // to N^2 groups. Pick the block dimension with the largest number of
    // ranks as the number of groups. We may want to tweak this as an optional
    // input later with this behavior as the default.
    int num_group = 0;
    for ( int d = 0; d < 3; ++d )
        if ( global_grid.dimNumBlock( d ) > num_group )
            num_group = global_grid.dimNumBlock( d );

    Cabana::Experimental::SiloParticleOutput::writePartialRangeTimeStep(
        prefix, global_grid.comm(), num_group, time_step_index, time, begin,
        end, coords, fields... );
}

/*!
  \brief Write output in Silo format for all particles using mesh information.
  \param prefix Filename prefix.
  \param global_grid Cajita global grid.
  \param time_step_index Current simulation step index.
  \param time Current simulation time.
  \param coords Particle coordinates.
  \param fields Variadic list of particle property fields.
*/
template <class GlobalGridType, class CoordSliceType, class... FieldSliceTypes>
void writeTimeStep( const std::string& prefix,
                    const GlobalGridType& global_grid,
                    const int time_step_index, const double time,
                    const CoordSliceType& coords, FieldSliceTypes&&... fields )
{
    writePartialRangeTimeStep( prefix, global_grid, time_step_index, time, 0,
                               coords.size(), coords, fields... );
}

} // namespace SiloParticleOutput
} // namespace Experimental
} // namespace Grid
} // namespace Cabana

namespace Cajita
{
namespace Experimental
{
namespace SiloParticleOutput
{
//! \cond Deprecated
template <class... Args>
CAJITA_DEPRECATED void writePartialRangeTimeStep( Args&&... args )
{
    Cabana::Grid::Experimental::SiloParticleOutput::writePartialRangeTimeStep(
        std::forward<Args>( args )... );
}

template <class... Args>
CAJITA_DEPRECATED void writeTimeStep( Args&&... args )
{
    Cabana::Grid::Experimental::SiloParticleOutput::writeTimeStep(
        std::forward<Args>( args )... );
}
//! \endcond
} // namespace SiloParticleOutput
} // namespace Experimental
} // namespace Cajita

#endif // CABANA_GRID_SILOPARTICLEOUTPUT_HPP
