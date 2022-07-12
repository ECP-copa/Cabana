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
  \file Cajita_ParticleDynamicPartitioner.hpp
  \brief Multi-node particle based dynamic grid partitioner
*/
#ifndef CAJITA_PARTICLEDYNAMICPARTITIONER_HPP
#define CAJITA_PARTICLEDYNAMICPARTITIONER_HPP

#include <Cajita_DynamicPartitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <vector>

#include <mpi.h>

namespace Cajita
{

template <class ParticlePosViewType, typename ArrayType, typename CellUnit,
          unsigned long long CellPerTileDim, int num_space_dim, typename Device>
class ParticleWorkloadSetter : public WorkloadSetter<Device>
{
    using memory_space = typename Device::memory_space;
    using execution_space = typename Device::execution_space;

    static constexpr unsigned long long cell_bits_per_tile_dim =
        bitCount( CellPerTileDim );

    const ParticlePosViewType& view;
    int particle_num;
    const ArrayType& global_lower_corner;
    const CellUnit dx;
    MPI_Comm comm;

  public:
    ParticleWorkloadSetter( const ParticlePosViewType& view, int particle_num,
                            const ArrayType& global_lower_corner,
                            const CellUnit dx, MPI_Comm comm )
        : view( view )
        , particle_num( particle_num )
        , global_lower_corner( global_lower_corner )
        , dx( dx )
        , comm( comm )
    {
    }

    void run( Kokkos::View<int***, memory_space>& workload ) override
    {
        Kokkos::Array<CellUnit, num_space_dim> lower_corner;
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            lower_corner[d] = global_lower_corner[d];
        }

        Kokkos::parallel_for(
            "compute_local_workload_parpos",
            Kokkos::RangePolicy<execution_space>( 0, particle_num ),
            KOKKOS_LAMBDA( const int i ) {
                int ti = static_cast<int>(
                             ( view( i, 0 ) - lower_corner[0] ) / dx - 0.5 ) >>
                         cell_bits_per_tile_dim;
                int tj = static_cast<int>(
                             ( view( i, 1 ) - lower_corner[1] ) / dx - 0.5 ) >>
                         cell_bits_per_tile_dim;
                int tz = static_cast<int>(
                             ( view( i, 2 ) - lower_corner[2] ) / dx - 0.5 ) >>
                         cell_bits_per_tile_dim;
                Kokkos::atomic_increment( &workload( ti + 1, tj + 1, tz + 1 ) );
            } );
        Kokkos::fence();
        // Wait for other ranks' workload to be ready
        MPI_Barrier( comm );
    }
};

/*!
    \brief compute the workload in the current MPI rank from particle
    positions (each particle count for 1 workload value). This function must
    be called before running optimizePartition() \param view particle
    positions view \param particle_num total particle number \param
    global_lower_corner the coordinate of the domain global lower corner
    \param dx cell dx size
    \param comm MPI communicator used for workload reduction
*/
template <unsigned long long CellPerTileDim, int num_space_dim, typename Device,
          class ParticlePosViewType, typename ArrayType, typename CellUnit>
ParticleWorkloadSetter<ParticlePosViewType, ArrayType, CellUnit, CellPerTileDim,
                       num_space_dim, Device>
createParticleWorkloadSetter( const ParticlePosViewType& view, int particle_num,
                              const ArrayType& global_lower_corner,
                              const CellUnit dx, MPI_Comm comm )
{
    return ParticleWorkloadSetter<ParticlePosViewType, ArrayType, CellUnit,
                                  CellPerTileDim, num_space_dim, Device>(
        view, particle_num, global_lower_corner, dx, comm );
}

} // end namespace Cajita

#endif // end CAJITA_PARTICLEDYNAMICPARTITIONER_HPP
