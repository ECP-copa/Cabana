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
  \file Cajita_SparseMapDynamicPartitioner.hpp
  \brief Multi-node sparse map based dynamic grid partitioner
*/
#ifndef CAJITA_SPARSEMAPDYNAMICPARTITIONER_HPP
#define CAJITA_SPARSEMAPDYNAMICPARTITIONER_HPP

#include <Cajita_DynamicPartitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <vector>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
/*!
  Dynamic mesh block partitioner. (Current Version: Support 3D only) Workload
  are computed from sparse map occupancy.

  \tparam Device Kokkos device type.
  \tparam CellPerTileDim Cells per tile per dimension.
  \tparam NumSpaceDim Dimemsion (The current version support 3D only)
*/
template <typename Device, unsigned long long CellPerTileDim = 4,
          std::size_t NumSpaceDim = 3>
class SparseMapDynamicPartitioner
    : public DynamicPartitioner<Device, CellPerTileDim, NumSpaceDim>
{
    using base = DynamicPartitioner<Device, CellPerTileDim, NumSpaceDim>;
    using base::base;

  protected:
    using base::_workload_per_tile;

  public:
    using base::cell_bits_per_tile_dim;
    using base::num_space_dim;
    using typename base::execution_space;

    /*!
      \brief compute the workload in the current MPI rank from sparseMap
      (the workload of a tile is 1 if the tile is occupied, 0 otherwise). This
      function must be called before running optimizePartition() \param
      sparseMap sparseMap in the current rank \param comm MPI communicator used
      for workload reduction
    */
    template <class SparseMapType>
    void setLocalWorkload( const SparseMapType& sparseMap, MPI_Comm comm )
    {
        base::resetWorkload();
        // make a local copy
        auto workload = _workload_per_tile;
        Kokkos::parallel_for(
            "compute_local_workload_sparsmap",
            Kokkos::RangePolicy<execution_space>( 0, sparseMap.capacity() ),
            KOKKOS_LAMBDA( uint32_t i ) {
                if ( sparseMap.valid_at( i ) )
                {
                    auto key = sparseMap.key_at( i );
                    int ti, tj, tk;
                    sparseMap.key2ijk( key, ti, tj, tk );
                    Kokkos::atomic_increment(
                        &workload( ti + 1, tj + 1, tk + 1 ) );
                }
            } );
        Kokkos::fence();
        // Wait for other ranks' workload to be ready
        MPI_Barrier( comm );
    }
};

} // end namespace Cajita

#endif // end CAJITA_SPARSEMAPDYNAMICPARTITIONER_HPP
