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
  \namespace Grid
  \brief Grid and particle-grid data structures and algorithms
*/
#ifndef CABANA_GRID_HPP
#define CABANA_GRID_HPP

#include <Cabana_Grid_Config.hpp>

#include <Cabana_Grid_Array.hpp>
#include <Cabana_Grid_BovWriter.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_GlobalParticleComm.hpp>
#include <Cabana_Grid_Halo.hpp>
#include <Cabana_Grid_IndexConversion.hpp>
#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Grid_Interpolation.hpp>
#include <Cabana_Grid_LocalGrid.hpp>
#include <Cabana_Grid_LocalMesh.hpp>
#include <Cabana_Grid_MpiTraits.hpp>
#include <Cabana_Grid_Parallel.hpp>
#include <Cabana_Grid_ParticleGridDistributor.hpp>
#include <Cabana_Grid_ParticleInit.hpp>
#include <Cabana_Grid_ParticleList.hpp>
#include <Cabana_Grid_Partitioner.hpp>
#include <Cabana_Grid_ReferenceStructuredSolver.hpp>
#ifndef KOKKOS_ENABLE_SYCL // FIXME_SYCL
#include <Cabana_Grid_SparseArray.hpp>
#include <Cabana_Grid_SparseDimPartitioner.hpp>
#include <Cabana_Grid_SparseHalo.hpp>
#include <Cabana_Grid_SparseIndexSpace.hpp>
#include <Cabana_Grid_SparseLocalGrid.hpp>
#endif
#include <Cabana_Grid_Splines.hpp>
#include <Cabana_Grid_Types.hpp>

#ifdef Cabana_ENABLE_HYPRE
#include <Cabana_Grid_Hypre.hpp>
#include <Cabana_Grid_HypreSemiStructuredSolver.hpp>
#include <Cabana_Grid_HypreStructuredSolver.hpp>
#endif

#ifdef Cabana_ENABLE_HEFFTE
#include <Cabana_Grid_FastFourierTransform.hpp>
#endif

#ifdef Cabana_ENABLE_SILO
#include <Cabana_Grid_SiloParticleOutput.hpp>
#endif

#ifdef Cabana_ENABLE_ALL
#include <Cabana_Grid_LoadBalancer.hpp>
#endif

#endif // end CABANA_GRID_HPP
