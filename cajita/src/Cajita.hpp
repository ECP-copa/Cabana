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
  \namespace Cajita
  \brief Cajita: grid and particle-grid data structures and algorithms
*/
#ifndef CAJITA_HPP
#define CAJITA_HPP

#include <Cajita_Config.hpp>

#include <Cajita_Array.hpp>
#include <Cajita_BovWriter.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_Halo.hpp>
#include <Cajita_IndexConversion.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_Interpolation.hpp>
#include <Cajita_LocalGrid.hpp>
#include <Cajita_LocalMesh.hpp>
#include <Cajita_ManualPartitioner.hpp>
#include <Cajita_MpiTraits.hpp>
#include <Cajita_Parallel.hpp>
#include <Cajita_ParticleGridDistributor.hpp>
#include <Cajita_Partitioner.hpp>
#include <Cajita_ReferenceStructuredSolver.hpp>
#include <Cajita_DynamicPartitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>
#include <Cajita_Splines.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>

#ifdef Cabana_ENABLE_HYPRE
#include <Cajita_HypreStructuredSolver.hpp>
#endif

#ifdef Cabana_ENABLE_HEFFTE
#include <Cajita_FastFourierTransform.hpp>
#endif

#ifdef Cabana_ENABLE_SILO
#include <Cajita_SiloParticleOutput.hpp>
#endif

#endif // end CAJITA_HPP
