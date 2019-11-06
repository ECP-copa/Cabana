/****************************************************************************
 * Copyright (c) 2019 by the Cajita authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cajita library. Cajita is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef CAJITA_HPP
#define CAJITA_HPP

#include <Cajita_Array.hpp>
#include <Cajita_Block.hpp>
#include <Cajita_BovWriter.hpp>
#include <Cajita_Config.hpp>
#include <Cajita_GlobalGrid.hpp>
#include <Cajita_GlobalMesh.hpp>
#include <Cajita_Halo.hpp>
#include <Cajita_IndexSpace.hpp>
#include <Cajita_LocalMesh.hpp>
#include <Cajita_ManualPartitioner.hpp>
#include <Cajita_MpiTraits.hpp>
#include <Cajita_Partitioner.hpp>
#include <Cajita_Types.hpp>
#include <Cajita_UniformDimPartitioner.hpp>
#include <Cajita_Version.hpp>

#ifdef CAJITA_HAVE_HYPRE
#include <Cajita_StructuredSolver.hpp>
#endif

#endif // end CAJITA_HPP
