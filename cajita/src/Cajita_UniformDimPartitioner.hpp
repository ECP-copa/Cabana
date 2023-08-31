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
  \file Cajita_UniformDimPartitioner.hpp
  \brief Uniform multi-node grid partitioner
*/
#ifndef CABANA_GRID_UNIFORMDIMPARTITIONER_HPP
#define CABANA_GRID_UNIFORMDIMPARTITIONER_HPP

#include <Cajita_Partitioner.hpp>

#include <array>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
/*!
  \brief Backwards compatibility wrapper for 3D DimBlockPartitioner
*/
class [[deprecated]] UniformDimPartitioner : public DimBlockPartitioner<3>
{
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CABANA_GRID_UNIFORMDIMPARTITIONER_HPP
