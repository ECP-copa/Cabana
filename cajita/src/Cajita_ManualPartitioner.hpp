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
  \file Cajita_ManualPartitioner.hpp
  \brief Manual multi-node grid partitioner
*/
#ifndef CAJITA_MANUALPARTITIONER_HPP
#define CAJITA_MANUALPARTITIONER_HPP

#include <Cajita_Partitioner.hpp>

#include <array>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
/*!
  \brief Backwards compatibility wrapper for 3D ManualBlockPartitioner
*/
class [[deprecated]] ManualPartitioner : public ManualBlockPartitioner<3>
{
  public:
    /*!
      \brief Constructor
      \param ranks_per_dim MPI ranks per dimension.
    */
    ManualPartitioner( const std::array<int, 3>& ranks_per_dim )
        : ManualBlockPartitioner<3>( ranks_per_dim )
    {
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_MANUALPARTITIONER_HPP
