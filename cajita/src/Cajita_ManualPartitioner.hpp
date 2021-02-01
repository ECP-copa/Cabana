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

#ifndef CAJITA_MANUALPARTITIONER_HPP
#define CAJITA_MANUALPARTITIONER_HPP

#include <Cajita_Partitioner.hpp>

#include <array>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
template <std::size_t NumSpaceDim>
class ManualPartitioner : public Partitioner<NumSpaceDim>
{
  public:
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    ManualPartitioner( const std::array<int, NumSpaceDim>& ranks_per_dim )
        : _ranks_per_dim( ranks_per_dim )
    {
    }

    std::array<int, NumSpaceDim>
    ranksPerDimension( MPI_Comm,
                       const std::array<int, NumSpaceDim>& ) const override
    {
        return _ranks_per_dim;
    }

  private:
    std::array<int, NumSpaceDim> _ranks_per_dim;
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_MANUALPARTITIONER_HPP
