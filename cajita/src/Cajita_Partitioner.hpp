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

#ifndef CAJITA_PARTITIONER_HPP
#define CAJITA_PARTITIONER_HPP

#include <array>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
template <std::size_t NumSpaceDim>
class Partitioner
{
  public:
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    ~Partitioner() = default;

    /*!
      \brief Get the number of MPI ranks in each dimension of the grid.
      \param comm The communicator to use for the partitioning.
      \param global_cells_per_dim The number of global cells in each dimension.
      \return The number of MPI ranks in each dimension of the grid.
    */
    virtual std::array<int, NumSpaceDim> ranksPerDimension(
        MPI_Comm comm,
        const std::array<int, NumSpaceDim>& global_cells_per_dim ) const = 0;
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_PARTITIONER_HPP
