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

#ifndef CAJITA_UNIFORMDIMPARTITIONER_HPP
#define CAJITA_UNIFORMDIMPARTITIONER_HPP

#include <Cajita_Partitioner.hpp>

#include <array>

#include <mpi.h>

namespace Cajita
{
//---------------------------------------------------------------------------//
template <std::size_t NumSpaceDim>
class UniformDimPartitioner : public Partitioner<NumSpaceDim>
{
  public:
    static constexpr std::size_t num_space_dim = NumSpaceDim;

    std::array<int, NumSpaceDim>
    ranksPerDimension( MPI_Comm comm,
                       const std::array<int, NumSpaceDim>& ) const override
    {
        int comm_size;
        MPI_Comm_size( comm, &comm_size );

        std::array<int, NumSpaceDim> ranks_per_dim;
        for ( std::size_t d = 0; d < NumSpaceDim; ++d )
            ranks_per_dim[d] = 0;
        MPI_Dims_create( comm_size, NumSpaceDim, ranks_per_dim.data() );

        return ranks_per_dim;
    }
};

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_UNIFORMDIMPARTITIONER_HPP
