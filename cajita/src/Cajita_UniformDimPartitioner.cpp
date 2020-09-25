/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cajita_UniformDimPartitioner.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
std::array<int, 3>
UniformDimPartitioner::ranksPerDimension( MPI_Comm comm,
                                          const std::array<int, 3> & ) const
{
    int comm_size;
    MPI_Comm_size( comm, &comm_size );

    std::array<int, 3> ranks_per_dim = { 0, 0, 0 };
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    return ranks_per_dim;
}

//---------------------------------------------------------------------------//

} // end namespace Cajita
