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

#include <Cajita_UniformDimPartitioner.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
std::vector<int>
UniformDimPartitioner::ranksPerDimension( MPI_Comm comm,
                                          const std::vector<int> & ) const
{
    int comm_size;
    MPI_Comm_size( comm, &comm_size );

    std::vector<int> ranks_per_dim( 3 );
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    return ranks_per_dim;
}

//---------------------------------------------------------------------------//

} // end namespace Cajita
