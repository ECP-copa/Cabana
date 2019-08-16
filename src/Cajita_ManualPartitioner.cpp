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

#include <Cajita_ManualPartitioner.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
ManualPartitioner::ManualPartitioner( const std::vector<int> &ranks_per_dim )
    : _ranks_per_dim( ranks_per_dim )
{
}

//---------------------------------------------------------------------------//
std::vector<int>
ManualPartitioner::ranksPerDimension( MPI_Comm, const std::vector<int> & ) const
{
    return _ranks_per_dim;
}

//---------------------------------------------------------------------------//

} // end namespace Cajita
