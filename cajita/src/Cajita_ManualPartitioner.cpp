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

#include <Cajita_ManualPartitioner.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
ManualPartitioner::ManualPartitioner( const std::array<int, 3> &ranks_per_dim )
    : _ranks_per_dim( ranks_per_dim )
{
}

//---------------------------------------------------------------------------//
std::array<int, 3>
ManualPartitioner::ranksPerDimension( MPI_Comm,
                                      const std::array<int, 3> & ) const
{
    return _ranks_per_dim;
}

//---------------------------------------------------------------------------//

} // end namespace Cajita
