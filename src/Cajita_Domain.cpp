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

#include <Cajita_Domain.hpp>

namespace Cajita
{
//---------------------------------------------------------------------------//
// Constructor.
Domain::Domain( const std::vector<double> &global_low_corner,
                const std::vector<double> &global_high_corner,
                const std::vector<bool> &periodic )
    : _global_low_corner( global_low_corner )
    , _global_high_corner( global_high_corner )
    , _periodic( periodic )
{
}

//---------------------------------------------------------------------------//
// Get the global low corner of the domain.
double Domain::lowCorner( const int dim ) const
{
    return _global_low_corner[dim];
}

//---------------------------------------------------------------------------//
// Get the global high corner of the domain.
double Domain::highCorner( const int dim ) const
{
    return _global_high_corner[dim];
}

//---------------------------------------------------------------------------//
// Get the extent of a given dimension.
double Domain::extent( const int dim ) const
{
    return _global_high_corner[dim] - _global_low_corner[dim];
}

//---------------------------------------------------------------------------//
// Get whether a given logical dimension is periodic.
bool Domain::isPeriodic( const int dim ) const { return _periodic[dim]; }

//---------------------------------------------------------------------------//
std::shared_ptr<Domain>
createDomain( const std::vector<double> &global_low_corner,
              const std::vector<double> &global_high_corner,
              const std::vector<bool> &periodic )
{
    return std::make_shared<Domain>( global_low_corner, global_high_corner,
                                     periodic );
}

//---------------------------------------------------------------------------//

} // end namespace Cajita
