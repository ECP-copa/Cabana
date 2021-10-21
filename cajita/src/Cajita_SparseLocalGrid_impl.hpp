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

#ifndef CAJITA_SPARSELOCALGRID_IMPL_HPP
#define CAJITA_SPARSELOCALGRID_IMPL_HPP

namespace Cajita
{

//---------------------------------------------------------------------------//
// Constructor
template <class Scalar, std::size_t NumSpaceDim>
LocalGrid<SparseMesh<Scalar, NumSpaceDim>>::LocalGrid(
    const std::shared_ptr<GlobalGrid<SparseMesh<Scalar, NumSpaceDim>>>&
        global_grid,
    const int halo_cell_width )
    : _global_grid( global_grid )
    , _halo_cell_width( halo_cell_width )
{
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_SPARSELOCALGRID_IMPL_HPP
