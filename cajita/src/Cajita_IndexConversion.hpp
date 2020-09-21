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

#ifndef CAJITA_INDEXCONVERSION_HPP
#define CAJITA_INDEXCONVERSION_HPP

#include <Cajita_IndexSpace.hpp>
#include <Cajita_Types.hpp>

#include <Kokkos_Core.hpp>

namespace Cajita
{
namespace IndexConversion
{
//---------------------------------------------------------------------------//
// Local-to-global indexer
template <class MeshType, class EntityType>
struct L2G
{
    // Mesh type
    using mesh_type = MeshType;

    // Entity type.
    using entity_type = EntityType;

    // Owned local indices minimum.
    Kokkos::Array<int, 3> local_own_min;

    // Owned local indices maximum.
    Kokkos::Array<int, 3> local_own_max;

    // Owned global indices minimum.
    Kokkos::Array<int, 3> global_own_min;

    // Global number of entities.
    Kokkos::Array<int, 3> global_num_entity;

    // Periodicity.
    Kokkos::Array<bool, 3> periodic;

    // True if block is on low boundary.
    Kokkos::Array<bool, 3> boundary_lo;

    // True if block is on high boundary.
    Kokkos::Array<bool, 3> boundary_hi;

    // Constructor.
    L2G( const LocalGrid<MeshType> & local_grid )
    {
        // Local index set of owned entities.
        auto local_own_space =
            local_grid.indexSpace( Own(), EntityType(), Local() );

        // Get the local owned min.
        for ( int d = 0; d < 3; ++d )
            local_own_min[d] = local_own_space.min( d );

        // Get the local owned max.
        for ( int d = 0; d < 3; ++d )
            local_own_max[d] = local_own_space.max( d );

        // Global index set of owned entities.
        auto global_own_space =
            local_grid.indexSpace( Own(), EntityType(), Global() );

        // Get the global owned min.
        for ( int d = 0; d < 3; ++d )
            global_own_min[d] = global_own_space.min( d );

        // Get the global grid.
        const auto & global_grid = local_grid.globalGrid();

        // Global number of entities.
        for ( int d = 0; d < 3; ++d )
            global_num_entity[d] =
                global_grid.globalNumEntity( EntityType(), d );

        // Periodicity
        for ( int d = 0; d < 3; ++d )
            periodic[d] = global_grid.isPeriodic( d );

        // Determine if a block is on the low or high boundaries.
        for ( int d = 0; d < 3; ++d )
        {
            auto block = global_grid.dimBlockId( d );
            boundary_lo[d] = ( 0 == block );
            boundary_hi[d] = ( global_grid.dimNumBlock( d ) - 1 == block );
        }
    }

    // Convert local indices to global indices.
    KOKKOS_INLINE_FUNCTION
    void operator()( const int li, const int lj, const int lk, int & gi,
                     int & gj, int & gk ) const
    {
        // I
        // Compute periodic wrap-around on low I boundary.
        if ( periodic[Dim::I] && li < local_own_min[Dim::I] &&
             boundary_lo[Dim::I] )
        {
            gi = global_num_entity[Dim::I] - local_own_min[Dim::I] + li;
        }

        // Compute periodic wrap-around on high I boundary.
        else if ( periodic[Dim::I] && local_own_max[Dim::I] <= li &&
                  boundary_hi[Dim::I] )
        {
            gi = li - local_own_max[Dim::I];
        }

        // Otherwise compute I indices as normal.
        else
        {
            gi = li - local_own_min[Dim::I] + global_own_min[Dim::I];
        }

        // J
        // Compute periodic wrap-around on low J boundary.
        if ( periodic[Dim::J] && lj < local_own_min[Dim::J] &&
             boundary_lo[Dim::J] )
        {
            gj = global_num_entity[Dim::J] - local_own_min[Dim::J] + lj;
        }

        // Compute periodic wrap-around on high J boundary.
        else if ( periodic[Dim::J] && local_own_max[Dim::J] <= lj &&
                  boundary_hi[Dim::J] )
        {
            gj = lj - local_own_max[Dim::J];
        }

        // Otherwise compute J indices as normal.
        else
        {
            gj = lj - local_own_min[Dim::J] + global_own_min[Dim::J];
        }

        // K
        // Compute periodic wrap-around on low K boundary.
        if ( periodic[Dim::K] && lk < local_own_min[Dim::K] &&
             boundary_lo[Dim::K] )
        {
            gk = global_num_entity[Dim::K] - local_own_min[Dim::K] + lk;
        }

        // Compute periodic wrap-around on high K boundary.
        else if ( periodic[Dim::K] && local_own_max[Dim::K] <= lk &&
                  boundary_hi[Dim::K] )
        {
            gk = lk - local_own_max[Dim::K];
        }

        // Otherwise compute K indices as normal.
        else
        {
            gk = lk - local_own_min[Dim::K] + global_own_min[Dim::K];
        }
    }
};

//---------------------------------------------------------------------------//
// Creation function.
template <class MeshType, class EntityType>
L2G<MeshType, EntityType> createL2G( const LocalGrid<MeshType> & local_grid,
                                     EntityType )
{
    return L2G<MeshType, EntityType>( local_grid );
}

//---------------------------------------------------------------------------//

} // end namespace IndexConversion
} // end namespace Cajita

#endif // end CAJITA_INDEXCONVERSION_HPP
