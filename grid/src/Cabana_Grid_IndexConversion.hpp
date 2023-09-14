/****************************************************************************
 * Copyright (c) 2018-2023 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file Cabana_Grid_IndexConversion.hpp
  \brief Local to global index conversion
*/
#ifndef CABANA_GRID_INDEXCONVERSION_HPP
#define CABANA_GRID_INDEXCONVERSION_HPP

#include <Cabana_Grid_IndexSpace.hpp>
#include <Cabana_Grid_Types.hpp>
#include <Cabana_Utils.hpp> // FIXME: remove after next release.

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Cabana
{
namespace Grid
{
namespace IndexConversion
{
//---------------------------------------------------------------------------//
//! Local-to-global indexer
template <class MeshType, class EntityType>
struct L2G
{
    //! Mesh type.
    using mesh_type = MeshType;

    //! Spatial dimension.
    static constexpr std::size_t num_space_dim = mesh_type::num_space_dim;

    //! Entity type.
    using entity_type = EntityType;

    //! Owned local indices minimum.
    Kokkos::Array<int, num_space_dim> local_own_min;

    //! Owned local indices maximum.
    Kokkos::Array<int, num_space_dim> local_own_max;

    //! Owned global indices minimum.
    Kokkos::Array<int, num_space_dim> global_own_min;

    //! Global number of entities.
    Kokkos::Array<int, num_space_dim> global_num_entity;

    //! Periodicity.
    Kokkos::Array<bool, num_space_dim> periodic;

    //! True if block is on low boundary.
    Kokkos::Array<bool, num_space_dim> boundary_lo;

    //! True if block is on high boundary.
    Kokkos::Array<bool, num_space_dim> boundary_hi;

    //! Constructor.
    L2G( const LocalGrid<MeshType>& local_grid )
    {
        // Local index set of owned entities.
        auto local_own_space =
            local_grid.indexSpace( Own(), EntityType(), Local() );

        // Get the local owned min.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            local_own_min[d] = local_own_space.min( d );

        // Get the local owned max.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            local_own_max[d] = local_own_space.max( d );

        // Global index set of owned entities.
        auto global_own_space =
            local_grid.indexSpace( Own(), EntityType(), Global() );

        // Get the global owned min.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            global_own_min[d] = global_own_space.min( d );

        // Get the global grid.
        const auto& global_grid = local_grid.globalGrid();

        // Global number of entities.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            global_num_entity[d] =
                global_grid.globalNumEntity( EntityType(), d );

        // Periodicity
        for ( std::size_t d = 0; d < num_space_dim; ++d )
            periodic[d] = global_grid.isPeriodic( d );

        // Determine if a block is on the low or high boundaries.
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            boundary_lo[d] = global_grid.onLowBoundary( d );
            boundary_hi[d] = global_grid.onHighBoundary( d );
        }
    }

    //! Convert local indices to global indices - general form.
    KOKKOS_INLINE_FUNCTION
    void operator()( const int lijk[num_space_dim],
                     int gijk[num_space_dim] ) const
    {
        for ( std::size_t d = 0; d < num_space_dim; ++d )
        {
            // Compute periodic wrap-around on low boundary.
            if ( periodic[d] && lijk[d] < local_own_min[d] && boundary_lo[d] )
            {
                gijk[d] = global_num_entity[d] - local_own_min[d] + lijk[d];
            }

            // Compute periodic wrap-around on high boundary.
            else if ( periodic[d] && local_own_max[d] <= lijk[d] &&
                      boundary_hi[d] )
            {
                gijk[d] = lijk[d] - local_own_max[d];
            }

            // Otherwise compute I indices as normal.
            else
            {
                gijk[d] = lijk[d] - local_own_min[d] + global_own_min[d];
            }
        }
    }

    //! Convert local indices to global indices - 3D.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<3 == NSD, void>
    operator()( const int li, const int lj, const int lk, int& gi, int& gj,
                int& gk ) const
    {
        int lijk[num_space_dim] = { li, lj, lk };
        int gijk[num_space_dim];
        this->operator()( lijk, gijk );
        gi = gijk[0];
        gj = gijk[1];
        gk = gijk[2];
    }

    //! Convert local indices to global indices - 3D.
    template <std::size_t NSD = num_space_dim>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<2 == NSD, void>
    operator()( const int li, const int lj, int& gi, int& gj ) const
    {
        int lijk[num_space_dim] = { li, lj };
        int gijk[num_space_dim];
        this->operator()( lijk, gijk );
        gi = gijk[0];
        gj = gijk[1];
    }
};

//---------------------------------------------------------------------------//
//! Creation function for local-to-global indexer.
//! \return L2G
template <class MeshType, class EntityType>
L2G<MeshType, EntityType> createL2G( const LocalGrid<MeshType>& local_grid,
                                     EntityType )
{
    return L2G<MeshType, EntityType>( local_grid );
}

//---------------------------------------------------------------------------//

} // end namespace IndexConversion
} // namespace Grid
} // namespace Cabana

namespace Cajita
{
//! \cond Deprecated
namespace IndexConversion
{
template <class MeshType, class EntityType>
using L2G CAJITA_DEPRECATED =
    Cabana::Grid::IndexConversion::L2G<EntityType, MeshType>;

template <class... Args>
CAJITA_DEPRECATED auto createL2G( Args&&... args )
{
    return Cabana::Grid::IndexConversion::createL2G(
        std::forward<Args>( args )... );
}
//! \endcond
} // namespace IndexConversion
} // namespace Cajita

#endif // end CABANA_GRID_INDEXCONVERSION_HPP
