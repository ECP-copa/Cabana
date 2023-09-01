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

#include <Kokkos_Core.hpp>

#include <Cabana_Fields.hpp>
#include <Cabana_Grid_GlobalGrid.hpp>
#include <Cabana_Grid_GlobalMesh.hpp>
#include <Cabana_Grid_ParticleList.hpp>
#include <Cabana_Grid_Partitioner.hpp>

#include <../../core/unit_test/particle_list_unit_test.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, particle_test )
{
    // Create the global mesh.
    std::array<double, 3> low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> high_corner = { -0.3, 9.5, 2.3 };
    double cell_size = 0.05;
    auto global_mesh = Cabana::Grid::createUniformGlobalMesh(
        low_corner, high_corner, cell_size );

    // Create the global grid.
    Cabana::Grid::DimBlockPartitioner<3> partitioner;
    std::array<bool, 3> is_dim_periodic = { true, true, true };
    auto global_grid = Cabana::Grid::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, is_dim_periodic, partitioner );

    // Create a local mesh.
    auto local_grid = Cabana::Grid::createLocalGrid( global_grid, 1 );

    Cabana::ParticleTraits<Cabana::Field::Position<3>, Foo, CommRank, Bar>
        fields;
    auto plist = Cabana::Grid::createParticleList<TEST_MEMSPACE>(
        "test_particles", fields );

    particleListTest( plist );

    particleViewTest( plist );

    // Use explicit field tag even though it could be done internally.
    plist.redistribute( *local_grid, Cabana::Field::Position<3>() );
}

//---------------------------------------------------------------------------//

} // end namespace Test
