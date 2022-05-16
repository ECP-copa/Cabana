/****************************************************************************
 * Copyright (c) 2018-2022 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/****************************************************************************
 * Copyright (c) 2022 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_AoSoA.hpp>
#include <Cabana_DeepCopy.hpp>
#include <Cabana_HDF5ParticleOutput.hpp>
#include <Cabana_ParticleInit.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <memory>

namespace Test
{

//---------------------------------------------------------------------------//
void writeTest()
{
    double low_corner = -2.8;
    double high_corner = 1.2;

    // Allocate particle properties.
    int num_particle = 100;
    using DataTypes = Cabana::MemberTypes<double[3],   // coords
                                          double[3],   // vec
                                          float[3][3], // matrix
                                          int>;        // id.
    Cabana::AoSoA<DataTypes, TEST_MEMSPACE> aosoa( "particles", num_particle );
    auto coords = Cabana::slice<0>( aosoa, "coords" );
    auto vec = Cabana::slice<1>( aosoa, "vec" );
    auto matrix = Cabana::slice<2>( aosoa, "matrix" );
    auto ids = Cabana::slice<3>( aosoa, "ids" );

    // Create random particles.
    Cabana::createRandomParticles( coords, num_particle, low_corner,
                                   high_corner );

    // Set other particle properties.
    auto aosoa_mirror =
        Cabana::create_mirror_view( Kokkos::HostSpace(), aosoa );
    auto coords_mirror = Cabana::slice<0>( aosoa_mirror, "coords" );
    auto vec_mirror = Cabana::slice<1>( aosoa_mirror, "vec" );
    auto matrix_mirror = Cabana::slice<2>( aosoa_mirror, "matrix" );
    auto ids_mirror = Cabana::slice<3>( aosoa_mirror, "ids" );
    for ( int p = 0; p < num_particle; ++p )
    {
        ids_mirror( p ) = p;

        for ( int d = 0; d < 3; ++d )
            vec_mirror( p, d ) = p * coords_mirror( p, d );

        for ( int d1 = 0; d1 < 3; ++d1 )
            for ( int d2 = 0; d2 < 3; ++d2 )
                matrix_mirror( p, d1, d2 ) = d1 * d2 / coords_mirror( p, d2 );
    }
    Cabana::deep_copy( aosoa, aosoa_mirror );

    Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;

    h5_config.collective = true;

    // Write a time step to file.
    double time = 7.64;
    double step = 892;
    Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
        h5_config, "particles", MPI_COMM_WORLD, step, time, coords.size(),
        coords, ids, matrix, vec );

    // FIXME: read data back in and compare.

    // Move the particles and write again.
    double time_step_size = 0.32;
    time += time_step_size;
    ++step;
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
            coords_mirror( p, d ) += 1.32;
    Cabana::deep_copy( coords, coords_mirror );
    Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
        h5_config, "particles", MPI_COMM_WORLD, step, time, coords.size(),
        coords, ids, matrix, vec );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, write_test ) { writeTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
