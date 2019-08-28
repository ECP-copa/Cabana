/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

// These initialize particles for a NaCl example structure

#include "definitions.h"
#include "mpi.h"

// function to initialize particles as a NaCl crystal of length crystal_size
void initializeParticles( ParticleList particles, int crystal_size )
{
    auto x = Cabana::slice<Position>( particles );
    auto v = Cabana::slice<Velocity>( particles );
    auto f = Cabana::slice<Force>( particles );
    auto q = Cabana::slice<Charge>( particles );
    auto u = Cabana::slice<Potential>( particles );
    auto i = Cabana::slice<Index>( particles );
    auto init_parts = KOKKOS_LAMBDA( const int idx )
    {
        // Calculate location of particle in crystal
        int idx_x = idx % crystal_size;
        int idx_y = ( idx / crystal_size ) % crystal_size;
        int idx_z = idx / ( crystal_size * crystal_size );

        // Initialize position.
        x( idx, 0 ) = ( (double)idx_x * 0.5 );
        x( idx, 1 ) = ( (double)idx_y * 0.5 );
        x( idx, 2 ) = ( (double)idx_z * 0.5 );

        // Initialize velocity.
        for ( int d = 0; d < SPACE_DIM; ++d )
            v( idx, d ) = 0.0;

        // Initialize field
        for ( int d = 0; d < SPACE_DIM; ++d )
            f( idx, d ) = 0.0;

        // Create alternating charge
        q( idx ) = ( ( ( idx_x + idx_y + idx_z ) % 2 ) ? 1.0 : -1.0 ) *
                   COULOMB_PREFACTOR_INV;

        // Set potential
        u( idx ) = 0.0;

        // Set global particle index
        i( idx ) = idx + 1l;
    };
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>( 0, particles.size() ),
        init_parts );
    Kokkos::fence();
}

// function to initialize particles as a NaCl crystal of length crystal_size
// MPI parallel version
void initializeParticles( ParticleList *particles, int c_size,
                          Kokkos::View<int *, MemorySpace> glob_dims,
                          Kokkos::View<double *, MemorySpace> domain_size,
                          Kokkos::View<int *, MemorySpace> loc_coords )
{
    // calculate the number of particles on the local domain
    Kokkos::View<int *, MemorySpace> indices( "indices for particle creation",
                                              6 );
    Kokkos::View<int *, MemorySpace> loc_edges( "indices for particle creation",
                                                3 );
    int n_particles = 1;

    for ( int dim = 0; dim < 3; ++dim )
    {
        indices( 2 * dim ) =
            (int)std::ceil( loc_coords( dim ) * domain_size( dim ) / 0.5 );
        indices( 2 * dim + 1 ) = (int)std::floor( ( loc_coords( dim ) + 1 ) *
                                                  domain_size( dim ) / 0.5 );
        if ( glob_dims( dim ) == loc_coords( dim ) + 1 ||
             fabs( ( loc_coords( dim ) + 1 ) * domain_size( dim ) / 0.5 -
                   (int)( ( loc_coords( dim ) + 1 ) * domain_size( dim ) /
                          0.5 ) ) < 1e-6 )
            indices( 2 * dim + 1 ) = indices( 2 * dim + 1 ) - 1;
        loc_edges( dim ) = indices( 2 * dim + 1 ) - indices( 2 * dim ) + 1;
        n_particles *= loc_edges( dim );
    }

    // debug output to check is there is any
    // error in the creation of the partial
    // chunks on each process
    /*
    std::cout << "local coords: " <<
                    loc_coords(0) << " " <<
                    loc_coords(1) << " " <<
                    loc_coords(2) << " " <<
                 "domain size: " <<
                    domain_size(0) << " " <<
                    domain_size(1) << " " <<
                    domain_size(2) << " " <<
                 "indices: " <<
                    indices(0) << " " <<
                    indices(1) << " " <<
                    indices(2) << " " <<
                    indices(3) << " " <<
                    indices(4) << " " <<
                    indices(5) << " " <<
                 "local edges: " <<
                    loc_edges(0) << " " <<
                    loc_edges(1) << " " <<
                    loc_edges(2) << " " <<
                 std::endl;
    */

    particles->resize( n_particles );

    auto x = Cabana::slice<Position>( *particles );
    auto v = Cabana::slice<Velocity>( *particles );
    auto f = Cabana::slice<Force>( *particles );
    auto q = Cabana::slice<Charge>( *particles );
    auto u = Cabana::slice<Potential>( *particles );
    auto i = Cabana::slice<Index>( *particles );
    auto init_parts = KOKKOS_LAMBDA( const int idx )
    {
        // Calculate location of particle in crystal
        int idx_z = idx % loc_edges( 2 ) + indices( 4 );
        int idx_y = ( idx / loc_edges( 2 ) ) % loc_edges( 1 ) + indices( 2 );
        int idx_x = idx / ( loc_edges( 1 ) * loc_edges( 2 ) ) + indices( 0 );

        // Initialize position.
        x( idx, 0 ) = ( (double)idx_x * 0.5 );
        x( idx, 1 ) = ( (double)idx_y * 0.5 );
        x( idx, 2 ) = ( (double)idx_z * 0.5 );

        // Initialize velocity.
        for ( int d = 0; d < SPACE_DIM; ++d )
            v( idx, d ) = 0.0;

        // Initialize field
        for ( int d = 0; d < SPACE_DIM; ++d )
            f( idx, d ) = 0.0;

        // Create alternating charge
        q( idx ) = ( ( ( idx_x + idx_y + idx_z ) % 2 ) ? 1.0 : -1.0 ) *
                   COULOMB_PREFACTOR_INV;

        // Set potential
        u( idx ) = 0.0;

        // Set global particle index
        i( idx ) = idx_z + idx_y * c_size + idx_x * c_size * c_size;
    };
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>( 0, particles->size() ),
        init_parts );
    Kokkos::fence();
}

// function to initialize uniform cubic mesh
void initializeMesh( ParticleList mesh, int width )
{
    int ptsx = std::round( std::pow(
        mesh.size(), 1.0 / 3.0 ) ); // number of points in each dimension
    auto x = Cabana::slice<Position>( mesh );
    auto v = Cabana::slice<Velocity>( mesh );
    auto f = Cabana::slice<Force>( mesh );
    auto q = Cabana::slice<Charge>( mesh );
    auto u = Cabana::slice<Potential>( mesh );
    auto i = Cabana::slice<Index>( mesh );
    auto init_mesh = KOKKOS_LAMBDA( const int idx )
    {
        // Calculate location of particle in crystal
        int idx_x = idx % ptsx;
        int idx_y = ( idx / ptsx ) % ptsx;
        int idx_z = idx / ( ptsx * ptsx );

        // Initialize position.
        x( idx, 0 ) = ( (double)idx_x * width / ( ptsx ) );
        x( idx, 1 ) = ( (double)idx_y * width / ( ptsx ) );
        x( idx, 2 ) = ( (double)idx_z * width / ( ptsx ) );

        // Initialize velocity.
        for ( int d = 0; d < SPACE_DIM; ++d )
            v( idx, d ) = 0.0;

        // Initialize field
        for ( int d = 0; d < SPACE_DIM; ++d )
            f( idx, d ) = 0.0;

        // Create charge = 0
        q( idx ) =
            0.0; //(((idx_x + idx_y + idx_z)%2)?1.0:-1.0)*COULOMB_PREFACTOR_INV;

        // Set potential
        u( idx ) = 0.0;

        // Set global particle index
        i( idx ) = idx + 1l;
    };
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, mesh.size() ),
                          init_mesh );
    Kokkos::fence();
}
