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

#include "ewald.cpp"
#include "example_definitions.h"
#include "particles.cpp"

#include <iomanip>

int main( int argc, char **argv )
{
    int provided_thread_env;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_env );

    // check if MPI environment supports threaded execution
    assert( provided_thread_env == MPI_THREAD_FUNNELED );

    // Initialize the kokkos runtime.
    Kokkos::initialize( argc, argv );
    {
        // crystal size
        const int c_size = ( argc == 2 ) ? atoi( argv[1] ) : 32;
        // accuracy parameter for the tuning of Ewald
        const double accuracy = 1e-6;
        // width of unit cell (assume cube)
        const double width = c_size / 2.0;
        // Number of particles, 3D
        const int n_particles = c_size * c_size * c_size;

        // create a MPI cartesian communicator
        std::vector<int> dims( 3 );
        std::vector<int> periods( 3 );

        dims.at( 0 ) = dims.at( 1 ) = dims.at( 2 ) = 0;
        periods.at( 0 ) = periods.at( 1 ) = periods.at( 2 ) = 1;

        int n_ranks;
        MPI_Comm_size( MPI_COMM_WORLD, &n_ranks );

        MPI_Dims_create( n_ranks, 3, dims.data() );

        MPI_Comm cart_comm;
        MPI_Cart_create( MPI_COMM_WORLD, 3, dims.data(), periods.data(), 0,
                         &cart_comm );

        int rank;
        MPI_Comm_rank( cart_comm, &rank );

        std::vector<int> loc_coords( 3 );
        std::vector<int> cart_dims( 3 );
        std::vector<int> cart_periods( 3 );
        MPI_Cart_get( cart_comm, 3, cart_dims.data(), cart_periods.data(),
                      loc_coords.data() );

        // domain size
        Kokkos::View<double *> domain_size( "domain size", 3 );
        domain_size( 0 ) = width / (double)cart_dims.at( 0 );
        domain_size( 1 ) = width / (double)cart_dims.at( 1 );
        domain_size( 2 ) = width / (double)cart_dims.at( 2 );

        // domain width
        Kokkos::View<double *> domain_width( "domain parameters", 6 );
        domain_width( 0 ) = loc_coords.at( 0 ) * domain_size( 0 );
        domain_width( 1 ) = ( loc_coords.at( 0 ) + 1 ) * domain_size( 0 );
        domain_width( 2 ) = loc_coords.at( 1 ) * domain_size( 1 );
        domain_width( 3 ) = ( loc_coords.at( 1 ) + 1 ) * domain_size( 1 );
        domain_width( 4 ) = loc_coords.at( 2 ) * domain_size( 2 );
        domain_width( 5 ) = ( loc_coords.at( 2 ) + 1 ) * domain_size( 2 );

        // Create an empty list of all the particles
        ParticleList *particles = new ParticleList( 10 );

        if ( rank == 0 )
            std::cout << std::setprecision( 12 );

        // Initialize the particles
        // Currently particles are initialized as alternating charges
        // in uniform cubic grid pattern like NaCl
        initializeParticles( particles, c_size, cart_dims, domain_size,
                             loc_coords );

        // Create a Kokkos timer to measure performance
        Kokkos::Timer timer;

        // Create the solver and tune it for decent values of alpha and r_max
        TEwald solver( accuracy, n_particles, width, width, width, domain_width,
                       cart_comm );
        auto tune_time = timer.seconds();
        timer.reset();
        // Perform the computation of real and imag space energies
        double total_energy = solver.compute( *particles, width, width, width );

        // compute sum of forces (nicer with functor)
        double tfx, tfy, tfz;
        auto forces = Cabana::slice<Force>( *particles );
        Kokkos::parallel_reduce( particles->size(),
                                 KOKKOS_LAMBDA( int idx, double &f_tmp ) {
                                     f_tmp += forces( idx, 0 );
                                 },
                                 tfx );
        Kokkos::parallel_reduce( particles->size(),
                                 KOKKOS_LAMBDA( int idx, double &f_tmp ) {
                                     f_tmp += forces( idx, 1 );
                                 },
                                 tfy );
        Kokkos::parallel_reduce( particles->size(),
                                 KOKKOS_LAMBDA( int idx, double &f_tmp ) {
                                     f_tmp += forces( idx, 2 );
                                 },
                                 tfz );

        auto exec_time = timer.seconds();
        timer.reset();

        auto elapsed_time = tune_time + exec_time;

        // Print out the timings and accuracy
        if ( rank == 0 )
        {
            std::cout << "Time for initialization in Ewald Sum solver: "
                      << ( tune_time ) << " s." << std::endl;
            std::cout << "Time for computation in Ewald Sum solver:        "
                      << ( exec_time ) << " s." << std::endl;
            std::cout << "Total time spent in Ewald Sum solver:            "
                      << ( elapsed_time ) << " s." << std::endl;
            std::cout << "Total potential energy (known): "
                      << MADELUNG_NACL * n_particles << std::endl;
            std::cout << "total potential energy (Ewald Sum): " << total_energy
                      << std::endl;
            std::cout << "absolute error (energy): "
                      << ( n_particles * MADELUNG_NACL ) - total_energy
                      << std::endl;
            std::cout << "relative error (energy): "
                      << 1.0 - ( n_particles * MADELUNG_NACL ) / total_energy
                      << std::endl;
            std::cout << "sum of forces: " << tfx << " " << tfy << " " << tfz
                      << std::endl;
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
