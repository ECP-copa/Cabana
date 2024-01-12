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

#include <Cabana_Core.hpp>

#include <iostream>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

// "h5fuse.sh" was changed to "h5fuse" in later HDF5 versions.
#if H5_VERSION_GE( 1, 14, 4 )
#define H5FUSE_EXEC "h5fuse"
#else
#define H5FUSE_EXEC "h5fuse.sh"
#endif

//---------------------------------------------------------------------------//
// HDF5 output example with subfiling
//---------------------------------------------------------------------------//
void hdf5OutputSubfiling()
{
    /*
      The previous HDF5 example shows how to write in parallel to a single file
      for all MPI ranks. In some cases, especially as the number of ranks grows,
      it is much more performant to write to separate files and later recombine
      into a final HDF5 file. This example is identical to the previous, but
      shows how to use HDF5 subfiling.
    */

    /*
       Get parameters from the communicator. We will use MPI_COMM_WORLD for
       this example but any MPI communicator may be used.
    */
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    if ( comm_rank == 0 )
        std::cout << "Cabana HDF5 subfiling output example\n" << std::endl;

    /*
      Start by declaring the types the particles will store. The first element
      will represent the coordinates, the second will be the particle's ID, the
      third velocity, and the fourth the radius of the particle.
    */
    using DataTypes = Cabana::MemberTypes<double[3], int, double[3], double>;

    /*
      Next declare the data layout of the AoSoA. We use the host space here
      for the purposes of this example but all memory spaces, vector lengths,
      and member type configurations are compatible.
    */
    const int VectorLength = 8;
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

    /*
      Create the AoSoA.
    */
    int num_particles = 9;
    Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> aosoa( "A",
                                                               num_particles );

    /*
      Get the particle ids, coordinates, velocity, radius
    */
    auto positions = Cabana::slice<0>( aosoa, "positions" );
    auto ids = Cabana::slice<1>( aosoa, "ids" );
    auto velocity = Cabana::slice<2>( aosoa, "velocity" );
    auto radius = Cabana::slice<3>( aosoa, "radius" );

    // initialize the particle properties
    for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
        // set ids of the particles
        ids( i ) = i;

        // set positions of the particles
        positions( i, 0 ) = 2. * i;
        positions( i, 1 ) = 0.;
        positions( i, 2 ) = 0.;

        // set the velocity of each particle
        velocity( i, 0 ) = 1.;
        velocity( i, 1 ) = 0.;
        velocity( i, 2 ) = 0.;

        // set the radius of each particle
        radius( i ) = 0.1;
    }

    // A configuration object is necessary for tuning HDF5 options.
    Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;

    // Configure HDF5 to use the subfiling VFD
    h5_config.subfiling = true;

    // Setting the HDF5 alignment equal to subfiling's stripe size
    // often achieves better performance
    //
    // Set the HDF5 alignment equal to subfiling's stripe size either by:
    //
    // (1) environment variable:
    //
    //     env_val = std::getenv("H5FD_SUBFILING_STRIPE_SIZE");
    //     if(env_val != NULL) {
    //        h5_config.align = true;
    //        h5_config.threshold = 0;
    //        h5_config.alignment = std::atoi(env_val);
    //     }
    //
    // or
    //
    // (2) in the source:

    h5_config.subfiling_stripe_size = 16 * 1024 * 124;
    h5_config.align = true;
    h5_config.threshold = 0;
    h5_config.alignment = h5_config.subfiling_stripe_size;

    /*
      We will evolve the system for a total time of 100 timesteps and
      output the properties at an interval of every 10 steps.
    */
    auto dt = 1e-2;
    auto final_time = 100. * dt;
    auto time = 0.;
    int steps = final_time / dt;
    int print_freq = 10;
    int nfork = 0;
    int shmrank;

    // Main timestep loop
    for ( int step = 0; step < steps; step++ )
    {
        // Update positions
        Kokkos::RangePolicy<ExecutionSpace> policy( 0, aosoa.size() );
        Kokkos::parallel_for(
            "move_particles", policy, KOKKOS_LAMBDA( const int i ) {
                positions( i, 0 ) += velocity( i, 0 ) * dt;
                positions( i, 1 ) += velocity( i, 1 ) * dt;
                positions( i, 2 ) += velocity( i, 2 ) * dt;
            } );
        Kokkos::fence();

        if ( step % print_freq == 0 )
        {
            /*
              Now we write all particle properties to HDF5 files, per node. See
              more detail in the previous HDF5 example.
            */
            Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
                h5_config, "particles", MPI_COMM_WORLD, step, time,
                num_particles, positions, ids, velocity, radius );

            if ( comm_rank == 0 )
                std::cout << "Output for step " << step << "/" << steps
                          << std::endl;

            // h5fuse is a tool for generating a hdf5 file from the subfiles.
            // h5fuse needs to be located in the same directory as the
            // executable.

            struct stat file_info;

            if ( stat( H5FUSE_EXEC, &file_info ) == 0 )
            {
                if ( h5_config.subfiling )
                {
                    if ( comm_rank == 0 )
                        std::cout << "Using HDF5 subfiling.\n" << std::endl;

                    MPI_Comm shmcomm;
                    MPI_Comm_split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                                         0, MPI_INFO_NULL, &shmcomm );

                    MPI_Comm_rank( shmcomm, &shmrank );

                    // One rank from each node executes h5fuse
                    if ( shmrank == 0 )
                    {
                        pid_t pid = 0;

                        pid = fork();
                        nfork++;
                        if ( pid == 0 )
                        {
                            std::stringstream filename_hdf5;
                            filename_hdf5 << "particles"
                                          << "_" << step << ".h5";

                            // Directory containing the subfiling configuration
                            // file
                            std::stringstream config_dir;
                            if ( const char* env_value = std::getenv(
                                     H5FD_SUBFILING_CONFIG_FILE_PREFIX ) )
                                config_dir << env_value;
                            else
                                config_dir << ".";

                            // Find the name of the subfiling configuration file

                            stat( filename_hdf5.str().c_str(), &file_info );

                            char config_filename[PATH_MAX];
                            snprintf(
                                config_filename, PATH_MAX,
                                "%s/" H5FD_SUBFILING_CONFIG_FILENAME_TEMPLATE,
                                config_dir.str().c_str(),
                                filename_hdf5.str().c_str(),
                                (uint64_t)file_info.st_ino );

                            // Call the h5fuse utility
                            // Removes the subfiles in the process
                            char* args[] = { strdup( H5FUSE_EXEC ),
                                             strdup( "-r" ), strdup( "-f" ),
                                             config_filename, NULL };

                            execvp( args[0], args );
                        }
                    }
                    else
                    {
                        if ( comm_rank == 0 )
                            std::cout << "HDF5 subfiling disabled.\n"
                                      << std::endl;
                    }
                    MPI_Comm_free( &shmcomm );
                }
            }
        }

        time += dt;
    }

    // Wait for all the h5fuse processes to complete
    if ( shmrank == 0 )
    {
        int status;
        for ( int i = 0; i < nfork; i++ )
        {
            waitpid( -1, &status, 0 );
            if ( WIFEXITED( status ) )
            {
                int ret;

                if ( ( ret = WEXITSTATUS( status ) ) != 0 )
                {
                    printf( "h5fuse process exited with error code %d\n", ret );
                    fflush( stdout );
                    MPI_Abort( MPI_COMM_WORLD, -1 );
                }
            }
            else
            {
                printf( "h5fuse process terminated abnormally\n" );
                fflush( stdout );
                MPI_Abort( MPI_COMM_WORLD, -1 );
            }
        }
    }

    /*
      The created HDF5 files with XMF metadata can be read by many visualization
      programs (once combined with h5fuse). This example can be run with any
      number of MPI ranks, for parallel particle output.
    */
}

int main( int argc, char* argv[] )
{
    // The HDF5 Subfiling VFD requires MPI_Init_thread with MPI_THREAD_MULTIPLE

    int mpi_thread_required = MPI_THREAD_MULTIPLE;
    int mpi_thread_provided = 0;

    MPI_Init_thread( &argc, &argv, mpi_thread_required, &mpi_thread_provided );
    if ( mpi_thread_provided < mpi_thread_required )
    {
        std::cout << "MPI_THREAD_MULTIPLE not supported" << std::endl;
        MPI_Abort( MPI_COMM_WORLD, -1 );
    }

    Kokkos::initialize( argc, argv );

    hdf5OutputSubfiling();

    Kokkos::finalize();

    MPI_Finalize();
    return 0;
}
