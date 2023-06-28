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

//---------------------------------------------------------------------------//
// HDF5 output example
//---------------------------------------------------------------------------//
void hdf5_output()
{
    /*
      In the current example, we will illustrate the process of storing a list
      of particles with properties, such as position, velocity, mass, radius,
      etc., in an hdf5 file format. This example is built up on previous
      examples such as, 04_aosoa example.
    */

    std::cout << "Cabana HDF5 output example\n" << std::endl;

    /*
      Start by declaring the types in our tuples will store. The first element
      will represent the coordinates, the second will be the particle's ID, the
      third will represent the velocity, and the fourth will represent the
      radius of the particle.
    */
    using DataTypes = Cabana::MemberTypes<double[3], int, double[3], double>;

    /*
      Next declare the data layout of the AoSoA. We use the host space here
      for the purposes of this example but all memory spaces, vector lengths,
      and member type configurations are compatible with neighbor lists.
    */
    const int VectorLength = 8;
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
      Create the AoSoA.
    */
    int num_tuple = 9;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "A", num_tuple );

    /*
      Define the parameters of the Cartesian grid over which we will build the
      particles. This is a simple 3x3x3 uniform grid on [0,3] in each
      direction. Each grid cell has a size of 1 in each dimension.
    */
    double grid_min[3] = { 0.0, 0.0, 0.0 };
    double grid_max[3] = { 3.0, 3.0, 3.0 };
    double grid_delta[3] = { 1.0, 1.0, 1.0 };

    /*
      Create the particle ids.
    */

    /*
      Get the particle ids, coordinates, velocity, radius
    */
    auto ids = Cabana::slice<1>( aosoa, "ids" );
    auto positions = Cabana::slice<0>( aosoa, "positions" );
    auto velocity = Cabana::slice<2>( aosoa, "velocity" );
    auto radius = Cabana::slice<3>( aosoa, "radius" );

    // initialize the particle properties
    for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {

        // set ids of the particles
        ids( i ) = i;

        // set positions of the particles
        positions( i, 0 ) = grid_min[0] + grid_delta[0] * ( 2. * i );
        positions( i, 1 ) = 0.;
        positions( i, 2 ) = 0.;

        // set the velocity of each particle
        velocity( i, 0 ) = 1.;
        velocity( i, 1 ) = 1.;
        velocity( i, 2 ) = 0.;

        // set the radius of each particle
        radius( i ) = 0.1;
    }

    /*
      We will evolve the system for a total time of 1000 timesteps. We will
      output the properties at an interval of every 100 steps.
    */
    auto dt = 1e-2;       // timestep
    auto tf = 1000. * dt; // total time
    auto t = 0.;          // time
    int nsteps = tf / dt; // total steps
    int pfreq = 100;      // print frequency
    // Main timestep loop
    for ( int step = 0; step < nsteps; step++ )
    {
        /*
          move positions
        */
        Kokkos::RangePolicy<ExecutionSpace> policy( 0, aosoa.size() );
        Kokkos::parallel_for(
            "move_particles", policy, KOKKOS_LAMBDA( const int i ) {
                positions( i, 0 ) += velocity( i, 0 ) * dt;
                positions( i, 1 ) += velocity( i, 0 ) * dt;
                positions( i, 2 ) += velocity( i, 0 ) * dt;
            } );
        Kokkos::fence();

        // This is for setting HDF5 options
        Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
        if ( step % pfreq == 0 )
        {

            /*
              Inorder to write the particle properties to an hdf5 files, we use

              Cabana::Experimental::HDF5ParticleOutput::writeTimeStep function.

              First seven arguments of `writeTimeStep` are mandotory arguments
              and rest are optional fields. The seven arguments are as follows


              1. config file
              2. name of the file
              3. MPI communication
              4. time step value (int)
              5. time of the simulation (double)
              6. number of particles (int)
              7. positions (double[3])
              8. Optional fields
              9. Optional fields
              10. ...

              Please look at the source for more details
              https://github.com/ECP-copa/Cabana/blob/master/core/src/Cabana_HDF5ParticleOutput.hpp#L496
            */
            Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
                h5_config,
                "particles",    // name of the file
                MPI_COMM_WORLD, //
                step,           // step
                t,              // time of the simulation
                num_tuple,      // no of points
                positions,      // positions of the points (double[3])
                ids,            // Optional fields
                velocity,       // Optional fields
                radius          // Optional fields
                // ...  // Optional fields
                // ...  // Optional fields
            );
        }

        t += dt;
    }
}

int main( int argc, char* argv[] )
{

    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    hdf5_output();

    Kokkos::finalize();

    MPI_Finalize();
    return 0;
}
