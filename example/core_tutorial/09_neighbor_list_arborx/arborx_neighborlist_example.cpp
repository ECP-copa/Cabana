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
// ArborX neighbor list example
//---------------------------------------------------------------------------//
void arborxNeighborListExample()
{
    /*
      ArborX is a Kokkos-based library for geometric search, included as an
      optional dependency in Cabana for building neighbor lists without a
      background acceleration grid, as with the Verlet list.

      This example follows very closely to the previous Verlet neighbor list
      example - more detail is included there.

      First we define the data types for the particles. We use the host space
      here for the purposes of this example but all memory spaces and other
      choices are configurable.
    */
    using DataTypes = Cabana::MemberTypes<double[3], int>;
    const int VectorLength = 8;
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Create the AoSoA.
    */
    int num_tuple = 81;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "A", num_tuple );

    /*
      Create the particle ids.
    */
    auto ids = Cabana::slice<1>( aosoa );
    for ( std::size_t i = 0; i < aosoa.size(); ++i )
        ids( i ) = i;

    /*
      Create the particle coordinates. We will put 3 particles in the center
      of each point on a regular grid. We will build the neighbor list such that
      each particle should only neighbor the other particles it shares a cell
      with.
    */
    auto positions = Cabana::slice<0>( aosoa );
    int ppc = 3;
    int particle_counter = 0;
    for ( int p = 0; p < ppc; ++p )
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                for ( int k = 0; k < 3; ++k, ++particle_counter )
                {
                    positions( particle_counter, 0 ) = ( 0.5 + i );
                    positions( particle_counter, 1 ) = ( 0.5 + j );
                    positions( particle_counter, 2 ) = ( 0.5 + k );
                }

    /*
      Create the neighbor list. Again, a candidate particle is only a
      neighbor if it is within the specified radius. Here we set the radius
      such that only particles that share a position will be neighbors.

      We also need to specify whether or not we want to build a full or half
      neighbor list and specify data layout as compressed sparse row (CSR) or
      2D (see the Verlet list example for more detail).

      We will build a full neighbor list with a CSR layout in this example. Note
      that to create a 2D neighbor list with ArborX, a separate function
      "make2DNeighborList" is provided.
     */
    double neighborhood_radius = 0.25;
    auto neighbor_list = Cabana::Experimental::makeNeighborList<DeviceType>(
        Cabana::FullNeighborTag{}, positions, 0, positions.size(),
        neighborhood_radius );

    /*
      Now we can get ArborX neighbor list data through the neighbor list
      interface - exactly as is done for the Verlet list - which is accessible
      on any device compatible with the memory space of the list. Each particle
      should have 2 neighbors.
     */
    using ListType = decltype( neighbor_list );
    for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
        int num_n =
            Cabana::NeighborList<ListType>::numNeighbor( neighbor_list, i );
        std::cout << "Particle " << i << " # neighbor = " << num_n << std::endl;
        for ( int j = 0; j < num_n; ++j )
            std::cout << "    neighbor " << j << " = "
                      << Cabana::NeighborList<ListType>::getNeighbor(
                             neighbor_list, i, j )
                      << std::endl;
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    arborxNeighborListExample();

    return 0;
}

//---------------------------------------------------------------------------//
