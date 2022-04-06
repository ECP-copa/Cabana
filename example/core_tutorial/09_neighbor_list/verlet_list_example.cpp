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
// Verlet list example
//---------------------------------------------------------------------------//
void verletListExample()
{
    /*
      Given a list of particle positions, for every particle in the list a
      Verlet list computes the other particles in the list that are within
      some specified cutoff distance from the particle. Once created, the
      Verlet list data can be accessed with the neighbor list interface. We
      will demonstrate building a Verlet list and accessing it's data in
      this example.
    */

    /*
       Start by declaring the types in our tuples will store. The first
       member will be the coordinates, the second an id.
    */
    using DataTypes = Cabana::MemberTypes<double[3], int>;

    /*
      Next declare the data layout of the AoSoA. We use the host space here
      for the purposes of this example but all memory spaces, vector lengths,
      and member type configurations are compatible with neighbor lists.
    */
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
    auto ids = Cabana::slice<1>( aosoa );
    for ( std::size_t i = 0; i < aosoa.size(); ++i )
        ids( i ) = i;

    /*
      Create the particle coordinates. We will put 3 particles in the center
      of each cell. We will set the Verlet list parameters such that each
      particle should only neighbor the other particles it shares a cell with.
    */
    auto positions = Cabana::slice<0>( aosoa );
    int ppc = 3;
    int particle_counter = 0;
    for ( int p = 0; p < ppc; ++p )
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                for ( int k = 0; k < 3; ++k, ++particle_counter )
                {
                    positions( particle_counter, 0 ) =
                        grid_min[0] + grid_delta[0] * ( 0.5 + i );
                    positions( particle_counter, 1 ) =
                        grid_min[1] + grid_delta[1] * ( 0.5 + j );
                    positions( particle_counter, 2 ) =
                        grid_min[2] + grid_delta[2] * ( 0.5 + k );
                }

    /*
      Create the Verlet list. We will only consider a particle to be a
      neighbor if it is within the specified radius. We will make this radius
      small enough so that only the particles that share cells will be found
      as neighbors.

      This variation of the Verlet list uses a background grid to accelerate
      the searching. The user must provide the minimum and maximum bounds of
      this grid and the ratio between the size of the cells in this grid and
      the neighborhood radius.

      This cell ratio parameter can be used to tweak performance. Larger cell
      ratios will give larger cells meaning more particles in each cell while
      a smaller cell ratio will give smaller cells and less particles in each
      cell.

      We also need to specify whether or not we want to build a full neighbor
      list or a half neighbor list.

          *: Full neighbor lists store all neighbors for all particles. This
             is specified by the Cabana::FullNeighborTag algorithm tag.

          *: Half neighbor lists only store a particle-particle neighbor pair
             once. If particle `i` is a neighbor of particle `j` then particle
             `i` will be listed as having 1 neighbor with index `j`. Particle
             `j` will be listed as having no neighbors as its relationship to
             `i` is implied. This is specified by the Cabana::HalfNeighborTag
             algorithm tag.

      In addition, we need to specify the layout of the neighhbor data,
      either as compressed sparse row (CSR) or 2D lists.

          *: CSR is specified by the Cabana::VerletLayoutCSR layout tag

          *: 2D is specified by the Cabana::VerletLayout2D layout tag

      We will build a full neighbor list with a CSR layout in this example.
      As an exercise, try changing the list algorithm tag to a half neighbor
      list and look at the difference in the output.

     */
    double neighborhood_radius = 0.25;
    double cell_ratio = 1.0;
    using ListAlgorithm = Cabana::FullNeighborTag;
    using ListType =
        Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayoutCSR,
                           Cabana::TeamOpTag>;
    ListType verlet_list( positions, 0, positions.size(), neighborhood_radius,
                          cell_ratio, grid_min, grid_max );

    /*
      Now lets get the Verlet list data using the neighbor list
      interface. This data is accessible on any device compatible with the
      memory space of the neighbor list. Each particle should have 2
      neighbors.
     */
    for ( std::size_t i = 0; i < aosoa.size(); ++i )
    {
        int num_n =
            Cabana::NeighborList<ListType>::numNeighbor( verlet_list, i );
        std::cout << "Particle " << i << " # neighbor = " << num_n << std::endl;
        for ( int j = 0; j < num_n; ++j )
            std::cout << "    neighbor " << j << " = "
                      << Cabana::NeighborList<ListType>::getNeighbor(
                             verlet_list, i, j )
                      << std::endl;
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    verletListExample();

    return 0;
}

//---------------------------------------------------------------------------//
