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

#include <Cabana_Core.hpp>

#include <iostream>

//---------------------------------------------------------------------------//
// Linked cell list.
//---------------------------------------------------------------------------//
void linkedCellListExample()
{
    /*
      A linked cell list is a data structure which bins particles based on the
      Cartesian grid cell in which they are located. An input uniform
      Cartesian grid description is used to locate which particles, defined by
      an input slice of 3D Cartesian coordinates, are located in which grid
      cells.

      The result of creating this data structure is binning data which may
      then be used to reorder the data in an AoSoA in identical fashion to the
      sorting and binning example we just completed. In effect, a linked cell
      list is simply a more sophisticated form of binning which uses a custom
      comparator for particles based on their spatial location relative to a
      uniform Cartesian grid.
    */

    std::cout << "Cabana Linked Cell List Example\n" << std::endl;

    /*
       Start by declaring the types in our tuples will store. The first
       member will be the coordinates, the second an id.
    */
    using DataTypes = Cabana::MemberTypes<double[3], int>;

    /*
      Next declare the data layout of the AoSoA. We use the host space here
      for the purposes of this example but all memory spaces, vector lengths,
      and member type configurations are compatible with sorting.
    */
    const int VectorLength = 8;
    using MemorySpace = Kokkos::HostSpace;

    /*
       Create the AoSoA.
    */
    int num_tuple = 54;
    Cabana::AoSoA<DataTypes, MemorySpace, VectorLength> aosoa( "A", num_tuple );

    /*
      Define the parameters of the Cartesian grid over which we will build the
      cell list. This is a simple 3x3x3 uniform grid on [0,3] in each
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
      Create the particle coordinates. We will put 2 particles in the center
      of each cell. We are ordering this such that each consecutive particle
      is in a different cell. When we use cell list to permute the particles
      later in the example, they will be regrouped by cell.
    */
    auto positions = Cabana::slice<0>( aosoa );
    int ppc = 2;
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
      Create the linked cell list for the AoSoA based on using the integer
      member as the sorting key. The linked cell list is a basic
      device-accessible data structure that describes how the data is to be
      reordered. For some algorithms, this data structure may be all that is
      necessary if the data does not have to be physically reordered.

      Note here that we are going to reorder the entire AoSoA. One may
      also construct a linked cell list over a subset of the particles. As an
      exercise, try adding both start and end (int) inputs to define a subset
      range of particles to permute:

      auto cell_list = Cabana::createLinkedCellList(
          positions, start, end, grid_delta, grid_min, grid_max );
     */
    auto cell_list = Cabana::createLinkedCellList( positions, grid_delta,
                                                   grid_min, grid_max );

    /*
      Now permute the AoSoA (i.e. reorder the data) using the linked cell list.
    */
    Cabana::permute( cell_list, aosoa );

    /*
       Now let's read the data we just binned using the linked cell
       list. Particles that are in the same cell should now be adjacent to
       each other.

       Note that even after the sorting has been completed the slices are
       still valid as long as we haven't resized, changes the capacity, or
       otherwise changed the memory associated with the AoSoA.
     */
    for ( std::size_t i = 0; i < aosoa.size(); ++i )
        std::cout << "Particle: id = " << ids( i ) << ", coords ("
                  << positions( i, 0 ) << "," << positions( i, 1 ) << ","
                  << positions( i, 2 ) << ")" << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    linkedCellListExample();

    return 0;
}

//---------------------------------------------------------------------------//
