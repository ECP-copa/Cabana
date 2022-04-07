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

#include <Kokkos_Core.hpp>

#include <iostream>

//---------------------------------------------------------------------------//
// neighbor parallel_for example.
//---------------------------------------------------------------------------//
void neighborParallelForExample()
{
    /*
      In previous examples we have demonstrated how to create a particle
      neighbor list using Cabana. Now we present a more flexible strategy for
      threading over slices of particle properties and their neighbors using
      Cabana extensions of Kokkos concepts.

      Depending on the kernel to be used with the slice(s), different types
      of threading algorithms and indexing schemes will give better
      performance.

      The purpose of this example is to demonstrate how to efficiently and
      portably use different options in threaded parallel code for kernels
      with:

      1. Central particles with neighbors in a given radius

      2. Central particles, with neighbors in a given radius, and second
         neighbors (forming triplets of particles) within the same radius

      We demonstrate both cases in this example.
    */

    /*
       Start by declaring the types in our tuples will store. The first
       member will be the coordinates, the second and third counters.
    */
    using DataTypes = Cabana::MemberTypes<double[3], int, int>;

    /*
      Declare the rest of the AoSoA parameters.
    */
    const int VectorLength = 8;
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::OpenMP;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Create the AoSoA.
    */
    const int num_tuple = 81;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "my_aosoa",
                                                              num_tuple );

    /*
      Define the Cartesian grid and particle positions within it - this matches
      the previous VerletList example exactly (and is described in detail
      there).
    */
    double grid_min[3] = { 0.0, 0.0, 0.0 };
    double grid_max[3] = { 3.0, 3.0, 3.0 };
    double grid_delta[3] = { 1.0, 1.0, 1.0 };

    /*
    One might consider using a parallel_for loop in this case - especially when
    the code being written is for an arbitrary memory space.
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
      Create a separate value per particle to sum neighbor values into,
      initialized to zero, and a value per particle to sum, one.
    */
    auto slice_i = Cabana::slice<1>( aosoa );
    auto slice_n = Cabana::slice<2>( aosoa );
    Cabana::deep_copy( slice_i, 0 );
    Cabana::deep_copy( slice_n, 1 );

    /*
      Create the Verlet list - again this is described in detail in the
      VerletList example.
    */
    double neighborhood_radius = 0.25;
    double cell_ratio = 1.0;
    using ListAlgorithm = Cabana::FullNeighborTag;
    using ListType =
        Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayout2D>;
    ListType verlet_list( positions, 0, positions.size(), neighborhood_radius,
                          cell_ratio, grid_min, grid_max );

    /*
      KERNEL 1 - First neighbors

      This kernel is used with the neighbor list created above and forwards
      it, along with indexing and threading tags to an underlying
      Kokkos::parallel_for. This first kernel thus indexes directly
      over both the central particle i and neighbors j.

      Note the atomic update to ensure multiple neighbors do not update the
      central particle simultaneously if threading over neighbors.
     */
    auto first_neighbor_kernel = KOKKOS_LAMBDA( const int i, const int j )
    {
        Kokkos::atomic_add( &slice_i( i ), slice_n( j ) );
    };

    /*
      We define a standard Kokkos execution policy to use for our outer loop.
    */
    Kokkos::RangePolicy<ExecutionSpace> policy( 0, aosoa.size() );

    /*
      Finally, perform the parallel loop. This parallel_for concept
      in Cabana is complementary to existing parallel_for
      implementations in Kokkos. The serial tag indicates that the neighbor loop
      is serial, while the central particle loop uses threading.

      Notice that we do not have to directly interact with the neighbor list we
      created as in the previous VerletList example. This is instead done
      internally through the neighbor_parallel_for interface.

      Note: We fence after the kernel is completed for safety but this may not
      be needed depending on the memory/execution space being used. When the
      CUDA UVM memory space is used this fence is necessary to ensure
      completion of the kernel on the device before UVM data is accessed on
      the host. Not fencing in the case of using CUDA UVM will typically
      result in a bus error.
    */
    Cabana::neighbor_parallel_for( policy, first_neighbor_kernel, verlet_list,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(), "ex_1st_serial" );
    Kokkos::fence();

    /*
      Print out the results. Each value should be equal to the number of
      neighbors. Note that this should also match the VerletList test -
      two neighbors each.
    */
    std::cout << "Cabana::neighbor_parallel_for results (first, serial)"
              << std::endl;
    for ( std::size_t i = 0; i < slice_i.size(); i++ )
        std::cout << slice_i( i ) << " ";
    std::cout << std::endl << std::endl;

    /*
      We can instead thread both over particles and neighbors, simply by
      changing the tag. Internally this uses a Kokkos team policy.
     */
    Cabana::neighbor_parallel_for( policy, first_neighbor_kernel, verlet_list,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::TeamOpTag(), "ex_1st_team" );
    Kokkos::fence();

    /*
      Print out the results again. Each value should now be doubled.
    */
    std::cout << "Cabana::neighbor_parallel_for results (first, team)"
              << std::endl;
    for ( std::size_t i = 0; i < slice_i.size(); i++ )
        std::cout << slice_i( i ) << " ";
    std::cout << std::endl << std::endl;

    /*
      KERNEL 2 - Second neighbors

      Next we use a kernel that depends on triplets of particles - both the
      neighbors and those neighbor's neighbors are required - but is
      otherwise extremely similar.
    */

    // First reset the slice to zero
    Cabana::deep_copy( slice_i, 0 );

    /*
      We define a new kernel for triplet interactions, indexing over central
      particle i, neighbor j, and second neighbor k.

      Again, note the atomic update to ensure multiple neighbors do not update
      the central particle simultaneously if threading over neighbors.
    */
    auto second_neighbor_kernel =
        KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        Kokkos::atomic_add( &slice_i( i ),
                            ( slice_n( j ) + slice_n( k ) ) / 2 );
    };

    /*
      Then we're ready to perform the parallel loop, reusing the Kokkos range
      policy (because this only defines the outer loop over central particles)
      and the neighbor list.

      We pass the new kernel involving second neighbors, as well as a different
      indexing tag to indicate this. In this case, the serial tag indicates that
      both neighbor loops are serial, with threading only over central
      particles.
     */
    Cabana::neighbor_parallel_for( policy, second_neighbor_kernel, verlet_list,
                                   Cabana::SecondNeighborsTag(),
                                   Cabana::SerialOpTag(), "ex_2nd_serial" );
    Kokkos::fence();

    /*
      Print out the results. Each value should be equal to the number of
      triplets - this matches the VerletList test as well, since each pair of
      neighbors corresponds to one triplet of particles.
    */
    std::cout << "Cabana::neighbor_parallel_for results (second, serial)"
              << std::endl;
    for ( std::size_t i = 0; i < slice_i.size(); i++ )
        std::cout << slice_i( i ) << " ";
    std::cout << std::endl << std::endl;

    /*
      We can again instead thread both over particles and neighbors, simply by
      changing the tag. The Kokkos team policy is still over particles and first
      neighbors, but now also includes a serial loop over second neighbors.
    */
    Cabana::neighbor_parallel_for( policy, second_neighbor_kernel, verlet_list,
                                   Cabana::SecondNeighborsTag(),
                                   Cabana::TeamOpTag(), "ex_2nd_team" );
    Kokkos::fence();

    /*
      Print out the results again. Each value should now be doubled.
    */
    std::cout << "Cabana::neighbor_parallel_for results (second, team)"
              << std::endl;
    for ( std::size_t i = 0; i < slice_i.size(); i++ )
        std::cout << slice_i( i ) << " ";
    std::cout << std::endl << std::endl;

    /*
      One additional option is available which again maps to a Kokkos construct.
      Only the tag changes to internally use a Kokkos TeamVector policy to
      thread over all three levels of particles, first neighbors, and second
      neighbors.

      This is likely only useful in very specific cases, e.g. when the number of
      particles is very small such that exposing all parallelism outweighs the
      overheads.
     */
    Cabana::neighbor_parallel_for( policy, second_neighbor_kernel, verlet_list,
                                   Cabana::SecondNeighborsTag(),
                                   Cabana::TeamVectorOpTag(), "ex_2nd_vector" );
    Kokkos::fence();

    /*
      Print out the results one more time. Each value should now be tripled.
    */
    std::cout << "Cabana::neighbor_parallel_for results (second, team_vector)"
              << std::endl;
    for ( std::size_t i = 0; i < slice_i.size(); i++ )
        std::cout << slice_i( i ) << " ";
    std::cout << std::endl;
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    neighborParallelForExample();

    return 0;
}

//---------------------------------------------------------------------------//
