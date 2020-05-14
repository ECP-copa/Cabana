/****************************************************************************
 * Copyright (c) 2018-2020 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Core.hpp>
#include <Cabana_NeighborList.hpp>
#include <Cabana_Sort.hpp>
#include <Cabana_VerletList.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

// Performance test function.
void perfTest( const double cutoff_ratio, const std::size_t num_data,
               const double cell_size_ratio )
{
    // Print test data.
    std::cout << std::endl;
    std::cout << "Number of particles: " << num_data << std::endl;
    std::cout << "Cutoff distance to minimum distance ratio: " << cutoff_ratio
              << std::endl;
    std::cout << "Cell size to cutoff distance ratio: " << cell_size_ratio
              << std::endl;

    // Declare the neighbor list type.
    using NeighborListTag = Cabana::FullNeighborTag;
    using LayoutTag = Cabana::VerletLayoutCSR;
    using BuildTag = Cabana::TeamVectorOpTag;

    // Declare the execution and memory spaces.
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    // Declare the inner array layout.
    const int vector_length = 64;

    // Declare data types.
    using DataTypes = Cabana::MemberTypes<double[3]>; // Position

    // Enumerate the types for convenience.
    enum MyTypes
    {
        Position = 0
    };

    // Declare the AoSoA type.
    using AoSoA_t = Cabana::AoSoA<DataTypes, DeviceType, vector_length>;

    // Create an Array-of-Structs-of-Arrays.
    AoSoA_t aosoa( "aosoa", num_data );

    // Get the particle postions.
    auto x = Cabana::slice<Position>( aosoa, "position" );

    // Build particles.
    std::cout << std::endl;
    std::cout << "Building Neighbors " << std::endl;
    double min_dist = 1.0;
    double interaction_cutoff = cutoff_ratio * min_dist;
    double min_dist_sqr = min_dist * min_dist;
    double x_min = 0.0;
    double x_max = 1.3 * min_dist * std::pow( num_data, 1.0 / 3.0 );
    std::default_random_engine rand_gen;
    std::uniform_real_distribution<double> rand_dist( x_min, x_max );

    // Create some coarse bins to accelerate construction.
    int nbinx = 15;
    int nbin = nbinx * nbinx * nbinx;
    double bin_dx = ( x_max - x_min ) / nbinx;
    std::vector<std::vector<int>> bins( nbin );
    auto bin_id = [=]( const int i, const int j, const int k ) {
        return k * nbinx * nbinx + j * nbinx + i;
    };

    // Seed the distribution with a particle at the origin.
    x( 0, 0 ) = ( x_max - x_min ) / 2.0;
    x( 0, 1 ) = ( x_max - x_min ) / 2.0;
    x( 0, 2 ) = ( x_max - x_min ) / 2.0;
    int p0_i = std::floor( ( x( 0, 0 ) - x_min ) / bin_dx );
    int p0_j = std::floor( ( x( 0, 1 ) - x_min ) / bin_dx );
    int p0_k = std::floor( ( x( 0, 2 ) - x_min ) / bin_dx );
    bins[bin_id( p0_i, p0_j, p0_k )].push_back( 0 );

    // Create particles. Only add particles that are outside a minimum
    // distance from other particles.
    for ( std::size_t p = 1; p < num_data; ++p )
    {
        if ( 0 == ( p - 1 ) % ( num_data / 10 ) )
            std::cout << "Inserting " << p << " / " << num_data << std::endl;

        bool found_neighbor = true;

        // Keep trying new random coordinates until we insert one that is not
        // within the minimum distance of any other particle.
        while ( found_neighbor )
        {
            found_neighbor = false;

            // Create particle coordinates.
            x( p, 0 ) = rand_dist( rand_gen );
            x( p, 1 ) = rand_dist( rand_gen );
            x( p, 2 ) = rand_dist( rand_gen );

            // Figure out which bin we are in.
            int bin_i = std::floor( ( x( p, 0 ) - x_min ) / bin_dx );
            int bin_j = std::floor( ( x( p, 1 ) - x_min ) / bin_dx );
            int bin_k = std::floor( ( x( p, 2 ) - x_min ) / bin_dx );
            int i_min = ( 0 < bin_i ) ? bin_i - 1 : 0;
            int j_min = ( 0 < bin_j ) ? bin_j - 1 : 0;
            int k_min = ( 0 < bin_k ) ? bin_k - 1 : 0;
            int i_max = ( nbinx > bin_i ) ? bin_i + 1 : nbinx;
            int j_max = ( nbinx > bin_j ) ? bin_j + 1 : nbinx;
            int k_max = ( nbinx > bin_k ) ? bin_k + 1 : nbinx;

            // Search the adjacent bins for neighbors.
            for ( int i = i_min; i < i_max; ++i )
            {
                for ( int j = j_min; j < j_max; ++j )
                {
                    for ( int k = k_min; k < k_max; ++k )
                    {
                        for ( auto &n : bins[bin_id( i, j, k )] )
                        {
                            double dx = x( n, 0 ) - x( p, 0 );
                            double dy = x( n, 1 ) - x( p, 1 );
                            double dz = x( n, 2 ) - x( p, 2 );
                            double dist = dx * dx + dy * dy + dz * dz;

                            if ( dist < min_dist_sqr )
                            {
                                found_neighbor = true;
                                break;
                            }
                        }
                    }
                    if ( found_neighbor )
                        break;
                }
                if ( found_neighbor )
                    break;
            }

            // Add the particle to its bin if we didn't find any neighbors.
            if ( !found_neighbor )
                bins[bin_id( bin_i, bin_j, bin_k )].push_back( p );
        }
    }

    // Neighbor grid list params.
    double grid_min[3] = { x_min, x_min, x_min };
    double grid_max[3] = { x_max, x_max, x_max };

    // Sort the particles to make them more realistic in terms of what we
    // would expect in an MD simulation. They aren't going to be randomly
    // scattered about but rather will be periodically sorted for spatial
    // locality. Bin them in cells the size of the cutoff distance.
    double sort_delta[3] = { interaction_cutoff, interaction_cutoff,
                             interaction_cutoff };
    Cabana::LinkedCellList<DeviceType> linked_cell_list(
        Cabana::slice<Position>( aosoa, "position" ), sort_delta, grid_min,
        grid_max );
    Cabana::permute( linked_cell_list, aosoa );

    // Create the list once to get some statistics.
    Cabana::VerletList<DeviceType, NeighborListTag, LayoutTag, BuildTag>
        stat_list( Cabana::slice<Position>( aosoa, "position" ), 0,
                   aosoa.size(), interaction_cutoff, cell_size_ratio, grid_min,
                   grid_max );
    Kokkos::MinMaxScalar<int> result;
    Kokkos::MinMax<int> reducer( result );
    Kokkos::parallel_reduce(
        "Cabana::countMinMax",
        Kokkos::RangePolicy<ExecutionSpace>( 0, num_data ),
        Kokkos::Impl::min_max_functor<Kokkos::View<int *, DeviceType>>(
            stat_list._data.counts ),
        reducer );
    Kokkos::fence();
    std::cout << std::endl;
    std::cout << "List min neighbors: " << result.min_val << std::endl;
    std::cout << "List max neighbors: " << result.max_val << std::endl;
    double count_average = stat_list._data.neighbors.extent( 0 ) / num_data;
    std::cout << "List avg neighbors: " << count_average << std::endl;
    std::cout << std::endl;

    // Create the neighbor list.
    int num_create = 10;
    std::vector<double> times( num_create );
    for ( int t = 0; t < num_create; ++t )
    {
        std::cout << "Run t: " << t << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

#if defined( Cabana_ENABLE_ARBORX )
        auto const list = Cabana::Experimental::makeNeighborList<DeviceType>(
            Cabana::FullNeighborTag{},
            Cabana::slice<Position>( aosoa, "position" ), 0, aosoa.size(),
            interaction_cutoff );
#else

        Cabana::VerletList<DeviceType, NeighborListTag, LayoutTag, BuildTag>
            list( Cabana::slice<Position>( aosoa, "position" ), 0, aosoa.size(),
                  interaction_cutoff, cell_size_ratio, grid_min, grid_max );
#endif

        auto end_time = std::chrono::high_resolution_clock::now();

        auto elapsed_time = end_time - start_time;
        auto ms_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            elapsed_time );

        times[t] = ms_elapsed.count();
    }

    double avg = 0.0;
    for ( auto &t : times )
        avg += t;
    avg /= num_create;
    std::cout << "Average run time: " << avg << "ms" << std::endl;
    std::cout << std::endl;
}

int main( int argc, char *argv[] )
{
    // Minimum particle distance to particle interaction cutoff distance ratio.
    double cutoff_ratio = std::atof( argv[1] );

    // Number of particles.
    std::size_t num_data = std::atoi( argv[2] );

    // Ratio between the cell size and the cutoff.
    double cell_size_ratio = std::atof( argv[3] );

    // Initialize the kokkos runtime.
    Kokkos::ScopeGuard scope_guard( argc, argv );

    // Run the test.
    perfTest( cutoff_ratio, num_data, cell_size_ratio );

    return 0;
}
