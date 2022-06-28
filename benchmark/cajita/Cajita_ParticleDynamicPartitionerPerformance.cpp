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

#include "../Cabana_BenchmarkUtils.hpp"
#include "Cabana_ParticleInit.hpp"

#include <Cajita_ParticleDynamicPartitioner.hpp>
#include <Cajita_SparseIndexSpace.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

//---------------------------------------------------------------------------//
// Helper functions.
struct ParticleWorkloadTag
{
};

// generate average partitioner
std::array<std::vector<int>, 3> computeAveragePartition(
    const int tile_per_dim, const std::array<int, 3>& ranks_per_dim )
{
    std::array<std::vector<int>, 3> rec_partitions;
    for ( int d = 0; d < 3; ++d )
    {
        int ele = tile_per_dim / ranks_per_dim[d];
        int part = 0;
        for ( int i = 0; i < ranks_per_dim[d]; ++i )
        {
            rec_partitions[d].push_back( part );
            part += ele;
        }
        rec_partitions[d].push_back( tile_per_dim );
    }
    return rec_partitions;
}

//---------------------------------------------------------------------------//
// Performance test.
template <class Device>
void performanceTest( ParticleWorkloadTag, std::ostream& stream, MPI_Comm comm,
                      const std::string& test_prefix,
                      std::vector<int> problem_sizes,
                      std::vector<int> num_cells_per_dim )
{
    using memory_space = typename Device::memory_space;

    // Get comm rank;
    int comm_rank;
    MPI_Comm_rank( comm, &comm_rank );

    // Get comm size;
    int comm_size;
    MPI_Comm_size( comm, &comm_size );

    // Domain size setup
    std::array<float, 3> global_low_corner = { 0.0, 0.0, 0.0 };
    std::array<float, 3> global_high_corner = { 1.0, 1.0, 1.0 };
    constexpr int cell_num_per_tile_dim = 4;
    constexpr int cell_bits_per_tile_dim = 2;

    // Declare the total number of particles
    int num_problem_size = problem_sizes.size();

    // Declare the size (cell nums) of the domain
    int num_cells_per_dim_size = num_cells_per_dim.size();

    // Number of runs in the test loops.
    int num_run = 10;

    // Basic settings for partitioenr
    float max_workload_coeff = 1.5;
    int max_optimize_iteration = 10;
    int num_step_rebalance = 100;

    // compute the max number of particles handled by the current MPI rank
    int max_par_num = problem_sizes.back() / comm_size +
                      ( problem_sizes.back() % comm_size < comm_rank ? 1 : 0 );

    // Create random sets of particle positions.
    using position_type = Kokkos::View<float* [3], memory_space>;
    std::vector<position_type> positions( num_problem_size );
    for ( int p = 0; p < num_problem_size; ++p )
    {
        positions[p] = position_type(
            Kokkos::ViewAllocateWithoutInitializing( "positions" ),
            problem_sizes[p] );
        Cabana::createRandomParticles( positions[p], problem_sizes[p],
                                       global_low_corner[0],
                                       global_high_corner[0] );
    }

    for ( int c = 0; c < num_cells_per_dim_size; ++c )
    {
        // init the sparse grid domain
        std::array<int, 3> global_num_cell = {
            num_cells_per_dim[c], num_cells_per_dim[c], num_cells_per_dim[c] };
        int num_tiles_per_dim = num_cells_per_dim[c] >> cell_bits_per_tile_dim;

        // set up partitioner
        Cajita::ParticleDynamicPartitioner<Device, cell_num_per_tile_dim>
            partitioner( comm, max_workload_coeff, max_par_num,
                         num_step_rebalance, global_num_cell,
                         max_optimize_iteration );
        auto ranks_per_dim =
            partitioner.ranksPerDimension( comm, global_num_cell );
        auto ave_partition =
            computeAveragePartition( num_tiles_per_dim, ranks_per_dim );

        // Create insertion timers
        std::stringstream local_workload_name;
        local_workload_name << test_prefix << "compute_local_workload_"
                            << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer local_workload_timer(
            local_workload_name.str(), num_problem_size );

        std::stringstream prefix_sum_name;
        prefix_sum_name << test_prefix << "compute_prefix_sum_"
                        << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer prefix_sum_timer( prefix_sum_name.str(),
                                                   num_problem_size );

        std::stringstream total_optimize_name;
        total_optimize_name << test_prefix << "total_optimize_"
                            << "domain_size(cell)_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer total_optimize_timer(
            total_optimize_name.str(), num_problem_size );

        // loop over all the particle numbers
        for ( int p = 0; p < num_problem_size; ++p )
        {
            // compute the number of particles handled by the current MPI rank
            int par_num = problem_sizes[p] / comm_size +
                          ( problem_sizes[p] % comm_size < comm_rank ? 1 : 0 );

            auto pos_view = Kokkos::subview(
                positions[p], Kokkos::pair<int, int>( 0, par_num ),
                Kokkos::pair<int, int>( 0, 3 ) );

            // try for num_run times
            for ( int t = 0; t < num_run; ++t )
            {
                // ensure every optimization process starts from the same status
                partitioner.initializeRecPartition(
                    ave_partition[0], ave_partition[1], ave_partition[2] );

                // compute local workload
                local_workload_timer.start( p );
                partitioner.setLocalWorkloadByParticles(
                    pos_view, par_num, global_low_corner,
                    1.0f / num_cells_per_dim[c], comm );
                local_workload_timer.stop( p );

                // compute prefix sum matrix
                prefix_sum_timer.start( p );
                partitioner.computeFullPrefixSum( comm );
                prefix_sum_timer.stop( p );

                // optimization
                bool is_changed = false;
                // full timer
                total_optimize_timer.start( p );
                for ( int i = 0; i < max_optimize_iteration; ++i )
                {
                    partitioner.optimizePartitionAlongDim( std::rand() % 3,
                                                           is_changed );
                    if ( !is_changed )
                        break;
                }
                total_optimize_timer.stop( p );
            }
        }
        // Output results
        outputResults( stream, "insert_tile_num", problem_sizes,
                       local_workload_timer, comm );
        outputResults( stream, "insert_tile_num", problem_sizes,
                       prefix_sum_timer, comm );
        outputResults( stream, "insert_tile_num", problem_sizes,
                       total_optimize_timer, comm );
        stream << std::flush;
    }
}

//---------------------------------------------------------------------------//
// main
int main( int argc, char* argv[] )
{
    // Initialize environment
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    // Check arguments.
    if ( argc < 2 )
        throw std::runtime_error( "Incorrect number of arguments. \n \
             First argument - file name for output \n \
             Optional second argument - run size (small or large) \n \
             \n \
             Example: \n \
             $/: ./SparseMapPerformance test_results.txt\n" );

    // Define run sizes.
    std::string run_type = "";
    if ( argc > 2 )
        run_type = argv[2];
    std::vector<int> problem_sizes = { 1000, 10000 };
    std::vector<int> num_cells_per_dim = { 32, 64 };
    if ( run_type == "large" )
    {
        problem_sizes = { 1000, 10000, 100000, 1000000 };
        num_cells_per_dim = { 32, 64, 128, 256 };
    }
    std::vector<double> occupy_fraction = { 0.01, 0.1, 0.5, 0.75, 1.0 };

    // Get the name of the output file.
    std::string filename = argv[1];

    // Barier before continuing.
    MPI_Barrier( MPI_COMM_WORLD );

    // Get comm rank;
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    // Get comm size;
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    // Get Cartesian comm
    std::array<int, 3> ranks_per_dim;
    for ( std::size_t d = 0; d < 3; ++d )
        ranks_per_dim[d] = 0;
    MPI_Dims_create( comm_size, 3, ranks_per_dim.data() );

    // Open the output file on rank 0.
    std::fstream file;
    if ( 0 == comm_rank )
        file.open( filename, std::fstream::out );

    // Output problem details.
    if ( 0 == comm_rank )
    {
        file << "\n";
        file << "Cajita Sparse Partitioner Performance Benchmark"
             << "\n";
        file << "----------------------------------------------"
             << "\n";
        file << "MPI Ranks: " << comm_size << "\n";
        file << "MPI Cartesian Dim Ranks: (" << ranks_per_dim[0] << ", "
             << ranks_per_dim[1] << ", " << ranks_per_dim[2] << ")\n";
        file << "----------------------------------------------"
             << "\n";
        file << "\n";
        file << std::flush;
    }

    // Do everything on the default CPU.
    using host_exec_space = Kokkos::DefaultHostExecutionSpace;
    using host_device_type = host_exec_space::device_type;
    // Do everything on the default device with default memory.
    using exec_space = Kokkos::DefaultExecutionSpace;
    using device_type = exec_space::device_type;

    // Don't run twice on the CPU if only host enabled.
    // Don't rerun on the CPU if already done or if turned off.
    if ( !std::is_same<device_type, host_device_type>{} )
    {
        performanceTest<device_type>( ParticleWorkloadTag(), file,
                                      MPI_COMM_WORLD, "device_particleWL_",
                                      problem_sizes, num_cells_per_dim );
    }
    performanceTest<host_device_type>( ParticleWorkloadTag(), file,
                                       MPI_COMM_WORLD, "host_particleWL_",
                                       problem_sizes, num_cells_per_dim );

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
