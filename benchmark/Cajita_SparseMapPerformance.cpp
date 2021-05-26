/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "Cabana_BenchmarkUtils.hpp"

#include <Cajita_SparseIndexSpace.hpp>

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

//---------------------------------------------------------------------------//
// Performance test.
template <class Device>
void performanceTest( std::ostream& stream, const std::string& test_prefix )
{
    using exec_space = typename Device::execution_space;
    // Domain size setup
    std::array<float, 3> global_low_corner = { 0.0, 0.0, 0.0 };
    std::array<float, 3> global_high_corner = { 1.0, 1.0, 1.0 };
    constexpr int cell_num_per_tile_dim = 4;
    constexpr int cell_bits_per_tile_dim = 2;

    // Declare the total number of particles to be inserted.
    std::vector<int> num_particles = { 100, 1000, 10000, 1000000 };
    int num_particles_size = num_particles.size();

    // Declare the size (cell nums) of the domain
    std::vector<int> num_cells_per_dim = { 32, 64, 128, 256 };
    int num_cells_per_dim_size = num_cells_per_dim.size();

    // Generate a random set of particles in domain [0.0, 1.0]
    auto poses_host = Cabana::Benchmark::createRandomParticles<float>(
        num_particles.back(), global_low_corner, global_high_corner );
    auto poses = Kokkos::create_mirror_view_and_copy(
        typename Device::memory_space(), poses_host );

    // Number of runs in the test loops.
    int num_run = 10;

    // Some helper views
    int max_num_tiles_per_dim =
        num_cells_per_dim.back() / cell_num_per_tile_dim;
    Kokkos::View<uint64_t***, Device> tile_ids(
        Kokkos::ViewAllocateWithoutInitializing( "tile_ids" ),
        max_num_tiles_per_dim, max_num_tiles_per_dim, max_num_tiles_per_dim );
    Kokkos::View<bool***, Device> is_tile_valid(
        Kokkos::ViewAllocateWithoutInitializing( "is_tile_valid" ),
        max_num_tiles_per_dim, max_num_tiles_per_dim, max_num_tiles_per_dim );

    // Loop over different size of the domain
    for ( int c = 0; c < num_cells_per_dim_size; ++c )
    {
        std::array<int, 3> global_num_cell = {
            num_cells_per_dim[c], num_cells_per_dim[c], num_cells_per_dim[c] };
        auto global_mesh = Cajita::createSparseGlobalMesh(
            global_low_corner, global_high_corner, global_num_cell );
        float cell_size = 1.0 / num_cells_per_dim[c];

        // create sparse map
        int pre_alloc_size = num_cells_per_dim[c] * num_cells_per_dim[c];
        auto sis =
            Cajita::createSparseMap<exec_space>( global_mesh, pre_alloc_size );

        // Create insertion timers
        std::stringstream insert_time_name;
        insert_time_name << test_prefix << "cell_insert_"
                         << "domain_size_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer insert_timer( insert_time_name.str(),
                                               num_particles_size );

        std::stringstream query_time_name;
        query_time_name << test_prefix << "cell_query_"
                        << "domain_size_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer query_timer( query_time_name.str(),
                                              num_particles_size );

        std::stringstream valid_tile_ijk_time_name;
        valid_tile_ijk_time_name << test_prefix << "get_valid_tile_ijk_"
                                 << "domain_size_" << num_cells_per_dim[c];
        Cabana::Benchmark::Timer valid_tile_ijk_timer(
            valid_tile_ijk_time_name.str(), num_particles_size );

        for ( int p = 0; p < num_particles_size; ++p )
        {
            auto range = Kokkos::RangePolicy<exec_space>( 0, num_particles[p] );
            for ( int t = 0; t < num_run; ++t )
            {
                // init helper views
                Kokkos::deep_copy( tile_ids, 0 );
                Kokkos::deep_copy( is_tile_valid, 0 );
                Kokkos::parallel_for(
                    "label_valid_cells", range,
                    KOKKOS_LAMBDA( const int par_id ) {
                        int ti = static_cast<int>( poses( par_id, 0 ) /
                                                   cell_size ) >>
                                 cell_bits_per_tile_dim;
                        int tj = static_cast<int>( poses( par_id, 1 ) /
                                                   cell_size ) >>
                                 cell_bits_per_tile_dim;
                        int tk = static_cast<int>( poses( par_id, 2 ) /
                                                   cell_size ) >>
                                 cell_bits_per_tile_dim;
                        is_tile_valid( ti, tj, tk ) = true;
                    } );

                // insertion
                insert_timer.start( p );
                Kokkos::parallel_for(
                    "insert_cells_to_sparse_map", range,
                    KOKKOS_LAMBDA( const int par_id ) {
                        int ci =
                            static_cast<int>( poses( par_id, 0 ) / cell_size );
                        int cj =
                            static_cast<int>( poses( par_id, 1 ) / cell_size );
                        int ck =
                            static_cast<int>( poses( par_id, 2 ) / cell_size );
                        sis.insertCell( ci, cj, ck );
                    } );
                insert_timer.stop( p );

                // query
                query_timer.start( p );
                Kokkos::parallel_for(
                    "query_cell_ids_from_cell_ijk", range,
                    KOKKOS_LAMBDA( const int par_id ) {
                        int ci =
                            static_cast<int>( poses( par_id, 0 ) / cell_size );
                        int cj =
                            static_cast<int>( poses( par_id, 1 ) / cell_size );
                        int ck =
                            static_cast<int>( poses( par_id, 2 ) / cell_size );
                        tile_ids( ci >> cell_bits_per_tile_dim,
                                  cj >> cell_bits_per_tile_dim,
                                  ck >> cell_bits_per_tile_dim ) =
                            sis.queryTile( ci, cj, ck );
                    } );
                query_timer.stop( p );

                // get valid tile_ijk
                valid_tile_ijk_timer.start( p );
                Kokkos::parallel_for(
                    "compute_tile_ijk_from_tile_id",
                    Kokkos::RangePolicy<exec_space>( 0, sis.capacity() ),
                    KOKKOS_LAMBDA( const int index ) {
                        if ( sis.valid_at( index ) )
                        {
                            auto tileKey = sis.key_at( index );
                            int qti, qtj, qtk;
                            sis.key2ijk( tileKey, qti, qtj, qtk );
                        }
                    } );
                valid_tile_ijk_timer.stop( p );
            }
        }

        // Output results
        outputResults( stream, "particle_num", num_particles, insert_timer );
        outputResults( stream, "particle_num", num_particles, query_timer );
        outputResults( stream, "particle_num", num_particles,
                       valid_tile_ijk_timer );
        stream << std::flush;
    }
}

//---------------------------------------------------------------------------//
// main
int main( int argc, char* argv[] )
{
    // Initialize environment
    Kokkos::initialize( argc, argv );

    // Check arguments.
    if ( argc < 2 )
        throw std::runtime_error( "Incorrect number of arguments. \n \
             First argument - file name for output \n \
             \n \
             Example: \n \
             $/: ./SparseMapPerformance test_results.txt\n" );

    // Get the name of the output file.
    std::string filename = argv[1];

    // Open the output file on rank 0.
    std::fstream file;
    file.open( filename, std::fstream::out );

    // Run the tests.
#ifdef KOKKOS_ENABLE_SERIAL
    using SerialDevice = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
    performanceTest<SerialDevice>( file, "serial_" );
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    using OpenMPDevice = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
    performanceTest<OpenMPDevice>( file, "openmp_" );
#endif

#ifdef KOKKOS_ENABLE_CUDA
    using CudaDevice = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
    performanceTest<CudaDevice>( file, "cuda_" );
#endif

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    return 0;
}
