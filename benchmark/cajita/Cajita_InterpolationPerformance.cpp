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

#include "../Cabana_BenchmarkUtils.hpp"

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <mpi.h>

using namespace Cajita;

//---------------------------------------------------------------------------//
// Performance test.
template <class Device>
void performanceTest( std::ostream& stream, const std::string& test_prefix,
                      std::vector<int> cells_per_dim,
                      std::vector<int> particles_per_cell )
{
    using exec_space = typename Device::execution_space;
    using memory_space = typename Device::memory_space;

    // Domain size setup
    std::array<double, 3> global_low_corner = { 0.0, 0.0, 0.0 };
    std::array<double, 3> global_high_corner = { 1.0, 1.0, 1.0 };
    std::array<bool, 3> is_dim_periodic = { false, false, false };

    // System sizes
    int num_problem_size = cells_per_dim.size();
    int num_particles_per_cell = particles_per_cell.size();

    // Define the particle types.
    using member_types =
        Cabana::MemberTypes<double[3][3], double[3], double[3], double>;
    using aosoa_type = Cabana::AoSoA<member_types, Device>;

    // Define properties that do not depend on mesh size.
    Cajita::DimBlockPartitioner<3> partitioner;
    int halo_width = 1;
    uint64_t seed = 1938347;

    for ( int ppc = 0; ppc < num_particles_per_cell; ++ppc )
    {
        // Create p2g value timers.
        std::stringstream p2g_scalar_value_time_name;
        p2g_scalar_value_time_name << test_prefix << "p2g_scalar_value_"
                                   << particles_per_cell[ppc];
        Cabana::Benchmark::Timer p2g_scalar_value_timer(
            p2g_scalar_value_time_name.str(), num_problem_size );

        std::stringstream p2g_vector_value_time_name;
        p2g_vector_value_time_name << test_prefix << "p2g_vector_value_"
                                   << particles_per_cell[ppc];
        Cabana::Benchmark::Timer p2g_vector_value_timer(
            p2g_vector_value_time_name.str(), num_problem_size );

        // Create p2g gradient timers.
        std::stringstream p2g_scalar_gradient_time_name;
        p2g_scalar_gradient_time_name << test_prefix << "p2g_scalar_gradient_"
                                      << particles_per_cell[ppc];
        Cabana::Benchmark::Timer p2g_scalar_gradient_timer(
            p2g_scalar_gradient_time_name.str(), num_problem_size );

        // Create p2g divergence timers.
        std::stringstream p2g_vector_divergence_time_name;
        p2g_vector_divergence_time_name << test_prefix
                                        << "p2g_vector_divergence_"
                                        << particles_per_cell[ppc];
        Cabana::Benchmark::Timer p2g_vector_divergence_timer(
            p2g_vector_divergence_time_name.str(), num_problem_size );

        std::stringstream p2g_tensor_divergence_time_name;
        p2g_tensor_divergence_time_name << test_prefix
                                        << "p2g_tensor_divergence_"
                                        << particles_per_cell[ppc];
        Cabana::Benchmark::Timer p2g_tensor_divergence_timer(
            p2g_tensor_divergence_time_name.str(), num_problem_size );

        // Create g2p value timers.
        std::stringstream g2p_scalar_value_time_name;
        g2p_scalar_value_time_name << test_prefix << "g2p_scalar_value_"
                                   << particles_per_cell[ppc];
        Cabana::Benchmark::Timer g2p_scalar_value_timer(
            g2p_scalar_value_time_name.str(), num_problem_size );

        std::stringstream g2p_vector_value_time_name;
        g2p_vector_value_time_name << test_prefix << "g2p_vector_value_"
                                   << particles_per_cell[ppc];
        Cabana::Benchmark::Timer g2p_vector_value_timer(
            g2p_vector_value_time_name.str(), num_problem_size );

        // Create g2p gradient timers.
        std::stringstream g2p_scalar_gradient_time_name;
        g2p_scalar_gradient_time_name << test_prefix << "g2p_scalar_gradient_"
                                      << particles_per_cell[ppc];
        Cabana::Benchmark::Timer g2p_scalar_gradient_timer(
            g2p_scalar_gradient_time_name.str(), num_problem_size );

        std::stringstream g2p_vector_gradient_time_name;
        g2p_vector_gradient_time_name << test_prefix << "g2p_vector_gradient_"
                                      << particles_per_cell[ppc];
        Cabana::Benchmark::Timer g2p_vector_gradient_timer(
            g2p_vector_gradient_time_name.str(), num_problem_size );

        // Create g2p divergence timers.
        std::stringstream g2p_vector_divergence_time_name;
        g2p_vector_divergence_time_name << test_prefix
                                        << "g2p_vector_divergence_"
                                        << particles_per_cell[ppc];
        Cabana::Benchmark::Timer g2p_vector_divergence_timer(
            g2p_vector_divergence_time_name.str(), num_problem_size );

        for ( int n = 0; n < num_problem_size; ++n )
        {
            // Create the global grid
            double cell_size = 1.0 / cells_per_dim[n];
            auto global_mesh = createUniformGlobalMesh(
                global_low_corner, global_high_corner, cell_size );
            auto global_grid = createGlobalGrid( MPI_COMM_WORLD, global_mesh,
                                                 is_dim_periodic, partitioner );

            // Create a local grid and local mesh
            auto local_grid = createLocalGrid( global_grid, halo_width );
            auto local_mesh = createLocalMesh<exec_space>( *local_grid );
            auto owned_cells = local_grid->indexSpace( Own(), Cell(), Local() );
            int num_cells = owned_cells.size();

            // Create a random number generator.
            Kokkos::Random_XorShift64_Pool<exec_space> pool;
            pool.init( seed, num_cells );

            // Create the particles.
            int num_ppc = particles_per_cell[ppc];
            aosoa_type aosoa( "aosoa", num_ppc * num_cells );

            auto tensor = Cabana::slice<0>( aosoa, "tensor" );
            auto vector = Cabana::slice<1>( aosoa, "vector" );
            auto position = Cabana::slice<2>( aosoa, "position" );
            auto scalar = Cabana::slice<3>( aosoa, "scalar" );

            Cajita::grid_parallel_for(
                "particles_init", exec_space{}, *local_grid, Own(), Cell(),
                KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                    int i_own = i - owned_cells.min( Dim::I );
                    int j_own = j - owned_cells.min( Dim::J );
                    int k_own = k - owned_cells.min( Dim::K );
                    int cell_id =
                        i_own +
                        owned_cells.extent( Dim::I ) *
                            ( j_own + k_own * owned_cells.extent( Dim::J ) );

                    // Random number generator.
                    auto rand = pool.get_state( cell_id );

                    // Get the coordinates of the low cell node.
                    int low_node[3] = { i, j, k };
                    double low_coords[3];
                    local_mesh.coordinates( Cajita::Node(), low_node,
                                            low_coords );

                    // Get the coordinates of the high cell node.
                    int high_node[3] = { i + 1, j + 1, k + 1 };
                    double high_coords[3];
                    local_mesh.coordinates( Cajita::Node(), high_node,
                                            high_coords );

                    for ( int ip = 0; ip < num_ppc; ++ip )
                    {
                        // Local particle id.
                        int pid = cell_id * num_ppc + ip;

                        for ( int d = 0; d < 3; ++d )
                            position( pid, d ) =
                                Kokkos::rand<decltype( rand ), double>::draw(
                                    rand, low_coords[d], high_coords[d] );

                        scalar( pid ) = 0.5;
                        for ( int ii = 0; ii < 3; ++ii )
                        {
                            vector( pid, ii ) = 0.25;
                            for ( int jj = 0; jj < 3; ++jj )
                            {
                                tensor( pid, ii, jj ) = 0.3;
                            }
                        }
                    }
                } );

            // Now perform the p2g and g2p interpolations and time them.

            // Create a scalar field on the grid.
            auto scalar_layout = createArrayLayout( local_grid, 1, Node() );
            auto scalar_grid_field = createArray<double, memory_space>(
                "scalar_grid_field", scalar_layout );
            auto scalar_halo =
                createHalo( NodeHaloPattern<3>{}, -1, *scalar_grid_field );

            // Create a vector field on the grid.
            auto vector_layout = createArrayLayout( local_grid, 3, Node() );
            auto vector_grid_field = createArray<double, memory_space>(
                "vector_grid_field", vector_layout );
            auto vector_halo =
                createHalo( NodeHaloPattern<3>{}, -1, *vector_grid_field );

            // Create a tensor field on the grid
            auto tensor_layout = createArrayLayout( local_grid, 9, Node() );
            auto tensor_grid_field = createArray<double, memory_space>(
                "tensor_grid_field", tensor_layout );
            auto tensor_halo =
                createHalo( NodeHaloPattern<3>{}, -1, *tensor_grid_field );

            // Interpolate a scalar point value to the grid.
            ArrayOp::assign( *scalar_grid_field, 0.0, Ghost() );

            // P2G scalar value
            auto scalar_p2g = createScalarValueP2G( scalar, -0.5 );
            p2g_scalar_value_timer.start( n );
            p2g( scalar_p2g, position, position.size(), Spline<1>(),
                 *scalar_halo, *scalar_grid_field );
            p2g_scalar_value_timer.stop( n );

            // P2G vector value
            auto vector_p2g = createVectorValueP2G( vector, -0.5 );
            p2g_vector_value_timer.start( n );
            p2g( vector_p2g, position, position.size(), Spline<1>(),
                 *vector_halo, *vector_grid_field );
            p2g_vector_value_timer.stop( n );

            // P2G scalar gradient
            auto scalar_grad_p2g = createScalarGradientP2G( scalar, -0.5 );
            p2g_scalar_gradient_timer.start( n );
            p2g( scalar_grad_p2g, position, position.size(), Spline<1>(),
                 *vector_halo, *vector_grid_field );
            p2g_scalar_gradient_timer.stop( n );

            // P2G vector divergence
            auto vector_div_p2g = createVectorDivergenceP2G( vector, -0.5 );
            p2g_vector_divergence_timer.start( n );
            p2g( vector_div_p2g, position, position.size(), Spline<1>(),
                 *scalar_halo, *scalar_grid_field );
            p2g_vector_divergence_timer.stop( n );

            // P2G tensor divergence
            auto tensor_div_p2g = createTensorDivergenceP2G( tensor, -0.5 );
            p2g_tensor_divergence_timer.start( n );
            p2g( tensor_div_p2g, position, position.size(), Spline<1>(),
                 *vector_halo, *vector_grid_field );
            p2g_tensor_divergence_timer.stop( n );

            // G2P scalar value
            auto scalar_value_g2p = createScalarValueG2P( scalar, -0.5 );
            g2p_scalar_value_timer.start( n );
            g2p( *scalar_grid_field, *scalar_halo, position, position.size(),
                 Spline<1>(), scalar_value_g2p );
            g2p_scalar_value_timer.stop( n );

            // G2P vector value
            auto vector_value_g2p = createVectorValueG2P( vector, -0.5 );
            g2p_vector_value_timer.start( n );
            g2p( *vector_grid_field, *vector_halo, position, position.size(),
                 Spline<1>(), vector_value_g2p );
            g2p_vector_value_timer.stop( n );

            // G2P scalar gradient
            auto scalar_gradient_g2p = createScalarGradientG2P( vector, -0.5 );
            g2p_scalar_gradient_timer.start( n );
            g2p( *scalar_grid_field, *scalar_halo, position, position.size(),
                 Spline<1>(), scalar_gradient_g2p );
            g2p_scalar_gradient_timer.stop( n );

            // G2P vector gradient
            auto vector_gradient_g2p = createVectorGradientG2P( tensor, -0.5 );
            g2p_vector_gradient_timer.start( n );
            g2p( *vector_grid_field, *vector_halo, position, position.size(),
                 Spline<1>(), vector_gradient_g2p );
            g2p_vector_gradient_timer.stop( n );

            // G2P vector divergence
            auto vector_div_g2p = createVectorDivergenceG2P( scalar, -0.5 );
            g2p_vector_divergence_timer.start( n );
            g2p( *vector_grid_field, *vector_halo, position, position.size(),
                 Spline<1>(), vector_div_g2p );
            g2p_vector_divergence_timer.stop( n );
        }

        // Output results
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       p2g_scalar_value_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       p2g_vector_value_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       p2g_scalar_gradient_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       p2g_vector_divergence_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       p2g_tensor_divergence_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       g2p_scalar_value_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       g2p_vector_value_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       g2p_scalar_gradient_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       g2p_vector_gradient_timer );
        outputResults( stream, "grid_size_per_dim", cells_per_dim,
                       g2p_vector_divergence_timer );

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
             First argument -  file name for output \n \
             Optional second argument - run size (small or large) \n \
             \n \
             Example: \n \
             $/: ./InterpolationPerformance test_results.txt\n" );

    // Get the name of the output file.
    std::string filename = argv[1];

    // Define run sizes.
    std::string run_type = "";
    if ( argc > 2 )
        run_type = argv[2];
    std::vector<int> cells_per_dim = { 16, 32 };
    std::vector<int> particles_per_cell = { 1, 4 };
    if ( run_type == "large" )
    {
        cells_per_dim = { 16, 32, 64, 128 };
        particles_per_cell = { 1, 8, 16 };
    }
    // Open the output file on rank 0.
    std::fstream file;
    file.open( filename, std::fstream::out );

// Run the tests.
#ifdef KOKKOS_ENABLE_SERIAL
    using SerialDevice = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
    performanceTest<SerialDevice>( file, "serial_", cells_per_dim,
                                   particles_per_cell );
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    using OpenMPDevice = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
    performanceTest<OpenMPDevice>( file, "openmp_", cells_per_dim,
                                   particles_per_cell );
#endif

#ifdef KOKKOS_ENABLE_CUDA
    using CudaDevice = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
    performanceTest<CudaDevice>( file, "cuda_", cells_per_dim,
                                 particles_per_cell );
#endif

#ifdef KOKKOS_ENABLE_HIP
    using HipDevice = Kokkos::Device<Kokkos::Experimental::HIP,
                                     Kokkos::Experimental::HIPSpace>;
    performanceTest<HipDevice>( file, "hip_", cells_per_dim,
                                particles_per_cell );
#endif

    // Close the output file on rank 0.
    file.close();

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
