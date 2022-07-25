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

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <ratio>
#include <string>
#include <vector>

#include <mpi.h>

//---------------------------------------------------------------------------//
// Performance test.
// Data device type is where the data to be communicated lives.
// Comm device type is the device we want to use for communication.
template <class DataDevice, class CommDevice>
void performanceTest( std::ostream& stream, const std::size_t num_particle,
                      const std::string& test_prefix,
                      std::vector<double> comm_fraction )
{
    // PROBLEM SETUP
    // -------------

    // Get comm world.
    auto comm = MPI_COMM_WORLD;

    // Get comm size;
    int comm_size;
    MPI_Comm_size( comm, &comm_size );

    // Partition the problem in 3 dimensions.
    const int space_dim = 3;
    std::vector<int> ranks_per_dim( space_dim );
    MPI_Dims_create( comm_size, space_dim, ranks_per_dim.data() );

    // Generate a communicator with a cartesian topology and periodic
    // boundaries.
    std::vector<int> periodic_dims( space_dim, 1 );
    MPI_Comm cart_comm;
    int reorder_cart_ranks = 1;
    MPI_Cart_create( comm, space_dim, ranks_per_dim.data(),
                     periodic_dims.data(), reorder_cart_ranks, &cart_comm );

    // Get the Cartesian topology index of this rank.
    int linear_rank;
    MPI_Comm_rank( cart_comm, &linear_rank );
    std::vector<int> cart_rank( space_dim );
    MPI_Cart_coords( cart_comm, linear_rank, space_dim, cart_rank.data() );

    // Compute the 27 ranks this rank communicates with. This includes self
    // communication. Put self communication first.
    std::vector<int> neighbor_ranks;
    neighbor_ranks.push_back( linear_rank );
    for ( int k = -1; k < 2; ++k )
        for ( int j = -1; j < 2; ++j )
            for ( int i = -1; i < 2; ++i )
                if ( !( i == 0 && j == 0 && k == 0 ) )
                {
                    std::vector<int> ncr = { cart_rank[0] + i, cart_rank[1] + j,
                                             cart_rank[2] + k };
                    int nr;
                    MPI_Cart_rank( cart_comm, ncr.data(), &nr );
                    neighbor_ranks.push_back( nr );
                }

    // Make a unique list of neighbors that will serve as the communication
    // topology.
    std::vector<int> unique_neighbors( neighbor_ranks );
    std::sort( unique_neighbors.begin(), unique_neighbors.end() );
    auto unique_end =
        std::unique( unique_neighbors.begin(), unique_neighbors.end() );
    unique_neighbors.resize(
        std::distance( unique_neighbors.begin(), unique_end ) );

    // Define the aosoa.
    using comm_memory_space = typename CommDevice::memory_space;
    using data_memory_space = typename DataDevice::memory_space;
    using member_types = Cabana::MemberTypes<double[3], double[3], double, int>;
    using aosoa_type = Cabana::AoSoA<member_types, data_memory_space>;

    // Get the byte size of a particle
    using tuple_type = Cabana::Tuple<member_types>;
    int bytes_per_particle = sizeof( tuple_type );

    // Number of runs in the test loops.
    int num_run = 10;

    // Fraction of particles on each rank that will be communicated to the
    // neighbors. We will sweep through these fractions to get an indicator of
    // performance as a function of message size.
    int num_fraction = comm_fraction.size();

    // Number of bytes we will send to each neighbor.
    std::vector<std::size_t> comm_bytes( num_fraction );

    // DISTRIBUTOR
    // -----------

    // Create distributor timers.
    Cabana::Benchmark::Timer distributor_fast_create(
        test_prefix + "distributor_fast_create", num_fraction );
    Cabana::Benchmark::Timer distributor_general_create(
        test_prefix + "distributor_general_create", num_fraction );
    Cabana::Benchmark::Timer distributor_aosoa_migrate(
        test_prefix + "distributor_aosoa_migrate", num_fraction );
    Cabana::Benchmark::Timer distributor_slice_migrate(
        test_prefix + "distributor_slice_migrate", num_fraction );

    // Loop over comm fractions.
    for ( int fraction = 0; fraction < num_fraction; ++fraction )
    {
        // Create the migrate distribution in the data memory space. This is
        // where it would likely be created. Note below that we divide by 26
        // and then multiply by 26 to get a number of sends via integer math
        // that is evenly divisible by the number of neighbors. This
        // guarantees that the same number of particles is sent to each
        // neighbor so we can rule out load imbalance in our timings.
        int num_send = num_particle * comm_fraction[fraction];
        int send_per_neighbor = num_send / 26;
        num_send = send_per_neighbor * 26;
        int num_stay = num_particle - num_send;
        comm_bytes[fraction] = send_per_neighbor * bytes_per_particle;
        Kokkos::View<int*, Kokkos::HostSpace> export_ranks_host( "export_ranks",
                                                                 num_particle );
        for ( int p = 0; p < num_stay; ++p )
        {
            export_ranks_host( p ) = neighbor_ranks[0];
        }
        for ( int n = 0; n < 26; ++n )
        {
            for ( int p = 0; p < send_per_neighbor; ++p )
            {
                export_ranks_host( num_stay + n * send_per_neighbor + p ) =
                    neighbor_ranks[n + 1];
            }
        }
        auto export_ranks = Kokkos::create_mirror_view_and_copy(
            data_memory_space(), export_ranks_host );

        // Run tests and time the ensemble.
        for ( int t = 0; t < num_run; ++t )
        {
            // Create source particles.
            aosoa_type src_particles( "src_particles", num_particle );

            // Create destination particles.
            aosoa_type dst_particles( "dst_particles" );

            // Create a distributor using the fast construction method.
            distributor_fast_create.start( fraction );
            auto comm_export_ranks = Kokkos::create_mirror_view_and_copy(
                comm_memory_space(), export_ranks );
            Cabana::Distributor<comm_memory_space> distributor_fast(
                comm, comm_export_ranks, unique_neighbors );
            distributor_fast_create.stop( fraction );

            // Create a distributor using the general construction method.
            distributor_general_create.start( fraction );
            comm_export_ranks = Kokkos::create_mirror_view_and_copy(
                comm_memory_space(), export_ranks );
            Cabana::Distributor<comm_memory_space> distributor_general(
                comm, comm_export_ranks );
            distributor_general_create.stop( fraction );

            // Resize the destination aosoa.
            dst_particles.resize( distributor_fast.totalNumImport() );

            // Migrate the aosoa as a whole. Do host/device
            // copies as needed.
            distributor_aosoa_migrate.start( fraction );
            auto comm_src_particles = Cabana::create_mirror_view_and_copy(
                comm_memory_space(), src_particles );
            auto comm_dst_particles = Cabana::create_mirror_view(
                comm_memory_space(), dst_particles );
            Cabana::migrate( distributor_fast, comm_src_particles,
                             comm_dst_particles );
            Cabana::deep_copy( dst_particles, comm_dst_particles );
            distributor_aosoa_migrate.stop( fraction );

            // Migrate the aosoa using individual slices. Do host/device
            // copies as needed.
            distributor_slice_migrate.start( fraction );

            comm_src_particles = Cabana::create_mirror_view_and_copy(
                comm_memory_space(), src_particles );
            comm_dst_particles = Cabana::create_mirror_view(
                comm_memory_space(), dst_particles );

            auto s0 = Cabana::slice<0>( comm_src_particles );
            auto d0 = Cabana::slice<0>( comm_dst_particles );
            Cabana::migrate( distributor_fast, s0, d0 );

            auto s1 = Cabana::slice<1>( comm_src_particles );
            auto d1 = Cabana::slice<1>( comm_dst_particles );
            Cabana::migrate( distributor_fast, s1, d1 );

            auto s2 = Cabana::slice<2>( comm_src_particles );
            auto d2 = Cabana::slice<2>( comm_dst_particles );
            Cabana::migrate( distributor_fast, s2, d2 );

            auto s3 = Cabana::slice<3>( comm_src_particles );
            auto d3 = Cabana::slice<3>( comm_dst_particles );
            Cabana::migrate( distributor_fast, s3, d3 );

            Cabana::deep_copy( dst_particles, comm_dst_particles );

            distributor_slice_migrate.stop( fraction );
        }
    }

    // Output results.
    outputResults( stream, "send_bytes", comm_bytes, distributor_fast_create,
                   comm );
    outputResults( stream, "send_bytes", comm_bytes, distributor_general_create,
                   comm );
    outputResults( stream, "send_bytes", comm_bytes, distributor_aosoa_migrate,
                   comm );
    outputResults( stream, "send_bytes", comm_bytes, distributor_slice_migrate,
                   comm );

    // HALO
    // ----

    // Create halo timers.
    Cabana::Benchmark::Timer halo_fast_create( test_prefix + "halo_fast_create",
                                               num_fraction );
    Cabana::Benchmark::Timer halo_general_create(
        test_prefix + "halo_general_create", num_fraction );
    Cabana::Benchmark::Timer halo_aosoa_gather(
        test_prefix + "halo_aosoa_gather", num_fraction );
    Cabana::Benchmark::Timer halo_slice_gather(
        test_prefix + "halo_slice_gather", num_fraction );
    Cabana::Benchmark::Timer halo_slice_scatter(
        test_prefix + "halo_slice_scatter", num_fraction );

    // Create halo timers.
    Cabana::Benchmark::Timer halo_buffer_aosoa_gather(
        test_prefix + "halo_buffer_aosoa_gather", num_fraction );
    Cabana::Benchmark::Timer halo_buffer_slice_gather(
        test_prefix + "halo_buffer_slice_gather", num_fraction );
    Cabana::Benchmark::Timer halo_buffer_slice_scatter(
        test_prefix + "halo_buffer_slice_scatter", num_fraction );

    // Create the particles.
    aosoa_type particles( "particles", num_particle );

    // Loop over comm fractions.
    for ( int fraction = 0; fraction < num_fraction; ++fraction )
    {
        // Create the halo distribution in the data memory space. This is
        // where it would likely be created. Note below that we divide by 26
        // and then multiply by 26 to get a number of sends via integer math
        // that is evenly divisible by the number of neighbors. This
        // guarantees that the same number of particles is sent to each
        // neighbor so we can rule out load imbalance in our timings.
        int num_send = num_particle * comm_fraction[fraction];
        int send_per_neighbor = num_send / 26;
        num_send = send_per_neighbor * 26;
        comm_bytes[fraction] = send_per_neighbor * bytes_per_particle;
        Kokkos::View<int*, Kokkos::HostSpace> export_ranks_host( "export_ranks",
                                                                 num_send );
        Kokkos::View<int*, Kokkos::HostSpace> export_ids_host( "export_ids",
                                                               num_send );
        for ( int n = 0; n < 26; ++n )
        {
            for ( int p = 0; p < send_per_neighbor; ++p )
            {
                export_ids_host( n * send_per_neighbor + p ) =
                    n * send_per_neighbor + p;
                export_ranks_host( n * send_per_neighbor + p ) =
                    neighbor_ranks[n + 1];
            }
        }
        auto export_ranks = Kokkos::create_mirror_view_and_copy(
            data_memory_space(), export_ranks_host );
        auto export_ids = Kokkos::create_mirror_view_and_copy(
            data_memory_space(), export_ids_host );

        // Run create tests and time the ensemble.
        for ( int t = 0; t < num_run; ++t )
        {
            // Create a halo using the fast construction method.
            halo_fast_create.start( fraction );
            auto comm_export_ids = Kokkos::create_mirror_view_and_copy(
                comm_memory_space(), export_ids );
            auto comm_export_ranks = Kokkos::create_mirror_view_and_copy(
                comm_memory_space(), export_ranks );
            Cabana::Halo<comm_memory_space> halo_fast(
                comm, num_particle, comm_export_ids, comm_export_ranks,
                unique_neighbors );
            halo_fast_create.stop( fraction );

            // Create a halo using the general construction method.
            halo_general_create.start( fraction );
            comm_export_ids = Kokkos::create_mirror_view_and_copy(
                comm_memory_space(), export_ids );
            comm_export_ranks = Kokkos::create_mirror_view_and_copy(
                comm_memory_space(), export_ranks );
            Cabana::Halo<comm_memory_space> halo_general(
                comm, num_particle, comm_export_ids, comm_export_ranks );
            halo_general_create.stop( fraction );
        }

        // Create one halo for gather/scatter tests.
        auto comm_export_ids = Kokkos::create_mirror_view_and_copy(
            comm_memory_space(), export_ids );
        auto comm_export_ranks = Kokkos::create_mirror_view_and_copy(
            comm_memory_space(), export_ranks );
        Cabana::Halo<comm_memory_space> halo(
            comm, num_particle, comm_export_ids, comm_export_ranks,
            unique_neighbors );
        // Resize the particles for gather.
        particles.resize( halo.numLocal() + halo.numGhost() );

        // Run gather/scatter tests and time the ensemble.
        for ( int t = 0; t < num_run; ++t )
        {
            // Gather the aosoa as a whole. Do host/device copies as needed.
            halo_aosoa_gather.start( fraction );
            auto comm_particles = Cabana::create_mirror_view_and_copy(
                comm_memory_space(), particles );
            Cabana::gather( halo, comm_particles );
            Cabana::deep_copy( particles, comm_particles );
            halo_aosoa_gather.stop( fraction );

            // Gather the aosoa using individual slices.
            halo_slice_gather.start( fraction );

            comm_particles = Cabana::create_mirror_view_and_copy(
                comm_memory_space(), particles );

            auto s0 = Cabana::slice<0>( comm_particles );
            Cabana::gather( halo, s0 );

            auto s1 = Cabana::slice<1>( comm_particles );
            Cabana::gather( halo, s1 );

            auto s2 = Cabana::slice<2>( comm_particles );
            Cabana::gather( halo, s2 );

            auto s3 = Cabana::slice<3>( comm_particles );
            Cabana::gather( halo, s3 );

            Cabana::deep_copy( particles, comm_particles );

            halo_slice_gather.stop( fraction );

            // Scatter the aosoa using individual slices.
            halo_slice_scatter.start( fraction );

            comm_particles = Cabana::create_mirror_view_and_copy(
                comm_memory_space(), particles );

            s0 = Cabana::slice<0>( comm_particles );
            Cabana::scatter( halo, s0 );

            s1 = Cabana::slice<1>( comm_particles );
            Cabana::scatter( halo, s1 );

            s2 = Cabana::slice<2>( comm_particles );
            Cabana::scatter( halo, s2 );

            s3 = Cabana::slice<3>( comm_particles );
            Cabana::scatter( halo, s3 );

            Cabana::deep_copy( particles, comm_particles );
            halo_slice_scatter.stop( fraction );
        }

        // Preallocated buffers.
        // ---
        double overallocation = 1.1;

        // Run preallocated AoSoA buffer tests and time the ensemble.
        auto comm_particles = Cabana::create_mirror_view_and_copy(
            comm_memory_space(), particles );
        auto gather = createGather( halo, comm_particles, overallocation );
        for ( int t = 0; t < num_run; ++t )
        {
            halo_buffer_aosoa_gather.start( fraction );
            auto comm_particles = Cabana::create_mirror_view_and_copy(
                comm_memory_space(), particles );
            gather.apply( comm_particles );
            Cabana::deep_copy( particles, comm_particles );
            halo_buffer_aosoa_gather.stop( fraction );
        }

        // Run preallocated slice buffer gather tests and time the ensemble.
        comm_particles = Cabana::create_mirror_view_and_copy(
            comm_memory_space(), particles );
        auto s0 = Cabana::slice<0>( comm_particles );
        auto s1 = Cabana::slice<1>( comm_particles );
        auto s2 = Cabana::slice<2>( comm_particles );
        auto s3 = Cabana::slice<3>( comm_particles );
        auto gather_s0 = createGather( halo, s0, overallocation );
        auto gather_s1 = createGather( halo, s1, overallocation );
        auto gather_s2 = createGather( halo, s2, overallocation );
        auto gather_s3 = createGather( halo, s3, overallocation );
        for ( int t = 0; t < num_run; ++t )
        {
            halo_buffer_slice_gather.start( fraction );
            comm_particles = Cabana::create_mirror_view_and_copy(
                comm_memory_space(), particles );

            auto s0 = Cabana::slice<0>( comm_particles );
            gather_s0.apply( s0 );

            auto s1 = Cabana::slice<1>( comm_particles );
            gather_s1.apply( s1 );

            auto s2 = Cabana::slice<2>( comm_particles );
            gather_s2.apply( s2 );

            auto s3 = Cabana::slice<3>( comm_particles );
            gather_s3.apply( s3 );

            Cabana::deep_copy( particles, comm_particles );
            halo_buffer_slice_gather.stop( fraction );
        }

        // Run preallocated slice buffer scatter tests and time the ensemble.
        comm_particles = Cabana::create_mirror_view_and_copy(
            comm_memory_space(), particles );
        s0 = Cabana::slice<0>( comm_particles );
        s1 = Cabana::slice<1>( comm_particles );
        s2 = Cabana::slice<2>( comm_particles );
        s3 = Cabana::slice<3>( comm_particles );
        auto scatter_s0 = createScatter( halo, s0, overallocation );
        auto scatter_s1 = createScatter( halo, s1, overallocation );
        auto scatter_s2 = createScatter( halo, s2, overallocation );
        auto scatter_s3 = createScatter( halo, s3, overallocation );
        for ( int t = 0; t < num_run; ++t )
        {
            halo_buffer_slice_scatter.start( fraction );
            comm_particles = Cabana::create_mirror_view_and_copy(
                comm_memory_space(), particles );

            s0 = Cabana::slice<0>( comm_particles );
            scatter_s0.apply( s0 );

            s1 = Cabana::slice<1>( comm_particles );
            scatter_s1.apply( s1 );

            s2 = Cabana::slice<2>( comm_particles );
            scatter_s2.apply( s2 );

            s3 = Cabana::slice<3>( comm_particles );
            scatter_s3.apply( s3 );

            Cabana::deep_copy( particles, comm_particles );
            halo_buffer_slice_scatter.stop( fraction );
        }
    }

    // Output results.
    outputResults( stream, "send_bytes", comm_bytes, halo_fast_create, comm );
    outputResults( stream, "send_bytes", comm_bytes, halo_general_create,
                   comm );
    outputResults( stream, "send_bytes", comm_bytes, halo_aosoa_gather, comm );
    outputResults( stream, "send_bytes", comm_bytes, halo_slice_gather, comm );
    outputResults( stream, "send_bytes", comm_bytes, halo_slice_scatter, comm );

    outputResults( stream, "send_bytes", comm_bytes, halo_buffer_aosoa_gather,
                   comm );
    outputResults( stream, "send_bytes", comm_bytes, halo_buffer_slice_gather,
                   comm );
    outputResults( stream, "send_bytes", comm_bytes, halo_buffer_slice_scatter,
                   comm );
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
             $/: ./CommPerformance test_results.txt large\n" );

    // Get the name of the output file.
    std::string filename = argv[1];

    // Define run sizes.
    std::string run_type = "";
    if ( argc > 2 )
        run_type = argv[2];
    std::vector<int> problem_sizes = { 1000 };
    std::vector<double> comm_fraction = { 0.0001, 0.001, 0.005, 0.01,
                                          0.05,   0.10,  0.25,  0.5 };
    if ( run_type == "large" )
    {
        problem_sizes = { 1000, 10000, 100000, 1000000 };
    }

    // Barier before continuing.
    MPI_Barrier( MPI_COMM_WORLD );

    // Get comm rank;
    int comm_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    // Get comm size;
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

    int num_problem_size = problem_sizes.size();
    for ( int p = 0; p < num_problem_size; ++p )
    {
        int num_particle = problem_sizes[p];

        // Open the output file on rank 0.
        std::fstream file;
        if ( 0 == comm_rank )
            file.open( filename + "_" + std::to_string( comm_size ) + "_" +
                           std::to_string( problem_sizes[p] ),
                       std::fstream::out );

        // Output problem details.
        if ( 0 == comm_rank )
        {
            std::size_t total_num_p = num_particle * comm_size;
            file << "\n";
            file << "Cabana Comm Performance Benchmark"
                 << "\n";
            file << "----------------------------------------------"
                 << "\n";
            file << "MPI Ranks: " << comm_size << "\n";
            file << "Particle per MPI Rank/GPU: " << num_particle << "\n";
            file << "Total number of particles: " << total_num_p << "\n";
            file << "----------------------------------------------"
                 << "\n";
            file << "\n";
        }

        // Do everything on the default CPU.
        using host_exec_space = Kokkos::DefaultHostExecutionSpace;
        using host_device_type = host_exec_space::device_type;
        // Do everything on the default device with default memory.
        using exec_space = Kokkos::DefaultExecutionSpace;
        using device_type = exec_space::device_type;

        // Don't run three times on the CPU if only host enabled.
        if ( !std::is_same<device_type, host_device_type>{} )
        {
            performanceTest<device_type, device_type>(
                file, num_particle, "device_device_", comm_fraction );

            // Transfer GPU data to CPU, communication on CPU, and transfer back
            // to GPU.
            performanceTest<device_type, host_device_type>(
                file, num_particle, "device_host_", comm_fraction );
        }
        performanceTest<host_device_type, host_device_type>(
            file, num_particle, "host_host_", comm_fraction );

        // Close the output file on rank 0.
        if ( 0 == comm_rank )
            file.close();
    }
    // Finalize
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}

//---------------------------------------------------------------------------//
