/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ewald.h"
#include "Cabana_DeepCopy.hpp"
#include "Cabana_Halo.hpp"
#include "Cabana_LinkedCellList.hpp"
#include "Cabana_VerletList.hpp"
#include "definitions.h"
#include "mpi.h"
#include <assert.h>
#include <cmath>
#include <iomanip>
#include <string>

TEwald::TEwald( double accuracy, int n_total, double lx, double ly, double lz,
                Kokkos::View<double *> &domain_width, MPI_Comm comm )
{
    // check if used communicator is cartesian
    int comm_type;
    MPI_Topo_test( comm, &comm_type );
    assert( comm_type == MPI_CART );
    this->comm = comm;

    _r_max = 0.0;
    tune( accuracy, n_total, lx, ly, lz );
    this->domain_width = domain_width;
}

// TODO: needs to be extended for parallel version
TEwald::TEwald( double alpha, double r_max, double k_max )
{
    _alpha = alpha;
    _r_max = r_max;
    _k_max = k_max;
}

void TEwald::tune( double accuracy, int N, double lx, double ly, double lz )
{
    double l = std::max( std::max( lx, ly ), lz );

    // Fincham 1994, Optimisation of the Ewald Sum for Large Systems
    // only valid for cubic systems (needs adjustement for non-cubic systems)
    constexpr double EXECUTION_TIME_RATIO_K_R = 2.0;
    double p = -log( accuracy );

    double tune_factor =
        pow( EXECUTION_TIME_RATIO_K_R, 1.0 / 6.0 ) * sqrt( p / PI );

    // to avoid problems with the real-space part,
    tune_factor = ( tune_factor / pow( N, 1.0 / 6.0 ) >= 1.0 )
                      ? pow( N, 1.0 / 6.0 ) * 0.99
                      : tune_factor;

    _r_max = tune_factor / pow( N, 1.0 / 6.0 ) * l;

    _alpha = tune_factor * pow( N, 1.0 / 6.0 ) / l;
    _k_max = tune_factor * pow( N, 1.0 / 6.0 ) / l * 2.0 * PI;
    _alpha = sqrt( p ) / _r_max;
    _k_max = 2.0 * sqrt( p ) * _alpha;

    int rank;
    MPI_Comm_rank( comm, &rank );
    if ( rank == 0 )
        std::cout << "tuned Ewald values: "
                  << "r_max: " << _r_max << " alpha: " << _alpha
                  << " k_max: " << _k_max << std::endl;
}

double TEwald::compute( ParticleList &particles, double lx, double ly,
                        double lz )
{

    double Ur = 0.0, Uk = 0.0, Uself = 0.0, Udip = 0.0;
    double Udip_vec[3];

    auto r = Cabana::slice<Position>( particles );
    auto q = Cabana::slice<Charge>( particles );
    auto p = Cabana::slice<Potential>( particles );
    auto f = Cabana::slice<Force>( particles );
    auto v = Cabana::slice<Velocity>( particles );
    auto i = Cabana::slice<Index>( particles );

    // compute domain size and make it available in the kernels
    Kokkos::View<double *> domain_size( "domain size", 3 );
    domain_size( 0 ) = domain_width( 1 ) - domain_width( 0 );
    domain_size( 1 ) = domain_width( 3 ) - domain_width( 2 );
    domain_size( 2 ) = domain_width( 5 ) - domain_width( 4 );

    Kokkos::View<double *> sys_size( "system size", 3 );
    sys_size( 0 ) = lx;
    sys_size( 1 ) = ly;
    sys_size( 2 ) = lz;

    // get the solver parameters
    double alpha = _alpha;
    double r_max = _r_max;
    double eps_r = _eps_r;
    double k_max = _k_max;

    // store MPI information
    int rank, n_ranks;
    std::vector<int> loc_dims( 3 );
    std::vector<int> cart_dims( 3 );
    std::vector<int> cart_periods( 3 );
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &n_ranks );
    MPI_Cart_get( comm, 3, cart_dims.data(), cart_periods.data(),
                  loc_dims.data() );

    // neighbor information
    std::vector<int> neighbor_low( 3 );
    std::vector<int> neighbor_up( 3 );

    // get neighbors in parallel decomposition
    for ( int dim = 0; dim < 3; ++dim )
    {
        MPI_Cart_shift( comm, dim, 1, &neighbor_low.at( dim ),
                        &neighbor_up.at( dim ) );
    }

    int n_max = particles.size();

    auto init_parameters = KOKKOS_LAMBDA( const int idx )
    {
        p( idx ) = 0.0;
        f( idx, 0 ) = 0.0;
        f( idx, 1 ) = 0.0;
        f( idx, 2 ) = 0.0;
    };
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
                          init_parameters );
    Kokkos::fence();

    // computation: k-space contribution
    if ( rank == 0 )
        std::cout << "starting k-space computations..." << std::endl;
    double start_time_Uk = MPI_Wtime();

    // In order to compute the k-space contribution in parallel
    // first the following sums need to be created for each
    // k-vector:
    //              sum(1<=i<=N_part) sin/cos (dot(k,r_i))
    // This can be achieved by computing partial sums on each
    // MPI process, reducing them over all processes and
    // afterward using the pre-computed values to compute
    // the forces and potentials acting on the particles
    // in parallel independently again.

    // determine number of required sine / cosine values
    int k_int = std::ceil( k_max );
    int n_kvec = ( 2 * k_int + 1 ) * ( 2 * k_int + 1 ) * ( 2 * k_int + 1 );

    // allocate View to store them
    Kokkos::View<double *> U_trigonometric( "sine and cosine contributions",
                                            2 * n_kvec );

    // set all values to zero
    Kokkos::parallel_for( 2 * n_kvec, KOKKOS_LAMBDA( const int idx ) {
        U_trigonometric( idx ) = 0.0;
    } );

    // compute partial sums
    Kokkos::parallel_for( n_max, KOKKOS_LAMBDA( const int idx ) {
        for ( int kz = -k_int; kz <= k_int; ++kz )
        {
            // compute wave vector component
            double _kz = 2.0 * PI / lz * (double)kz;
            for ( int ky = -k_int; ky <= k_int; ++ky )
            {
                // compute wave vector component
                double _ky = 2.0 * PI / ly * (double)ky;
                for ( int kx = -k_int; kx <= k_int; ++kx )
                {
                    // no values required for the central box
                    if ( kx == 0 && ky == 0 && kz == 0 )
                        continue;
                    // compute index in contribution array
                    int kidx =
                        ( kz + k_int ) * ( 2 * k_int + 1 ) * ( 2 * k_int + 1 ) +
                        ( ky + k_int ) * ( 2 * k_int + 1 ) + ( kx + k_int );
                    // compute wave vector component
                    double _kx = 2.0 * PI / lx * (double)kx;
                    // compute dot product with local particle and wave vector
                    double kr = _kx * r( idx, 0 ) + _ky * r( idx, 1 ) +
                                _kz * r( idx, 2 );
                    // add contributions
                    Kokkos::atomic_add( &U_trigonometric( 2 * kidx ),
                                        q( idx ) * cos( kr ) );
                    Kokkos::atomic_add( &U_trigonometric( 2 * kidx + 1 ),
                                        q( idx ) * cos( kr ) );
                }
            }
        }
    } );
    Kokkos::fence();

    // TODO: check if there is a better way to do this
    // reduce the partial results

    double U_trigon_array[2 * n_kvec];
    for ( int idx = 0; idx < 2 * n_kvec; ++idx )
        U_trigon_array[idx] = U_trigonometric( idx );

    MPI_Allreduce( MPI_IN_PLACE, U_trigon_array, 2 * n_kvec, MPI_DOUBLE,
                   MPI_SUM, comm );

    for ( int idx = 0; idx < 2 * n_kvec; ++idx )
        U_trigonometric( idx ) = U_trigon_array[idx];

    Kokkos::parallel_reduce(
        n_max,
        KOKKOS_LAMBDA( const int idx, double &Uk_part ) {
            // general coefficient
            double coeff = 4.0 * PI / ( lx * ly * lz );
            double k[3];

            for ( int kz = -k_int; kz <= k_int; ++kz )
            {
                // compute wave vector component
                k[2] = 2.0 * PI / lz * (double)kz;
                for ( int ky = -k_int; ky <= k_int; ++ky )
                {
                    // compute wave vector component
                    k[1] = 2.0 * PI / ly * (double)ky;
                    for ( int kx = -k_int; kx <= k_int; ++kx )
                    {
                        // no values required for the central box
                        if ( kx == 0 && ky == 0 && kz == 0 )
                            continue;
                        // compute index in contribution array
                        int kidx = ( kz + k_int ) * ( 2 * k_int + 1 ) *
                                       ( 2 * k_int + 1 ) +
                                   ( ky + k_int ) * ( 2 * k_int + 1 ) +
                                   ( kx + k_int );
                        // compute wave vector component
                        k[0] = 2.0 * PI / lx * (double)kx;
                        // compute dot product of wave vector with itself
                        double kk = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                        ;
                        // compute dot product with local particle and wave
                        // vector
                        double kr = k[0] * r( idx, 0 ) + k[1] * r( idx, 1 ) +
                                    k[2] * r( idx, 2 );

                        // coefficient dependent on wave vector
                        double k_coeff =
                            exp( -kk / ( 4 * alpha * alpha ) ) / kk;

                        // contribution to potential energy
                        double contrib =
                            coeff * k_coeff *
                            ( U_trigonometric( 2 * kidx ) *
                                  U_trigonometric( 2 * kidx ) +
                              U_trigonometric( 2 * kidx + 1 ) *
                                  U_trigonometric( 2 * kidx + 1 ) );

                        p( idx ) += contrib;
                        Uk_part += contrib;

                        for ( int dim = 0; dim < 3; ++dim )
                            f( idx, dim ) +=
                                k_coeff * 2.0 * q( idx ) * k[dim] *
                                ( U_trigonometric( 2 * kidx + 1 ) * cos( kr ) -
                                  U_trigonometric( 2 * kidx ) * sin( kr ) );
                    }
                }
            }
        },
        Uk );
    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &Uk, 1, MPI_DOUBLE, MPI_SUM, comm );

    double end_time_Uk = MPI_Wtime();
    double elapsed_Uk = end_time_Uk - start_time_Uk;

    if ( rank == 0 )
        std::cout << "starting real-space computations..." << std::endl;
    double start_time_Ur = MPI_Wtime();

    // computation real-space contribution

    // In order to compute the real-space contribution to potentials and
    // forces the Cabana implementation of halos and Verlet lists is
    // used. The halos are used to communicate particles along the
    // borders of MPI domains to their respective neighbors, so that
    // complete Verlet lists can be created. To save computation time
    // the half shell variant is used, that means that Newton's third
    // law of motion is used: F(i,j) = -F(j,i). The downside of this
    // is that the computed partial forces and potentials of the
    // ghost particles need to be communicated back to the source
    // process, which is done by using the 'scatter' implementation
    // of Cabana.

    // TODO: cellsize dynamic (if still necessary?)
    double lc = 1.0;
    // double cfactor = std::ceil( r_max / lc );
    // int nneigdim = 2 * (int)cfactor + 1;

    /*
    if (rank == 0)
        std::cout << "l: " << lx
                  << " lc: " << lc
                  << " r_max: " << r_max << std::endl;
    */

    // create a Cabana linked list
    // lower end of system
    double grid_min[3] = {-r_max + domain_width( 0 ),
                          -r_max + domain_width( 2 ),
                          -r_max + domain_width( 4 )};
    // upper end of system
    double grid_max[3] = {domain_width( 1 ) + r_max, domain_width( 3 ) + r_max,
                          domain_width( 5 ) + r_max};

    using ListAlgorithm = Cabana::HalfNeighborTag;
    using ListType =
        Cabana::VerletList<DeviceType, ListAlgorithm, Cabana::VerletLayoutCSR>;

    // store the number of local particles
    int n_local = n_max;
    // offset due to the previously received particles
    int offset = 0;

    // communicate particles along the edges of the system

    // six halo regions required for transport in all directions
    std::vector<int> n_export = {0, 0, 0, 0, 0, 0};
    std::vector<Cabana::Halo<DeviceType> *> halos( 6 );

    // do three-step communication, x -> y -> z
    for ( int dim = 0; dim < 3; ++dim )
    {
        // check if the cut-off is not larger then two times the system size
        assert( r_max <= 2.0 * sys_size( dim ) );

        // find out how many particles are close to the -dim border
        Kokkos::parallel_reduce(
            n_max,
            KOKKOS_LAMBDA( const int idx, int &low ) {
                low += ( r( idx, dim ) <= domain_width( 2 * dim ) + r_max ) ? 1
                                                                            : 0;
            },
            n_export.at( 2 * dim ) );

        // find out how many particles are close to the +dim border
        Kokkos::parallel_reduce(
            n_max,
            KOKKOS_LAMBDA( const int idx, int &up ) {
                up += ( r( idx, dim ) >= domain_width( 2 * dim + 1 ) - r_max )
                          ? 1
                          : 0;
            },
            n_export.at( 2 * dim + 1 ) );

        // list with the ranks and target processes for the halo
        Kokkos::View<int *, DeviceType> export_ranks_low(
            "export_ranks_low", n_export.at( 2 * dim ) );
        Kokkos::View<int *, DeviceType> export_ranks_up(
            "export_ranks_up", n_export.at( 2 * dim + 1 ) );

        Kokkos::View<int *, DeviceType> export_ids_low(
            "export_ids_low", n_export.at( 2 * dim ) );
        Kokkos::View<int *, DeviceType> export_ids_up(
            "export_ids_up", n_export.at( 2 * dim + 1 ) );

        // TODO: parallel implementation
        // fill the export arrays for the halo construction
        int idx_up = 0, idx_low = 0;
        for ( int idx = 0; idx < n_max; ++idx )
        {
            if ( r( idx, dim ) <= domain_width( 2 * dim ) + r_max )
            {
                export_ranks_low( idx_low ) = neighbor_low.at( dim );
                export_ids_low( idx_low ) = idx;
                ++idx_low;
            }
            if ( r( idx, dim ) >= domain_width( 2 * dim + 1 ) - r_max )
            {
                export_ranks_up( idx_up ) = neighbor_up.at( dim );
                export_ids_up( idx_up ) = idx;
                ++idx_up;
            }
        }

        // create neighbor list
        std::vector<int> neighbors = {neighbor_low.at( dim ), rank,
                                      neighbor_up.at( dim )};
        std::sort( neighbors.begin(), neighbors.end() );
        auto unique_end = std::unique( neighbors.begin(), neighbors.end() );
        neighbors.resize( std::distance( neighbors.begin(), unique_end ) );
        halos.at( 2 * dim ) = new Cabana::Halo<DeviceType>(
            comm, n_max, export_ids_low, export_ranks_low, neighbors );
        n_max += halos.at( 2 * dim )->numGhost();

        // resize particle list to contain all particles
        particles.resize( n_max );

        // update slices
        r = Cabana::slice<Position>( particles );
        q = Cabana::slice<Charge>( particles );
        p = Cabana::slice<Potential>( particles );
        f = Cabana::slice<Force>( particles );
        i = Cabana::slice<Index>( particles );

        // gather data for halo regions

        Cabana::gather( *( halos.at( 2 * dim ) ), r );
        Cabana::gather( *( halos.at( 2 * dim ) ), q );
        Cabana::gather( *( halos.at( 2 * dim ) ), p );
        Cabana::gather( *( halos.at( 2 * dim ) ), f );
        Cabana::gather( *( halos.at( 2 * dim ) ), i );

        // do periodic corrections and reset partial forces
        // and potentials of ghost particles
        // (they are accumulated during the scatter step)
        for ( int idx = n_local + offset;
              idx < n_local + offset + halos.at( 2 * dim )->numGhost(); ++idx )
        {
            p( idx ) = 0.0;
            f( idx, 0 ) = 0.0;
            f( idx, 1 ) = 0.0;
            f( idx, 2 ) = 0.0;
            if ( loc_dims.at( dim ) == cart_dims.at( dim ) - 1 )
            {
                r( idx, dim ) += sys_size( dim );
            }
        }

        offset += halos.at( 2 * dim )->numGhost();

        // do transfer of particles in upward direction
        halos.at( 2 * dim + 1 ) = new Cabana::Halo<DeviceType>(
            comm, n_max, export_ids_up, export_ranks_up, neighbors );
        n_max += halos.at( 2 * dim + 1 )->numGhost();

        // resize particle list to contain all particles
        particles.resize( n_max );

        // update slices
        r = Cabana::slice<Position>( particles );
        q = Cabana::slice<Charge>( particles );
        p = Cabana::slice<Potential>( particles );
        f = Cabana::slice<Force>( particles );
        i = Cabana::slice<Index>( particles );

        // gather data for halo regions

        Cabana::gather( *( halos.at( 2 * dim + 1 ) ), r );
        Cabana::gather( *( halos.at( 2 * dim + 1 ) ), q );
        Cabana::gather( *( halos.at( 2 * dim + 1 ) ), p );
        Cabana::gather( *( halos.at( 2 * dim + 1 ) ), f );
        Cabana::gather( *( halos.at( 2 * dim + 1 ) ), i );

        // do periodic corrections and reset partial forces
        // and potentials of ghost particles
        // (they are accumulated during the scatter step)
        for ( int idx = n_local + offset;
              idx < n_local + offset + halos.at( 2 * dim + 1 )->numGhost();
              ++idx )
        {
            p( idx ) = 0.0;
            f( idx, 0 ) = 0.0;
            f( idx, 1 ) = 0.0;
            f( idx, 2 ) = 0.0;
            if ( loc_dims.at( dim ) == 0 )
            {
                r( idx, dim ) -= sys_size( dim );
            }
        }

        offset += halos.at( 2 * dim + 1 )->numGhost();
    }

    // DEBUG output to check the particles on each process
    // prepare to be flooded in slowly created output
    /*
    for (int n = 0; n < n_ranks; ++n)
    {
        if (rank == n)
        {
            std::cout
                << grid_min[0] << " " << grid_max[0] << " "
                << grid_min[1] << " " << grid_max[1] << " "
                << grid_min[2] << " " << grid_max[2] << " "
                << sys_size(0) << " "
                << sys_size(1) << " "
                << sys_size(2) << " "
                << std::endl;
            for (int idx = 0; idx < n_max; ++idx)
                std::cout
                    << rank << " | "
                    << idx << ": "
                    << r( idx, 0 ) << " "
                    << r( idx, 1 ) << " "
                    << r( idx, 2 ) << " "
                    << q( idx ) << " "
                    << i( idx ) << " "
                    << std::endl;
        }
        MPI_Barrier(comm);
    }
    */

    // create VerletList to iterate over
    ListType verlet_list( r, 0, n_local, r_max, 1.0, grid_min, grid_max );

    // compute forces and potential
    // loop only over local particles, as the forces for the
    // ghost particles are accumulated on them and the
    // source particles are processed on source process
    Kokkos::parallel_for( n_local, KOKKOS_LAMBDA( const int idx ) {
        int num_n =
            Cabana::NeighborList<ListType>::numNeighbor( verlet_list, idx );

        double rx = r( idx, 0 );
        double ry = r( idx, 1 );
        double rz = r( idx, 2 );

        for ( int ij = 0; ij < num_n; ++ij )
        {
            int j = Cabana::NeighborList<ListType>::getNeighbor( verlet_list,
                                                                 idx, ij );
            double dx = r( j, 0 ) - rx;
            double dy = r( j, 1 ) - ry;
            double dz = r( j, 2 ) - rz;
            double d = sqrt( dx * dx + dy * dy + dz * dz );

            // potential computation
            double contrib = 0.5 * q( idx ) * q( j ) * erfc( alpha * d ) / d;
            Kokkos::atomic_add( &p( idx ), contrib );
            Kokkos::atomic_add( &p( j ), contrib );

            // force computation
            double f_fact = q( idx ) * q( j ) *
                            ( 2.0 * sqrt( alpha / PI ) * exp( -alpha * d * d ) +
                              erfc( sqrt( alpha ) * d ) ) /
                            ( d * d );
            Kokkos::atomic_add( &f( idx, 0 ), f_fact * dx );
            Kokkos::atomic_add( &f( idx, 1 ), f_fact * dy );
            Kokkos::atomic_add( &f( idx, 2 ), f_fact * dz );
            Kokkos::atomic_add( &f( j, 0 ), -f_fact * dx );
            Kokkos::atomic_add( &f( j, 1 ), -f_fact * dy );
            Kokkos::atomic_add( &f( j, 2 ), -f_fact * dz );

            // Debug to compare pair-wise interactions
            // LOTS of output
            /*
            std::cout << "contrib i-j: " <<
                         "i: " << idx << " " <<
                         "j: " << j << " " <<
                         "contrib: " << contrib << " " <<
                         "idx(i): " << i(idx) << " " <<
                         "idx(j): " << i(j) << " " <<
                         "r(i): " << r(idx,0) << " " << r(idx,1) << " " <<
            r(idx,2) << " " << "r(j): " << r(j,0) << " " << r(j,1) << " " <<
            r(j,2) << " " << std::endl;
            */
        }
    } );

    // send the force and potential contributions of the
    // ghost particles back to the origin processes
    for ( int n_halo = 5; n_halo >= 0; --n_halo )
    {
        Cabana::scatter( *( halos.at( n_halo ) ), p );
        Cabana::scatter( *( halos.at( n_halo ) ), f );

        n_max -= halos.at( n_halo )->numGhost();
        particles.resize( n_max );
        r = Cabana::slice<Position>( particles );
        q = Cabana::slice<Charge>( particles );
        p = Cabana::slice<Potential>( particles );
        f = Cabana::slice<Force>( particles );
        i = Cabana::slice<Index>( particles );
    }

    // check if the particle array was reduced to the correct size again
    assert( n_max == n_local );

    double Ur_local;
    Kokkos::parallel_reduce(
        n_max,
        KOKKOS_LAMBDA( int idx, double &Ur_part ) { Ur_part += p( idx ); },
        Ur_local );

    MPI_Allreduce( &Ur_local, &Ur, 1, MPI_DOUBLE, MPI_SUM, comm );

    double end_time_Ur = MPI_Wtime();
    double elapsed_Ur = end_time_Ur - start_time_Ur;

    if ( rank == 0 )
        std::cout << "starting self correction..." << std::endl;
    double start_time_Uself = MPI_Wtime();

    double Uself_loc;
    // computation of self-energy contribution
    Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
                             KOKKOS_LAMBDA( int idx, double &Uself_part ) {
                                 Uself_part +=
                                     -alpha / PI_SQRT * q( idx ) * q( idx );
                                 p( idx ) += Uself_part;
                             },
                             Uself_loc );
    Kokkos::fence();
    MPI_Allreduce( &Uself_loc, &Uself, 1, MPI_DOUBLE, MPI_SUM, comm );

    double end_time_Uself = MPI_Wtime();
    double elapsed_Uself = end_time_Uself - start_time_Uself;

    if ( rank == 0 )
        std::cout << "starting dipole correction..." << std::endl;
    auto start_time_Udip = MPI_Wtime();

    // computation of the dipole correction
    // (for most cases probably irrelevant)
    // TODO: enhancement - combine these 3 reductions
    Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
                             KOKKOS_LAMBDA( int idx, double &Udip_part ) {
                                 double V = lx * ly * lz;
                                 double Udip_prefactor =
                                     2 * PI / ( ( 1.0 + 2.0 * eps_r ) * V );
                                 Udip_part +=
                                     Udip_prefactor * q( idx ) * r( idx, 0 );
                             },
                             Udip_vec[0] );
    Kokkos::fence();

    Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
                             KOKKOS_LAMBDA( int idx, double &Udip_part ) {
                                 double V = lx * ly * lz;
                                 double Udip_prefactor =
                                     2 * PI / ( ( 1.0 + 2.0 * eps_r ) * V );
                                 Udip_part +=
                                     Udip_prefactor * q( idx ) * r( idx, 1 );
                             },
                             Udip_vec[1] );
    Kokkos::fence();

    Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
                             KOKKOS_LAMBDA( int idx, double &Udip_part ) {
                                 double V = lx * ly * lz;
                                 double Udip_prefactor =
                                     2 * PI / ( ( 1.0 + 2.0 * eps_r ) * V );
                                 Udip_part +=
                                     Udip_prefactor * q( idx ) * r( idx, 2 );
                             },
                             Udip_vec[2] );
    Kokkos::fence();

    Udip = Udip_vec[0] * Udip_vec[0] + Udip_vec[1] * Udip_vec[1] +
           Udip_vec[2] * Udip_vec[2];

    MPI_Allreduce( MPI_IN_PLACE, &Udip, 1, MPI_DOUBLE, MPI_SUM, comm );

    double end_time_Udip = MPI_Wtime();
    double elapsed_Udip = end_time_Udip - start_time_Udip;

    // display minimum, maximum and avarage run-time over
    // all processes as well as the single contributions
    // of each partial contribution

    double times[4];
    double times_min[4];
    double times_max[4];
    double times_sum[4];

    times[0] = elapsed_Uk;
    times[1] = elapsed_Ur;
    times[2] = elapsed_Uself;
    times[3] = elapsed_Udip;

    MPI_Reduce( times, times_min, 4, MPI_DOUBLE, MPI_MIN, 0, comm );
    MPI_Reduce( times, times_max, 4, MPI_DOUBLE, MPI_MAX, 0, comm );
    MPI_Reduce( times, times_sum, 4, MPI_DOUBLE, MPI_SUM, 0, comm );

    if ( rank == 0 )
    {
        double times_avg[4];

        for ( int idx = 0; idx < 4; ++idx )
            times_avg[idx] = times_sum[idx] / (double)n_ranks;

        std::cout << std::endl;
        std::cout << "part                     |  "
                  << "minimum runtime [s]      |  "
                  << "maximum runtime [s]      |  "
                  << "average runtime [s]      |  "
                  << "contribution             |  " << std::endl
                  << "----------------------------"
                  << "----------------------------"
                  << "----------------------------"
                  << "----------------------------"
                  << "----------------------------" << std::endl;

        std::cout << std::setprecision( 9 );
        std::cout << std::scientific;
        std::cout << std::showpos;

        for ( int idx = 0; idx < 4; ++idx )
        {
            switch ( idx )
            {
            case 0:
                std::cout << "k-space contribution     |  ";
                break;
            case 1:
                std::cout << "real-space contribution  |  ";
                break;
            case 2:
                std::cout << "self energy correction   |  ";
                break;
            case 3:
                std::cout << "dipole correction        |  ";
                break;
            default:
                break;
            }
            std::cout << times_min[idx] << "         |  ";
            std::cout << times_max[idx] << "         |  ";
            std::cout << times_avg[idx] << "         |  ";
            switch ( idx )
            {
            case 0:
                std::cout << Uk << "        |  ";
                break;
            case 1:
                std::cout << Ur << "        |  ";
                break;
            case 2:
                std::cout << Uself << "        |  ";
                break;
            case 3:
                std::cout << Udip << "        |  ";
                break;
            default:
                break;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::resetiosflags( std::ios::scientific );
        std::cout << std::resetiosflags( std::ios::showpos );
    }

    // return total potential
    return Ur + Uk + Uself + Udip;
}
