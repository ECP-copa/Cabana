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
#include "Cabana_LinkedCellList.hpp"
#include "definitions.h"
#include <cmath>

TEwald::TEwald( double accuracy, ParticleList particles, double lx, double ly,
                double lz )
{
    _r_max = 0.0;
    tune( accuracy, particles, lx, ly, lz );
}

TEwald::TEwald( double alpha, double r_max, double k_max )
{
    _alpha = alpha;
    _r_max = r_max;
    _k_max = k_max;
}

void TEwald::tune( double accuracy, ParticleList particles, double lx,
                   double ly, double lz )
{
    auto q = Cabana::slice<Charge>( particles );

    const int N = particles.size();

    double l = std::max( std::max( lx, ly ), lz );

    // Fincham 1994, Optimisation of the Ewald Sum for Large Systems
    // only valid for cubic systems (needs adjustement for non-cubic systems)
    constexpr double EXECUTION_TIME_RATIO_K_R = 2.0;
    double p = -log( accuracy );
    _alpha = pow( EXECUTION_TIME_RATIO_K_R, 1.0 / 6.0 ) * sqrt( p / PI ) *
             pow( N, 1.0 / 6.0 ) / l;
    _k_max = pow( EXECUTION_TIME_RATIO_K_R, 1.0 / 6.0 ) * sqrt( p / PI ) *
             pow( N, 1.0 / 6.0 ) / l * 2.0 * PI;
    _r_max = pow( EXECUTION_TIME_RATIO_K_R, 1.0 / 6.0 ) * sqrt( p / PI ) /
             pow( N, 1.0 / 6.0 ) * l;
    _alpha = sqrt( p ) / _r_max;
    _k_max = 2.0 * sqrt( p ) * _alpha;

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

    double alpha = _alpha;
    double r_max = _r_max;
    double eps_r = _eps_r;

    auto start_time_kf = std::chrono::high_resolution_clock::now();
    EwaldUkFunctor ukf( particles, _k_max, _alpha, lx, ly, lz );
    std::cout << "k-space potentials functor created" << std::endl;
    int k_int = std::ceil( _k_max );
    int n_k = 8 * k_int * k_int * k_int + 12 * k_int * k_int + 6 * k_int + 1;
    Kokkos::parallel_reduce( n_k, ukf, Uk );
    std::cout << "k-space potentials computed" << std::endl;
    Kokkos::fence();
    auto end_time_kf = std::chrono::high_resolution_clock::now();
    auto start_time_kff = std::chrono::high_resolution_clock::now();
    EwaldUkForcesFunctor<ExecutionSpace> uk_fi( r, q, f, _k_max, _alpha, n_k,
                                                lx, ly, lz );
    std::cout << "k-space forces functor created" << std::endl;
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecutionSpace>( n_max, Kokkos::AUTO ), uk_fi );
    Kokkos::fence();

    auto end_time_kff = std::chrono::high_resolution_clock::now();
    auto elapsed_time_kf = end_time_kf - start_time_kf;
    auto elapsed_time_kff = end_time_kff - start_time_kff;
    auto ns_elapsed_kf =
        std::chrono::duration_cast<std::chrono::nanoseconds>( elapsed_time_kf );
    auto ns_elapsed_kff = std::chrono::duration_cast<std::chrono::nanoseconds>(
        elapsed_time_kff );

    std::cout << "k-space contribution: "
              << ( ( ns_elapsed_kf + ns_elapsed_kff ).count() / 1000000000.0 )
              << " s = "
              << " potential: " << ( ns_elapsed_kf.count() / 1000000000.0 )
              << " s + "
              << " forces: " << ( ns_elapsed_kff.count() / 1000000000.0 )
              << " s " << Uk << " " << std::endl;

    // computation real-space contribution
    Kokkos::fence();
    if ( r_max > 0.5 * lx )
    {
        /*
         *
         *  Computation real-space contribution
         *
         */
        // plain all to all comparison and regard of periodic images with
        // distance > 1
        auto start_time_r = std::chrono::high_resolution_clock::now();
        Kokkos::parallel_reduce(
            Kokkos::TeamPolicy<>( n_max, Kokkos::AUTO ),
            KOKKOS_LAMBDA( Kokkos::TeamPolicy<>::member_type member,
                           double &Ur_i ) {
                int i = member.league_rank(); // * member.team_size() +
                                              // member.team_rank();
                if ( i < n_max )
                {
                    double Ur_inner = 0.0;
                    int per_shells = std::ceil( r_max / lx );
                    Kokkos::single( Kokkos::PerTeam( member ), [=] {
                        if ( i == 0 )
                            printf( "number of periodic shells in real-space "
                                    "computation: %d\n",
                                    per_shells );
                    } );
                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange( member, n_max ),
                        [&]( int &j, double &Ur_j ) {
                            for ( int pz = -per_shells; pz <= per_shells; ++pz )
                                for ( int py = -per_shells; py <= per_shells;
                                      ++py )
                                    for ( int px = -per_shells;
                                          px <= per_shells; ++px )
                                    {
                                        double dx =
                                            r( i, 0 ) -
                                            ( r( j, 0 ) + (double)px * lx );
                                        double dy =
                                            r( i, 1 ) -
                                            ( r( j, 1 ) + (double)py * ly );
                                        double dz =
                                            r( i, 2 ) -
                                            ( r( j, 2 ) + (double)pz * lz );
                                        double d =
                                            sqrt( dx * dx + dy * dy + dz * dz );
                                        double contrib =
                                            ( d <= r_max &&
                                              std::abs( d ) >= 1e-12 )
                                                ? 0.5 * q( i ) * q( j ) *
                                                      erfc( alpha * d ) / d
                                                : 0.0;
                                        double f_fact =
                                            ( d <= r_max &&
                                              std::abs( d ) >= 1e-12 ) *
                                            q( i ) * q( j ) *
                                            ( 2.0 * sqrt( alpha / PI ) *
                                                  exp( -alpha * d * d ) +
                                              erfc( sqrt( alpha ) * d ) ) /
                                            ( d * d +
                                              ( std::abs( d ) <= 1e-12 ) );
                                        Kokkos::atomic_add( &f( i, 0 ),
                                                            f_fact * dx );
                                        Kokkos::atomic_add( &f( i, 1 ),
                                                            f_fact * dy );
                                        Kokkos::atomic_add( &f( i, 2 ),
                                                            f_fact * dz );
                                        Kokkos::single(
                                            Kokkos::PerThread( member ),
                                            [&] { Ur_j += contrib; } );
                                    }
                        },
                        Ur_inner );
                    Kokkos::single( Kokkos::PerTeam( member ), [&] {
                        p( i ) += Ur_inner;
                        Ur_i += Ur_inner;
                    } );
                }
            },
            Ur );
        Kokkos::fence();
        auto end_time_r = std::chrono::high_resolution_clock::now();
        auto elapsed_time_r = end_time_r - start_time_r;
        auto ns_elapsed_r =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                elapsed_time_r );

        std::cout << "real-space contribution: "
                  << ( ns_elapsed_r.count() / 1000000000.0 ) << " s " << Ur
                  << std::endl;
    }
    else
    {

        // In this branch a linked-cell list is used, which is using the Cabana implementation
        // of linked cell neighbor list. To compute the real-space contribution the particles
        // are sorted into cells of a given size and then for each cell the distances to
        // particles in surrounding cells are computed. If the distance is smaller than the
        // cut-off distance, the real-space interaction is computed and summed up for the
        // total energy potential


        // TODO: cellsize dynamic!
        double lc = 1.0;
        double cfactor = std::ceil( r_max / lc );
        int nneigdim = 2 * (int)cfactor + 1;
        int nneig = nneigdim * nneigdim * nneigdim;

        std::cout << "l: " << lx << " cfactor: " << cfactor << " lc: " << lc
                  << " r_max: " << r_max << std::endl;

        int n_cells =
            std::ceil( lx / lc ) * std::ceil( ly / lc ) * std::ceil( lz / lc );
        Kokkos::View<int *, MemorySpace> cdim( "cdim", 3 );
        cdim( 0 ) = std::ceil( lx / lc );
        cdim( 1 ) = std::ceil( ly / lc );
        cdim( 2 ) = std::ceil( lz / lc );

        auto start_time_rsort = std::chrono::high_resolution_clock::now();
        // create a Cabana linked list
        // lower end of system
        double grid_min[3] = {0.0, 0.0, 0.0};
        // upper end of system
        double grid_max[3] = {lx, ly, lz};
        // cell size
        double grid_delta[3] = {lc, lc, lc};
        // create LinkedCellList
        Cabana::LinkedCellList<MemorySpace> cell_list( r, grid_delta, grid_min,
                                                       grid_max );

        // resort particles in original ParticleList to be sorted according to
        // LinkedCellList
        Cabana::permute( cell_list, particles );
        auto end_time_rsort = std::chrono::high_resolution_clock::now();
        auto elapsed_time_rsort = end_time_rsort - start_time_rsort;

        auto start_time_rij = std::chrono::high_resolution_clock::now();

        // energy contribution from neighbor cells
        double Ur_ij = 0.0;

        for ( int n = nneig / 2 + 1; n < nneig; ++n )
        {
            double Ur_ijn;
            Kokkos::parallel_reduce(
                n_cells,
                KOKKOS_LAMBDA( const int idx, double &Ur_ic ) {
                    // get coordinates of local domain
                    int ix, iy, iz;
                    cell_list.ijkBinIndex( idx, ix, iy, iz );

                    // index of neighbor (to dermine which neighbor it is
                    int jcidx = n;

                    // compute position in relation to local cell
                    int jnx = jcidx % nneigdim - (int)cfactor;
                    int jny = ( jcidx / nneigdim ) % nneigdim - (int)cfactor;
                    int jnz = jcidx / ( nneigdim * nneigdim ) - (int)cfactor;

                    // compute global cell index
                    int jc = ( iz + jnz + cdim( 2 ) ) % cdim( 2 ) +
                             ( iy + jny + cdim( 1 ) ) % cdim( 1 ) * cdim( 2 ) +
                             ( ix + jnx + cdim( 0 ) ) % cdim( 0 ) * cdim( 1 ) *
                                 cdim( 2 );

                    // get neighbor coodinates
                    int jx, jy, jz;
                    cell_list.ijkBinIndex( jc, jx, jy, jz );

                    double contrib = 0;

                    // for all particles in the current cell
                    for ( int ii = 0; ii < cell_list.binSize( ix, iy, iz );
                          ++ii )
                    {
                        // get index of particle,
                        int i = cell_list.binOffset( ix, iy, iz ) + ii;
                        // and get the position of the particle
                        double rx = r( i, 0 );
                        double ry = r( i, 1 );
                        double rz = r( i, 2 );

                        double dx, dy, dz, d, pij;
                        int j;

                        double fx_i = 0.0;
                        double fy_i = 0.0;
                        double fz_i = 0.0;

                        double Ur_i = 0.0;
                        //then loop over the particles in the chosen neighbor cell
                        for ( int ij = 0; ij < cell_list.binSize( jx, jy, jz );
                              ++ij )
                        {
                            j = cell_list.binOffset( jx, jy, jz ) + ij;

                            // compute the distance between particles
                            dx = rx - r( j, 0 );
                            dy = ry - r( j, 1 );
                            dz = rz - r( j, 2 );
                            // use minimum image convention to get the shortest distance
                            dx -= round( dx / lx ) * lx;
                            dy -= round( dy / lx ) * lx;
                            dz -= round( dz / lx ) * lx;
                            d = sqrt( dx * dx + dy * dy + dz * dz );
                            // compute prefactor
                            double qij = q( i ) * q( j );
                            pij = ( d <= r_max ) * 0.5 * qij *
                                  erfc( alpha * d ) / d;
                            // compute force prefactor
                            double f_fact = ( d <= r_max ) * qij *
                                            ( 2.0 * sqrt( alpha / PI ) *
                                                  exp( -alpha * d * d ) +
                                              erfc( sqrt( alpha ) * d ) ) /
                                            ( d * d );
                            // compute force contributions
                            double fx = f_fact * dx;
                            double fy = f_fact * dy;
                            double fz = f_fact * dz;

                            // use Newton's 3rd law to add contributions to both particles
                            fx_i += fx;
                            fy_i += fy;
                            fz_i += fz;

                            Kokkos::atomic_add( &f( j, 0 ), -fx );
                            Kokkos::atomic_add( &f( j, 1 ), -fy );
                            Kokkos::atomic_add( &f( j, 2 ), -fz );

                            // update global energy reduction variable
                            Ur_i += pij;
                            Kokkos::atomic_add( &p( j ), pij );
                        }
                        // update forces for local particle
                        Kokkos::atomic_add( &p( i ), Ur_i );
                        Kokkos::atomic_add( &f( i, 0 ), fx_i );
                        Kokkos::atomic_add( &f( i, 1 ), fy_i );
                        Kokkos::atomic_add( &f( i, 2 ), fz_i );

                        contrib += 2.0 * Ur_i;
                    }
                    Ur_ic += contrib;
                },
                Ur_ijn );
            if ( Ur_ijn != Ur_ijn )
                printf( "ERROR: nan in n = %d\n", n );
            Ur_ij += Ur_ijn;
        }

        auto end_time_rij = std::chrono::high_resolution_clock::now();
        auto elapsed_time_rij = end_time_rij - start_time_rij;
        auto start_time_rii = std::chrono::high_resolution_clock::now();

        // energy contribution from local cells
        double Ur_ii = 0.0;
        Kokkos::parallel_reduce(
            n_cells,
            KOKKOS_LAMBDA( const int ic, double &Ur_ic ) {
                int ix, iy, iz;
                // get index of local cell
                cell_list.ijkBinIndex( ic, ix, iy, iz );

                double contrib = 0.0;

                // loop over the contents of the local cell
                for ( int ii = 0; ii < cell_list.binSize( ix, iy, iz ); ++ii )
                {
                    int i = cell_list.binOffset( ix, iy, iz ) + ii;
                    // get position of current particle
                    double rx = r( i, 0 );
                    double ry = r( i, 1 );
                    double rz = r( i, 2 );
                    double Ur_i = 0.0;
                    double fx_i = 0.0;
                    double fy_i = 0.0;
                    double fz_i = 0.0;

                    // loop over all other particles in current cell
                    for ( int ij = ii + 1; ij < cell_list.binSize( ix, iy, iz );
                          ++ij )
                    {
                        int j = cell_list.binOffset( ix, iy, iz ) + ij;
                        // compute distance and apply minimum image convention
                        // for periodic boundary conditions
                        double dx = rx - r( j, 0 );
                        double dy = ry - r( j, 1 );
                        double dz = rz - r( j, 2 );
                        dx -= round( dx / lx ) * lx;
                        dy -= round( dy / lx ) * lx;
                        dz -= round( dz / lx ) * lx;
                        double d = sqrt( dx * dx + dy * dy + dz * dz );
                        double qij = q( i ) * q( j );
                        // compute potential and force computations for
                        // both interacting particles
                        double pij =
                            ( d <= r_max ) * 0.5 * qij * erfc( alpha * d ) / d;
                        double f_fact =
                            ( d <= r_max ) * qij *
                            ( 2.0 * sqrt( alpha / PI ) * exp( -alpha * d * d ) +
                              erfc( sqrt( alpha ) * d ) ) /
                            ( d * d );

                        double fx = f_fact * dx;
                        double fy = f_fact * dy;
                        double fz = f_fact * dz;

                        // update redution variables for particle i
                        fx_i += fx;
                        fy_i += fy;
                        fz_i += fz;

                        // update forces for particle j
                        Kokkos::atomic_add( &f( j, 0 ), -fx );
                        Kokkos::atomic_add( &f( j, 1 ), -fy );
                        Kokkos::atomic_add( &f( j, 2 ), -fz );

                        Ur_i += pij;
                        // update potential energy for particle j
                        Kokkos::atomic_add( &p( j ), pij );
                    }
                    // update particle i
                    Kokkos::atomic_add( &p( i ), Ur_i );
                    Kokkos::atomic_add( &f( i, 0 ), fx_i );
                    Kokkos::atomic_add( &f( i, 1 ), fy_i );
                    Kokkos::atomic_add( &f( i, 2 ), fz_i );
                    contrib += 2.0 * Ur_i;
                }
                Ur_ic += contrib;
            },
            Ur_ii );
        
        Kokkos::fence();
        auto end_time_rii = std::chrono::high_resolution_clock::now();
        auto elapsed_time_rii = end_time_rii - start_time_rii;

        // total real-space contribution is the sum of interactions with
        // surrounding boxes and local boxes
        Ur = Ur_ij + Ur_ii;

        auto elapsed_time_r =
            elapsed_time_rsort + elapsed_time_rii + elapsed_time_rij;
        auto ns_elapsed_r =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                elapsed_time_r );

        std::cout << "real-space contribution: "
                  << ( ns_elapsed_r.count() / 1000000000.0 ) << " s "
                  << " = " << ( elapsed_time_rsort.count() / 1000000000.0 )
                  << " s (sorting) + "
                  << ( elapsed_time_rii.count() / 1000000000.0 )
                  << " s (local cell) + "
                  << ( elapsed_time_rij.count() / 1000000000.0 )
                  << " s (neighbor cells) | " << Ur << " = " << Ur_ii << " + "
                  << Ur_ij << std::endl;
    }

    // computation of self-energy contribution
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
        KOKKOS_LAMBDA( int idx, double &Uself_part ) {
            Uself_part += -alpha / PI_SQRT * q( idx ) * q( idx );
            p( idx ) += Uself_part;
        },
        Uself );
    Kokkos::fence();

    // TODO: enhancement - combine these 3 reductions
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
        KOKKOS_LAMBDA( int idx, double &Udip_part ) {
            double V = lx * ly * lz;
            double Udip_prefactor = 2 * PI / ( ( 1.0 + 2.0 * eps_r ) * V );
            Udip_part += Udip_prefactor * q( idx ) * r( idx, 0 );
        },
        Udip_vec[0] );
    Kokkos::fence();

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
        KOKKOS_LAMBDA( int idx, double &Udip_part ) {
            double V = lx * ly * lz;
            double Udip_prefactor = 2 * PI / ( ( 1.0 + 2.0 * eps_r ) * V );
            Udip_part += Udip_prefactor * q( idx ) * r( idx, 1 );
        },
        Udip_vec[1] );
    Kokkos::fence();

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
        KOKKOS_LAMBDA( int idx, double &Udip_part ) {
            double V = lx * ly * lz;
            double Udip_prefactor = 2 * PI / ( ( 1.0 + 2.0 * eps_r ) * V );
            Udip_part += Udip_prefactor * q( idx ) * r( idx, 2 );
        },
        Udip_vec[2] );
    Kokkos::fence();

    Udip = Udip_vec[0] * Udip_vec[0] + Udip_vec[1] * Udip_vec[1] +
           Udip_vec[2] * Udip_vec[2];

    std::cout << "Ur: " << Ur << " Uk: " << Uk << " Uself: " << Uself
              << " Udip: " << Udip << std::endl;

    return Ur + Uk + Uself + Udip;
}
