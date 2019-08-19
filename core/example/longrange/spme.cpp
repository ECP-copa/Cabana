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

#include "spme.h"
#include "definitions.h"
#include "Cabana_LinkedCellList.hpp"
#include <chrono>
#include <cmath>
#include <sys/time.h>

#ifdef Cabana_ENABLE_Cuda
#include <cufft.h>
#include <cufftw.h>
#else
#include <fftw3.h>
#endif

/* Smooth particle mesh Ewald (SPME) solver
 * - This method, from Essman et al. (1995) computes long-range Coulombic forces
 *   with O(nlogN) scaling by using 3D FFT and interpolation to a mesh for the
 *   reciprocal space part of the Ewald sum.
 * - Here the method is used to compute electrostatic energies from an example
 *   arrangement of charged particles. Currently, we assume periodic boundary
 * conditions and a cubic mesh and arrangement of particles in 3 dimensions.
 * - Future versions will allow for assymetric meshes and non-uniform particle
 *   distributions, as well as 1 or 2 dimensions.
 */

// constructor given an accuracy
TPME::TPME( double accuracy, ParticleList particles, double lx, double ly,
            double lz )
{
    _r_max = 0.0;
    tune( accuracy, particles, lx, ly, lz );
}

// set base values for alpha, r_max, k_max
TPME::TPME( double alpha, double r_max )
{
    _alpha = alpha;
    _r_max = r_max;
}

// Tune to a given accuracy
void TPME::tune( double accuracy, ParticleList particles, double lx, double ly,
                 double lz )
{
    //Force symmetry for now
    if( lx != ly or lx != lz) {
       std::cout << "Must have cubes for now!";
       return;
    }

    auto q = Cabana::slice<Charge>( particles );

    const int N = particles.size();

    // Fincham 1994, Optimisation of the Ewald Sum for Large Systems
    // only valid for cubic systems (needs adjustement for non-cubic systems)
    constexpr double EXECUTION_TIME_RATIO_K_R = 2.0;
    double p = -log( accuracy );
    _alpha = pow( EXECUTION_TIME_RATIO_K_R, 1.0 / 6.0 ) * sqrt( p / PI ) *
             pow( N, 1.0 / 6.0 ) / lx;
    _k_max = pow( EXECUTION_TIME_RATIO_K_R, 1.0 / 6.0 ) * sqrt( p / PI ) *
             pow( N, 1.0 / 6.0 ) / lx * 2.0 * PI;
    _r_max = pow( EXECUTION_TIME_RATIO_K_R, 1.0 / 6.0 ) * sqrt( p / PI ) /
             pow( N, 1.0 / 6.0 ) * lx;
    _alpha = sqrt( p ) / _r_max;
    _k_max = 2.0 * sqrt( p ) * _alpha;

    std::cout << "tuned SPME values: " 
              << "N: " << N << " "
              << "accuracy: " << accuracy << " "
              << "r_max: " << _r_max 
              << " alpha: " << _alpha
              << " " << _k_max << std::endl;
}
//

// Compute a 1D cubic cardinal B-spline value, used in spreading charge to mesh
// points
//   Given the distance from the particle (x) in units of mesh spaces, this
//   computes the fraction of that charge to place at a mesh point x mesh spaces
//   away The cubic B-spline used here is shifted so that it is symmetric about
//   zero All cubic B-splines are smooth functions that go to zero and are
//   defined piecewise
KOKKOS_INLINE_FUNCTION
double TPME::oneDspline( double x )
{
    if ( x >= 0.0 and x < 1.0 )
    {
        return ( 1.0 / 6.0 ) * x * x * x;
    }
    else if ( x >= 1.0 and x <= 2.0 )
    {
        return -( 1.0 / 2.0 ) * x * x * x + 2.0 * x * x - 2.0 * x +
               ( 2.0 / 3.0 );
    }
    // Using the symmetry here, only need to define function between 0 and 2
    // Beware: This means all input to this function should be made positive
    else
    {
        return 0.0; // Zero if distance is >= 2 mesh spacings
    }
}

// Compute a 1-D Euler spline. This function is part of the "lattice structure
// factor" and is given by:
//   b(k, meshwidth) = exp(2*PI*i*3*k/meshwidth) / SUM_{l=0,2}(1Dspline(l+1) *
//   exp(2*PI*i*k*l/meshwidth)) when using a non-shifted cubic B-spline in the
//   charge spread, where meshwidth is the number of mesh points in that
//   dimension and k is the scaled fractional coordinate
KOKKOS_INLINE_FUNCTION
double TPME::oneDeuler( int k, int meshwidth )
{
    double denomreal = 0.0;
    double denomimag = 0.0;
    // Compute the denominator sum first, splitting the complex exponential into
    // sin and cos
    for ( int l = 0; l < 3; l++ )
    {
        denomreal +=
            TPME::oneDspline( fmin( 4.0 - ( l + 1.0 ), l + 1.0 ) ) *
            cos( 2.0 * PI * double( k ) * l / double( meshwidth ) );
        denomimag +=
            TPME::oneDspline( fmin( 4.0 - ( l + 1.0 ), l + 1.0 ) ) *
            sin( 2.0 * PI * double( k ) * l / double( meshwidth ) );
    }
    // Compute the numerator, again splitting the complex exponential
    double numreal = cos( 2.0 * PI * 3.0 * double( k ) / double( meshwidth ) );
    double numimag = sin( 2.0 * PI * 3.0 * double( k ) / double( meshwidth ) );
    // Returning the norm of the 1-D Euler spline
    return ( numreal * numreal + numimag * numimag ) /
           ( denomreal * denomreal + denomimag * denomimag );
}

// Compute the energy
double TPME::compute( ParticleList &particles, ParticleList &mesh, double lx,
                      double ly, double lz )
{
    //For now, force symmetry
    ly = lx;
    lz = lx;
            
    // Initialize energies: real-space, k-space (reciprocal space), self-energy
    // correction, dipole correction
    double Ur = 0.0, Uk = 0.0, Uself = 0.0, Udip = 0.0;
    double Udip_vec[3];

    // Particle slices
    auto r = Cabana::slice<Position>( particles );
    auto q = Cabana::slice<Charge>( particles );
    auto p = Cabana::slice<Potential>( particles );
    auto f = Cabana::slice<Force>( particles );

    // Mesh slices
    auto meshr = Cabana::slice<Position>( mesh );
    auto meshq = Cabana::slice<Charge>( mesh );

    // Number of particles
    const int n_max = particles.size();

    // Number of mesh points
    const int meshsize = mesh.size();

    double total_energy = 0.0;

    // Set the potential of each particle to zero
    auto init_p = KOKKOS_LAMBDA( const int idx ) { p( idx ) = 0.0; };
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
                          init_p );
    Kokkos::fence();

    double alpha = _alpha;
    // double k_max = _k_max;
    double r_max = _r_max;
    double eps_r = _eps_r;

    // std::chrono::time_point<std::chrono::steady_clock> starttime, starttime2,
    // endtime, endtime2; starttime = std::chrono::steady_clock::now();

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

                    for ( int ii = 0; ii < cell_list.binSize( ix, iy, iz );
                          ++ii )
                    {
                        int i = cell_list.binOffset( ix, iy, iz ) + ii;
                        double rx = r( i, 0 );
                        double ry = r( i, 1 );
                        double rz = r( i, 2 );

                        double dx, dy, dz, d, pij;
                        int j;

                        double fx_i = 0.0;
                        double fy_i = 0.0;
                        double fz_i = 0.0;

                        double Ur_i = 0.0;
                        for ( int ij = 0; ij < cell_list.binSize( jx, jy, jz );
                              ++ij )
                        {
                            j = cell_list.binOffset( jx, jy, jz ) + ij;

                            dx = rx - r( j, 0 );
                            dy = ry - r( j, 1 );
                            dz = rz - r( j, 2 );
                            dx -= round( dx / lx ) * lx;
                            dy -= round( dy / lx ) * lx;
                            dz -= round( dz / lx ) * lx;
                            d = sqrt( dx * dx + dy * dy + dz * dz );
                            double qij = q( i ) * q( j );
                            pij = ( d <= r_max ) * 0.5 * qij *
                                  erfc( alpha * d ) / d;
                            double f_fact = ( d <= r_max ) * qij *
                                            ( 2.0 * sqrt( alpha / PI ) *
                                                  exp( -alpha * d * d ) +
                                              erfc( sqrt( alpha ) * d ) ) /
                                            ( d * d );

                            double fx = f_fact * dx;
                            double fy = f_fact * dy;
                            double fz = f_fact * dz;

                            fx_i += fx;
                            fy_i += fy;
                            fz_i += fz;

                            Kokkos::atomic_add( &f( j, 0 ), -fx );
                            Kokkos::atomic_add( &f( j, 1 ), -fy );
                            Kokkos::atomic_add( &f( j, 2 ), -fz );

                            Ur_i += pij;
                            Kokkos::atomic_add( &p( j ), pij );
                        }
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
                cell_list.ijkBinIndex( ic, ix, iy, iz );

                double contrib = 0.0;
                for ( int ii = 0; ii < cell_list.binSize( ix, iy, iz ); ++ii )
                {
                    int i = cell_list.binOffset( ix, iy, iz ) + ii;
                    double rx = r( i, 0 );
                    double ry = r( i, 1 );
                    double rz = r( i, 2 );
                    double Ur_i = 0.0;
                    double fx_i = 0.0;
                    double fy_i = 0.0;
                    double fz_i = 0.0;

                    for ( int ij = ii + 1; ij < cell_list.binSize( ix, iy, iz );
                          ++ij )
                    {
                        int j = cell_list.binOffset( ix, iy, iz ) + ij;
                        double dx = rx - r( j, 0 );
                        double dy = ry - r( j, 1 );
                        double dz = rz - r( j, 2 );
                        dx -= round( dx / lx ) * lx;
                        dy -= round( dy / lx ) * lx;
                        dz -= round( dz / lx ) * lx;
                        double d = sqrt( dx * dx + dy * dy + dz * dz );
                        double qij = q( i ) * q( j );
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

                        fx_i += fx;
                        fy_i += fy;
                        fz_i += fz;

                        Kokkos::atomic_add( &f( j, 0 ), -fx );
                        Kokkos::atomic_add( &f( j, 1 ), -fy );
                        Kokkos::atomic_add( &f( j, 2 ), -fz );

                        Ur_i += pij;
                        Kokkos::atomic_add( &p( j ), pij );
                    }
                    Kokkos::atomic_add( &p( i ), Ur_i );
                    Kokkos::atomic_add( &f( i, 0 ), fx_i );
                    Kokkos::atomic_add( &f( i, 1 ), fy_i );
                    Kokkos::atomic_add( &f( i, 2 ), fz_i );
                    contrib += 2.0 * Ur_i;
                }
                Ur_ic += contrib;
            },
            Ur_ii );

        auto end_time_rii = std::chrono::high_resolution_clock::now();
        auto elapsed_time_rii = end_time_rii - start_time_rii;

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
    


    // computation reciprocal-space contribution

    // First, spread the charges onto the mesh

    double spacing =
        meshr( 1, 0 ) -
        meshr( 0,
               0 ); // how far apart the mesh points are (assumed uniform cubic)

    // Current method: Each mesh point loops over *all* particles, and gathers
    // charge to it
    //                 according to spline interpolation.
    // Alternatives: Looping over all particles, using atomics to scatter charge
    // to mesh points Also, would be nice to loop only over neighbors - spline
    // is only 2 mesh points away maximum
    auto spread_q = KOKKOS_LAMBDA( const int idx )
    {
        double xdist, ydist, zdist;
        for ( size_t pidx = 0; pidx < particles.size(); ++pidx )
        {
            // x-distance between mesh point and particle
            xdist =
                fmin( fmin( std::abs( meshr( idx, 0 ) - r( pidx, 0 ) ),
                                    std::abs( meshr( idx, 0 ) -
                                              ( r( pidx, 0 ) + 1.0 ) ) ),
                          std::abs( meshr( idx, 0 ) -
                                    ( r( pidx, 0 ) -
                                      1.0 ) ) ); // account for periodic bndry
            // y-distance between mesh point and particle
            ydist =
                fmin( fmin( std::abs( meshr( idx, 1 ) - r( pidx, 1 ) ),
                                    std::abs( meshr( idx, 1 ) -
                                              ( r( pidx, 1 ) + 1.0 ) ) ),
                          std::abs( meshr( idx, 1 ) -
                                    ( r( pidx, 1 ) -
                                      1.0 ) ) ); // account for periodic bndry
            // z-distance between mesh point and particle
            zdist =
                fmin( fmin( std::abs( meshr( idx, 2 ) - r( pidx, 2 ) ),
                                    std::abs( meshr( idx, 2 ) -
                                              ( r( pidx, 2 ) + 1.0 ) ) ),
                          std::abs( meshr( idx, 2 ) -
                                    ( r( pidx, 2 ) -
                                      1.0 ) ) ); // account for periodic bndry

            if ( xdist <= 2.0 * spacing and ydist <= 2.0 * spacing and
                 zdist <= 2.0 * spacing ) // more efficient way to do this? Skip
                                          // it? May be unnecessary.
            {
                // add charge to mesh point according to spline
                meshq( idx ) += q( pidx ) *
                                TPME::oneDspline( 2.0 - ( xdist / spacing ) ) *
                                TPME::oneDspline( 2.0 - ( ydist / spacing ) ) *
                                TPME::oneDspline( 2.0 - ( zdist / spacing ) );
            }
        }
    };
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, meshsize ),
                          spread_q );
    Kokkos::fence();

    // starttime2 = std::chrono::steady_clock::now();

    // std::cout << "Creating BC array" << std::endl;
    // Create "B*C" array (called theta in Eqn 4.7 SPME paper by Essman)
    // Can be done at start of run and stored
    //"meshwidth" should be number of mesh points along any axis.
    int meshwidth =
        std::round( std::pow( meshsize, 1.0 / 3.0 ) ); // Assuming cubic mesh

// Calculating the values of the BC array involves first shifting the fractional
// coords then compute the B and C arrays as described in the paper This can be
// done once at the start of a run if the mesh stays constant
#ifdef Cabana_ENABLE_Cuda
    cufftDoubleComplex *BC;
    cudaMallocManaged( (void **)&BC, sizeof( cufftDoubleComplex ) * meshsize );
#else
    fftw_complex *BC;
    BC = (fftw_complex *)fftw_malloc( sizeof( fftw_complex ) * meshsize );
#endif
    // TODO: Is this a good place for Kokkos Hierarchical parallelism?
    auto BC_functor = KOKKOS_LAMBDA( const int kx )
    {
        int ky, kz, mx, my, mz, idx;
        for ( ky = 0; ky < meshwidth; ky++ )
        {
            for ( kz = 0; kz < meshwidth; kz++ )
            {
                idx = kx + ( ky * meshwidth ) + ( kz * meshwidth * meshwidth );
                if ( kx + ky + kz > 0 )
                {
                    // Shift the C array
                    mx = kx;
                    my = ky;
                    mz = kz;
                    if ( mx > meshwidth / 2.0 )
                    {
                        mx = kx - meshwidth;
                    }
                    if ( my > meshwidth / 2.0 )
                    {
                        my = ky - meshwidth;
                    }
                    if ( mz > meshwidth / 2.0 )
                    {
                        mz = kz - meshwidth;
                    }
                    double m2 = ( mx * mx + my * my +
                                  mz * mz ); // Unnecessary extra variable

// Calculate BC. Why store the imag part at all?
#ifdef Cabana_ENABLE_Cuda
                    BC[idx].x = TPME::oneDeuler( kx, meshwidth ) *
                                TPME::oneDeuler( ky, meshwidth ) *
                                TPME::oneDeuler( kz, meshwidth ) *
                                exp( -PI * PI * m2 / ( alpha * alpha ) ) /
                                ( PI * lx * ly * lz * m2 );
                    BC[idx].y = 0.0; // imag part
#else
                    BC[idx][0] = TPME::oneDeuler( kx, meshwidth ) *
                                 TPME::oneDeuler( ky, meshwidth ) *
                                 TPME::oneDeuler( kz, meshwidth ) *
                                 exp( -PI * PI * m2 / ( alpha * alpha ) ) /
                                 ( PI * lx * ly * lz * m2 );
                    BC[idx][1] = 0.0; // imag part
#endif
                }
                else
                {
#ifdef Cabana_ENABLE_Cuda
                    BC[idx].x = 0.0;
                    BC[idx].y = 0.0; // set origin element to zero
#else
                    BC[idx][0] = 0.0;
                    BC[idx][1] = 0.0; // set origin element to zero
#endif
                }
            }
        }
    };
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, meshwidth ),
                          BC_functor );
    Kokkos::fence();

// endtime = std::chrono::steady_clock::now();

// Next, solve Poisson's equation taking some FFTs of charges on mesh grid
// The plan here is to perform an inverse FFT on the mesh charge, then multiply
//  the norm of that result (in reciprocal space) by the BC array

// Set up the real-space charge and reciprocal-space charge
#ifdef Cabana_ENABLE_Cuda
    cufftDoubleComplex *Qr, *Qktest;
    cufftHandle plantest;
    cudaMallocManaged( (void **)&Qr, sizeof( fftw_complex ) * meshsize );
    cudaMallocManaged( (void **)&Qktest, sizeof( fftw_complex ) * meshsize );
    // Copy charges into real input array
    auto copy_charge = KOKKOS_LAMBDA( const int idx )
    {
        Qr[idx].x = meshq( idx );
        Qr[idx].y = 0.0;
    };
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, meshsize ),
                          copy_charge );
#else
    fftw_complex *Qr, *Qktest;
    fftw_plan plantest;
    Qr = (fftw_complex *)fftw_malloc( sizeof( fftw_complex ) * meshsize );
    Qktest = (fftw_complex *)fftw_malloc( sizeof( fftw_complex ) * meshsize );
    // Copy charges into real input array
    auto copy_charge = KOKKOS_LAMBDA( const int idx )
    {
        Qr[idx][0] = meshq( idx );
        Qr[idx][1] = 0.0;
    };
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, meshsize ),
                          copy_charge );
#endif
    Kokkos::fence();

// Plan out that IFFT on the real-space charge mesh
#ifdef Cabana_ENABLE_Cuda
    cufftPlan3d( &plantest, meshwidth, meshwidth, meshwidth, CUFFT_Z2Z );
    cufftExecZ2Z( plantest, Qr, Qktest, CUFFT_INVERSE ); // IFFT on Q

    Kokkos::parallel_reduce(
        meshsize,
        KOKKOS_LAMBDA( const int idx, double &Uk_part ) {
            Uk_part += BC[idx].x * ( ( Qktest[idx].x * Qktest[idx].x ) +
                                     ( Qktest[idx].y * Qktest[idx].y ) );
        },
        Uk );
    Kokkos::fence();
    cufftDestroy( plantest );
#else
    plantest = fftw_plan_dft_3d( meshwidth, meshwidth, meshwidth, Qr, Qktest,
                                 FFTW_BACKWARD, FFTW_ESTIMATE );
    fftw_execute( plantest ); // IFFT on Q
    Kokkos::parallel_reduce(
        meshsize,
        KOKKOS_LAMBDA( const int idx, double &Uk_part ) {
            Uk_part += BC[idx][0] * ( ( Qktest[idx][0] * Qktest[idx][0] ) +
                                      ( Qktest[idx][1] * Qktest[idx][1] ) );
        },
        Uk );
    Kokkos::fence();
    fftw_destroy_plan( plantest );
#endif

    Uk *= 0.5;

    // computation of self-energy contribution
    Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
                             KOKKOS_LAMBDA( int idx, double &Uself_part ) {
                                 Uself_part +=
                                     -alpha / PI_SQRT * q( idx ) * q( idx );
                                 p( idx ) += Uself_part;
                             },
                             Uself );
    Kokkos::fence();

    // computation of dipole correction to energy
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
    
    total_energy = Ur + Uk + Uself + Udip;

    std::cout << "SPME (real-space): " << Ur << std::endl;
    std::cout << "SPME (k-space):    " << Uk << std::endl;
    std::cout << "SPME (self):       " << Uself << std::endl;
    std::cout << "SPME (dipole):     " << Udip << std::endl;
    std::cout << "SPME (total):      " << total_energy << std::endl;

#ifndef Cabana_ENABLE_Cuda
    fftw_cleanup();
#endif
    return total_energy;
}
