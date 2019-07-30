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
    _k_max_int[0] = std::ceil( _k_max );
    _k_max_int[1] = std::ceil( _k_max );
    _k_max_int[2] = std::ceil( _k_max );
}

void TEwald::tune( double accuracy, ParticleList particles, double lx,
                   double ly, double lz )
{
    typedef Kokkos::MinLoc<double, int> reducer_type;
    typedef reducer_type::value_type value_type;
    value_type error_estimate;

    auto q = Cabana::slice<Charge>( particles );
    double q_sum;

    const int N_alpha = 200;
    const int N_k = 2000;

    const int n_max = particles.size();

    // calculate sum of charge squares
    Kokkos::parallel_reduce( n_max,
                             KOKKOS_LAMBDA( const int idx, double &q_part ) {
                                 q_part += q( idx ) * q( idx );
                             },
                             q_sum );
    Kokkos::fence();
    // Not sure, but think fence required since default memory or exec space
    // could be CudaUVM

    double r_max = _r_max = std::min( 0.49 * lx, 0.1 * lx + 1.0 );
    Kokkos::parallel_reduce(
        "MinLocReduce", N_alpha * N_k,
        KOKKOS_LAMBDA( const int &idx, value_type &team_errorest ) {
            int ia = idx % N_alpha;
            int ik = idx / N_alpha;

            double alpha = (double)ia * 0.05 + 1.0;
            double k_max = (double)ik * 0.05;

            double delta_Ur = q_sum * sqrt( 0.5 * r_max / ( lx * ly * lz ) ) *
                              std::pow( alpha * r_max, -2.0 ) *
                              exp( -alpha * alpha * r_max * r_max );

            double delta_Uk =
                q_sum * alpha / PI_SQ * std::pow( k_max, -1.5 ) *
                exp( -std::pow( PI * k_max / ( alpha * lx ), 2 ) );

            double delta = delta_Ur + delta_Uk;
            Kokkos::pair<double, double> values( alpha, k_max );

            if ( ( delta < team_errorest.val ) && ( delta > 0.8 * accuracy ) )
            {
                team_errorest.val = delta;
                team_errorest.loc = idx;
            }
        },
        reducer_type( error_estimate ) );
    Kokkos::fence();
    // Not sure, but think fence required since default memory or exec space
    // could be CudaUVM

    _alpha = (double)( error_estimate.loc % N_alpha ) * 0.05 + 1.0;
    _k_max = (double)( error_estimate.loc / N_alpha ) * 0.05;

    _k_max_int[0] = _k_max_int[1] = _k_max_int[2] = std::ceil( _k_max );
}

double TEwald::compute( ParticleList &particles, double lx, double ly,
                        double lz )
{

    double Ur = 0.0, Uk = 0.0, Uself = 0.0, Udip = 0.0;
    double Udip_vec[3];

    auto r = Cabana::slice<Position>( particles );
    auto q = Cabana::slice<Charge>( particles );
    auto p = Cabana::slice<Potential>( particles );

    int n_max = particles.size();

    auto init_p = KOKKOS_LAMBDA( const int idx ) { p( idx ) = 0.0; };
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
                          init_p );
    Kokkos::fence();

    double alpha = _alpha;
    double k_max = _k_max;
    double r_max = _r_max;
    double eps_r = _eps_r;

#ifdef Cabana_ENABLE_Cuda
    Kokkos::View<int *, MemorySpace> k_max_int( "k_max_int", 3 );
    for ( auto i = 0; i < 3; ++i )
    {
        k_max_int[i] = _k_max_int[i];
    } // TODO: enhancement - use Views instead. No need for macros.
#else
    int *k_max_int = &( _k_max_int[0] );
#endif
    // computation real-space contribution
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
        KOKKOS_LAMBDA( int idx, double &Ur_part ) {
            double d[SPACE_DIM];
            double k;
            // For each particle with charge q, the real space contribution to
            // energy is Ur_part = 0.5*q*SUM_i(q_i*erfc(alpha*dist)/dist) The
            // sum is over all other particles in the cell and in neighboring
            // images up to some real-space cutoff distance "r_max"
            //
            // Self-energy terms are included then corrected for later, with one
            // exception:
            // the self-energy term where kx=ky=kz here is explicitly excluded
            // in the method as this would cause a division by zero
            for ( auto i = 0; i < n_max;
                  ++i ) // For each particle (including self)
            {
                // compute distance in x,y,z and charge multiple
                for ( auto j = 0; j < 3; ++j )
                    d[j] = r( idx, j ) - r( i, j );
                double qiqj = q( idx ) * q( i );
                for ( auto kx = 0; kx <= k_max_int[0]; ++kx )
                {
                    // check if cell within r_max distance in x
                    k = (double)kx * lx;
                    if ( k - lx > r_max )
                        continue;
                    for ( auto ky = 0; ky <= k_max_int[1]; ++ky )
                    {
                        // check if cell within r_max distance in x+y
                        k = sqrt( (double)kx * (double)kx * lx * lx +
                                  (double)ky * (double)ky * ly * ly );
                        if ( k - lx > r_max )
                            continue;
                        for ( auto kz = 0; kz <= k_max_int[2]; ++kz )
                        {
                            // Exclude self-energy term when kx=ky=kz
                            if ( kx == 0 && ky == 0 && kz == 0 && i == idx )
                                continue;
                            // check if cell within r_max distance in x+y+z
                            k = sqrt( (double)kx * (double)kx * lx * lx +
                                      (double)ky * (double)ky * ly * ly +
                                      (double)kz * (double)kz * lz * lz );
                            if ( k - lx > r_max )
                                continue;
                            // check if particle distance is less than r_max
                            double scal = ( d[0] + (double)kx * lx ) *
                                              ( d[0] + (double)kx * lx ) +
                                          ( d[1] + (double)ky * ly ) *
                                              ( d[1] + (double)ky * ly ) +
                                          ( d[2] + (double)kz * lz ) *
                                              ( d[2] + (double)kz * lz );
                            scal = sqrt( scal );
                            if ( scal > r_max )
                                continue;
                            // Compute real-space energy contribution of
                            // interaction
                            Ur_part += qiqj * erfc( alpha * scal ) / scal;
                        }
                    }
                }
            }
            Ur_part *= 0.5;
            p( idx ) += Ur_part;
        },
        Ur );
    Kokkos::fence();

    // computation reciprocal-space contribution
    int k_int = std::ceil( _k_max );
    // The reciprocal space energy contribution of the Ewald Sum
    // iterates over all vectors k (reciprocal lattice vector) within the bounds
    // k^2 < kmax^2 Indexing in 3D gives (k_int+1)^3 vectors to sum over
    const int max_idx =
        8 * k_int * k_int * k_int + 12 * k_int * k_int + 6 * k_int + 1;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>( 0, max_idx ),
        KOKKOS_LAMBDA( int idx, double &Uk_part ) {
            double kk;
            double kr;
            double coeff;
            double sum_re = 0.0;
            double sum_im = 0.0;
            // Math to go from indexing to kx,ky,kz
            int kx = idx % ( 2 * k_int + 1 ) - k_int;
            int ky = idx % ( 4 * k_int * k_int + 4 * k_int + 1 ) /
                         ( 2 * k_int + 1 ) -
                     k_int;
            int kz = idx / ( 4 * k_int * k_int + 4 * k_int + 1 ) - k_int;

            if ( kx == 0 && ky == 0 && kz == 0 )
                return; // The kx=ky=kz term is excluded from the sum
            kk = kx * kx + ky * ky + kz * kz;
            if ( kk > k_max * k_max )
                return; // Check to ensure within kmax bounds
            // This sum involves calculating a coefficient given by:
            // coeff(k) = (2/L^2) * exp(-(PI*k/(alpha*L))^2)/(k^2)
            coeff = 2.0 / ( lx * lx ) *
                    exp( -PI_SQ / ( alpha * alpha * lx * lx ) * kk ) / kk;
            // The sum's terms for each k-vector are then
            // sum(k) = coeff(k) * |S(k)|^2
            // where S(k) is a sum over all particles in the unit cell S(k) =
            // SUM_i[ q_i * exp(I* k.r) ] Now compute S(k), summing over all
            // particles and computing this influence function S(k) with real
            // and imag parts for the complex exponential
            for ( auto j = 0; j < n_max; ++j )
            {
                kr = 2.0 * PI *
                     ( kx * r( j, 0 ) + ky * r( j, 1 ) + kz * r( j, 2 ) ) /
                     lx;                      // 2*PI*(dotproduct of k and r)/L
                sum_re += q( j ) * cos( kr ); // real part
                sum_im += q( j ) * sin( kr ); // imag part
            }
            for ( auto j = 0; j < n_max; ++j )
            {
                // Compute the norm of the influence function for each particle
                // and k combo
                kr = 2.0 * PI *
                     ( kx * r( j, 0 ) + ky * r( j, 1 ) + kz * r( j, 2 ) ) / lx;
                double re = sum_re * cos( kr ); // realpart * realpart = real
                double im = sum_im * sin( kr ); // imagpart * imagpart = real

                // For a given k, each particle's reciprocal-space contribution
                // to the energy is given by p_k = (L/(4*PI)) * q * coeff *
                // |S(k)|^2
                Kokkos::atomic_add( &p( j ), q( j ) * coeff * ( re + im ) * lx /
                                                 ( 4.0 * PI ) );
                // add all particle's contributions to recip energy for this
                // k-vector
                Uk_part += q( j ) * coeff * ( re + im ) * lx / ( 4.0 * PI );
            }
        },
        Uk );
    Kokkos::fence();

    // computation of self-energy contribution
    Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>( 0, n_max ),
                             KOKKOS_LAMBDA( int idx, double &Uself_part ) {
                                 Uself_part +=
                                     -alpha / PI_SQRT * q( idx ) * q( idx );
                                 p( idx ) += Uself_part;
                             },
                             Uself );
    Kokkos::fence();

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

    return Ur + Uk + Uself + Udip;
}
