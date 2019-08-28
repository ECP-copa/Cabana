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

#ifndef TDS_EWALD_INCLUDED
#define TDS_EWALD_INCLUDED

#include <iostream>
#include "definitions.h"
#include "mpi.h"

class TEwald
{
  public:
    //constructor with accuracy
    TEwald(const double accuracy_threshold, 
           const long N,
           const double x_width, 
           const double y_width, 
           const double z_width,
           Kokkos::View<double*>& domain_width,
           MPI_Comm comm);
    
    //set base values for alpha, r_max, k_max
    TEwald(const double alpha, const double r_max, const double k_max);

    //compute Ewald Sum
    double compute(ParticleList& particles, const double x_width, const double y_width, const double z_width);

    // tune alpha, r_max, k_max to adhere to given accuracy
    void tune(const double accuracy_threshold, long N, const double x_width, const double y_width, const double z_width);

    // setter functions for parameters
    void set_alpha(double);
    void set_r_max(double);
    void set_k_max(double);

    // getter functions
    double get_alpha() {return _alpha;}
    double get_r_max() {return _r_max;}
    double get_k_max() {return _k_max;}

  private:
    double _alpha;
    double _r_max;
    double _k_max;

    // dielectric constant (1.0 = vacuum)
    double _eps_r = 1.0;

    double* EwaldUk_coeffs;

    Kokkos::View<double*> domain_width;

    MPI_Comm comm;
};

/// Functor to compute the total k-space energy contribution
struct EwaldUkFunctor 
{
    typedef double value_type;

    typedef Kokkos::View<double*>::size_type size_type;

    size_type value_count;

    /// k-space cutoff
    double k_max;
    /// splitting factor
    double alpha;
    /// system size
    double lx, ly, lz;
    /// number of particles
    int n_max;
    /// list of particle charges
    ParticleList::member_slice_type<Charge> q;
    /// list of particle positions
    ParticleList::member_slice_type<Position> r;
    /// list of particle forces
    ParticleList::member_slice_type<Force> f;

    /// constructor of the functor
    /// @param p        reference to the list of particles and their parameters
    /// @param _k_max   k-space cutoff
    /// @param _alpha   splitting factor
    /// @param _l       system size
    KOKKOS_INLINE_FUNCTION EwaldUkFunctor ( ParticleList& p, double _k_max, double _alpha, double _lx, double _ly, double _lz ) :
        value_count (1),
        k_max(_k_max),
        alpha(_alpha), 
        lx(_lx),
        ly(_ly),
        lz(_lz)
    {
        n_max = p.size();
        q = Cabana::slice<Charge>(p);
        r = Cabana::slice<Position>(p);
        f = Cabana::slice<Force>(p);
    }

    /// operator() to compute the k-space contribution to the energy
    /// @param kidx     index of the local k-vector
    /// @param sum      reduction variable
    KOKKOS_INLINE_FUNCTION void operator() (const size_type kidx, value_type& sum) const
    {
        // get discrete k-space cutoff
        int k_int = std::ceil(k_max);
        // compute coefficient
        double coeff = 4.0 * PI / (lx * ly * lz);
        /* Calculate kx, ky, kz. We iterate from -k_int to k_int in each dimension
        * Therefore number_of_wave_vectors = 2 * k_int + 1 and
        *
        * kx = kidx % number_of_wave_vectors - k_int
        *
        * ky = (kidx // number_of_wave_vectors) % number_of_wave_vectors - k_int
        *
        * or
        *
        * ky = (kidx % number_of_wave_vectors * number_of_wave_vectors) // number_of_wave_vectors - k_int
        *
        * kz = kidx // (number_of_wave_vectors * number_of_wave_vectors) - k_int
        */
        int kx = kidx % (2 * k_int + 1) - k_int;
        int ky = kidx % (4 * k_int * k_int + 4 * k_int + 1) / (2 * k_int + 1) - k_int;
        int kz = kidx / (4 * k_int * k_int + 4 * k_int + 1) - k_int;
        // check if the k-vector is not the zero-vector
        if (kx == 0 && ky == 0 && kz == 0) return;
        // check if k-vector is smaller than the k-space cutoff
        double kk = kx*kx + ky*ky + kz*kz;
        if (kk >= k_max*k_max) return;
        // compute wavevector
        double _kx = 2.0 * PI / lx * (double)kx;
        double _ky = 2.0 * PI / ly * (double)ky;
        double _kz = 2.0 * PI / lz * (double)kz;
        // compute self scalar product of wave vector
        kk = _kx*_kx + _ky*_ky + _kz*_kz;
        // compute k-vector specific coefficient
        double k_coeff = exp(-kk/(4*alpha*alpha))/kk;

        // sine and cosine contributions
        double U_cos = 0.0; 
        double U_sin = 0.0;

        // add up sine and cosine contributions
        for (int i = 0; i < n_max; ++i)
        {
            double kr = _kx * r(i,0) + _ky * r(i,1) + _kz * r(i,2);
            U_cos += q(i) * cos(kr);
            U_sin += q(i) * sin(kr);
        }
        // add results to the reduction variable
        sum += coeff * k_coeff * ( U_cos * U_cos + U_sin * U_sin );

        // force computation
        for (int i = 0; i < n_max; ++i)
        {
            double kr = _kx * r(i,0) + _ky * r(i,1) + _kz * r(i,2);
            Kokkos::atomic_add(&f(i, 0), k_coeff * 2.0 * q(i) * _kx * ( U_sin * cos(kr) - U_cos * sin(kr) ));
            Kokkos::atomic_add(&f(i, 1), k_coeff * 2.0 * q(i) * _ky * ( U_sin * cos(kr) - U_cos * sin(kr) ));
            Kokkos::atomic_add(&f(i, 2), k_coeff * 2.0 * q(i) * _kz * ( U_sin * cos(kr) - U_cos * sin(kr) ));
        }
    }

    /// join procedure during reduction
    /// @param dst       reduction variable
    /// @param src       contribution
    KOKKOS_INLINE_FUNCTION void join (volatile value_type& dst, const volatile value_type& src) const
    {
        dst += src;
    }

    /// initialization method of the reduction variable
    /// @param sum       reduction variable
    KOKKOS_INLINE_FUNCTION void init (value_type& sum) const
    {
        sum = 0.0;
    }
};

#endif
