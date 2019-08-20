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

class TEwald
{
  public:
    //constructor with accuracy
    TEwald(const double accuracy_threshold, ParticleList particles, const double x_width, const double y_width, const double z_width);
    
    //set base values for alpha, r_max, k_max
    TEwald(const double alpha, const double r_max, const double k_max);

    //compute Ewald Sum
    double compute(ParticleList& particles, const double x_width, const double y_width, const double z_width);

    // tune alpha, r_max, k_max to adhere to given accuracy
    void tune(const double accuracy_threshold, ParticleList particles, const double x_width, const double y_width, const double z_width);

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
    int _k_max_int[3];

    // dielectric constant (1.0 = vacuum)
    double _eps_r = 1.0;

    double* EwaldUk_coeffs;
};

/// struct used in the Kokkos functors in order to reduce the forces
struct EwaldKSpaceValueType
{
    /// first component
    double vx;
    /// second component
    double vy; 
    /// third component
    double vz;

    /// constexpr constructor without input arguments (required from Kokkos)
    KOKKOS_FORCEINLINE_FUNCTION constexpr EwaldKSpaceValueType() : vx(0.0), vy(0.0), vz(0.0)
    {
    }

    /// constexpr constructor with input arguments (required from Kokkos)
    /// @param d     value with which each component is initialized
    KOKKOS_FORCEINLINE_FUNCTION constexpr EwaldKSpaceValueType(double d) : vx(d), vy(d), vz(d)
    {
    }

    /// overloaded incremental operator to add the content of another EwaldKSpaceValueType object
    /// @param other     the object which is added to the local object
    KOKKOS_FORCEINLINE_FUNCTION void operator+=(EwaldKSpaceValueType const& other) 
    {
        vx += other.vx;
        vy += other.vy;
        vz += other.vz;
    }
   
    /// overloaded incremental operator to add the content of another EwaldKSpaceValueType object
    /// (volatile version)
    /// @param other     the object which is added to the local object
    KOKKOS_FORCEINLINE_FUNCTION void operator+=(EwaldKSpaceValueType const volatile& other) volatile 
    {
        vx += other.vx;
        vy += other.vy;
        vz += other.vz;
    }
};

/// forward definition of the Kokkos::reduction_identity for the EwaldKSpaceValueType
namespace Kokkos {
template<>
struct reduction_identity<EwaldKSpaceValueType> 
{
    KOKKOS_FORCEINLINE_FUNCTION constexpr static EwaldKSpaceValueType sum()  {return static_cast<EwaldKSpaceValueType>(0.0);}
    KOKKOS_FORCEINLINE_FUNCTION constexpr static EwaldKSpaceValueType prod() {return static_cast<EwaldKSpaceValueType>(1.0);}
    KOKKOS_FORCEINLINE_FUNCTION constexpr static EwaldKSpaceValueType max()  {return static_cast<EwaldKSpaceValueType>(-DBL_MAX);}
    KOKKOS_FORCEINLINE_FUNCTION constexpr static EwaldKSpaceValueType min()  {return static_cast<EwaldKSpaceValueType>(DBL_MAX);}
};
}

/// Functor to compute the k-space contribution of a give k-vector to a given pair of particles i,j.
/// Lowest level of the nested parallelism hierarchy.
template <class ExecutionSpace>
struct EwaldUkForcesIJKFunctor
{
    typedef Kokkos::View<double*>::size_type size_type;


    /// TeamPolicy used in nested parallel hierarchy
    typename Kokkos::TeamPolicy<ExecutionSpace>::member_type member;
    /// array size (forces therefore hardcoded to 3)
    size_type value_count;
    /// k-space cutoff
    double k_max;
    /// spliting factor
    double alpha;
    /// list of particles positions
    const ParticleList::member_slice_type<Position> r;
    /// system size
    double lx, ly, lz;
    /// index of particle i
    int ii;
    /// index of particle j
    int ij;

    /// constructor of the functor
    /// @param mem       TeamPolicy used for the nested parallelism
    /// @param p         reference to the list of positions of the particles
    /// @param _k_max    k-space cutoff
    /// @param _alpha    splitting factor
    /// @param _l        system size
    /// @param _i        index of first particle
    /// @param _j        index of second particle
    KOKKOS_INLINE_FUNCTION EwaldUkForcesIJKFunctor ( 
            typename Kokkos::TeamPolicy<ExecutionSpace>::member_type mem, 
            const ParticleList::member_slice_type<Position>& p, 
            double _k_max, 
            double _alpha,
            double _lx,
            double _ly,
            double _lz,
            int _i, 
            int _j ) :
        member(mem),
        value_count (3),
        k_max(_k_max),
        alpha(_alpha),
        r(p),
        lx(_lx),
        ly(_ly),
        lz(_lz),
        ii(_i),
        ij(_j) 
    {
    }

    /// operator() to compute the contribution of k-vector kidx to the forces between
    /// particles ii and ij.
    /// @param kidx     local k-vector to be computed
    /// @param sum      reduction variable to sum up all contributions for the particle pair ii, ij 
    KOKKOS_INLINE_FUNCTION void operator() (const int kidx, EwaldKSpaceValueType& sum) const
    {
        // get discretized k-space cutoff
        int k_int = std::ceil(k_max);
        // compute coefficient
        double coeff = 4.0 * PI / (lx * ly * lz);
        // compute correct k-vector
        int kx = kidx % (2 * k_int + 1) - k_int;
        int ky = kidx % (4 * k_int * k_int + 4 * k_int + 1) / (2 * k_int + 1) - k_int;
        int kz = kidx / (4 * k_int * k_int + 4 * k_int + 1) - k_int;
        // check for if k-vector is not the central box
        if (kx == 0 && ky == 0 && kz == 0) return;
        // scalar product of the k-vector to
        double kk = kx*kx + ky*ky + kz*kz;
        // check if the k-vector is within the k-space cutoff
        if (kk >= k_max*k_max) return;
        // compute wave vector
        double _kx = 2.0 * PI / lx * (double)kx;
        double _ky = 2.0 * PI / ly * (double)ky;
        double _kz = 2.0 * PI / lz * (double)kz;
        // compute self product for the wave vector
        kk = _kx*_kx + _ky*_ky + _kz*_kz;
        // update coefficient
        coeff *= exp(-kk)/(4.0 * alpha) / kk;
        // one thread of the ThreadTeam adds up local contribution 
        Kokkos::single(Kokkos::PerThread(member), [&]
        {
            sum.vx += coeff * kx * sin ( kx * (r( ij,0 ) -r( ii,0 )));
            sum.vy += coeff * ky * sin ( ky * (r( ij,1 ) -r( ii,1 )));
            sum.vz += coeff * kz * sin ( kz * (r( ij,2 ) -r( ii,2 )));
        });
    }

    /// join procedure during reduction
    /// @param dst       reduction variable
    /// @param src       contribution
    KOKKOS_INLINE_FUNCTION void join (volatile EwaldKSpaceValueType dst, const volatile EwaldKSpaceValueType src) const
    {
        dst += src;
    }

    /// initialization method of the reduction variable
    /// @param sum       reduction variable
    KOKKOS_INLINE_FUNCTION void init (EwaldKSpaceValueType sum) const
    {
        sum.vx = 0.0;
        sum.vy = 0.0;
        sum.vz = 0.0;
    }
};


/// Functor to compute the contribution of particles j to the forces acting on particle i.
/// Second level of nested parallelism hierarchy.
template <class ExecutionSpace>
struct EwaldUkForcesIJFunctor 
{
    typedef ExecutionSpace execution_space;

    typedef Kokkos::View<double*>::size_type size_type;


    /// used TeamPolicy for the nested parallelism
    typename Kokkos::TeamPolicy<ExecutionSpace>::member_type member;
    /// array size (forces, therefore hardcoded to 3)
    size_type value_count;
    /// k-space cutoff
    double k_max;
    /// splitting factor
    double alpha;
    /// list of positions
    ParticleList::member_slice_type<Position> r;
    /// list of charges
    ParticleList::member_slice_type<Charge> q;
    /// system size
    double lx, ly, lz;
    /// particle index
    int i;

    /// constructor of the functor
    /// @param mem       TeamPolicy used to create nested parallelism
    /// @param p         reference to the used positions
    /// @param q_        reference to the used charges
    /// @param _k_max    k-space cutoff
    /// @param _alpha    splitting factor
    /// @param _l        system size
    /// @param _i        index of the particle the forces are computed for
    KOKKOS_INLINE_FUNCTION EwaldUkForcesIJFunctor ( 
            typename Kokkos::TeamPolicy<ExecutionSpace>::member_type mem, 
            const ParticleList::member_slice_type<Position>& p, 
            const ParticleList::member_slice_type<Charge>& q_,  
            double _k_max, 
            double _alpha,
            double _lx,
            double _ly,
            double _lz,
            int _i ) :
        member(mem),
        value_count (3),
        k_max(_k_max),
        alpha(_alpha), 
        r(p),
        q(q_),
        lx(_lx),
        ly(_ly),
        lz(_lz),
        i(_i)
    {}

    /// operator() to compute the force contribution of other particles j to the particle i
    /// @param j        index of partner particle
    /// @param sum      reduction variable for the forces
    KOKKOS_INLINE_FUNCTION void operator() (const int j, EwaldKSpaceValueType& sum) const
    {
        // compute discrete k-space cutoff
        int k_int = std::ceil(k_max);
        // compute number of needed k-vectors
        int n_k = 8 * k_int * k_int * k_int + 12 * k_int * k_int + 6 * k_int + 1;
        
        // prepare variable to store force contributions to
        EwaldKSpaceValueType EwaldUk_fij;
        EwaldUk_fij.vx = 0.0;
        EwaldUk_fij.vy = 0.0;
        EwaldUk_fij.vz = 0.0;
        // initialize functor for the computation of the k-vector contributions
        EwaldUkForcesIJKFunctor<ExecutionSpace> ukijk(member, r, k_max, alpha, lx, ly, lz, i, j);

        // call parallel_reduce on the functor to reduce all the contributions
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange( member, n_k ), ukijk, EwaldUk_fij);

        // one Thread adds the contributions to the local contribution for pair i,j
        Kokkos::single(Kokkos::PerThread(member), [&]
        {
            sum.vx += q(j) * EwaldUk_fij.vx;
            sum.vy += q(j) * EwaldUk_fij.vy;
            sum.vz += q(j) * EwaldUk_fij.vz;
        });
    }

    /// join procedure during reduction
    /// @param dst       reduction variable
    /// @param src       contribution
    KOKKOS_INLINE_FUNCTION void join (volatile EwaldKSpaceValueType dst, const volatile EwaldKSpaceValueType src) const
    {
        dst += src;
    }

    /// initialization method of the reduction variable
    /// @param sum       reduction variable
    KOKKOS_INLINE_FUNCTION void init (EwaldKSpaceValueType sum) const
    {
        sum.vx = 0.0;
        sum.vy = 0.0;
        sum.vz = 0.0;
    }
};

/// Functor to compute all forces acting on a particle i.
/// Highest level of nested parallelism
template <class ExecutionSpace>
struct EwaldUkForcesFunctor 
{

    typedef Kokkos::View<double*>::size_type size_type;

    size_type value_count;

    /// number of k-vectors
    int n_k;
    /// k-space cutoff
    double k_max;
    /// splitting factor
    double alpha;
    /// particle positions
    ParticleList::member_slice_type<Position> r;
    /// particles charges
    ParticleList::member_slice_type<Charge> q;
    /// particles forces
    ParticleList::member_slice_type<Force> f;
    /// system size
    double lx, ly, lz;

    /// constructor of functor
    /// @param p         reference to the list of particles and their parameters
    /// @param _k_max    k-space cutoff
    /// @param _alpha    splitting factor
    /// @param n         number of k-vectors
    /// @param _l        system size
    KOKKOS_INLINE_FUNCTION EwaldUkForcesFunctor (
            ParticleList::member_slice_type<Position>& _r,
            ParticleList::member_slice_type<Charge>& _q,
            ParticleList::member_slice_type<Force>& _f,
            double _k_max, 
            double _alpha, 
            int n, 
            double _lx,
            double _ly,
            double _lz
            ) :
        value_count (3),
        k_max(_k_max),
        alpha(_alpha), 
        n_k(n),
        r(_r),
        q(_q),
        f(_f),
        lx(_lx),
        ly(_ly),
        lz(_lz)
    {
    }

    /// operator() to compute the k-space contributions of forces acting on a single particle
    /// @param member   used TeamPolicy for the nested parallelism
    KOKKOS_INLINE_FUNCTION void operator() (const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type& member) const
    {
        // get the local particle index
        int i = member.league_rank();
        // get the number of particles
        int n_max = f.extent(0);

        // create functor to compute the k-space force contributions of the other particles
        EwaldUkForcesIJFunctor<ExecutionSpace> ukfij (member, r, q, k_max, alpha, lx, ly, lz, i);

        // setup reduction variable
        EwaldKSpaceValueType EwaldUk_fi;
        EwaldUk_fi.vx = 0.0;
        EwaldUk_fi.vy = 0.0;
        EwaldUk_fi.vz = 0.0;

        // dispatch Kokkos parallel_reduce to collect the force contributions of the other particles
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange (member, n_max), ukfij, EwaldUk_fi);
    
        // one time update of the forces of the local particle with the results
        Kokkos::single(Kokkos::PerTeam(member), [=] () 
        {
            f(i, 0) = q(i) * EwaldUk_fi.vx;
            f(i, 1) = q(i) * EwaldUk_fi.vy;
            f(i, 2) = q(i) * EwaldUk_fi.vz;
        });
    }

    /// join procedure during reduction
    /// @param dst       reduction variable
    /// @param src       contribution
    KOKKOS_INLINE_FUNCTION void join (volatile EwaldKSpaceValueType dst, const volatile EwaldKSpaceValueType src) const
    {
        dst += src;
    }

    /// initialization method of the reduction variable
    /// @param sum       reduction variable
    KOKKOS_INLINE_FUNCTION void init (EwaldKSpaceValueType sum) const
    {
        sum.vx = 0.0;
        sum.vy = 0.0;
        sum.vz = 0.0;
    }
};

/// Functor to compute the total k-space energy contribution
struct EwaldUkFunctor 
{
    typedef double value_type;

    typedef Kokkos::View<double*>::size_type size_type;

    size_type value_count;

    /// splitting factor
    double alpha;
    /// system size
    double lx, ly, lz;
    /// k-space cutoff
    double k_max;
    /// list of particle charges
    //Cabana::slice<Charge> q;
    ParticleList::member_slice_type<Charge> q;
    /// list of particle positions
    ParticleList::member_slice_type<Position> r;
    /// number of particles
    int n_max;

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
