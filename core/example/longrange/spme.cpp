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

#include "definitions.h"
#include "spme.h"
#include <cmath>
#include <sys/time.h>
#include <chrono>

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
 *   arrangement of charged particles. Currently, we assume periodic boundary conditions
 *   and a cubic mesh and arrangement of particles in 3 dimensions.
 * - Future versions will allow for assymetric meshes and non-uniform particle
 *   distributions, as well as 1 or 2 dimensions. 
 */


//constructor given an accuracy
TPME::TPME(double accuracy, ParticleList particles, double lx, double ly, double lz)
{
  _r_max = 0.0;
  tune(accuracy,particles,lx, ly, lz);
}

//set base values for alpha, r_max, k_max
TPME::TPME(double alpha, double r_max)
{
  _alpha = alpha;
  _r_max = r_max;
}

//Tune to a given accuracy
void TPME::tune(double accuracy, ParticleList particles, double lx, double ly, double lz)
{
  typedef Kokkos::MinLoc<double, int> reducer_type;
  typedef reducer_type::value_type value_type;
  value_type error_estimate;

  auto q = particles.slice<Charge>();
  double q_sum;  

  const int N_alpha = 200;
  const int N_k = 2000;

  const int n_max = particles.size();
  // calculate sum of charge squares
  Kokkos::parallel_reduce( n_max, KOKKOS_LAMBDA( const int idx, double& q_part )
      {
      q_part += q(idx) * q(idx);
      },
      q_sum
      );
  Kokkos::fence();

  double r_max = _r_max = std::min(0.49*lx,0.1*lx + 1.0);//real space cutoff value
  Kokkos::parallel_reduce( "MinLocReduce", N_alpha*N_k,
    KOKKOS_LAMBDA (const int& idx, value_type& team_errorest)
    {
      int ia = idx%N_alpha;
      int ik = idx/N_alpha;

      double alpha = (double)ia * 0.05 + 1.0;//splitting parameter
      double k_max = (double)ik * 0.05;//max k-vector
      
      //Compute real part of error estimate
      double delta_Ur = q_sum * 
      sqrt(0.5 * r_max / (lx * ly * lz )) *
      std::pow(alpha*r_max,-2.0) *
      exp(- alpha * alpha * r_max * r_max);

      //Compute k-space part of error estimate
      double delta_Uk = q_sum * alpha / PI_SQ *
      std::pow(k_max,-1.5) *
      exp(-std::pow(PI * k_max / (alpha * lx),2));

      //Total error estimate
      double delta = delta_Ur + delta_Uk;
      Kokkos::pair<double,double> values(alpha,k_max);

      if ( (delta < team_errorest.val) && (delta > 0.8 * accuracy))
      {
        team_errorest.val = delta;
        team_errorest.loc = idx;
      }
    }, reducer_type(error_estimate)
  );
  Kokkos::fence();

  _alpha = (double)(error_estimate.loc%N_alpha)*0.05+1.0;
  _k_max = (double)(error_estimate.loc/N_alpha)*0.05;


  _k_max_int[0] = _k_max_int[1] = _k_max_int[2] = std::ceil(_k_max);
}
//

//Compute a 1D cubic cardinal B-spline value, used in spreading charge to mesh points
//   Given the distance from the particle (x) in units of mesh spaces, this computes the 
//   fraction of that charge to place at a mesh point x mesh spaces away
//   The cubic B-spline used here is shifted so that it is symmetric about zero
//   All cubic B-splines are smooth functions that go to zero and are defined piecewise
KOKKOS_INLINE_FUNCTION
double TPME::oneDspline(double x)
{
  if ( x >= 0.0 and x < 1.0 ) {
     return (1.0/6.0) * x*x*x;
  } 
  else if ( x >= 1.0 and x <= 2.0 )
  {
     return -(1.0/2.0)*x*x*x + 2.0*x*x - 2.0*x + (2.0/3.0);
  } 
  //Using the symmetry here, only need to define function between 0 and 2
  //Beware: This means all input to this function should be made positive
  else
  {
     return 0.0;//Zero if distance is >= 2 mesh spacings
  }
} 

//Compute a 1-D Euler spline. This function is part of the "lattice structure factor"
//and is given by:
//   b(k, meshwidth) = exp(2*PI*i*3*k/meshwidth) / SUM_{l=0,2}(1Dspline(l+1) * exp(2*PI*i*k*l/meshwidth))
//   when using a non-shifted cubic B-spline in the charge spread, where meshwidth is the number of 
//   mesh points in that dimension and k is the scaled fractional coordinate
KOKKOS_INLINE_FUNCTION
double TPME::oneDeuler(int k, int meshwidth)
{ 
  double denomreal = 0.0; 
  double denomimag = 0.0;
  //Compute the denominator sum first, splitting the complex exponential into sin and cos
  for(int l = 0; l < 3; l++)
  {
     denomreal += TPME::oneDspline(min(4.0-(l+1.0),l+1.0)) * cos( 2.0 * PI * double(k) * l / double(meshwidth));
     denomimag += TPME::oneDspline(min(4.0-(l+1.0),l+1.0)) * sin( 2.0 * PI * double(k) * l / double(meshwidth));
  }
  //Compute the numerator, again splitting the complex exponential
  double numreal = cos(2.0*PI*3.0*double(k)/double(meshwidth));
  double numimag = sin(2.0*PI*3.0*double(k)/double(meshwidth));
  //Returning the norm of the 1-D Euler spline
  return (numreal*numreal + numimag*numimag) / (denomreal*denomreal + denomimag*denomimag);
}


//Compute the energy
double TPME::compute( ParticleList& particles, ParticleList& mesh, double lx, double ly, double lz)
{
  //Initialize energies: real-space, k-space (reciprocal space), self-energy correction, dipole correction
  double Ur = 0.0, Uk = 0.0, Uself = 0.0, Udip = 0.0;
  double Udip_vec[3];

  //Particle slices
  auto r = particles.slice<Position>();
  auto q = particles.slice<Charge>();
  auto p = particles.slice<Potential>();
  
  //Mesh slices
  auto meshr = mesh.slice<Position>();
  auto meshq = mesh.slice<Charge>();

  //Number of particles
  const int n_max = particles.size();

  //Number of mesh points
  const int meshsize = mesh.size();  

  double total_energy = 0.0; 

  //Set the potential of each particle to zero
  auto init_p = KOKKOS_LAMBDA( const int idx )
  {
    p(idx) = 0.0;
  };
  Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>(0,n_max), init_p ); 
  Kokkos::fence();
  
  double alpha = _alpha;
  //double k_max = _k_max;
  double r_max = _r_max;
  double eps_r = _eps_r;


#ifdef Cabana_ENABLE_Cuda
  Kokkos::View<int*, Kokkos::CudaUVMSpace> k_max_int("k_max_int",3);
  for ( auto i = 0; i < 3; ++i)
  {
    k_max_int[i] = _k_max_int[i];
  }
#else
  int* k_max_int = &(_k_max_int[0]);
#endif

  //std::chrono::time_point<std::chrono::steady_clock> starttime, starttime2, endtime, endtime2;
  //starttime = std::chrono::steady_clock::now();
  
  // computation real-space contribution
  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0,n_max), KOKKOS_LAMBDA(int idx, double& Ur_part)
     {
        double d[SPACE_DIM];
        double k; 
        //For each particle with charge q, the real space contribution to energy is
        //Ur_part = 0.5*q*SUM_i(q_i*erfc(alpha*dist)/dist)
        //The sum is over all other particles in the cell and in neighboring images
        //up to some real-space cutoff distance "r_max"
        //
        //Self-energy terms are included then corrected for later, with one exception:
        // the self-energy term where kx=ky=kz here is explicitly excluded in the method
        // as this would cause a division by zero
        for (auto i = 0; i < n_max; ++i)
        {
          //compute distance in x,y,z and charge multiple
          for (auto j = 0; j < 3; ++j)
            d[j] = r( idx, j ) - r( i, j );
            double qiqj = q( idx ) * q( i );
            for (auto kx = 0; kx <= k_max_int[0]; ++kx)
            {
              //check if cell within r_max distance in x
              k = (double)kx * lx;
              if (k - lx > r_max) continue;
              for (auto ky = 0; ky <= k_max_int[1]; ++ky)
              {
                //check if cell within r_max distance in x+y
                k = sqrt( (double)kx * (double)kx * lx * lx +
                  (double)ky * (double)ky * ly * ly);
                if (k - lx  > r_max) continue;
                for (auto kz = 0; kz <= k_max_int[2]; ++kz)
                {
                  //Exclude self-energy term when kx=ky=kz
                  if ( kx == 0 && ky == 0 && kz == 0 && i == idx) continue;
                  //check if cell within r_max distance in x+y+z
                  k = sqrt( (double)kx * (double)kx * lx * lx +
                            (double)ky * (double)ky * ly * ly +
                            (double)kz * (double)kz * lz * lz );
                  if (k - lx > r_max) continue;
                  //check if particle distance is less than r_max
                  double scal = (d[0] + (double)kx * lx) * (d[0] + (double)kx * lx) +
                    (d[1] + (double)ky * ly) * (d[1] + (double)ky * ly) +
                    (d[2] + (double)kz * lz) * (d[2] + (double)kz * lz);
                  scal = sqrt(scal);
                  if (scal > r_max) 
                    continue;
                  //Compute real-space energy contribution of interaction
                  Ur_part += qiqj * erfc(alpha * scal)/scal;
                }
              }
            }
          }
        Ur_part *= 0.5;
        p(idx) += Ur_part;
      },
    Ur);
  Kokkos::fence();

  
  // computation reciprocal-space contribution
 
  //First, spread the charges onto the mesh
  
  double spacing = meshr(1,0)-meshr(0,0);//how far apart the mesh points are (assumed uniform cubic)   
  
  //Current method: Each mesh point loops over *all* particles, and gathers charge to it 
  //                 according to spline interpolation. 
  //Alternatives: Looping over all particles, using atomics to scatter charge to mesh points
  //Also, would be nice to loop only over neighbors - spline is only 2 mesh points away maximum
  auto spread_q = KOKKOS_LAMBDA( const int idx )
  {
     double xdist, ydist, zdist;
     for ( size_t pidx = 0; pidx < particles.size(); ++pidx )
     {
        //x-distance between mesh point and particle
        xdist = min(min(abs(meshr(idx,0) - r(pidx,0)),
                     abs(meshr(idx,0) - (r(pidx,0) +  1.0))),
                     abs(meshr(idx,0) - (r(pidx,0) -  1.0)) );//account for periodic bndry
        //y-distance between mesh point and particle
        ydist = min(min(abs(meshr(idx,1) - r(pidx,1)),
                     abs(meshr(idx,1) - (r(pidx,1) +  1.0))),
                     abs(meshr(idx,1) - (r(pidx,1) -  1.0)) );//account for periodic bndry
        //z-distance between mesh point and particle
        zdist = min(min(abs(meshr(idx,2) - r(pidx,2)),
                     abs(meshr(idx,2) - (r(pidx,2) +  1.0))),
                     abs(meshr(idx,2) - (r(pidx,2) -  1.0)) );//account for periodic bndry

        if ( xdist <= 2.0*spacing and ydist <= 2.0*spacing and zdist <= 2.0*spacing ) //more efficient way to do this? Skip it? May be unnecessary.
        {
           //add charge to mesh point according to spline
           meshq(idx) += q(pidx) * TPME::oneDspline(2.0-(xdist/spacing))
                                 * TPME::oneDspline(2.0-(ydist/spacing)) 
                                 * TPME::oneDspline(2.0-(zdist/spacing));
        }
     }       
  };
  Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>(0,meshsize), spread_q ); 
  Kokkos::fence();
  
  //starttime2 = std::chrono::steady_clock::now();

  //std::cout << "Creating BC array" << std::endl;
  //Create "B*C" array (called theta in Eqn 4.7 SPME paper by Essman)
  //Can be done at start of run and stored
  //"meshwidth" should be number of mesh points along any axis. 
  int meshwidth = std::round(std::pow(meshsize, 1.0/3.0));//Assuming cubic mesh

  //Calculating the values of the BC array involves first shifting the fractional coords
  //then compute the B and C arrays as described in the paper
  //This can be done once at the start of a run if the mesh stays constant
  #ifdef Cabana_ENABLE_Cuda
  cufftDoubleComplex *BC;
  cudaMallocManaged((void**)&BC,sizeof(cufftDoubleComplex) * meshsize);
  #else
  fftw_complex* BC;
  BC = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * meshsize);
  #endif
  //TODO: Is this a good place for Kokkos Hierarchical parallelism?
  auto BC_functor = KOKKOS_LAMBDA(const int kx)
  {
  int ky,kz,mx,my,mz,idx;
      for (ky=0; ky<meshwidth; ky++)
      {
          for (kz=0; kz<meshwidth; kz++)
          {
              idx = kx + (ky*meshwidth) + (kz*meshwidth*meshwidth);
              if (kx + ky + kz > 0)
              {
                  //Shift the C array
                  mx = kx;
                  my = ky;
                  mz = kz;
                  if (mx > meshwidth/2.0)
                  {
                     mx = kx - meshwidth;
                  }
                  if (my > meshwidth/2.0)
                  {
                     my = ky - meshwidth;
                  }
                  if (mz > meshwidth/2.0)
                  {
                     mz = kz - meshwidth;
                  }
                  double m2 = (mx*mx + my*my + mz*mz);//Unnecessary extra variable

                  //Calculate BC. Why store the imag part at all?
                  BC[idx].x =  TPME::oneDeuler(kx,meshwidth) * TPME::oneDeuler(ky,meshwidth) 
                               * TPME::oneDeuler(kz,meshwidth) 
                               * exp( -PI*PI*m2 / (alpha*alpha) ) / (PI * lx*ly*lz * m2 );
                  BC[idx].y = 0.0;//imag part
              }
              else
              {
                  BC[idx].x = 0.0;
                  BC[idx].y = 0.0;//set origin element to zero
              }
          }
      }
  };
  Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>(0,meshwidth),BC_functor );
  Kokkos::fence();

  //endtime = std::chrono::steady_clock::now();

  //Next, solve Poisson's equation taking some FFTs of charges on mesh grid  
  //The plan here is to perform an inverse FFT on the mesh charge, then multiply
  //  the norm of that result (in reciprocal space) by the BC array

  //Set up the real-space charge and reciprocal-space charge
  #ifdef Cabana_ENABLE_Cuda 
  cufftDoubleComplex *Qr,*Qktest;
  cufftHandle plantest;
  cudaMallocManaged((void**)&Qr,sizeof(fftw_complex) * meshsize);
  cudaMallocManaged((void**)&Qktest,sizeof(fftw_complex) * meshsize);
  //Copy charges into real input array
  auto copy_charge = KOKKOS_LAMBDA( const int idx)
  {
     Qr[idx].x = meshq(idx);
     Qr[idx].y = 0.0;
  };
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0,meshsize), copy_charge);
  #else
  fftw_complex *Qr,*Qktest;
  fftw_plan plantest;
  Qr = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * meshsize);
  Qktest = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * meshsize);
  //Copy charges into real input array
  auto copy_charge = KOKKOS_LAMBDA( const int idx)
  {
     Qr[idx][0] = meshq(idx);
     Qr[idx][1] = 0.0;
  };
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0,meshsize), copy_charge);
  #endif  
  Kokkos::fence();
  
  //Plan out that IFFT on the real-space charge mesh
  #ifdef Cabana_ENABLE_Cuda
  cufftPlan3d(&plantest,meshwidth, meshwidth, meshwidth,CUFFT_Z2Z);
  cufftExecZ2Z(plantest,Qr,Qktest,CUFFT_INVERSE);//IFFT on Q

  Kokkos::parallel_reduce( meshsize, KOKKOS_LAMBDA(const int idx, double& Uk_part) 
  {
    Uk_part += BC[idx].x * ((Qktest[idx].x * Qktest[idx].x) + (Qktest[idx].y * Qktest[idx].y));
  },
  Uk
  );
  Kokkos::fence();
  cufftDestroy(plantest);
  #else
  plantest = fftw_plan_dft_3d(meshwidth, meshwidth, meshwidth, Qr, Qktest, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(plantest);//IFFT on Q
  Kokkos::parallel_reduce( meshsize, KOKKOS_LAMBDA(const int idx, double& Uk_part) 
  {
    Uk_part += BC[idx][0] * ((Qktest[idx][0] * Qktest[idx][0]) + (Qktest[idx][1] * Qktest[idx][1]));
  },
  Uk
  );
  Kokkos::fence();
  fftw_destroy_plan(plantest);
  #endif 

  Uk *= 0.5;
 
  //endtime2 = std::chrono::steady_clock::now();

  //std::chrono::duration<double> elapsed_time = starttime2 - starttime + endtime2 - endtime;

  // computation of self-energy contribution
  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0, n_max), KOKKOS_LAMBDA(int idx, double& Uself_part)
      {
      Uself_part += -alpha / PI_SQRT * q(idx) * q(idx);
      p(idx) += Uself_part;
      },
      Uself
      );
  Kokkos::fence();

  // computation of dipole correction to energy
  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0, n_max), KOKKOS_LAMBDA(int idx, double& Udip_part)
      {
      double V = lx * ly * lz;
      double Udip_prefactor = 2*PI/( (1.0 + 2.0 * eps_r) * V);
      Udip_part += Udip_prefactor * q(idx) * r(idx, 0);
      },
      Udip_vec[0]
      );
  Kokkos::fence();

  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0, n_max), KOKKOS_LAMBDA(int idx, double& Udip_part)
      {
      double V = lx * ly * lz;
      double Udip_prefactor = 2*PI/( (1.0 + 2.0 * eps_r) * V);
      Udip_part += Udip_prefactor * q(idx) * r(idx, 1);
      },
      Udip_vec[1]
      );
  Kokkos::fence();

  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0, n_max), KOKKOS_LAMBDA(int idx, double& Udip_part)
      {
      double V = lx * ly * lz;
      double Udip_prefactor = 2*PI/( (1.0 + 2.0 * eps_r) * V);
      Udip_part += Udip_prefactor * q(idx) * r(idx, 2);
      },
      Udip_vec[2]
      );
  Kokkos::fence();

  Udip = Udip_vec[0] * Udip_vec[0] +
    Udip_vec[1] * Udip_vec[1] +
    Udip_vec[2] * Udip_vec[2];

  total_energy = Ur + Uk + Uself + Udip;
  #ifndef Cabana_ENABLE_Cuda
    fftw_cleanup();
  #endif
  return total_energy;
}


