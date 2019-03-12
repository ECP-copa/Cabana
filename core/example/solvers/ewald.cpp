#include "definitions.h"
#include "ewald.h"
#include <cmath>
#include <gsl/gsl_sf_lambert.h>
#include <Cabana_AoSoA.hpp>
#include <Cabana_Parallel.hpp>
#include <Cabana_ExecutionPolicy.hpp>
#include <atomic>
#include <Kokkos_Pair.hpp>

TEwald::TEwald(double accuracy, ParticleList particles, double lx, double ly, double lz)
{
  _r_max = 0.0;
  tune(accuracy,particles,lx, ly, lz);
}

TEwald::TEwald(double alpha, double r_max, double k_max)
{
  _alpha = alpha;
  _r_max = r_max;
  _k_max = k_max;
  _k_max_int[0] = std::ceil(_k_max);
  _k_max_int[1] = std::ceil(_k_max);
  _k_max_int[2] = std::ceil(_k_max);
}

TEwald::~TEwald()
{
}

void TEwald::tune(double accuracy, ParticleList particles, double lx, double ly, double lz)
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

  double r_max = _r_max = std::min(0.49*lx,0.1*lx + 1.0);
  Kokkos::parallel_reduce( "MinLocReduce", N_alpha*N_k,
    KOKKOS_LAMBDA (const int& idx, value_type& team_errorest)
    {
      int ia = idx%N_alpha;
      int ik = idx/N_alpha;

      double alpha = (double)ia * 0.05 + 1.0;
      double k_max = (double)ik * 0.05;

      double delta_Ur = q_sum * 
      sqrt(0.5 * r_max / (lx * ly * lz )) *
      std::pow(alpha*r_max,-2.0) *
      exp(- alpha * alpha * r_max * r_max);

      double delta_Uk = q_sum * alpha / PI_SQ *
      std::pow(k_max,-1.5) *
      exp(-std::pow(PI * k_max / (alpha * lx),2));

      double delta = delta_Ur + delta_Uk;
      Kokkos::pair<double,double> values(alpha,k_max);

      if ( (delta < team_errorest.val) && (delta > 0.8 * accuracy))
      {
      //  std::cout << alpha << " " << k_max << " " << delta_Ur << " " << delta_Uk << " " << delta << " " << accuracy << std::endl;      
 
        team_errorest.val = delta;
        team_errorest.loc = idx;
      }
    }, reducer_type(error_estimate)
  );

  _alpha = (double)(error_estimate.loc%N_alpha)*0.05+1.0;
  _k_max = (double)(error_estimate.loc/N_alpha)*0.05;

  std::cout << "estimated error: " << error_estimate.val << std::endl;
  _k_max_int[0] = _k_max_int[1] = _k_max_int[2] = std::ceil(_k_max);
  std::cout << "Tuned values: " << "r_max: " << _r_max << " alpha: " << _alpha << " k_max: " << _k_max_int[0] << "  " << _k_max_int[1] << " " << _k_max_int[2] << " " << _k_max << std::endl;
}

void TEwald::compute(ParticleList& particles, double lx, double ly, double lz)
{

  double Ur = 0.0, Uk = 0.0, Uself = 0.0, Udip = 0.0;
  double Udip_vec[3];


  // Create an execution policy over the entire AoSoA.
  Cabana::Experimental::RangePolicy<INNER_ARRAY_SIZE,ExecutionSpace> range_policy( particles );

  auto r = particles.slice<Position>();
  auto q = particles.slice<Charge>();
  auto p = particles.slice<Potential>();

  int n_max = particles.size();

  total_energy = 0.0; 

  auto init_p = KOKKOS_LAMBDA( const int idx )
  {
    p(idx) = 0.0;
  };
  Cabana::Experimental::parallel_for( range_policy, init_p ); 

  double alpha = _alpha;
  double k_max = _k_max;
  double r_max = _r_max;
  double eps_r = _eps_r;

#ifdef TDS_CUDA
  Kokkos::View<int*, Kokkos::CudaUVMSpace> k_max_int("k_max_int",3);
  for ( auto i = 0; i < 3; ++i)
  {
    k_max_int[i] = _k_max_int[i];
  }
#else
  int* k_max_int = &(_k_max_int[0]);
#endif

  // computation real-space contribution
  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0,n_max), KOKKOS_LAMBDA(int idx, double& Ur_part)
      {
        double d[SPACE_DIM];
        double k; 
          for (auto i = 0; i < n_max; ++i)
          {
            for (auto j = 0; j < 3; ++j)
              d[j] = r( idx, j ) - r( i, j );
            double qiqj = q( idx ) * q( i );
            for (auto kx = 0; kx <= k_max_int[0]; ++kx)
            {
              k = (double)kx * lx;
              if (k - lx > r_max) continue;
              for (auto ky = 0; ky <= k_max_int[1]; ++ky)
              {
                k = sqrt( (double)kx * (double)kx * lx * lx +
                  (double)ky * (double)ky * ly * ly);
                if (k - lx  > r_max) continue;
                for (auto kz = 0; kz <= k_max_int[2]; ++kz)
                {
                  if ( kx == 0 && ky == 0 && kz == 0 && i == idx) continue;

                  k = sqrt( (double)kx * (double)kx * lx * lx +
                            (double)ky * (double)ky * ly * ly +
                            (double)kz * (double)kz * lz * lz );
                  if (k - lx > r_max) continue;

                  double scal = (d[0] + (double)kx * lx) * (d[0] + (double)kx * lx) +
                    (d[1] + (double)ky * ly) * (d[1] + (double)ky * ly) +
                    (d[2] + (double)kz * lz) * (d[2] + (double)kz * lz);
                  scal = sqrt(scal);
                  if (scal > r_max) 
                    continue;
                  //std::cout << idx << " " <<
                  //              i << " " << kx << " " << ky << " " << kz << " " <<
                  //              "dist = " << scal << " " << r_max << " " << qiqj * erfc(alpha * scal)/scal << std::endl;
                  Ur_part += qiqj * erfc(alpha * scal)/scal;
                }
              }
            }
          }
        Ur_part *= 0.5;
        p(idx) += Ur_part;
      },
    Ur
      );

  // computation reciprocal-space contribution
  int k_int = std::ceil(_k_max);
  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0,8*k_int*k_int*k_int+12*k_int*k_int+6*k_int+1), KOKKOS_LAMBDA(int idx, double& Uk_part)
      {
      double kk;
      double kr;
      double coeff;
      double sum_re = 0.0;
      double sum_im = 0.0;
      int kx = idx % (2 * k_int + 1) - k_int;
      int ky = idx % (4 * k_int * k_int + 4 * k_int + 1) / (2 * k_int + 1) - k_int;
      int kz = idx / (4 * k_int * k_int + 4 * k_int + 1) - k_int;

      if (kx == 0 && ky == 0 && kz == 0) return;
      kk = kx*kx + ky*ky + kz*kz;
      if (kk > k_max*k_max) return;
      coeff = 2.0 / (lx * lx) * exp( - PI_SQ / (alpha * alpha * lx * lx) * kk) / kk;
      for (auto j = 0; j < n_max; ++j)
      {
      kr = 2.0 * PI * (kx*r(j,0) + ky*r(j,1) + kz*r(j,2)) / lx;
      sum_re += q(j)*cos(kr);
      sum_im += q(j)*sin(kr);
      }
      for (auto j = 0; j < n_max; ++j)
      {
        kr = 2.0 * PI * (kx*r(j,0) + ky*r(j,1) + kz*r(j,2)) / lx;
        double re = sum_re * cos(kr);
        double im = sum_im * sin(kr);

        Kokkos::atomic_add(&p(j),q(j)*coeff*(re + im) * lx / (4.0*PI));

        Uk_part += q(j)*coeff*(re + im) * lx/(4.0*PI);
      }
      }, Uk);

  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0,n_max), KOKKOS_LAMBDA(int idx, double& Uk_part)
      {
      Uk_part += p(idx);
      },
      Uk);

  // computation of self-energy contribution
  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0, n_max), KOKKOS_LAMBDA(int idx, double& Uself_part)
      {
      Uself_part += -alpha / PI_SQRT * q(idx) * q(idx);
      p(idx) += Uself_part;
      },
      Uself
      );

  // TODO: using TeamPolicies?
  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0, n_max), KOKKOS_LAMBDA(int idx, double& Udip_part)
      {
      double V = lx * ly * lz;
      double Udip_prefactor = 2*PI/( (1.0 + 2.0 * eps_r) * V);
      Udip_part += Udip_prefactor * q(idx) * r(idx, 0);
      },
      Udip_vec[0]
      );

  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0, n_max), KOKKOS_LAMBDA(int idx, double& Udip_part)
      {
      double V = lx * ly * lz;
      double Udip_prefactor = 2*PI/( (1.0 + 2.0 * eps_r) * V);
      Udip_part += Udip_prefactor * q(idx) * r(idx, 1);
      },
      Udip_vec[1]
      );

  Kokkos::parallel_reduce( Kokkos::RangePolicy<ExecutionSpace>(0, n_max), KOKKOS_LAMBDA(int idx, double& Udip_part)
      {
      double V = lx * ly * lz;
      double Udip_prefactor = 2*PI/( (1.0 + 2.0 * eps_r) * V);
      Udip_part += Udip_prefactor * q(idx) * r(idx, 2);
      },
      Udip_vec[2]
      );

  Udip = Udip_vec[0] * Udip_vec[0] +
    Udip_vec[1] * Udip_vec[1] +
    Udip_vec[2] * Udip_vec[2];


#ifndef TDS_BENCHMARKING
  std::cout << "real-space contribution: " << Ur << std::endl;
  std::cout << "k-space contribution: " << Uk << std::endl;
  std::cout << "self-energy contribution: " << Uself << std::endl;
  std::cout << "dipole correction: " << Udip << std::endl;
#endif

  total_energy = Ur + Uk + Uself + Udip;
}


