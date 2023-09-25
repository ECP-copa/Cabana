/****************************************************************************
 * Copyright (c) 2023 by the Cabana authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

/*!
  \file impl/Cabana_GaussianMixtureModel.hpp
  \brief Creation of a Gaussian Mixture Model
*/
#ifndef CABANA_GMM_IMPL_HPP
#define CABANA_GMM_IMPL_HPP

#include <typeinfo>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Cabana_Parallel.hpp>
#include <impl/Cabana_GaussianWeight.hpp>

namespace Cabana {

template<unsigned int dims>
class GMMImpl {
public:

/*!
  Do a scan of the particles and compute the variance of the 1d velocity data

  This is done for the particles where the entry in the cell slide matches c,
  respecting the possibly different weights in the macro slice and the result is
  reported in cov[0][0].
*/
template <typename CellSliceType, typename WeightSliceType, typename VelocitySliceType>
static void variance(const CellSliceType& cell, const WeightSliceType& macro, const VelocitySliceType& velocity_x, const int c, double(&cov)[1][1]) {
	using gmm_float_type = typename WeightSliceType::value_type;

	gmm_float_type sum_x = 0.;
	gmm_float_type sum_xx = 0.;
	gmm_float_type N = 0;

	// How does a single particle contribute to the total
	auto _add = KOKKOS_LAMBDA(const int& i, gmm_float_type& local_N,
	                          gmm_float_type& local_sum_x, gmm_float_type& local_sum_xx) {
		if(cell(i) == c) {
			local_N      += macro(i);
			local_sum_x  += macro(i) * velocity_x(i);
			local_sum_xx += macro(i) * velocity_x(i)*velocity_x(i);
		}
	};

	// Execute
	Kokkos::parallel_reduce("Particle Scan", velocity_x.size(), _add, N, sum_x, sum_xx);
	const gmm_float_type mu_x = sum_x/gmm_float_type(N);
	const gmm_float_type var_x = sum_xx/gmm_float_type(N) - mu_x*mu_x;

	cov[0][0] = var_x;
}

/*!
  Do a scan of the particles and compute the variance of the 2d velocity data

  This is done for the particles where the entry in the cell slide matches c,
  respecting the possibly different weights in the macro slice and the result is
  reported in cov[0..1][0..1].
*/
template <typename CellSliceType, typename WeightSliceType, typename VelocitySliceType>
static void variance(const CellSliceType& cell, const WeightSliceType& macro, const VelocitySliceType& velocity_par, const VelocitySliceType& velocity_per, const int c, double(&cov)[2][2]) {
	using gmm_float_type = typename WeightSliceType::value_type;

	gmm_float_type sum_par = 0.;
	gmm_float_type sum_per = 0.;
	gmm_float_type sum_parpar = 0.;
	gmm_float_type sum_perper = 0.;
	gmm_float_type sum_parper = 0.;
	gmm_float_type N = 0;

	// How does a single particle contribute to the total
	auto _add = KOKKOS_LAMBDA(const int& i, gmm_float_type& local_N,
	                          gmm_float_type& local_sum_par,  gmm_float_type& local_sum_per,
	                          gmm_float_type& local_sum_parpar, gmm_float_type& local_sum_perper, gmm_float_type& local_sum_parper) {
		if(cell(i) == c) {
			local_N     += macro(i);
			local_sum_par  += macro(i) * velocity_par(i);
			local_sum_per  += macro(i) * velocity_per(i);
			local_sum_parpar += macro(i) * velocity_par(i)*velocity_par(i);
			local_sum_perper += macro(i) * velocity_per(i)*velocity_per(i);
			local_sum_parper += macro(i) * velocity_par(i)*velocity_per(i);
		}
	};

	// Execute
	Kokkos::parallel_reduce("Particle Scan", velocity_par.size(), _add, N, sum_par, sum_per, sum_parpar, sum_perper, sum_parper);

	printf("N = %e, sum_par=%e, sum_per = %e, sum_parpar = %e, sum_perper = %e, sum_parper = %e\n", N, sum_par, sum_per, sum_parpar, sum_perper, sum_parper);
	const gmm_float_type mu_par = sum_par/gmm_float_type(N);
	const gmm_float_type mu_per = 0.; // in cylindrical coordinates we don't expect this to be the perpendicular drift speed but sqrt(pi/2)*vthper
	const gmm_float_type var_par = sum_parpar/gmm_float_type(N) - mu_par*mu_par;
	const gmm_float_type var_per = sum_perper/gmm_float_type(N);// - mu_per*mu_per;
	printf("var_par = %e, var_per = %e\n", var_par, var_per);

	cov[0][0] = var_par;  cov[0][1] = 0.;
	cov[1][0] = 0.;       cov[1][1] = var_per;

	printf("variance: cell %d. var = %e %e, %e %e\n", c, cov[0][0], cov[0][1], cov[1][0], cov[1][1]);
}

/*!
  Do a scan of the particles and compute the variance of the 3d velocity data

  This is done for the particles where the entry in the cell slide matches c,
  respecting the possibly different weights in the macro slice and the result is
  reported in the 3x3 covariance matrix cov
*/
template <typename CellSliceType, typename WeightSliceType, typename VelocitySliceType>
static void variance(const CellSliceType& cell, const WeightSliceType& macro, const VelocitySliceType& velocity_x, const VelocitySliceType& velocity_y, const VelocitySliceType& velocity_z, const int c, double(&cov)[3][3]) {
	using gmm_float_type = typename WeightSliceType::value_type;

	gmm_float_type sum_x = 0.;
	gmm_float_type sum_y = 0.;
	gmm_float_type sum_z = 0.;
	gmm_float_type sum_xx = 0.;
	gmm_float_type sum_yy = 0.;
	gmm_float_type sum_zz = 0.;
	gmm_float_type sum_xy = 0.;
	gmm_float_type sum_xz = 0.;
	gmm_float_type sum_yz = 0.;
	gmm_float_type N = 0;

	// How does a single particle contribute to the total
	auto _add = KOKKOS_LAMBDA(const int& i, gmm_float_type& local_N,
	                          gmm_float_type& local_sum_x,  gmm_float_type& local_sum_y,  gmm_float_type& local_sum_z,
	                          gmm_float_type& local_sum_xx, gmm_float_type& local_sum_yy, gmm_float_type& local_sum_zz,
	                          gmm_float_type& local_sum_xy, gmm_float_type& local_sum_xz, gmm_float_type& local_sum_yz) {
		if(cell(i) == c) {
			local_N     += macro(i);
			local_sum_x  += macro(i) * velocity_x(i);
			local_sum_y  += macro(i) * velocity_y(i);
			local_sum_z  += macro(i) * velocity_z(i);
			local_sum_xx += macro(i) * velocity_x(i)*velocity_x(i);
			local_sum_yy += macro(i) * velocity_y(i)*velocity_y(i);
			local_sum_zz += macro(i) * velocity_z(i)*velocity_z(i);
			local_sum_xy += macro(i) * velocity_x(i)*velocity_y(i);
			local_sum_xz += macro(i) * velocity_x(i)*velocity_z(i);
			local_sum_yz += macro(i) * velocity_y(i)*velocity_z(i);
		}
	};

	// Execute
	Kokkos::parallel_reduce("Particle Scan", velocity_x.size(), _add, N, sum_x, sum_y, sum_z, sum_xx, sum_yy, sum_zz, sum_xy, sum_xz, sum_yz);
	const gmm_float_type mu_x = sum_x/gmm_float_type(N);
	const gmm_float_type mu_y = sum_y/gmm_float_type(N);
	const gmm_float_type mu_z = sum_z/gmm_float_type(N);
	const gmm_float_type var_x = sum_xx/gmm_float_type(N) - mu_x*mu_x;
	const gmm_float_type var_y = sum_yy/gmm_float_type(N) - mu_y*mu_y;
	const gmm_float_type var_z = sum_zz/gmm_float_type(N) - mu_z*mu_z;
	const gmm_float_type cov_xy = sum_xy/gmm_float_type(N) - mu_x*mu_y;
	const gmm_float_type cov_xz = sum_xz/gmm_float_type(N) - mu_x*mu_z;
	const gmm_float_type cov_yz = sum_yz/gmm_float_type(N) - mu_y*mu_z;
	//printf("N = %e, sum_x=%e, sum_y = %e, sum_z = %e, sum_xx = %e, sum_yy = %e, sum_zz = %e\n", N, sum_x, sum_y, sum_z, sum_xx, sum_yy, sum_zz);
	//printf("var_x = %e, var_y = %e var_z = %e\n", var_x, var_y, var_z);
	printf("var_iance: cell %d. sum_ = %e %e %e\n", c, sum_x,sum_y,sum_z);
	printf("var_iance: cell %d. sum_^2 = %e %e %e\n", c, sum_xx,sum_yy,sum_zz);
	printf("var_iance: cell %d. sum_ offdiag = %e %e %e\n", c, sum_xy,sum_xz,sum_yz);

	printf("var_iance: cell %d. mu_ = %e %e %e\n", c, mu_x,mu_y,mu_z);
	printf("var_iance: cell %d. var_ = %e %e %e\n", c, var_x,var_y,var_z);
	printf("var_iance: cell %d. cov_ = %e %e %e\n", c, cov_xy,cov_xz,cov_yz);

	cov[0][0] = var_x;  cov[0][1] = cov_xy; cov[0][2] = cov_xz;
	cov[1][0] = cov_xy; cov[1][1] = var_y;  cov[1][2] = cov_yz;
	cov[2][0] = cov_xz; cov[2][1] = cov_yz; cov[2][2] = var_z;

	printf("variance: cell %d. var = %e %e %e, %e %e %e, %e %e %e\n", c, cov[0][0], cov[0][1], cov[0][2], cov[1][0], cov[1][1], cov[1][2], cov[2][0], cov[2][1], cov[2][2]);

	//return (var_x+var_y+var_z)/3. ;
}

/*!
  Compute the sum of alpha and return a version normalized to 1 in alpha_norm
*/
template<typename AlphaType>
static void normalize(AlphaType& alpha_norm, const AlphaType& alpha) {
	using gmm_float_type = typename AlphaType::value_type;

	for(size_t c = 0; c < alpha.extent(0); c++) {
		gmm_float_type sum = 0.;
		int k_max = alpha.extent(1);
		auto _add = KOKKOS_LAMBDA(const int& m, gmm_float_type& lsum) {
			lsum += alpha(c,m);
		};
		Kokkos::parallel_reduce("Sum alpha", k_max, _add, sum);

		auto _norm = KOKKOS_LAMBDA(const int&m) {
			alpha_norm(c,m) = alpha(c,m) / sum;
		};
		Kokkos::parallel_for("Norm alpha", k_max, _norm);
	}
}


/*!
  Compute the w for each combination of a Gaussian and a particle by multiplying  alpha into u
*/
template<typename WType, typename AlphaType, typename UType>
static void prefillW(WType w, const AlphaType& alpha_norm, const UType& u) {
	using gmm_float_type = typename AlphaType::value_type;

	auto _mult = KOKKOS_LAMBDA(const int& i) {
		for(size_t c = 0; c < w.extent(0); c++) {
			for(size_t m = 0; m < w.extent(1); m++) {
				w(c,m,i) = alpha_norm(c,m) * u(c,m,i);
			}
		}
	};
	// We have way more particles than gaussians, so that is what we parallelize over
	Kokkos::parallel_for("Prefill w", w.extent(2), _mult);
}

/*!
  Compute the w_norm by normalizing w such that the sum over all Gaussians for a given particle is one
*/
template<typename WType>
static void prefillWNorm(WType& w_norm, const WType& w) {
	using gmm_float_type = typename WType::value_type;

	auto _norm = KOKKOS_LAMBDA(const int& i) {
		for(size_t c = 0; c < w.extent(0); c++) {
			gmm_float_type sum = 0.;
			for(size_t m = 0; m < w.extent(1); m++) {
				sum += w(c,m,i);
			}
			for(size_t m = 0; m < w.extent(1); m++) {
				if(sum == 0.) {
					w_norm(c,m,i) = 0.;
				} else {
					w_norm(c,m,i) = w(c,m,i)/sum;
				}
			}
		}
	};
	// We have way more particles than gaussians, so that is what we parallelize over
	Kokkos::parallel_for("Prefill wnorm", w.extent(2), _norm);
}


/*!
  recompute alpha as per line 10
*/
template<typename AlphaType, typename WType>
static void updateAlpha(AlphaType& alpha, const WType& w_norm, const int c, const int m) {
	using gmm_float_type = typename AlphaType::value_type;

	int N = dims + dims*(dims+1)/2; // Number of actually independent degrees of freedom
	if (dims == 1) {
		N = 1 + 1*(1+1)/2;          // Number of actually independent degrees of freedom for each 1d Gaussian
	} else if (dims == 2) {
		N = 2 + 2;                  // Number of actually independent degrees of freedom for each uncorrelated(!) 2d Gaussian
	} else if (dims == 3) {
		N = 3 + 3*(3+1)/2;          // Number of actually independent degrees of freedom for each 3d Gaussian
	}

	Kokkos::View<gmm_float_type*> tmp1("tmp1", w_norm.extent(1));
	auto _sett = KOKKOS_LAMBDA(const int j) {
		gmm_float_type sum = 0.;
		for(size_t n = 0; n < w_norm.extent(2); n++) {
			sum += w_norm(c,j,n);
		}
		tmp1(j) = Kokkos::max(0., sum-N/2.);
	};
	Kokkos::parallel_for("compute tmp1", w_norm.extent(1), _sett);

	gmm_float_type sum = 0.;
	auto _sum = KOKKOS_LAMBDA(const int n, gmm_float_type& lsum) {
		lsum += tmp1(n);
	};
	Kokkos::parallel_reduce("sum tmp1", w_norm.extent(1), _sum, sum);

	auto _seta = KOKKOS_LAMBDA(const int j) {
		if(j == m) {
			alpha(c,j) = tmp1(j) / sum;
		}
	};

	// Not actually parallel but should run on-device
	Kokkos::parallel_for("update alpha", w_norm.extent(1), _seta);
}


/*!
  compute weights u from the probability to get particles given one gaussian
*/
template <typename GaussianType, typename CellSliceType, typename VelocitySliceType, typename weight_type>
static void updateWeights(weight_type u, const CellSliceType& cell, const VelocitySliceType vx, const VelocitySliceType& vy, const VelocitySliceType vz, const GaussianType& g_dev, const int c, const int m) {

	using gmm_float_type = typename GaussianType::value_type;

	if(u.extent(0) != g_dev.extent(0)) {
		fprintf(stderr, "We need a separate value of the first index in u for each cell\n");
		exit(1);
	}
	if(u.extent(1) != g_dev.extent(1)) {
		fprintf(stderr, "We need a separate value of the second index in u for each gaussian\n");
		exit(1);
	}
	if(u.extent(2) != vx.size()) {
		fprintf(stderr, "We need a separate value of the third index in u for each particle\n");
		exit(1);
	}

	if (dims == 1) {
		VelocitySliceType velocity_x = vx;

		// The value of u is given by the probability to draw a single particle from a Gaussian
		auto _weight = KOKKOS_LAMBDA(const int s, const int i) {
			int n = (s)*cell.vector_length + i;
			if(cell(n) == c) {
				gmm_float_type p = 0.;
				gmm_float_type v[1] = {velocity_x(n)};
				gmm_float_type Mu[1] = {g_dev(c,m,MuX)};
				gmm_float_type C[1][1] = {{g_dev(c,m,Cxx)}};
				p = Impl::GaussianWeight<gmm_float_type>::weight_1d(v, Mu, C);

				u(c,m,n) = p;
			} else {
				u(c,m,n) = 0.;
			}
		};

		// Define an execution policy
		SimdPolicy<cell.vector_length, Kokkos::DefaultExecutionSpace> vec_policy(0, cell.size());

		// Execute for all particles in parallel
		simd_parallel_for(vec_policy, _weight, "weight()");

	} else if (dims == 2) {
		VelocitySliceType velocity_par = vx;
		VelocitySliceType velocity_per = vy;

		// The value of u is given by the probability to draw a single particle from a Gaussian
		auto _weight = KOKKOS_LAMBDA(const int s, const int i) {
			int n = (s)*cell.vector_length + i;
			if(cell(n) == c) {
				gmm_float_type p = 0.;
				gmm_float_type v[2] = {velocity_par(n), velocity_per(n)};
				gmm_float_type Mu[2] = {g_dev(c,m,MuPar), g_dev(c,m,MuPer)};
				gmm_float_type C[2][2] = {{g_dev(c,m,Cparpar), g_dev(c,m,Cparper)},
										{g_dev(c,m,Cperpar), g_dev(c,m,Cperper)}};
				p = Impl::GaussianWeight<gmm_float_type>::weight_2d(v, Mu, C);

				u(c,m,n) = p;
			} else {
				u(c,m,n) = 0.;
			}
		};

		// Define an execution policy
		SimdPolicy<cell.vector_length, Kokkos::DefaultExecutionSpace> vec_policy(0, cell.size());

		// Execute for all particles in parallel
		simd_parallel_for(vec_policy, _weight, "weight()");

	} else if (dims == 3) {
		VelocitySliceType velocity_x = vx;
		VelocitySliceType velocity_y = vy;
		VelocitySliceType velocity_z = vz;

		// The value of u is given by the probability to draw a single particle from a Gaussian
		auto _weight = KOKKOS_LAMBDA(const int s, const int i) {
			int n = (s)*cell.vector_length + i;
			if(cell(n) == c) {
				gmm_float_type p = 0.;
				gmm_float_type v[3] = {velocity_x(n), velocity_y(n), velocity_z(n)};
				gmm_float_type Mu[3] = {g_dev(c,m,MuX), g_dev(c,m,MuY), g_dev(c,m,MuZ)};
				gmm_float_type C[3][3] = {{g_dev(c,m,Cxx), g_dev(c,m,Cxy), g_dev(c,m,Cxz)},
										{g_dev(c,m,Cyx), g_dev(c,m,Cyy), g_dev(c,m,Cyz)},
										{g_dev(c,m,Czx), g_dev(c,m,Czy), g_dev(c,m,Czz)}};
				p = Impl::GaussianWeight<gmm_float_type>::weight_3d(v, Mu, C);

				u(c,m,n) = p;
			} else {
				u(c,m,n) = 0.;
			}
		};

		// Define an execution policy
		SimdPolicy<cell.vector_length, Kokkos::DefaultExecutionSpace> vec_policy(0, cell.size());

		// Execute for all particles in parallel
		simd_parallel_for(vec_policy, _weight, "weight()");
	}

}

/*!
  update gaussian m from the particles where we consider weights w_norm for the impact
*/
template <typename GaussianType, typename CellSliceType, typename VelocitySliceType, typename weight_type>
static void updateGMM(GaussianType& g_dev, const CellSliceType& cell, const VelocitySliceType& vx, const VelocitySliceType& vy, const VelocitySliceType& vz, const weight_type& w_norm, const int c, const int m) {
	using gmm_float_type = typename GaussianType::value_type;

	if (dims == 1) {
		auto velocity_x = vx;

		auto _moments = KOKKOS_LAMBDA(const int i, gmm_float_type& lM0, gmm_float_type& lM1x, gmm_float_type& lM2xx) {
			if(cell(i) == c) {
				lM0   +=                               w_norm(c,m,i);
				lM1x  += velocity_x(i)               * w_norm(c,m,i);
				lM2xx += velocity_x(i)*velocity_x(i) * w_norm(c,m,i);
			}
		};

		// Compute moments
		gmm_float_type M0   = 0.;
		gmm_float_type M1x  = 0.;
		gmm_float_type M2xx = 0.;
		Kokkos::parallel_reduce("update gaussian", w_norm.extent(2), _moments, M0, M1x, M2xx);

		// Compute underlying parameters
		auto _update = KOKKOS_LAMBDA(const int j) {
			if(j == m) {
				g_dev(c,m,MuX) = M1x/M0;
				g_dev(c,m,Cxx) = M2xx/M0 - (M1x/M0)*(M1x/M0);
			}
		};

		// Not actually parallel but should be on-device
		Kokkos::parallel_for("update Gaussian", g_dev.extent(1), _update);
	} else if (dims == 2) {
		auto velocity_par = vx;
		auto velocity_per = vy;

		auto _moments = KOKKOS_LAMBDA(const int i, gmm_float_type& lM0,
								   gmm_float_type& lM1par, gmm_float_type& lM1per,
								   gmm_float_type& lM2parpar, gmm_float_type& lM2parper, gmm_float_type& lM2perper) {
			if(cell(i) == c) {
				lM0       +=                                   w_norm(c,m,i) / velocity_per(i);
				lM1par    += velocity_par(i)                 * w_norm(c,m,i) / velocity_per(i);
				lM1per    += velocity_per(i)                 * w_norm(c,m,i) / velocity_per(i);
				lM2parpar += velocity_par(i)*velocity_par(i) * w_norm(c,m,i) / velocity_per(i);
				lM2parper += velocity_par(i)*velocity_per(i) * w_norm(c,m,i) / velocity_per(i);
				lM2perper += velocity_per(i)*velocity_per(i) * w_norm(c,m,i) / velocity_per(i);
			}
		};

		// Compute moments
		gmm_float_type M0 = 0.;
		gmm_float_type M1par = 0.;
		gmm_float_type M1per = 0.;
		gmm_float_type M2parpar = 0.;
		gmm_float_type M2parper = 0.;
		gmm_float_type M2perper = 0.;
		Kokkos::parallel_reduce("update gaussian", w_norm.extent(2), _moments, M0, M1par,M1per, M2parpar,M2parper,M2perper);

		// Compute underlying parameters
		auto _update = KOKKOS_LAMBDA(const int j) {
			if(j == m) {
				g_dev(c,m,MuPar) = M1par / M0;
				g_dev(c,m,MuPer) = 0.;
				g_dev(c,m,Cparpar) = M2parpar/M0 - (M1par/M0)*(M1par/M0);
				g_dev(c,m,Cparper) = 0.;
				g_dev(c,m,Cperpar) = 0.;
				g_dev(c,m,Cperper) = M2perper/M0; // Eq. 93 has a factor 2 here

				gmm_float_type disc = 4.*M1per*M1per - M_PI*M0*M2perper;
				//printf("c=%d,m=%d, M0=%f,M1per=%f,M2perper=%f, disc=%f\n", c,m, M0, M1per, M2perper, disc);
				gmm_float_type alpha, cperper;
				if(disc < 0.) {
					alpha = 0.;
					cperper =  M2perper/M0;
				} else {
					gmm_float_type rad = (disc + 2.*M1per*Kokkos::sqrt(disc)) / (M_PI*M_PI*M0*M2perper);
					alpha = Kokkos::sqrt(rad);
					cperper = 0.5*M_PI*M2perper*M2perper / (8.*M1per*M1per - M_PI*M0*M2perper + 4.*M1per*Kokkos::sqrt(disc));
				}
				gmm_float_type uper = 2.*alpha*cperper;
				//printf("c=%d,m=%d, alpha=%f, MuPer=%f, Cperper=%f\n", c,m, alpha, uper, cperper);
				if(alpha > 0.01) {
					// Iterative solution
					double I0 = Kokkos::Experimental::cyl_bessel_i0<Kokkos::complex<double>, double, int>(Kokkos::complex(0.25*uper*uper/cperper)).real();
					double I1 = Kokkos::Experimental::cyl_bessel_i1<Kokkos::complex<double>, double, int>(Kokkos::complex(0.25*uper*uper/cperper)).real();

					double Exp = Kokkos::exp(-0.25*uper*uper/cperper);
					for(int i = 0; i<500; i++) {
						uper    = Kokkos::sqrt((-M_PI*Exp*Exp*I0*I1*M0*M2perper + 2*M1per*(-M1per + Kokkos::sqrt(M_PI*Exp*Exp*I0*I1*M0*M2perper + M_PI*Exp*Exp*I1*I1*M0*M2perper + M1per*M1per)))/M0/M0)/(Kokkos::sqrt(M_PI)*Exp*I1);
						cperper = (M_PI*Exp*Exp*I0*I1*M0*M2perper + M_PI*Exp*Exp*I1*I1*M0*M2perper + 2*M1per*(M1per - Kokkos::sqrt(M_PI*Exp*Exp*I0*I1*M0*M2perper + M_PI*Exp*Exp*I1*I1*M0*M2perper + M1per*M1per)))/(2*M_PI*Exp*Exp*I1*I1*M0*M0);
						I0  = Kokkos::Experimental::cyl_bessel_i0<Kokkos::complex<double>, double, int>(Kokkos::complex(0.25*uper*uper/cperper)).real();
						I1  = Kokkos::Experimental::cyl_bessel_i1<Kokkos::complex<double>, double, int>(Kokkos::complex(0.25*uper*uper/cperper)).real();
						Exp = Kokkos::exp(-0.25*uper*uper/cperper);
					}
					//printf("c=%d,m=%d, alpha=%f, MuPer=%f, Cperper=%f\n", c,m, alpha, uper, cperper);
				}
				g_dev(c,m,MuPer)  = uper;
				g_dev(c,m,Cperper) = cperper;
			}
		};

		// This shouldn't actually be parallel, but it should be on-device
		Kokkos::parallel_for("update Gaussian", g_dev.extent(1), _update);
	} else if (dims == 3) {
		auto velocity_x = vx;
		auto velocity_y = vy;
		auto velocity_z = vz;

		auto _moments = KOKKOS_LAMBDA(const int i, gmm_float_type& lM0,
								   gmm_float_type& lM1x, gmm_float_type& lM1y, gmm_float_type& lM1z,
								   gmm_float_type& lM2xx, gmm_float_type& lM2xy, gmm_float_type& lM2xz,
								   gmm_float_type& lM2yx, gmm_float_type& lM2yy, gmm_float_type& lM2yz,
								   gmm_float_type& lM2zx, gmm_float_type& lM2zy, gmm_float_type& lM2zz) {
			if(cell(i) == c) {
				lM0   +=                               w_norm(c,m,i);
				lM1x  += velocity_x(i)               * w_norm(c,m,i);
				lM1y  += velocity_y(i)               * w_norm(c,m,i);
				lM1z  += velocity_z(i)               * w_norm(c,m,i);
				lM2xx += velocity_x(i)*velocity_x(i) * w_norm(c,m,i);
				lM2xy += velocity_x(i)*velocity_y(i) * w_norm(c,m,i);
				lM2xz += velocity_x(i)*velocity_z(i) * w_norm(c,m,i);
				lM2yx += velocity_y(i)*velocity_x(i) * w_norm(c,m,i);
				lM2yy += velocity_y(i)*velocity_y(i) * w_norm(c,m,i);
				lM2yz += velocity_y(i)*velocity_z(i) * w_norm(c,m,i);
				lM2zx += velocity_z(i)*velocity_x(i) * w_norm(c,m,i);
				lM2zy += velocity_z(i)*velocity_y(i) * w_norm(c,m,i);
				lM2zz += velocity_z(i)*velocity_z(i) * w_norm(c,m,i);
			}
		};

		// Compute moments
		gmm_float_type M0 = 0.;
		gmm_float_type M1x = 0.;
		gmm_float_type M1y = 0.;
		gmm_float_type M1z = 0.;
		gmm_float_type M2xx = 0.;
		gmm_float_type M2xy = 0.;
		gmm_float_type M2xz = 0.;
		gmm_float_type M2yx = 0.;
		gmm_float_type M2yy = 0.;
		gmm_float_type M2yz = 0.;
		gmm_float_type M2zx = 0.;
		gmm_float_type M2zy = 0.;
		gmm_float_type M2zz = 0.;
		Kokkos::parallel_reduce("update gaussian", w_norm.extent(2), _moments, M0, M1x,M1y,M1z, M2xx,M2xy,M2xz, M2yx,M2yy,M2yz, M2zx,M2zy,M2zz);

		// Compute underlying parameters
		auto _update = KOKKOS_LAMBDA(const int j) {
			if(j == m) {
				g_dev(c,m,MuX) = M1x/M0;
				g_dev(c,m,MuY) = M1y/M0;
				g_dev(c,m,MuZ) = M1z/M0;
				g_dev(c,m,Cxx) = M2xx/M0 - (M1x/M0)*(M1x/M0);
				g_dev(c,m,Cxy) = M2xy/M0 - (M1x/M0)*(M1y/M0);
				g_dev(c,m,Cxz) = M2xz/M0 - (M1x/M0)*(M1z/M0);
				g_dev(c,m,Cyx) = M2yx/M0 - (M1y/M0)*(M1x/M0);
				g_dev(c,m,Cyy) = M2yy/M0 - (M1y/M0)*(M1y/M0);
				g_dev(c,m,Cyz) = M2yz/M0 - (M1y/M0)*(M1z/M0);
				g_dev(c,m,Czx) = M2zx/M0 - (M1z/M0)*(M1x/M0);
				g_dev(c,m,Czy) = M2zy/M0 - (M1z/M0)*(M1y/M0);
				g_dev(c,m,Czz) = M2zz/M0 - (M1z/M0)*(M1z/M0);
			}
		};

		// This shouldn't actually be parallel, but it should be on-device
		Kokkos::parallel_for("update Gaussian", g_dev.extent(1), _update);
	}
}


/*!
  Find the component that has the smallest remaining non-zero alpha_norm
*/
template<typename AlphaType>
static size_t findWeakestComponent(const AlphaType& alpha_norm, const int c) {
	using gmm_float_type = typename AlphaType::value_type;

	auto alpha_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), alpha_norm);
	Kokkos::deep_copy(alpha_host, alpha_norm);
	size_t idx = alpha_host.extent(1) + 1;
	gmm_float_type min_weight = std::numeric_limits<gmm_float_type>::infinity();
	for(size_t m = 0; m < alpha_host.extent(1); m++) {
		if(alpha_host(c,m) > 0. && alpha_host(c,m) < min_weight) {
			min_weight = alpha_host(c,m);
			idx = m;
		}
	}

	if(idx == alpha_host.extent(1) + 1) {
		printf("No component left to remove\n");
		exit(1);
	}

	return idx;
}


/*!
  remove the selected component from the Gaussian Mixture Model
*/
template <typename AlphaType, typename WType>
static void removeGaussianComponent(AlphaType& alpha, AlphaType& alpha_norm, WType& w, WType& w_norm, WType& u, const size_t c, const size_t m) {
	printf("removing component m = %zd from cell %zd\n", m, c);

	auto _remove = KOKKOS_LAMBDA(const int n) {
		if(n == 0) { // Should we allow conflicting writes instead of branching?
			alpha(c,m) = 0.;
			alpha_norm(c,m) = 0.;
		}
		w(c,m,n) = 0.;
		w_norm(c,m,n) = 0.;
		u(c,m,n) = 0.;
	};

	Kokkos::parallel_for("remove smallest Gaussian", w_norm.extent(2), _remove);
}

/*!
  Compute the Gaussian Mixture Model given the particle information
*/
template <typename GaussianType, typename CellSliceType, typename WeightSliceType, typename VelocitySliceType>
static void implReconstructGMM(GaussianType& gaussians, const double eps, const int seed, CellSliceType const& cell, WeightSliceType const& weight, VelocitySliceType const& vx, VelocitySliceType const& vy, VelocitySliceType const& vz) {
	using gmm_float_type = typename GaussianType::value_type;

	const int Nparticles = vx.size();
	const int c_max  = gaussians.extent(0); // Maximum number of Cells
	const size_t k_max  = gaussians.extent(1); // Maximum number of Gaussians
	int N;
	if(dims == 1) {
		N     = 1 + 1*(1+1)/2;       // Number of actually independent degrees of freedom for each 1d Gaussian
	} else if (dims == 2) {
		N     = 2 + 2;               // Number of actually independent degrees of freedom for each uncorrelated(!) 2d Gaussian
	} else if (dims == 3) {
		N     = 3 + 3*(3+1)/2;       // Number of actually independent degrees of freedom for each 3d Gaussian
	}

	// Line 1 in Fig. 2

	Kokkos::View<gmm_float_type**> alpha("alpha", c_max, k_max);
	Kokkos::View<gmm_float_type**> alpha_norm("alpha norm", c_max, k_max);

	auto theta = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace::memory_space(), gaussians);
	Kokkos::deep_copy(theta, gaussians);

	// Parameters for the case where we have the best number of gaussians
	auto theta_best = Kokkos::create_mirror(theta);
	auto alpha_best = Kokkos::create_mirror(alpha_norm);

	// Generate initial guesses for all Gaussians
	Kokkos::Random_XorShift64_Pool<> random_pool(seed);

	if (dims == 1) {
		VelocitySliceType velocity_x = vx;

		for(int c = 0; c < c_max; c++) {
			double cov[1][1];
			GMMImpl<1>::variance(cell, weight, velocity_x, c, cov);
			printf("cell %d. var = %e\n", c, cov[0][0]);

			// Define how to initialize one Gaussian
			auto _init = KOKKOS_LAMBDA(const int m) {
				auto generator = random_pool.get_state();
				theta(c,m,Weight) = 0.; // can we use this instead of alpha?
				// Are we worried that me might draw the same particle multiple times?
				theta(c,m,MuX) = velocity_x(generator.drand(0, Nparticles)); // FIXME: Draw a particle in the right cell
				theta(c,m,Cxx) = cov[0][0];
				alpha(c,m) = 1.;
				random_pool.free_state(generator);
			};

			// execute for all Gaussians
			Kokkos::parallel_for("initial guesses", k_max, _init);
		}
	} else if (dims == 2) {
		VelocitySliceType velocity_par = vx;
		VelocitySliceType velocity_per = vz;

		for(int c = 0; c < c_max; c++) {
			double cov[2][2];
			GMMImpl<2>::variance(cell, weight, velocity_par, velocity_per, c, cov);
			printf("cell %d. var = %e %e, %e %e\n", c, cov[0][0], cov[0][1], cov[1][0], cov[1][1]);

			// Define how to initialize one Gaussian
			auto _init = KOKKOS_LAMBDA(const int m) {
				auto generator = random_pool.get_state();
				theta(c,m,Weight) = 0.; // can we use this instead of alpha?
				// Are we worried that me might draw the same particle multiple times?
				const int particle_idx = generator.drand(0, Nparticles); // FIXME: Draw a particle in the right cell
				theta(c,m,MuPar) = velocity_par(particle_idx);
				theta(c,m,MuPer) = velocity_per(particle_idx);
				theta(c,m,Cparpar) = cov[0][0];
				theta(c,m,Cparper) = cov[0][1];
				theta(c,m,Cperpar) = cov[1][0];
				theta(c,m,Cperper) = cov[1][1];
				alpha(c,m) = 1.;
				random_pool.free_state(generator);
			};

			// execute for all Gaussians
			Kokkos::parallel_for("initial guesses", k_max, _init);
		}
	} else if (dims == 3) {
		VelocitySliceType velocity_x = vx;
		VelocitySliceType velocity_y = vy;
		VelocitySliceType velocity_z = vz;

		for(int c = 0; c < c_max; c++) {
			double cov[3][3];
			GMMImpl<3>::variance(cell, weight, velocity_x, velocity_y, velocity_z, c, cov);
			printf("cell %d. var = %e %e %e, %e %e %e, %e %e %e\n", c, cov[0][0], cov[0][1], cov[0][2], cov[1][0], cov[1][1], cov[1][2], cov[2][0], cov[2][1], cov[2][2]);

			// Define how to initialize one Gaussian
			auto _init = KOKKOS_LAMBDA(const int m) {
				auto generator = random_pool.get_state();
				theta(c,m,Weight) = 0.; // can we use this instead of alpha?
				// Are we worried that me might draw the same particle multiple times?
				const int particle_idx = generator.drand(0, Nparticles); // FIXME: Draw a particle in the right cell
				theta(c,m,MuX) = velocity_x(particle_idx);
				theta(c,m,MuY) = velocity_y(particle_idx);
				theta(c,m,MuZ) = velocity_z(particle_idx);
				theta(c,m,Cxx) = cov[0][0];
				theta(c,m,Cxy) = cov[0][1];
				theta(c,m,Cxz) = cov[0][2];
				theta(c,m,Cyx) = cov[1][0];
				theta(c,m,Cyy) = cov[1][1];
				theta(c,m,Cyz) = cov[1][2];
				theta(c,m,Czx) = cov[2][0];
				theta(c,m,Czy) = cov[2][1];
				theta(c,m,Czz) = cov[2][2];
			};

			// execute for all Gaussians
			Kokkos::parallel_for("initial guesses", k_max, _init);
		}
	}

	// print initial gaussians
	auto thetahost = Kokkos::create_mirror(theta);
	Kokkos::deep_copy(thetahost, theta);
	printf("Initital Guess for gaussians (using diag entries):\n");
	for(size_t c = 0; c < thetahost.extent(0); c++) {
		for(size_t m = 0; m < k_max; m++) {
#if VelocityDimensions == 1
			printf("Gaussian(%zd) in cell %zd weight=%f, Mu=%f, Cov=%f\n", m, c, thetahost(c,m,Weight), thetahost(c,m,MuX), thetahost(c,m,Cxx));
#elif VelocityDimensions == 2
			printf("Gaussian(%zd) in cell %zd weight=%f, Mu=(%f,%f), Cov=((%f,%f),(%f,%f))\n", m, c, thetahost(c,m,Weight), thetahost(c,m,MuPar),thetahost(c,m,MuPer),
				thetahost(c,m,Cparpar), thetahost(c,m,Cparper),
				thetahost(c,m,Cperpar), thetahost(c,m,Cperper));
#elif VelocityDimensions == 3
			printf("Gaussian(%zd) in cell %zd weight=%f, Mu=(%f,%f,%f), Cov=((%f,%f,%f),(%f,%f,%f),(%f,%f,%f))\n", m, c, thetahost(c,m,Weight), thetahost(c,m,MuX),thetahost(c,m,MuY),thetahost(c,m,MuZ),
				thetahost(c,m,Cxx), thetahost(c,m,Cxy), thetahost(c,m,Cxz),
				thetahost(c,m,Cyx), thetahost(c,m,Cyy), thetahost(c,m,Cyz),
				thetahost(c,m,Czx), thetahost(c,m,Czy), thetahost(c,m,Czz));
#endif
		}
	}


	// normalize alpha to sum 1
	normalize(alpha_norm, alpha);

	// at some points we need to have alpha_norm on the host
	auto alpha_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), alpha_norm);

	// Line 4 in Fig. 2
	// Compute the probabilities of all particles relative to all gaussians
	Kokkos::View<gmm_float_type***> u("u", c_max,k_max,Nparticles);
	for(int c = 0; c < c_max; c++) {
		for(size_t m = 0; m < k_max; m++) {
			updateWeights(u, cell, vx,vy,vz, theta, c,m);
		}
	}

	// Compute weights
	Kokkos::View<gmm_float_type***> w("w", c_max,k_max,Nparticles);
	prefillW(w, alpha_norm, u);

	// Normalize weights
	Kokkos::View<gmm_float_type***> w_norm("w_norm", c_max,k_max,Nparticles);
	prefillWNorm(w_norm, w);

	for(int c = 0; c < c_max; c++) { // For now we just do things cell by cell
		printf("#\n# c = %d\n#\n", c);

		// Line 3 in Fig. 2
		int knz = k_max;
		gmm_float_type Lmin = std::numeric_limits<gmm_float_type>::infinity();

		// Line 5 in Fig. 2
		while(knz>0) {
			int t = 0;
			printf("\nknz = %d\n", knz);
			fflush(NULL);

			bool is_first_iter = true;
			gmm_float_type L = std::numeric_limits<gmm_float_type>::infinity();
			gmm_float_type Lold = std::numeric_limits<gmm_float_type>::infinity();
			// Line 6 in Fig. 2
			bool converged = false;
			while(!converged) {
				// Line 7 in Fig. 2
				t += 1;

				// Line 8 in Fig. 2
				for(size_t m = 0; m < k_max; m++) {
					// Get a current copy of alpha_norm
					Kokkos::deep_copy(alpha_host, alpha_norm);
					if(alpha_host(c,m) <= 0.) {
						continue; // with next m
					}

					// Line 9 of Fig. 2
					auto _updatew = KOKKOS_LAMBDA(const int& i) {
						w(c,m,i) = alpha_norm(c,m) * u(c,m,i);
					};
					Kokkos::parallel_for("Update w", Nparticles, _updatew);
					auto _updatewnorm = KOKKOS_LAMBDA(const int& i) {
						gmm_float_type sum = 0.;
						for(size_t mprime = 0; mprime < k_max; mprime++) {
							sum += w(c,mprime,i);
						}
						if(sum == 0.) {
							w_norm(c,m,i) = 0.;
						} else {
							w_norm(c,m,i) = weight(i) * w(c,m,i) / sum;
						}
						};
					Kokkos::parallel_for("Update wnorm", Nparticles, _updatewnorm);

					// Line 10 of Fig. 2
					updateAlpha(alpha, w_norm, c,m);

					// Line 11 of Fig. 2
					// normalize alpha to sum 1
					normalize(alpha_norm, alpha);

					Kokkos::deep_copy(alpha_host, alpha_norm);

					// Line 12 of Fig. 2
					// Get a current copy of alpha_norm
					Kokkos::deep_copy(alpha_host, alpha_norm);
					if(alpha_host(c,m) > 0.) {
						// Line 13 of Fig. 2
						updateGMM(theta, cell, vx,vy,vz, w_norm, c,m);

						// Line 14 of Fig. 2
						// update u
						updateWeights(u, cell, vx,vy,vz, theta, c,m);
						// update w
						Kokkos::parallel_for("Update w", Nparticles, _updatew);
						// update w_norm
						Kokkos::parallel_for("Update wnorm", Nparticles, _updatewnorm);
					} else { // Line 15 of Fig. 2
						printf("alpha(%d,%lu) <= 0\n", c,m);
						removeGaussianComponent(alpha, alpha_norm, w, w_norm, u, c,m);
						// Line 16 of Fig. 2
						// count how many gaussians have alpha_norm(m)>0
						int new_knz = 0;
						for(size_t mprime = 0; mprime < k_max; mprime++) {
							if(alpha_host(c,mprime) > 0.) {
								new_knz++;
							}
						}
						knz = new_knz;
						printf("knz = %d\n", knz);
						if(knz <= 0) {
							Kokkos::abort("Unexpectedly all gaussian weights have fallen below zero. Construction of Gaussian mixture model failed.");
						}
					} // Line 17 of Fig. 2
				} // Line 18 of Fig. 2
				// Line 19 of Fig. 2
				auto thetat = Kokkos::create_mirror(theta);
				Kokkos::deep_copy(thetat, theta);

				// Line 20 of Fig. 2
				gmm_float_type term1b = 0.;
				Kokkos::deep_copy(alpha_host, alpha_norm);
				for(size_t mprime = 0; mprime < k_max; mprime++) {
					if(alpha_host(c,mprime) > 0.) {
						term1b += std::log(alpha_host(c,mprime));
					}
				}

				gmm_float_type term2 = 0.;
				auto _term2 = KOKKOS_LAMBDA(const int i, gmm_float_type& lsum) {
					gmm_float_type tmp2 = 0.;
					for(size_t mprime = 0; mprime < k_max; mprime++) {
						if(alpha_norm(c,mprime) > 0.) {
							tmp2 += alpha_norm(c,mprime) * u(c,mprime,i);
						}
					}
					if(tmp2 > 0.) {
						lsum += weight(i) * Kokkos::log(tmp2);
					}
				};
				Kokkos::parallel_reduce("get term2", Nparticles, _term2, term2);

				Lold = L;
				gmm_float_type d = knz * N + knz - 1.;
				L = N/2. * term1b + 0.5*d*std::log(Nparticles) - term2;

				// Line 21 of Fig 2.
				if(is_first_iter) {
					is_first_iter = false;
				} else {
					//printf("knz = %d, Lold = %f, L = %f\n", knz, Lold, L);
					if(fabs(Lold-L) < eps*fabs(L)) {
						printf("converged on %d gaussians with L= %f in %d iterations!\n", knz, L, t);
						converged = true;
					}
				}
			}

			// Line 22 of Fig. 2
			if(L < Lmin) {
				size_t mu_min = findWeakestComponent(alpha_norm, c);
				Kokkos::deep_copy(alpha_host, alpha_norm);
				if(alpha_host(c,mu_min) * Nparticles >= pow(2,dims)) {
					// Line 23
					Lmin = L;
					// Line 24
					Kokkos::deep_copy(theta_best, theta);
					Kokkos::deep_copy(alpha_best, alpha_norm);
					printf("new Lmin = %f for knz = %d\n", Lmin, knz);
				}
			} // Line 25

			// Line 26 of Fig. 2
			if(knz > 1) { // don't remove the last Gaussian
				size_t mu_min = findWeakestComponent(alpha_norm, c);
				Kokkos::deep_copy(alpha_host, alpha_norm);
				if(alpha_host(c,mu_min) * Nparticles < pow(2,dims)) {
					printf("component predicts zero particles, removing\n");
				} else {
					printf("weakest component isn't that weak, why should this make anything better?\n");
				}
				removeGaussianComponent(alpha, alpha_norm, w, w_norm, u, c, mu_min);
			}
			knz--;

		} // Line 27

		Kokkos::deep_copy(gaussians, theta_best); // When we are done, copy the gaussians back to the host

		// Copy back alpha norm and set the weights of the Gaussians from that
		Kokkos::deep_copy(alpha_host, alpha_best);
		for(size_t m = 0; m < k_max; m++) {
			gaussians(c,m,Weight) = alpha_host(c,m);
		}

		// copy that value of theta_best back to the device, because we trashed
		// out theta there, trying smaller numbers of gaussians in that cell
		Kokkos::deep_copy(theta, gaussians);
	}

	printf("Reconstruction done\n");
}


};

} // end namespace Cabana

#endif // end CABANA_GMM_IMPL_HPP
