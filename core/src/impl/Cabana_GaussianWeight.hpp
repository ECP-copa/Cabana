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
  \file impl/Cabana_GaussianWeight.hpp
*/
#ifndef CABANA_GAUSSIAN_WEIGHT_HPP
#define CABANA_GAUSSIAN_WEIGHT_HPP

#include <impl/Cabana_Matrix2d.hpp>

namespace Cabana {
namespace Impl {

template<typename GMMFloatType>
class GaussianWeight {

public:

/*!
  Compute the value of a 1d Gaussian distribution function with mean Mu and covariance C at velocity v
*/
static GMMFloatType KOKKOS_INLINE_FUNCTION weight_1d(const GMMFloatType (&v)[1], const GMMFloatType (&Mu)[1], const GMMFloatType (&C)[1][1]) {
	GMMFloatType I[1][1];
	Matrix2d<GMMFloatType,1>::invert(I,C);
	const GMMFloatType det = Matrix2d<GMMFloatType,1>::determinant(C);

	const GMMFloatType arg = (v[0]-Mu[0]) * I[0][0] * (v[0]-Mu[0]);
	return pow(2.*M_PI, -0.5*1)/Kokkos::sqrt(det) * Kokkos::exp(-0.5*arg);
}

/*!
  Compute the value of a 2d ring distribution function with mean Mu and covariance C at velocity v
*/
static GMMFloatType KOKKOS_INLINE_FUNCTION weight_2d(const GMMFloatType (&v)[2], const GMMFloatType (&Mu)[2], const GMMFloatType (&C)[2][2]) {
	return v[1]/Kokkos::sqrt(2.*M_PI*C[0][0])/C[1][1] * Kokkos::exp(-0.5*Mu[1]*Mu[1]/C[1][1]) * Kokkos::exp(-0.5*(v[0]-Mu[0])*(v[0]-Mu[0])/C[0][0]) * Kokkos::exp(-0.5*v[1]*v[1]/C[1][1]) *
	       Kokkos::Experimental::cyl_bessel_i0<Kokkos::complex<GMMFloatType>, double, int>(Kokkos::complex(v[1]*Mu[1]/C[1][1])).real();
}

/*!
  Compute the value of a 3d Gaussian distribution function with mean Mu and covariance C at velocity v
*/
static GMMFloatType KOKKOS_INLINE_FUNCTION weight_3d(const GMMFloatType (&v)[3], const GMMFloatType (&Mu)[3], const GMMFloatType (&C)[3][3]) {
	GMMFloatType I[3][3];
	Matrix2d<GMMFloatType,3>::invert(I,C);
	const GMMFloatType det = Matrix2d<GMMFloatType,3>::determinant(C);

	const GMMFloatType rx = I[0][0]*(v[0]-Mu[0]) + I[0][1]*(v[1]-Mu[1]) + I[0][2]*(v[2]-Mu[2]);
	const GMMFloatType ry = I[1][0]*(v[0]-Mu[0]) + I[1][1]*(v[1]-Mu[1]) + I[1][2]*(v[2]-Mu[2]);
	const GMMFloatType rz = I[2][0]*(v[0]-Mu[0]) + I[2][1]*(v[1]-Mu[1]) + I[2][2]*(v[2]-Mu[2]);

	const GMMFloatType arg = (v[0]-Mu[0])*rx + (v[1]-Mu[1])*ry + (v[2]-Mu[2])*rz;
	return pow(2.*M_PI, -0.5*3)/Kokkos::sqrt(det) * Kokkos::exp(-0.5*arg);
}

};


} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_GAUSSIAN_WEIGHT_HPP
