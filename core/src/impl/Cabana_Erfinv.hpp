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
  \file impl/Cabana_Erfinv.hpp
*/
#ifndef CABANA_ERFINV_HPP
#define CABANA_ERFINV_HPP

#include <cstdio>
#include <cmath>

namespace Cabana {
namespace Impl {

/*!
 This produces the inverse of
 Phi(zp) = integrate(1/sqrt(2*pi) * exp(-zeta**2/2), (zeta,-oo,zp))
         = erf(sqrt(2)*zp/2)/2 + 1/2
 This means that ppnd7(p) = sqrt(2)*erfinv(2*p - 1)
*/
double KOKKOS_INLINE_FUNCTION ppnd7(double p) {
	/*
	Algorithm AS 241    Appl. Statist. (1988) Vol. 37, No. 3
	Produce the normal deviate  z  corresponding to a given lower
	tail area of  p;  z  is accurate to about 1 part in 10**7
	*/

	const double split1 = 0.425E0;
	const double split2 = 5.0E0;
	const double const1 = 0.180625E0;
	const double const2 = 1.6E0;

	// coefficicents for  p  close to  1/2
	const double A0 = 3.38713'27179E0;
	const double A1 = 5.04342'71938E1;
	const double A2 = 1.59291'13202E2;
	const double A3 = 5.91093'74720E1;
	const double B1 = 1.78951'69469E1;
	const double B2 = 7.87577'57664E1;
	const double B3 = 6.71875'63600E1;

	// coefficients for  p  neither close to  1/2  nor 0 or 1
	const double C0 = 1.42343'72777E0;
	const double C1 = 2.75681'53900E0;
	const double C2 = 1.30672'84816E0;
	const double C3 = 1.70238'21103E-1;
	const double D1 = 7.37001'64250E-1;
	const double D2 = 1.20211'32975E-1;

	// coefficients for  p  near 0 or 1
	const double E0 = 6.65790'51150E0;
	const double E1 = 3.08122'63860E0;
	const double E2 = 4.28682'94337E-1;
	const double E3 = 1.73372'03997E-2;
	const double F1 = 2.41978'94225E-1;
	const double F2 = 1.22582'02635E-2;

	const double q = p - 0.5;
	if(fabs(q) < split1) {
		const double r = const1 - q*q;
		const double out = q * (((A3 * r + A2) * r + A1) * r + A0) / (((B3 * r + B2) * r + B1) * r + 1.0);
		return out;
	} else {
		double r, out;
		if(q < 0.) {
			r = p;
		} else {
			r = 1.0 - p;
		}
		if((r < 0.) || (r > 1.)) {
			Kokkos::abort("unexpected problem in ppnd7() inside erfinv()");
		}
		r = Kokkos::sqrt(-log(r));
		if(r < split2) {
			r = r - const2;
			out = (((C3 * r + C2) * r + C1) * r + C0) / ((D2 * r + D1) * r + 1.0);
		} else {
			r = r - split2;
			out = (((E3 * r + E2) * r + E1) * r + E0) / ((F2 * r + F1) * r + 1.0);
		}

		if(q < 0.) {
			out = -out;
		}

		return out;
	}
}

double KOKKOS_INLINE_FUNCTION erfinv(double x) {
	const double out = ppnd7((x+1.)/2.)/Kokkos::sqrt(2.);
	return out;
}


} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_ERFINV_HPP
