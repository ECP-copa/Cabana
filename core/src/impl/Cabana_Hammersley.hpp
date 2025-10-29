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
  \file Cabana_Hammersley.hpp
*/
#ifndef CABANA_HAMMERSLEY_HPP
#define CABANA_HAMMERSLEY_HPP

#include <Kokkos_Vector.hpp>

namespace Cabana {
namespace Impl {

/*!
  Reverse the bits in the unsigned 64bit input
*/
uint64_t KOKKOS_INLINE_FUNCTION reverseBits64(uint64_t n) {
	n = ((n & 0x00000000ffffffff) << 32) | ((n & 0xffffffff00000000) >> 32);
	n = ((n & 0x0000ffff0000ffff) << 16) | ((n & 0xffff0000ffff0000) >> 16);
	n = ((n & 0x00ff00ff00ff00ff) <<  8) | ((n & 0xff00ff00ff00ff00) >>  8);
	n = ((n & 0x0f0f0f0f0f0f0f0f) <<  4) | ((n & 0xf0f0f0f0f0f0f0f0) >>  4);
	n = ((n & 0x3333333333333333) <<  2) | ((n & 0xcccccccccccccccc) >>  2);
	n = ((n & 0x5555555555555555) <<  1) | ((n & 0xaaaaaaaaaaaaaaaa) >>  1);
	return n;
}

/*!
  Reverse the digits of the input n in the number system with basis base to
  compute the nth entry in the Van der Corput sequence.
*/
template <int base> struct Corput {
static double KOKKOS_INLINE_FUNCTION value(int n){
	const double inv_Base = (double)1. / (double)base;
	uint64_t out = 0;
	double power = 1.;
	while (n > 0.) {
		uint64_t next  = n / base;        // next value with the last digit removed / after the right shift
		uint64_t digit = n - next * base; // last digit that we pop off
		out = out * base + digit;         // shift output left and append digit
		power *= inv_Base;                // keep track of how often we have done that
		n = next;                         // shift input right
	}
	// At this point out is the reversed string when writing n as digits in base
	return out * power;                   // get a value between 0 and 1
}
};

/*!
  In base 2 reversing the digits is just reversing the bits and then scaling by
  the appropriate constant to shift things to the right of the decimal point.
  This is much faster than the general case.
*/
template<> struct Corput<2> {
static double KOKKOS_INLINE_FUNCTION value(int n) {
	return reverseBits64(n) * 0x1p-64;
}
};


/*!
  Generate the nth sample (of N) of the hammersley sequence and return the value in dimension i
*/
double KOKKOS_INLINE_FUNCTION hammersley(const int i, const uint64_t n, const uint64_t N) {
	if(n >= N) {
		printf("Called hammersely with n = %zd, N=%zd, which is wrong\n", n, N);
	}
	if((i < 0) || (i > 5)) {
		printf("Called hammersely with i = %d, which is wrong\n", i);
	}

	if(i == 0) {
		return (double)n/(double)N;
	} else if (i == 1) {
		return Corput<2>::value(n);
	} else if (i == 2) {
		return Corput<3>::value(n);
	} else if (i == 3) {
		return Corput<5>::value(n);
	} else if (i == 4) {
		return Corput<7>::value(n);
	} else if (i == 5) {
		return Corput<11>::value(n);
	} else {
		return 0. / 0.;
	}
}


} // end namespace Impl
} // end namespace Cabana

#endif // end CABANA_HAMMERSLEY_HPP
