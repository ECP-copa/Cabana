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

#include <Kokkos_Vector.hpp>
#include <impl/Cabana_Erfinv.hpp>

#include <gtest/gtest.h>

namespace Test
{

//---------------------------------------------------------------------------//
// Erfinv test
void testErfinv()
{
	// Test values from the original paper
	EXPECT_NEAR(Cabana::Impl::ppnd7(0.25),  -0.6744897501960817, 1e-7);
	EXPECT_NEAR(Cabana::Impl::ppnd7(0.001), -3.090232306167814,  1e-6);
	EXPECT_NEAR(Cabana::Impl::ppnd7(1e-20), -9.262340089798408,  1e-6);

	// test that erfinv() is actually the inverse of erf()
	double testpoints[] = {-0.999, -0.9, -0.5, -0.1, -1e-3, -1e-8, 1e-8, 1e-3, 0.1, 0.5, 0.9, 0.999};
	for(int i = 0; i < sizeof(testpoints)/sizeof(testpoints[0]); i++) {
		const double x = testpoints[i];
		EXPECT_NEAR(x, Kokkos::erf(Cabana::Impl::erfinv(x)), 1e-7);
	}
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( cabana_erfinv, erfinv_test ) { testErfinv(); }

//---------------------------------------------------------------------------//

} // end namespace Test
