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
#include <impl/Cabana_Hammersley.hpp>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <algorithm>

#include <gtest/gtest.h>

namespace Test
{

//---------------------------------------------------------------------------//
// Corput test
void testCorput()
{
	EXPECT_NEAR(Cabana::Impl::Corput<2>::value(0), 0./8., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<2>::value(1), 4./8., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<2>::value(2), 2./8., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<2>::value(3), 6./8., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<2>::value(4), 1./8., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<2>::value(5), 5./8., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<2>::value(6), 3./8., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<2>::value(7), 7./8., 1e-7);

	EXPECT_NEAR(Cabana::Impl::Corput<3>::value(0), 0./9., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<3>::value(1), 3./9., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<3>::value(2), 6./9., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<3>::value(3), 1./9., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<3>::value(4), 4./9., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<3>::value(5), 7./9., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<3>::value(6), 2./9., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<3>::value(7), 5./9., 1e-7);
	EXPECT_NEAR(Cabana::Impl::Corput<3>::value(8), 8./9., 1e-7);

	const int N = 1020; // 34 * 2*3*5
	auto _print_hammersley = KOKKOS_LAMBDA(const int& i) {
		printf("%f %f %f %f\n", Cabana::Impl::hammersley(0,i,N), Cabana::Impl::hammersley(1,i,N), Cabana::Impl::hammersley(2,i,N), Cabana::Impl::hammersley(3,i,N));
	};
	Kokkos::parallel_for("print hammersley", N, _print_hammersley);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( cabana_corput, corput_test ) { testCorput(); }

//---------------------------------------------------------------------------//

} // end namespace Test
