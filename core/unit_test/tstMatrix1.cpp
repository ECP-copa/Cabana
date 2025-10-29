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
#include <impl/Cabana_Matrix2d.hpp>

#include <gtest/gtest.h>

namespace Test
{

//---------------------------------------------------------------------------//
// Matrix test
void testMatrix()
{
	// Hard code a random matrix
	const float A1[1][1] = {4.f};

	// Check determinant
	auto _determinant_A1 = KOKKOS_LAMBDA(const int& i) {
		float det = Cabana::Impl::Matrix2d<float,1>::determinant(A1);

		assert(det == 4.f);
	};
	Kokkos::parallel_for("Determinant A1", 1, _determinant_A1);

	// Check transposition
	auto _transpose_A1 = KOKKOS_LAMBDA(const int& i) {
		float T1[1][1];
		Cabana::Impl::Matrix2d<float,1>::transpose(T1, A1);

		assert(T1[0][0] == 4.f);
	};
	Kokkos::parallel_for("Transpose A1", 1, _transpose_A1);

	// Check inverse
	auto _invert_A1 = KOKKOS_LAMBDA(const int& i) {
		float I1[1][1];
		Cabana::Impl::Matrix2d<float,1>::invert(I1, A1);

		assert(I1[0][0] ==  0.25f);
	};
	Kokkos::parallel_for("Invert A1", 1, _invert_A1);

	// Test cholesky decomposition
	auto _cholesky_A1 = KOKKOS_LAMBDA(const int& i) {
		float C1[1][1];
		Cabana::Impl::Matrix2d<float,1>::cholesky(C1, A1);

		assert(C1[0][0] ==  2.f);
	};
	Kokkos::parallel_for("Decompose A1", 1, _cholesky_A1);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( cabana_matrix, matrix_test ) { testMatrix(); }

//---------------------------------------------------------------------------//

} // end namespace Test




