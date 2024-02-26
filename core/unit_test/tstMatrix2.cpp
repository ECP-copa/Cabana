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
	const float A2[2][2] = {{1.94f, 1.14f},
							{5.58f, 6.66f}};

	// Hard code a second, symmetrix matrix
	const float B2[2][2] = {{ 4.f, 12.f},
							{12.f, 37.f}};

	// Check determinant
	auto _determinant_A2 = KOKKOS_LAMBDA(const int& i) {
		float det = Cabana::Impl::Matrix2d<float,2>::determinant(A2);

		assert(det == 6.5592f);
	};
	Kokkos::parallel_for("Determinant A2", 1, _determinant_A2);

	auto _determinant_B2 = KOKKOS_LAMBDA(const int& i) {
		float det = Cabana::Impl::Matrix2d<float,2>::determinant(B2);

		assert(det == 4.f);
	};
	Kokkos::parallel_for("Determinant B2", 1, _determinant_B2);

	// Check transposition
	auto _transpose_A2 = KOKKOS_LAMBDA(const int& i) {
		float T2[2][2];
		Cabana::Impl::Matrix2d<float,2>::transpose(T2, A2);

		assert(T2[0][0] == 1.94f);
		assert(T2[0][1] == 5.58f);
		assert(T2[1][0] == 1.14f);
		assert(T2[1][1] == 6.66f);
	};
	Kokkos::parallel_for("Transpose A2", 1, _transpose_A2);

	auto _transpose_B2 = KOKKOS_LAMBDA(const int& i) {
		float T2[2][2];
		Cabana::Impl::Matrix2d<float,2>::transpose(T2, B2);

		assert(T2[0][0] == B2[0][0]);
		assert(T2[0][1] == B2[0][1]);
		assert(T2[1][0] == B2[1][0]);
		assert(T2[1][1] == B2[1][1]);
	};
	Kokkos::parallel_for("Transpose B2", 1, _transpose_B2);

	// Check inverse
	auto _invert_A2 = KOKKOS_LAMBDA(const int& i) {
		float I2[2][2];
		Cabana::Impl::Matrix2d<float,2>::invert(I2, A2);

		assert(I2[0][0] ==  1.015370f);
		assert(I2[0][1] == -0.173802f);
		assert(I2[1][0] == -0.850714f);
		assert(I2[1][1] ==  0.295768f);
	};
	Kokkos::parallel_for("Invert A2", 1, _invert_A2);

	auto _invert_B2 = KOKKOS_LAMBDA(const int& i) {
		float I2[2][2];
		Cabana::Impl::Matrix2d<float,2>::invert(I2, B2);

		assert(I2[0][0] ==  9.25f);
		assert(I2[0][1] == -3.00f);
		assert(I2[1][0] == -3.00f);
		assert(I2[1][1] ==  1.00f);
	};
	Kokkos::parallel_for("Invert B2", 1, _invert_B2);

	// Test cholesky decomposition
	auto _cholesky_B2 = KOKKOS_LAMBDA(const int& i) {
		float C2[2][2];
		Cabana::Impl::Matrix2d<float,2>::cholesky(C2, B2);

		assert(C2[0][0] == 2.f);
		assert(C2[0][1] == 0.f);
		assert(C2[1][0] == 6.f);
		assert(C2[1][1] == 1.f);
	};
	Kokkos::parallel_for("Decompose B2", 1, _cholesky_B2);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( cabana_matrix, matrix_test ) { testMatrix(); }

//---------------------------------------------------------------------------//

} // end namespace Test
