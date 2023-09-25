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
	const float A3[3][3] = {{1.94f, 1.14f, 1.47f},
							{5.58f, 6.66f, 0.25f},
							{8.16f, 0.83f, 8.81f}};

	// Hard code a second, symmetrix matrix
	const float B3[3][3] = {{  4.f,  12.f, -16.f},
							{ 12.f,  37.f, -43.f},
							{-16.f, -43.f,  98.f}};

	// Check determinant
	auto _determinant_A3 = KOKKOS_LAMBDA(const int& i) {
		float det = Cabana::Impl::Matrix2d<float,3>::determinant(A3);

		assert(det == -13.3703f);
	};
	Kokkos::parallel_for("Determinant A3", 1, _determinant_A3);

	auto _determinant_B3 = KOKKOS_LAMBDA(const int& i) {
		float det = Cabana::Impl::Matrix2d<float,3>::determinant(B3);

		assert(det == 36.f);
	};
	Kokkos::parallel_for("Determinant B3", 1, _determinant_B3);

	// Check transposition
	auto _transpose_A3 = KOKKOS_LAMBDA(const int& i) {
		float T3[3][3];
		Cabana::Impl::Matrix2d<float,3>::transpose(T3, A3);

		assert(T3[0][0] == 1.94f);
		assert(T3[0][1] == 5.58f);
		assert(T3[0][2] == 8.16f);
		assert(T3[1][0] == 1.14f);
		assert(T3[1][1] == 6.66f);
		assert(T3[1][2] == 0.83f);
		assert(T3[2][0] == 1.47f);
		assert(T3[2][1] == 0.25f);
		assert(T3[2][2] == 8.81f);
	};
	Kokkos::parallel_for("Transpose A3", 1, _transpose_A3);

	auto _transpose_B3 = KOKKOS_LAMBDA(const int& i) {
		float T3[3][3];
		Cabana::Impl::Matrix2d<float,3>::transpose(T3, B3);

		assert(T3[0][0] == B3[0][0]);
		assert(T3[0][1] == B3[0][1]);
		assert(T3[0][2] == B3[0][2]);
		assert(T3[1][0] == B3[1][0]);
		assert(T3[1][1] == B3[1][1]);
		assert(T3[1][2] == B3[1][2]);
		assert(T3[2][0] == B3[2][0]);
		assert(T3[2][1] == B3[2][1]);
		assert(T3[2][2] == B3[2][2]);
	};
	Kokkos::parallel_for("Transpose B3", 1, _transpose_B3);

	// Check inverse
	auto _invert_A3 = KOKKOS_LAMBDA(const int& i) {
		float I3[3][3];
		Cabana::Impl::Matrix2d<float,3>::invert(I3, A3);

		assert(I3[0][0] == -4.372920f);
		assert(I3[0][1] ==  0.659919f);
		assert(I3[0][2] ==  0.710920f);
		assert(I3[1][0] ==  3.524220f);
		assert(I3[1][1] == -0.381159f);
		assert(I3[1][2] == -0.577221f);
		assert(I3[2][0] ==  3.718266f);
		assert(I3[2][1] == -0.575321f);
		assert(I3[2][2] == -0.490581f);
	};
	Kokkos::parallel_for("Invert A3", 1, _invert_A3);

	auto _invert_B3 = KOKKOS_LAMBDA(const int& i) {
		float I3[3][3];
		Cabana::Impl::Matrix2d<float,3>::invert(I3, B3);

		assert(I3[0][0] ==  49.361111f);
		assert(I3[0][1] == -13.555555f);
		assert(I3[0][2] ==   2.111111f);
		assert(I3[1][0] == -13.555555f);
		assert(I3[1][1] ==   3.777777f);
		assert(I3[1][2] ==  -0.555555f);
		assert(I3[2][0] ==   2.111111f);
		assert(I3[2][1] ==  -0.555555f);
		assert(I3[2][2] ==   0.111111f);
	};
	Kokkos::parallel_for("Invert B3", 1, _invert_B3);

	// Test cholesky decomposition
	auto _cholesky_B3 = KOKKOS_LAMBDA(const int& i) {
		float C3[3][3];
		Cabana::Impl::Matrix2d<float,3>::cholesky(C3, B3);

		assert(C3[0][0] ==  2.f);
		assert(C3[0][1] ==  0.f);
		assert(C3[0][2] ==  0.f);
		assert(C3[1][0] ==  6.f);
		assert(C3[1][1] ==  1.f);
		assert(C3[1][2] ==  0.f);
		assert(C3[2][0] == -8.f);
		assert(C3[2][1] ==  5.f);
		assert(C3[2][2] ==  3.f);
	};
	Kokkos::parallel_for("Decompose B3", 1, _cholesky_B3);
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( cabana_matrix, matrix_test ) { testMatrix(); }

//---------------------------------------------------------------------------//

} // end namespace Test
