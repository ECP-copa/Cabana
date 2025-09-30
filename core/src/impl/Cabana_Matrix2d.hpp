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
  \file impl/Cabana_Matrix2d.hpp
  \brief
  Linear algebra helpers. A lot of them are not implemented for generic
  matricies and we probably shouldn't do that either, but call to some
  appropriate linear algebra package. Special cases for small matricies are
  implemented below.
*/
#ifndef CABANA_MATRIX2D_IMPL_HPP
#define CABANA_MATRIX2D_IMPL_HPP

#include <cmath>

namespace Cabana {

namespace Impl {

template<typename FloatType, int dim>
class Matrix2d {

public:
/*!
  Compute the determinant of a dim-by-dim matrix
*/
	static FloatType KOKKOS_INLINE_FUNCTION determinant(const FloatType (&A)[dim][dim]) {
		// I would use fprintf to stderr, but that isn't possible from device code
		printf("Somebody should implement matrix determinants for larger than 3x3\n");
		return 0./0.;
	}

/*!
  Compute the Cholesky decomposition of a dim-by-dim matrix A and return B such that B * B.T = A
*/
	static void KOKKOS_INLINE_FUNCTION cholesky(FloatType (&B)[dim][dim], const FloatType (&A)[dim][dim]) {
		printf("Somebody should implement Cholesky decomposition for larger than 3x3\n");
		for(size_t i = 0; i < dim; i++) {
			for(size_t j = 0; j < dim; j++) {
				B[i][j] = 0./0.;
			}
		}
	}

/*!
  Transpose a dim-by-dim matrix matrix A and returns B such that B.T = A
*/
	static void KOKKOS_INLINE_FUNCTION transpose(FloatType (&B)[dim][dim], const FloatType (&A)[dim][dim]) {
		for(size_t i = 0; i < dim; i++) {
			for(size_t j = 0; j < dim; j++) {
				B[i][j] = A[j][i];
			}
		}
	}

/*!
  Invert a dim-by-dim matrix matrix A and returns B such that B.A = 1
*/
	static void KOKKOS_INLINE_FUNCTION invert(FloatType (&B)[dim][dim], const FloatType (&A)[dim][dim]) {
		printf("Somebody should implement matrix inversion for larger than 3x3\n");
		for(size_t i = 0; i < dim; i++) {
			for(size_t j = 0; j < dim; j++) {
				B[i][j] = 0./0.;
			}
		}
	}
};

//
// The trivial 1x1 case
//
template<typename FloatType> struct Matrix2d<FloatType,1> {
/*!
  Compute the determinant of a trivial 1-by-1 matrix
*/
	static FloatType KOKKOS_INLINE_FUNCTION determinant(const FloatType (&A)[1][1]) {
		return A[0][0];
	}

/*!
  Compute the Cholesky decomposition of a trival 1-by-1 matrix A and return B such that B * B.T = A
*/
	static void KOKKOS_INLINE_FUNCTION cholesky(FloatType (&B)[1][1], const FloatType (&A)[1][1]) {
		B[0][0] = Kokkos::sqrt(A[0][0]);
	}

/*!
  Transpose a trivial 1-by-1 matrix matrix A and returns B such that B.T = A
*/
	static void KOKKOS_INLINE_FUNCTION transpose(FloatType (&B)[1][1], const FloatType (&A)[1][1]) {
		B[0][0] = A[0][0];
	}

/*!
  Invert a trivial 1-by-1 matrix matrix A and returns B such that B.A = 1
*/
	static void KOKKOS_INLINE_FUNCTION invert(FloatType (&B)[1][1], const FloatType (&A)[1][1]) {
		// determinant
		const FloatType det = determinant(A);

		// elements of the inverse
		B[0][0] =  1./det;
	}
};

//
// The 2x2 case
//
template<typename FloatType> struct Matrix2d<FloatType,2> {
/*!
  Compute the determinant of a 2-by-2 matrix
*/
	static FloatType KOKKOS_INLINE_FUNCTION determinant(const FloatType (&A)[2][2]) {
		return A[0][0]*A[1][1] - A[1][0]*A[0][1];
	}

/*!
  Compute the Cholesky decomposition of a 2-by-2 matrix A and return B such that B * B.T = A
*/
	static void KOKKOS_INLINE_FUNCTION cholesky(FloatType (&B)[2][2], const FloatType (&A)[2][2]) {
		B[0][0] = Kokkos::sqrt(A[0][0]);
		B[0][1] = 0.;
		B[1][0] = A[1][0] / B[0][0];
		B[1][1] = Kokkos::sqrt(A[1][1] - B[1][0]*B[1][0]);
	}

/*!
  Transpose a 2-by-2 matrix matrix A and returns B such that B.T = A
*/
	static void KOKKOS_INLINE_FUNCTION transpose(FloatType (&B)[2][2], const FloatType (&A)[2][2]) {
		B[0][0] = A[0][0];
		B[0][1] = A[1][0];
		B[1][0] = A[0][1];
		B[1][1] = A[1][1];
	}

/*!
  Invert a 2-by-2 matrix matrix A and returns B such that B.A = 1
*/
	static void KOKKOS_INLINE_FUNCTION invert(FloatType (&B)[2][2], const FloatType (&A)[2][2]) {
		// determinant
		const FloatType det = determinant(A);

		// elements of the inverse
		B[0][0] =  A[1][1]/det;
		B[0][1] = -A[0][1]/det;
		B[1][0] = -A[1][0]/det;
		B[1][1] =  A[0][0]/det;
	}
};

//
// The 3x3 case
//
template<typename FloatType> struct Matrix2d<FloatType,3> {
/*!
  Compute the determinant of a 3-by-3 matrix
*/
	static FloatType KOKKOS_INLINE_FUNCTION determinant(const FloatType (&A)[3][3]) {
		return A[0][0]*A[1][1]*A[2][2]
			 + A[0][1]*A[1][2]*A[2][0]
			 + A[0][2]*A[1][0]*A[2][1]
			 - A[2][0]*A[1][1]*A[0][2]
			 - A[2][1]*A[1][2]*A[0][0]
			 - A[2][2]*A[1][0]*A[0][1];
	}

/*!
  Compute the Cholesky decomposition of a 3-by-3 matrix A and return B such that B * B.T = A
*/
	static void KOKKOS_INLINE_FUNCTION cholesky(FloatType (&B)[3][3], const FloatType (&A)[3][3]) {
		B[0][0] = Kokkos::sqrt(A[0][0]);
		B[0][1] = 0.;
		B[0][2] = 0.;
		B[1][0] = A[1][0] / B[0][0];
		B[1][1] = Kokkos::sqrt(A[1][1] - B[1][0]*B[1][0]);
		B[1][2] = 0.;
		B[2][0] = A[2][0] / B[0][0];
		B[2][1] = (A[2][1] - B[2][0]*B[1][0]) / B[1][1];
		B[2][2] = Kokkos::sqrt(A[2][2] - B[2][0]*B[2][0] - B[2][1]*B[2][1]);
	}

/*!
  Transpose a 3-by-3 matrix matrix A and returns B such that B.T = A
*/
	static void KOKKOS_INLINE_FUNCTION transpose(FloatType (&B)[3][3], const FloatType (&A)[3][3]) {
		B[0][0] = A[0][0];
		B[0][1] = A[1][0];
		B[0][2] = A[2][0];
		B[1][0] = A[0][1];
		B[1][1] = A[1][1];
		B[1][2] = A[2][1];
		B[2][0] = A[0][2];
		B[2][1] = A[1][2];
		B[2][2] = A[2][2];
	}

/*!
  Invert a 3-by-3 matrix matrix A and returns B such that B.A = 1
*/
	static void KOKKOS_INLINE_FUNCTION invert(FloatType (&B)[3][3], const FloatType (&A)[3][3]) {
		// determinant
		const FloatType det = determinant(A);

		// elements of the inverse
		B[0][0] = (A[1][1]*A[2][2]-A[1][2]*A[2][1])/det;
		B[0][1] = (A[0][2]*A[2][1]-A[0][1]*A[2][2])/det;
		B[0][2] = (A[0][1]*A[1][2]-A[0][2]*A[1][1])/det;
		B[1][0] = (A[1][2]*A[2][0]-A[1][0]*A[2][2])/det;
		B[1][1] = (A[0][0]*A[2][2]-A[0][2]*A[2][0])/det;
		B[1][2] = (A[0][2]*A[1][0]-A[0][0]*A[1][2])/det;
		B[2][0] = (A[1][0]*A[2][1]-A[1][1]*A[2][0])/det;
		B[2][1] = (A[0][1]*A[2][0]-A[0][0]*A[2][1])/det;
		B[2][2] = (A[0][0]*A[1][1]-A[0][1]*A[1][0])/det;
	}
};

} // end namespace Impl

} // end namespace Cabana

#endif // end CABANA_MATRIX2D_IMPL_HPP
