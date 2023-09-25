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
  \file Cabana_GaussianMixtureModel.hpp
  \brief Creation of a Gaussian Mixture Model
*/
#ifndef CABANA_GMM_HPP
#define CABANA_GMM_HPP

#include <typeinfo>

/*!
  Possible parameters to describe a Gaussian

  In one dimension we use MuX and Cxx. In two dimension we assume a cylindrical
  velocity space and use MuPar, MuPer, Cparpar and Cperper. The off-diagonal
  elements Cparper and Cperpar are not currently in use. In three dimensions we
  use all MuX, MuY, MuZ, as well as a full (symmetric) 3-by-3 covariance matrix.
*/
enum GaussianFields {
    Weight,
    // Can we make only the relevant fields vissible based on <dims>?
    MuPar, MuPer,
    Cparpar, Cparper,
    Cperpar, Cperper,
    MuX, MuY, MuZ,
    Cxx, Cxy, Cxz,
    Cyx, Cyy, Cyz,
    Czx, Czy, Czz,
    n_gaussian_param
};


#include <impl/Cabana_GaussianMixtureModel.hpp>

namespace Cabana {

/*!
  Reconstruct a Gaussian Mixture Model in one dimension

  eps set the limit on relative change of the penalized log liklyhood function
  when we consider the reconstruction to be converged. seed is used when drawing
  the initial random Gaussians that the algorithm starts from. Reconstruction is
  done separately for each unique value in cell. This slice is assume to be a
  dense set of integers from 0 to gaussians.extent(0). The entries in weight can
  be all equal or can be different for particles with different statistical
  weight / macro factor, but it is assumed that the sum of all weights is equal
  to the number of particles. All particle slices have to have the same extent.
*/
template <typename GaussianType, typename CellSliceType, typename WeightSliceType, typename VelocitySliceType>
void reconstructGMM(GaussianType& gaussians, const double eps, const int seed, CellSliceType const& cell, WeightSliceType const& weight, VelocitySliceType const& vx) {
	GMMImpl<1>::implReconstructGMM(gaussians, eps, seed, cell, weight, vx, vx, vx);
}

/*!
  Reconstruct a Gaussian Mixture Model in two (cylindrical) dimensions

  eps set the limit on relative change of the penalized log liklyhood function
  when we consider the reconstruction to be converged. seed is used when drawing
  the initial random Gaussians that the algorithm starts from. Reconstruction is
  done separately for each unique value in cell. This slice is assume to be a
  dense set of integers from 0 to gaussians.extent(0). The entries in weight can
  be all equal or can be different for particles with different statistical
  weight / macro factor, but it is assumed that the sum of all weights is equal
  to the number of particles. All particle slices have to have the same extent.
*/
template <typename GaussianType, typename CellSliceType, typename WeightSliceType, typename VelocitySliceType>
void reconstructGMM(GaussianType& gaussians, const double eps, const int seed, CellSliceType const& cell, WeightSliceType const& weight, VelocitySliceType const& vpar, VelocitySliceType const& vper) {
	GMMImpl<2>::implReconstructGMM(gaussians, eps, seed, cell, weight, vpar, vper, vper);
}

/*!
  Reconstruct a Gaussian Mixture Model in three (cartesian) dimensions

  eps set the limit on relative change of the penalized log liklyhood function
  when we consider the reconstruction to be converged. seed is used when drawing
  the initial random Gaussians that the algorithm starts from. Reconstruction is
  done separately for each unique value in cell. This slice is assume to be a
  dense set of integers from 0 to gaussians.extent(0). The entries in weight can
  be all equal or can be different for particles with different statistical
  weight / macro factor, but it is assumed that the sum of all weights is equal
  to the number of particles. All particle slices have to have the same extent.
*/
template <typename GaussianType, typename CellSliceType, typename WeightSliceType, typename VelocitySliceType>
void reconstructGMM(GaussianType& gaussians, const double eps, const int seed, CellSliceType const& cell, WeightSliceType const& weight, VelocitySliceType const& vx, VelocitySliceType const& vy, VelocitySliceType const& vz) {
	GMMImpl<3>::implReconstructGMM(gaussians, eps, seed, cell, weight, vx, vy, vz);
}

} // end namespace Cabana

#endif // end CABANA_GMM_HPP
