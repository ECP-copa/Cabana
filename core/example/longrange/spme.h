/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "definitions.h"
#include <iostream>

class TPME
{
  public:
    // constructor with accuracy
    TPME( const double accuracy_threshold, ParticleList particles,
          const double x_width, const double y_width, const double z_width );

    // set base values for alpha, r_max
    TPME( const double alpha, const double r_max );

    // compute 1D cubic cardinal B-spline value given distance from point in
    // mesh spacings (mesh_dist)
    KOKKOS_INLINE_FUNCTION
    double oneDspline(double mesh_dist);
    
    //compute deriv of 1D cubic cardinal B-spline value given distance from point in mesh spacings (mesh_dist)
    KOKKOS_INLINE_FUNCTION
    double oneDsplinederiv(double mesh_dist);

    // computes Euler exponential spline in 1D
    KOKKOS_INLINE_FUNCTION
    double oneDeuler( int k, int meshwidth );

    // short and long range energy computation
    double compute( ParticleList &particles, ParticleList &mesh,
                    const double x_width, const double y_width,
                    const double z_width );

    // tune alpha, r_max, k_max to adhere to given accuracy
    void tune( double accuracy_threshold, ParticleList particles,
               const double x_width, const double y_width,
               const double z_width );

    // setter functions for parameters
    void set_alpha( double alpha );
    void set_r_max( double r_max );
    void set_k_max( double k_max );

    // getter functions
    double get_alpha() { return _alpha; }
    double get_r_max() { return _r_max; }
    double get_k_max() { return _k_max; }

  private:
    double _alpha;
    double _r_max;
    double _k_max;

    int _k_max_int[3];

    // dielectric constant (1.0 = vacuum)
    double _eps_r = 1.0;
};
