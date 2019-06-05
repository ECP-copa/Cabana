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

#ifndef TDS_EWALD_INCLUDED
#define TDS_EWALD_INCLUDED

#include <iostream>
#include "definitions.h"

class TEwald
{
  public:
    //constructor with accuracy
    TEwald(const double accuracy_threshold, ParticleList particles, const double x_width, const double y_width, const double z_width);
    
    //set base values for alpha, r_max, k_max
    TEwald(const double alpha, const double r_max, const double k_max);

    //compute Ewald Sum
    double compute(ParticleList& particles, const double x_width, const double y_width, const double z_width);

    // tune alpha, r_max, k_max to adhere to given accuracy
    void tune(const double accuracy_threshold, ParticleList particles, const double x_width, const double y_width, const double z_width);

    // setter functions for parameters
    void set_alpha(double);
    void set_r_max(double);
    void set_k_max(double);

    // getter functions
    double get_alpha() {return _alpha;}
    double get_r_max() {return _r_max;}
    double get_k_max() {return _k_max;}

  private:
    double _alpha;
    double _r_max;
    double _k_max;
    int _k_max_int[3];

    // dielectric constant (1.0 = vacuum)
    double _eps_r = 1.0;

    double* Uk_coeffs;
};

#endif
