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

#ifndef TDS_TDS_INCLUDED
#define TDS_TDS_INCLUDED

#include <iostream>
#include "definitions.h"

class TDS
{
  public:
    TDS(int periodic = 0);

    //compute direct sum
    double compute(ParticleList& particles, double x_width, double y_width, double z_width);

  private:
    int _periodic_shells;
};

#endif
