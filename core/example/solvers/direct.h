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
