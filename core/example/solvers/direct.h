#ifndef TDS_TDS_INCLUDED
#define TDS_TDS_INCLUDED

#include <iostream>
#include "definitions.h"

class TDS
{
  public:
    TDS(int periodic = 0);

    //compute direct sum
    void compute(ParticleList& particles, double x_width, double y_width, double z_width);

    double get_energy();
  private:
    double total_energy;
    int _periodic_shells;
};

#endif
