#ifndef TDS_TDS_INCLUDED
#define TDS_TDS_INCLUDED

#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberTypes.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include "definitions.h"

class TDS
{
  public:
    TDS(int periodic = 0);
    ~TDS();

    //compute direct sum
    void compute(ParticleList& particles, double x_width, double y_width, double z_width);

    double get_energy();
  private:
    double total_energy;
    int _periodic_shells;
};

#endif
