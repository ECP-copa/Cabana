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

    //compute(particles, x_width, y_width, z_width)
       //particles is a ParticleList of particle information for all in the unit cell
          //(positions[NDIM],  velocities[NDIM], 1 charge, 1 potential, 1 index)
       //x_width is the length of the NaCl unit cell in the x-direction
       //y_width is the length of the NaCl unit cell in the y-direction
       //z_width is the length of the NaCl unit cell in the z-direction
    void compute(ParticleList&, double, double, double);

    double get_energy();
  private:
    double total_energy;
    int _periodic_shells;
};

#endif
