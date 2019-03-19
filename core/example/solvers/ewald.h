#ifndef TDS_EWALD_INCLUDED
#define TDS_EWALD_INCLUDED

#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberTypes.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include "definitions.h"

class TEwald
{
  public:
    //TEwald(accuracy_threshold, particles, x_width, y_width, z_width)
       //accuracy_threshold is a value used in tuning the Ewald parameters (alpha, r_max, and k_max)
       //particles is a ParticleList of particle information for all in the unit cell
          //(positions[NDIM],  velocities[NDIM], 1 charge, 1 potential, 1 index)
       //mesh is also a ParticleList, but for mesh points
          //same setup as particles
       //x_width is the length of the NaCl unit cell in the x-direction
       //y_width is the length of the NaCl unit cell in the y-direction
       //z_width is the length of the NaCl unit cell in the y-direction
    TEwald(double,ParticleList,double,double,double);
    
    //set base values for alpha, r_max, k_max
    //TEwald(alpha, r_max, k_max)
    TEwald(double, double, double);
    ~TEwald();

    //compute(particles, x_width, y_width, z_width)
       //particles is a ParticleList of particle information for all in the unit cell
          //(positions[NDIM],  velocities[NDIM], 1 charge, 1 potential, 1 index)
       //mesh is also a ParticleList, but for mesh points
          //same setup as particles
       //x_width is the length of the NaCl unit cell in the x-direction
       //y_width is the length of the NaCl unit cell in the y-direction
       //z_width is the length of the NaCl unit cell in the y-direction
    void compute(ParticleList&, double,double,double);

    // tune alpha, r_max, k_max to adhere to given accuracy
    void tune(double,ParticleList,double,double,double);

    // setter functions for parameters
    void set_alpha(double);
    void set_r_max(double);
    void set_k_max(double);

    // getter functions
    double get_alpha() {return _alpha;}
    double get_r_max() {return _r_max;}
    double get_k_max() {return _k_max;}

    double get_energy() {return total_energy;}
  private:
    double _alpha;
    double _r_max;
    double _k_max;
    double total_energy;
    int _k_max_int[3];

    // dielectric constant (1.0 = vacuum)
    double _eps_r = 1.0;

    double* Uk_coeffs;
};

#endif
