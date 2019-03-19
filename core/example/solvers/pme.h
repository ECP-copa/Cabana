#ifndef TDS_EWALD_INCLUDED
#define TDS_EWALD_INCLUDED

#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberTypes.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include "definitions.h"

class TPME
{
  public:
    //TPME(accuracy_threshold, particles, mesh, x_width, y_width, z_width)
       //accuracy_threshold is a value used in tuning the Ewald parameters (alpha, r_max, and k_max)
       //particles is a ParticleList of particle information for all in the unit cell
          //(positions[NDIM],  velocities[NDIM], 1 charge, 1 potential, 1 index)
       //mesh is also a ParticleList, but for mesh points
          //same setup as particles
       //x_width is the length of the NaCl unit cell in the x-direction
       //y_width is the length of the NaCl unit cell in the y-direction
       //z_width is the length of the NaCl unit cell in the z-direction
    TPME(double,ParticleList,ParticleList,double,double,double);

    //TPME(alpha, r_max, k_max)
       //set base values for alpha, r_max, k_max
    TPME(double, double, double);

    ~TPME();
    
    //oneDspline(mesh_dist)
       //Computes a cubic cardinal B-spline in one dimension
       //The cubic spline interpolates to find the fraction of the value to spread to
       //   mesh points up to 2 mesh spacings away
       //mesh_dist in this usage is the distance from the charged particle 
       //   to the mesh point, measure in mesh spacings
    double oneDspline(double);

    //oneDeuler(k, meshwidth)
    //Computes the Euler exponential spline in one-dimension 
    //k is the index of the mesh point position in one dimension, and varies from 0 to meshwidth-1
    //meshwidth is the *number of mesh points* in each direction (assuming cubic for now)
    double oneDeuler(int, int);

    //compute(particles, mesh, x_width, y_width, z_width)
       //particles is a ParticleList of particle information for all in the unit cell
          //(positions[NDIM],  velocities[NDIM], 1 charge, 1 potential, 1 index)
       //mesh is also a ParticleList, but for mesh points
          //same setup as particles
       //x_width is the length of the NaCl unit cell in the x-direction
       //y_width is the length of the NaCl unit cell in the y-direction
       //z_width is the length of the NaCl unit cell in the z-direction
    void compute(ParticleList&,ParticleList&,double,double,double);

    //tune(accuracy_threshold, particles, x_width, y_width, z_width) 
    //tune alpha, r_max, k_max to adhere to given accuracy
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

    //double* Uk_coeffs;
};

#endif
