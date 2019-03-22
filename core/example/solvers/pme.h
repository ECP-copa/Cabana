#ifndef TDS_EWALD_INCLUDED
#define TDS_EWALD_INCLUDED

//#include <Cabana_AoSoA.hpp>
//#include <Cabana_MemberTypes.hpp>
//#include <Kokkos_Core.hpp>

#include <iostream>
#include "definitions.h"

class TPME
{
  public:
    //constructor with accuracy
    TPME(double accuracy_threshold, ParticleList particles, double x_width, double y_width, double z_width);
    
    //set base values for alpha, r_max, k_max
    TPME(double alpha, double r_max, double k_max);

    //compute 1D cubic cardinal B-spline value given distance from point in mesh spacings (mesh_dist)
    double oneDspline(double mesh_dist);

    //computes Euler exponential spline in 1D
    double oneDeuler(int k, int meshwidth);

    //short and long range energy computation
    void compute(ParticleList& particles, ParticleList& mesh, double x_width, double y_width, double z_width);

    //tune alpha, r_max, k_max to adhere to given accuracy
    void tune(double accuracy_threshold, ParticleList particles, double x_width, double y_width, double z_width); 

    // setter functions for parameters
    void set_alpha(double alpha);
    void set_r_max(double r_max);
    void set_k_max(double k_max);

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
