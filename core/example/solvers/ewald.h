#ifndef TDS_EWALD_INCLUDED
#define TDS_EWALD_INCLUDED

#include <iostream>
#include "definitions.h"

class TEwald
{
  public:
    //constructor with accuracy
    TEwald(double accuracy_threshold, ParticleList particles, double x_width, double y_width, double z_width);
    
    //set base values for alpha, r_max, k_max
    TEwald(double alpha, double r_max, double k_max);

    //compute Ewald Sum
    void compute(ParticleList& particles, double x_width, double y_width, double z_width);

    // tune alpha, r_max, k_max to adhere to given accuracy
    void tune(double accuracy_threshold, ParticleList particles, double x_width, double y_width, double z_width);

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
