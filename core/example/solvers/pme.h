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
    // constructor with accuracy
    TPME(double,ParticleList,ParticleList,double,double,double);
    // set base values for alpha, r_max, k_max
    TPME(double, double, double);
    ~TPME();

    double oneDspline(double);
    double oneDeuler(int, int);


    void compute(ParticleList&,ParticleList&,double,double,double);

    // tune alpha, r_max, k_max to adhere to given accuracy
    void tune(double,ParticleList,ParticleList,double,double,double);

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
