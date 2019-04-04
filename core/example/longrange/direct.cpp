#include "definitions.h"
#include "direct.h"

TDS::TDS(int periodic)
{
  // number of periodic images (number of shells surrounding systems)
  _periodic_shells = periodic;
}

double TDS::compute(ParticleList& particles, double lx, double ly, double lz)
{
  // Create an execution policy over the entire AoSoA.
  //Kokkos::RangePolicy<ExecutionSpace>(0,n_max) range_policy( particles );

  // Create slices
  auto r = particles.slice<Position>();
  auto f = particles.slice<Force>();
  auto p = particles.slice<Potential>();
  auto q = particles.slice<Charge>();

  int n_max = particles.size();
  int periodic_shells = _periodic_shells;

  // functor to calculate potentials
  auto work_func = KOKKOS_LAMBDA( const int idx )
  {
    double d[SPACE_DIM];//3-D is supported only for now
    double d_;
    double shift[SPACE_DIM];
    double shift_l;
    p( idx ) = 0.0;//set potential to zero

    //For each particle, we'll find the potential by summing
    //the electrostatic interaction between it and every other 
    //particle in the cell, as well as every particle in the 
    //surrounding periodic spherical shells
    for (auto i = 0; i < n_max; ++i)
    {
      for (int kx = -periodic_shells; kx <= periodic_shells; ++kx)
      {
        //x-dist shift for particles in periodic cells
        shift[0] = (double)kx * lx; 
        for (int ky = -periodic_shells; ky <= periodic_shells; ++ky)
        {
          //y-dist shift for particles in periodic cells
          shift[1] = (double)ky * ly;
          //check if still in bounds of spherical periodic shells
          shift_l = sqrt(shift[0] * shift[0] + shift[1] * shift[1]);
          if (shift_l > (double)periodic_shells) continue;
          for (int kz = -periodic_shells; kz <= periodic_shells; ++kz)
          {
            //z-dist shift for particles in periodic cells
            shift[2] = (double)kz * lz;
            //check if still in bounds of spherical periodic shells
            shift_l = sqrt(shift[0] * shift[0] + shift[1] * shift[1] + shift[2] * shift[2]);
            if (shift_l > (double)periodic_shells) continue;
            // no self-interaction
            if (i == idx && kx == 0 && ky == 0 && kz == 0) continue;
            // set distance to zero
            d_ = 0.0;
            // compute distance
            for (auto j = 0; j < SPACE_DIM; ++j)
            {
              d[j] = r( idx, j ) - ( r( i, j ) + shift[j] );
              d_ += d[j]*d[j];
            }
            d_ = sqrt(d_);
            // compute potential
            p( idx ) += 0.5*q(i)/d_;
            // compute forces
            for (auto j = 0; j < SPACE_DIM; ++j)
            {
              f(idx, j) += 0.5 * COULOMB_PREFACTOR * q( idx ) * q( i ) / ( d_ * d_ ) * d[j];
            }
          }
        }
      }
    }
    p( idx ) *= COULOMB_PREFACTOR * q( idx );
  };

  Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>(0,n_max), work_func );

  //Compute total energy of particles
  double total_energy = 0.0;
  Kokkos::parallel_reduce( "Sum", Kokkos::RangePolicy<ExecutionSpace>(0,n_max), KOKKOS_LAMBDA(int idx, double& energy)
    {
      energy += p( idx );
    },
    total_energy);

  return total_energy;
}
