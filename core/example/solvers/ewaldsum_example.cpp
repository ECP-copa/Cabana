#include "example_definitions.h"
#include "ewald.cpp"
#include "particles.cpp"

#include <iomanip>

int main(int argc, char** argv)
{
  // Initialize the kokkos runtime.
  Kokkos::initialize( argc, argv );

  // crystal size
  const int c_size = 2;
  // accuracy parameter for the tuning of Ewald
  const double accuracy = 1e-6;
  //width of unit cell (assume cube)
  const double width = 1.0;
  //Number of particles, 3D
  const int n_particles = c_size * c_size * c_size;

  //Create an empty list of all the particles
  ParticleList *particles = new ParticleList( n_particles );

  std::cout << std::setprecision(12);

  //Initialize the particles
  //Currently particles are initialized as alternating charges 
  //in uniform cubic grid pattern like NaCl
  initializeParticles( *particles, c_size );

  //Create a Kokkos timer to measure performance
  Kokkos::Timer timer;

  //Create the solver and tune it for decent values of alpha and r_max
  TEwald solver(accuracy,*particles,width,width,width);
  auto tune_time = timer.seconds();
  timer.reset();
  //Perform the computation of real and imag space energies
  solver.compute(*particles,width,width,width);
  auto exec_time = timer.seconds();
  timer.reset();

  auto elapsed_time = tune_time + exec_time;
  
  //Print out the timings and accuracy
  std::cout << "Time for initialization in Ewald Sum solver: " << (tune_time) << " s." << std::endl;
  std::cout << "Time for computation in Ewald Sum solver:        " << (exec_time) << " s." << std::endl;
  std::cout << "Total time spent in Ewald Sum solver:            " << (elapsed_time) << " s." << std::endl;
  std::cout << "Total potential energy (known): " << MADELUNG_NACL*n_particles << std::endl;
  std::cout << "total potential energy (Ewald Sum): " << solver.get_energy() << std::endl;
  std::cout << "absolute error (energy): " << (n_particles * MADELUNG_NACL)-solver.get_energy() << std::endl;
  std::cout << "relative error (energy): " << 1.0 - (n_particles * MADELUNG_NACL)/solver.get_energy() << std::endl;
  
  //Clean up
  //delete mesh;
  //delete particles;
  //Kokkos::fence();
  //Kokkos::Cuda::finalize();
  Kokkos::finalize();
  return 0;
}
