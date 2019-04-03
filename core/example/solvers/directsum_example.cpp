#include "example_definitions.h"
#include "particles.cpp"
#include "direct.cpp"
#include <iomanip>
int main(int argc, char** argv)
{
  // Initialize the kokkos runtime.
  Kokkos::initialize( argc, argv );

  // crystal size
  const int c_size = 2;
  // accuracy parameter for the tuning of Ewald
//  const double accuracy = 1e-6;
  //width of unit cell (assume cube)
  const double width = 1.0;
  //Number of mesh points in each direction for SPME
//  const int n_meshpoints = 4096;//16*16*16;
  //Number of particles, 3D
  const int n_particles = c_size * c_size * c_size;

  //Number of periodic shells
  const int periodic_shells = 3;
  
  //Create an empty list of all the particles
  ParticleList *particles = new ParticleList( n_particles );
  //Create an empty list of all the mesh points
//  ParticleList *mesh = new ParticleList( n_meshpoints );

  std::cout << std::setprecision(12);

  //Initialize the particles and mesh
  //Currently particles are initialized as alternating charges 
  //in uniform cubic grid pattern like NaCl
  initializeParticles( *particles, c_size );
  //Mesh is by default cubic and uniform
//  initializeMesh( *mesh, width );  

  //Create a Kokkos timer to measure performance
  Kokkos::Timer timer;

  //Create the solver and tune it for decent values of alpha and r_max
  TDS solver(periodic_shells);
  auto tune_time = timer.seconds();
  timer.reset();
  //Perform the computation of real and imag space energies
  solver.compute(*particles,width,width,width);
  auto exec_time = timer.seconds();
  timer.reset();

  auto elapsed_time = tune_time + exec_time;
  
  //Print out the timings and accuracy
  std::cout << "Time for initialization in Direct Sum solver: " << (tune_time) << " s." << std::endl;
  std::cout << "Time for computation in Direct Sum solver:        " << (exec_time) << " s." << std::endl;
  std::cout << "Total time spent in Direct Sum solver:            " << (elapsed_time) << " s." << std::endl;
  std::cout << "Total potential energy (known): " << MADELUNG_NACL*n_particles << std::endl;
  std::cout << "total potential energy (Direct Sum): " << solver.get_energy() << std::endl;
  std::cout << "absolute error (energy): " << (n_particles * MADELUNG_NACL)-solver.get_energy() << std::endl;
  std::cout << "relative error (energy): " << 1.0 - (n_particles * MADELUNG_NACL)/solver.get_energy() << std::endl;
  
  //Clean up
  //delete mesh;
  //delete particles;
  //Kokkos::fence();
  Kokkos::Cuda::finalize();
  return 0;
}
