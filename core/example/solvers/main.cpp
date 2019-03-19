#include "definitions.h"
#include "particles.h"
//#include "ewald.h"
#include "direct.h"
#include "pme.h"
//#include "mesh.h"

#include <iomanip>
#ifdef TDS_BENCHMARKING
#include <fstream>
#endif

#define CRYSTAL_SIZE 2

int main(int argc, char** argv)
{
  // Initialize the kokkos runtime.
  Kokkos::initialize( argc, argv );


#ifdef TDS_BENCHMARKING
  std::ofstream directfile;
  std::ofstream ewaldfile;
  std::ofstream pmefile;
#endif

  // crystal size
  int c_size = (argc >= 2)?atoi(argv[1]):CRYSTAL_SIZE;
  // number of periodic shells (and k_max for Ewald)
  int periodic_shells = (argc >= 3)?atoi(argv[2]):0;
  // accuracy parameter for the tuning of Ewald
  double accuracy = (argc >= 4)?atof(argv[3]):1e-6;
  // flag to set for the activation of solvers:
  // 0 -> none
  // 1 -> direct
  // 2 -> Ewald
  // 3 -> both
  int flag = (argc >= 5)?atoi(argv[4]):3;
  double l[3];
  l[0] = l[1] = l[2] = 0.5*(double)c_size;
  // alpha splitting parameter for Ewald
  double alpha = (argc >= 6)?atof(argv[5]):2.0;
  // cutoff radius for real-space part of Ewald
  double r_max = (argc >= 7)?atof(argv[6]):0.499*(double)c_size;

  int n_particles = c_size * c_size * c_size;
  int n_meshpoints = 16*16*16;//Arbitrary value for now...
 
#ifndef TDS_BENCHMARKING
#else
  directfile.open("direct.res", std::ofstream::out | std::ofstream::app);
  ewaldfile.open("ewald.res", std::ofstream::out | std::ofstream::app);
  pmefile.open("pme.res", std::ofstream::out | std::ofstream::app);
#endif

  ParticleList* particles = new ParticleList( n_particles ); 
  //ParticleList particles = *plist;

  ParticleList* mesh = new ParticleList( n_meshpoints );

  std::cout << std::setprecision(12);

/*  if (flag & 1)
  {
    initializeParticles( *particles, c_size );

    Kokkos::Timer timer;
    TDS solver(periodic_shells);
    //auto init_time = timer.seconds();
    //timer.reset();
#ifndef TDS_BENCHMARKING
    std::cout << "Direct summation solver: " << std::endl;
#endif
    solver.compute(*particles,l[0],l[1],l[2]);

    auto exec_time = timer.seconds();
    auto elapsed_time = init_time + exec_time;

#ifndef TDS_BENCHMARKING
    std::cout << "Time for initialization of direct summation solver: " << init_time << " s." << std::endl;
    std::cout << "Time for computation in direct summation solver:    " << exec_time << " s." << std::endl;
    std::cout << "Total time spent in direct summation solver:        " << elapsed_time << " s." << std::endl;
    std::cout << "total potential energy: " << solver.get_energy() << std::endl;
    std::cout << "absolute error (potential): " << MADELUNG_NACL-particles->view(Cabana::MemberTag<Potential>())(0) << std::endl;
    std::cout << "relative error (potential): " << 1.0 - MADELUNG_NACL/particles->view(Cabana::MemberTag<Potential>())(0) << std::endl;
    std::cout << "absolute error (energy): " << (n_particles * MADELUNG_NACL)-solver.get_energy() << std::endl;
    std::cout << "relative error (energy): " << 1.0 - (n_particles * MADELUNG_NACL)/solver.get_energy() << std::endl;
    std::cout << std::endl;
#else
    std::cout << "total potential energy (direct): " << solver.get_energy() << std::endl;
    std::cout << "absolute error (potential): " << MADELUNG_NACL-particles->view(Cabana::MemberTag<Potential>())(0) << std::endl;
    std::cout << "relative error (potential): " << 1.0 - MADELUNG_NACL/particles->view(Cabana::MemberTag<Potential>())(0) << std::endl;
    std::cout << "absolute error (energy): " << (n_particles * MADELUNG_NACL)-solver.get_energy() << std::endl;
    std::cout << "relative error (energy): " << 1.0 - (n_particles * MADELUNG_NACL)/solver.get_energy() << std::endl;
    directfile << elapsed_time << " ";
#endif
  }*/
  if (flag & 2)
  {
    int width = 1.0;//64*c_size;
    initializeParticles( *particles, c_size );
    initializeMesh( *mesh, width );  
    
    Kokkos::Timer timer;
    double kmax = (double)periodic_shells;

#ifndef TDS_BENCHMARKING
    std::cout << std::endl;
    std::cout << "Ewald summation solver:" << std::endl;
#endif
    std::cout << "starting parameters: " << kmax << " " << alpha << " " << r_max << std::endl;
    TPME solver(alpha,r_max,kmax);
    auto init_time = timer.seconds();
    timer.reset();
    //accuracy *= -n_particles * MADELUNG_NACL;
    if (argc < 6) solver.tune(accuracy,*particles,l[0],l[1],l[2]);
    std::cout << "req. acc: " << accuracy << std::endl; 
    auto tune_time = timer.seconds();
    timer.reset();
    solver.compute(*particles,*mesh,l[0],l[1],l[2]);
    std::cout << "Done" << std::endl;
    auto exec_time = timer.seconds();
    timer.reset();

    auto elapsed_time = init_time + tune_time + exec_time;
#ifndef TDS_BENCHMARKING
    std::cout << "Time for initialization in Ewald summation solver:     " << (init_time) << " s." << std::endl;
    std::cout << "Time for tuning parameters  in Ewald summation solver: " << (tune_time) << " s." << std::endl;
    std::cout << "Time for computation in Ewald summation solver:        " << (exec_time) << " s." << std::endl;
    std::cout << "Total time spent in Ewald summation solver:            " << (elapsed_time) << " s." << std::endl;
    std::cout << "total potential energy (code): " << solver.get_energy() << std::endl;
    std::cout << "Total potential energy (known): " << MADELUNG_NACL*n_particles << std::endl;
    //std::cout << "absolute error (potential): " << MADELUNG_NACL-particles->slice<Potential>()(0) << std::endl;
    //std::cout << "relative error (potential): " << 1.0 - MADELUNG_NACL/particles->slice<Potential>()(0) << std::endl;
    std::cout << "absolute error (energy): " << (n_particles * MADELUNG_NACL)-solver.get_energy() << std::endl;
    std::cout << "relative error (energy): " << 1.0 - (n_particles * MADELUNG_NACL)/solver.get_energy() << std::endl;
    //printParticles(*particles,0);
    std::cout << std::endl;
#else
    std::cout << "total potential energy (ewald): " << solver.get_energy() << std::endl;
    //std::cout << "absolute error (potential): " << MADELUNG_NACL-particles->slice<Potential>()(0) << std::endl;
    //std::cout << "relative error (potential): " << 1.0 - MADELUNG_NACL/particles->slice<Potential>()(0) << std::endl;
    std::cout << "absolute error (energy): " << (n_particles * MADELUNG_NACL)-solver.get_energy() << std::endl;
    std::cout << "relative error (energy): " << 1.0 - (n_particles * MADELUNG_NACL)/solver.get_energy() << std::endl;
    ewaldfile << elapsed_time << " ";
#endif
  }
#ifdef TDS_BENCHMARKING
  directfile.close();
  ewaldfile.close();
  pmefile.close();
#endif
  delete particles;
  delete mesh;
  Kokkos::fence();
  Kokkos::finalize();
  return 0;
}
