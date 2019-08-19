#include "example_definitions.h"
#include "particles.cpp"
#include "spme.cpp"
#include <iomanip>

int main( int argc, char **argv )
{
    // Initialize the kokkos runtime.
    Kokkos::initialize( argc, argv );

    // crystal size
    const int c_size = ( argc == 2 ) ? atoi( argv[1] ) : 32;
    // accuracy parameter for the tuning of Ewald
    const double accuracy = 1e-6;
    // width of unit cell (assume cube)
    const double width = (double)c_size / 2.0;;
    // Number of mesh points in each direction for SPME
    const int n_meshpoints = 4096; // 16*16*16;
    // Declare alpha and rmax, but just let the tuner select their values later
    // double alpha, r_max;
    // Number of particles, 3D
    const int n_particles = c_size * c_size * c_size;

    // Create an empty list of all the particles
    ParticleList *particles = new ParticleList( "particles", n_particles );
    // Create an empty list of all the mesh points
    ParticleList *mesh = new ParticleList( "mesh", n_meshpoints );

    std::cout << std::setprecision( 12 );

    // Initialize the particles and mesh
    // Currently particles are initialized as alternating charges
    // in uniform cubic grid pattern like NaCl
    initializeParticles( *particles, c_size );
    // Mesh is by default cubic and uniform
    initializeMesh( *mesh, width );

    // Create a Kokkos timer to measure performance
    Kokkos::Timer timer;

    // Create the solver and tune it for decent values of alpha and r_max
    TPME solver( accuracy, *particles, width, width, width );
    auto tune_time = timer.seconds();
    timer.reset();
    // Perform the computation of real and imag space energies
    double total_energy =
        solver.compute( *particles, *mesh, width, width, width );
    auto exec_time = timer.seconds();
    timer.reset();

    auto elapsed_time = tune_time + exec_time;

    // Print out the timings and accuracy
    std::cout << "Time for init+tuning parameters in SPME solver: "
              << ( tune_time ) << " s." << std::endl;
    std::cout << "Time for computation in SPME solver:        " << ( exec_time )
              << " s." << std::endl;
    std::cout << "Total time spent in SPME solver:            "
              << ( elapsed_time ) << " s." << std::endl;
    std::cout << "Total potential energy (known): "
              << MADELUNG_NACL * n_particles << std::endl;
    std::cout << "total potential energy (SPME): " << total_energy << std::endl;
    std::cout << "absolute error (energy): "
              << ( n_particles * MADELUNG_NACL ) - total_energy << std::endl;
    std::cout << "relative error (energy): "
              << 1.0 - ( n_particles * MADELUNG_NACL ) / total_energy
              << std::endl;

    // Clean up
    delete mesh;
    delete particles;
    Kokkos::fence();
    Kokkos::finalize();
    return 0;
}
