#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberTypes.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

#include "fcs.h"

//---------------------------------------------------------------------------//
// Define particle data.
//---------------------------------------------------------------------------//
// Spatial dimension.
const int space_dim = 3;

// Dimension of the NaCl crystal
const int default_crystal_size = 2;

// User field enumeration. These will be used to index into the data set. Must
// start at 0 and increment contiguously.
//
// NOTE: Users don't have to make this enum (or some other set of integral
// constants) but it is a nice way to provide meaning to the different data
// types and values assigned to the particles.
//
// NOTE: These enums are also ordered in the same way as the data in the
// template parameters below.
enum UserParticleFields
{
    PositionX = 0,
    PositionY,
    PositionZ,
    Velocity,
    Charge,
    Potential,
    Field,
    Index
};

// Designate the types that the particles will hold.
using ParticleDataTypes =
    Cabana::MemberTypes<float,                        // (0) x-position type
                        float,                        // (1) y-position type
                        float,                        // (2) z-position type
                        double[space_dim],            // (3) velocity type
                        double,			              // (4) charge
		                double,			              // (5) potential
		                double[space_dim],		      // (6) electric field values
                        int                           // (7) status type
                        >;

// Declare the memory space.
using MemorySpace = Cabana::HostSpace;

// Declare the length of the internal vectors
const int VectorLength = 8;

// Set the type for the particle AoSoA.
using ParticleList = Cabana::AoSoA<ParticleDataTypes,MemorySpace,VectorLength>;

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//
// Function to intitialize the particles.
void initializeParticles( ParticleList particles, int crystal_size )
{
    auto p_x = particles.slice<PositionX>();
    auto p_y = particles.slice<PositionY>();
    auto p_z = particles.slice<PositionZ>();
    auto v = particles.slice<Velocity>();
    auto q = particles.slice<Charge>();
    auto pot = particles.slice<Potential>();
    auto field = particles.slice<Field>();
    auto indx = particles.slice<Index>();

    for ( auto idx = 0; idx != particles.size(); ++idx )
    {
        // Calculate location of particle in crystal
        int idx_x = idx % crystal_size;
        int idx_y = (idx / crystal_size) % crystal_size;
        int idx_z = idx / (crystal_size * crystal_size);	
	
        // Initialize position.
        p_x(idx) = (double)idx_x * 1.0;
        p_y(idx) = (double)idx_y * 1.0;
        p_z(idx) = (double)idx_z * 1.0;

        // Initialize velocity.
        for ( int d = 0; d < space_dim; ++d )
            v( idx, d ) = 0.0;

        // Initialize field
        for ( int d = 0; d < space_dim; ++d )
            field( idx, d ) = 0.0;

        // Create alternating charge
        q(idx) = ((idx_x + idx_y + idx_z)%2)?1.0:-1.0;

        // Set potential
        pot(idx) = 0.0;

        // Set global particle index
        indx(idx) = idx+1;
    }
}

// Function to print out the data for every particle.
void printParticles( const ParticleList particles )
{

    // get slices for the corresponding particle data
    auto p_x = particles.slice<PositionX>();
    auto p_y = particles.slice<PositionY>();
    auto p_z = particles.slice<PositionZ>();
    auto v = particles.slice<Velocity>();
    auto q = particles.slice<Charge>();
    auto pot = particles.slice<Potential>();
    auto field = particles.slice<Field>();
    auto indx = particles.slice<Index>();

    for ( auto idx = 0; idx < particles.size(); ++idx )
    {
        std::cout << "Position: "
                  << p_x(idx) << " "
                  << p_y(idx) << " "
                  << p_z(idx) << std::endl;

        std::cout << "Velocity ";
        for ( int d = 0; d < space_dim; ++d )
            std::cout << v(idx, d) << " ";
        std::cout << std::endl;

        std::cout << "Charge " << q(idx) << std::endl;

        std::cout << "Potential " << pot(idx) << std::endl;

        std::cout << "Field ";
        for ( int d = 0; d < space_dim; ++d )
            std::cout << field(idx, d) << " ";
        std::cout << std::endl;

        std::cout << "Index " << indx(idx) << std::endl;

        std::cout << std::endl;
    }
}

// check of ScaFaCoS result
static bool check_result(FCSResult result, int comm_rank, bool force_abort = false) {
  if (result) {
    std::cout << "ERROR: Caught error on task " << comm_rank << "!" << std::endl;
    fcs_result_print_result(result);
    fcs_result_destroy(result);
    if (force_abort)
      MPI_Abort(MPI_COMM_WORLD, 1);
    else
      std::cout << "WARNING: Continuing after error!" << std::endl;
    return false;
  }
  return true;
}

// example main
void exampleMain(int num_particle, int crystal_size, std::string method, MPI_Comm mpi_comm, int rank)
{
    // Create the particle list.
    ParticleList particles( num_particle );

    // Initialize particles.
    initializeParticles( particles, crystal_size );

    // Print particles.
    if (num_particle <= 216)  printParticles( particles );

    // ScaFaCoS handle (contains common parameters, like box size, periodicity, etc.)
    FCS fcs = FCS_NULL;
    // ScaFaCoS result (contains information about function call, e.g. error messages when
    // the call failed)
    FCSResult result;

    // initialize the handle with the chosen method
    result = fcs_init(&fcs,method.c_str(),mpi_comm);
    if (!check_result(result, rank)) return;

    // prepare the arrays containing positions (pos), charges (q), field values (f) and potentials (pot)
    std::vector<double> pos;
    std::vector<double> q;
    std::vector<double> f(3*num_particle);
    std::vector<double> pot(num_particle);

    auto p_x = particles.slice<PositionX>();
    auto p_y = particles.slice<PositionY>();
    auto p_z = particles.slice<PositionZ>();
    auto qp = particles.slice<Charge>();

    for (int i = 0; i < num_particle; ++i)
    {
      // ScaFaCoS expects postions in a (x,y,z) AoS format
      pos.push_back(p_x(i));
      pos.push_back(p_y(i));
      pos.push_back(p_z(i));
      q.push_back(qp(i));
    }

    // parameter string to set the periodicity and disable the calculation of near field parts from the calling program
    std::string common_parameters = "periodicity,1,1,1,offset,0.0,0.0,0.0,near_field_flag,0";

    // set the box size, so that the crystal should be stable in a periodic system
    std::vector<double> box_a{(double)(crystal_size),0.0,0.0};
    std::vector<double> box_b{0.0,(double)(crystal_size),0.0};
    std::vector<double> box_c{0.0,0.0,(double)(crystal_size)};

    result = fcs_set_box_a(fcs, box_a.data());
    if (!check_result(result,rank)) return;
    
    result = fcs_set_box_b(fcs, box_b.data());
    if (!check_result(result,rank)) return;
    
    result = fcs_set_box_c(fcs, box_c.data());
    if (!check_result(result,rank)) return;

    // set the other common parameters (see above)
    result = fcs_set_parameters(fcs, common_parameters.c_str(), FCS_FALSE);
    if (!check_result(result,rank)) return;

    // set the total number of particles
    result = fcs_set_total_particles(fcs, num_particle);
    if (!check_result(result,rank)) return;

    // print out the chosen parameters
    fcs_print_parameters(fcs);

    // call the tuning routine in order to setup the solver for the system
    result = fcs_tune(fcs,num_particle,pos.data(),q.data());
    if (!check_result(result,rank)) return;

    // calculate electro-static interactions
    result = fcs_run(fcs,num_particle,pos.data(),q.data(),f.data(),pot.data());
    if (!check_result(result,rank)) return;

    auto field = particles.slice<Field>();
    auto poten = particles.slice<Potential>();

    // copy results from the call to the particle structures
    for (int i = 0; i < num_particle; ++i)
    {
      field(i,0) = f.at(3*i);
      field(i,1) = f.at(3*i+1);
      field(i,2) = f.at(3*i+2);
      poten(i) = pot.at(i);
    }

    // Print particles (to check if any calculation took place and check results)
    if (num_particle <= 216)  printParticles( particles );
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{

    MPI_Init(&argc, &argv);

    // Initialize the kokkos runtime.
    Kokkos::initialize( argc, argv );

    std::string method = "direct";

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Run the test.
    if (argc == 3)
    {
      // to change the system size at run time, call:
      // ./scafacos_example <crystal_dimension> <solver_name>
      // solver name is one of:
      // direct  	direct solver (does not work for periodic systems, uses 'halo-systems' (expensive!))
      // ewald  	ewald solver
      // fmm  	fast multipole methode
      // p3m  	particle - particle particle - mesh method
      // p2nfft  	FFT based solver
      //
      // crystal dimension should be divisible by two for an un-charged system
      int c_size = atoi(argv[1]);
      method = argv[2];
      exampleMain(c_size * c_size * c_size, c_size, method, comm, rank);
    }
    else
    {
      exampleMain(	default_crystal_size * default_crystal_size * default_crystal_size,
            default_crystal_size,
            method,
            comm,
            rank);
    }

    // Finalize.
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}

//---------------------------------------------------------------------------//
