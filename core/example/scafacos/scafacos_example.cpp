/****************************************************************************
 * Copyright (c) 2018-2019 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Cabana_Types.hpp>
#include <Cabana_AoSoA.hpp>
#include <Cabana_MemberTypes.hpp>

#include <Kokkos_Core.hpp>

// ScaFaCoS library
#include <fcs.h>

#include <cstdlib>
#include <iostream>
#include <string>

//---------------------------------------------------------------------------//
// Define particle data.
//---------------------------------------------------------------------------//
// Spatial dimension.
const int space_dim = 3;

// Dimension of the NaCl crystal
const int default_crystal_size = 2;

// Gap between the charges of the NaCl crystal
const double default_gap_space = 1.0;

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

typedef struct
{
    // MPI rank
    int rank;
    // number of MPI processes
    int n_proc;
    // number of local particles
    int n_local_particles;
    // number of particles in each cartesian direction
    // within the crystal
    int loc_crystal[3];
    // offset within the crystal
    int off_crystal[3];
    // cartesian dimensions of communicator
    int dim[3];
    // cartesian location of the local rank
    int loc[3];
    // communicator
    MPI_Comm comm;
} parallel_info;

// Designate the types that the particles will hold.
using ParticleDataTypes =
    Cabana::MemberTypes<double,                       // (0) x-position type
                        double,                       // (1) y-position type
                        double,                       // (2) z-position type
                        double[space_dim],            // (3) velocity type
                        double,                       // (4) charge
                        double,                       // (5) potential
                        double[space_dim],            // (6) electric field values
                        int                           // (7) global index
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
void initializeParticles( ParticleList& particles,
                          double gap_space,
                          parallel_info& info )
{
    auto p_x = particles.slice<PositionX>();
    auto p_y = particles.slice<PositionY>();
    auto p_z = particles.slice<PositionZ>();
    auto v = particles.slice<Velocity>();
    auto q = particles.slice<Charge>();
    auto pot = particles.slice<Potential>();
    auto field = particles.slice<Field>();
    auto indx = particles.slice<Index>();

    int offset;

    // compute offset with MPI_Exscan
    MPI_Exscan(&(info.n_local_particles), &offset, 1, MPI_INT, MPI_SUM, info.comm);
    // MPI standard: receive buffer on rank 0 is undefined
    if (info.rank == 0) offset = 0;

    for ( int idx = 0; idx < info.n_local_particles; ++idx )
    {
        // Calculate location of particle in crystal
        int idx_x = idx % info.loc_crystal[0] + info.off_crystal[0];
        int idx_y = (idx / info.loc_crystal[0]) % info.loc_crystal[1] + info.off_crystal[1];
        int idx_z = idx / (info.loc_crystal[0] * info.loc_crystal[1]) + info.off_crystal[2];

        // Initialize position.
        p_x(idx) = idx_x * gap_space;
        p_y(idx) = idx_y * gap_space;
        p_z(idx) = idx_z * gap_space;

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
        indx(idx) = offset + idx;
    }
}

// Function to print out the data for every particle.
void printParticles( const ParticleList particles, parallel_info& info )
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

    std::cout << "Rank: "
              << info.rank
              << '\n'
              << '\n';

    for ( size_t idx = 0; idx < particles.size(); ++idx )
    {
        std::cout << "Position: "
                  << p_x(idx) << " "
                  << p_y(idx) << " "
                  << p_z(idx) << '\n';

        std::cout << "Velocity ";
        for ( int d = 0; d < space_dim; ++d )
            std::cout << v(idx, d) << " ";
        std::cout << '\n';

        std::cout << "Charge " << q(idx) << '\n';

        std::cout << "Potential " << pot(idx) << '\n';

        std::cout << "Field ";
        for ( int d = 0; d < space_dim; ++d )
            std::cout << field(idx, d) << " ";
        std::cout << '\n';

        std::cout << "Index " << indx(idx) << '\n';

        std::cout << '\n';
    }
}

// check of ScaFaCoS result
static bool check_result(FCSResult result, int comm_rank, bool force_abort = false) {
  if (result) {
    std::cout << "ERROR: Caught error on task " << comm_rank << "!" << '\n';
    fcs_result_print_result(result);
    fcs_result_destroy(result);
    if (force_abort)
      MPI_Abort(MPI_COMM_WORLD, 1);
    else
      std::cout << "WARNING: Continuing after error!" << '\n';
    return false;
  }
  return true;
}

// example main
void exampleMain(int num_particle,
                 int crystal_size,
                 const std::string& method,
                 parallel_info& info,
                 double gap_space)
{


    // Create the particle list.
    ParticleList particles( info.n_local_particles );

    // Initialize particles.
    initializeParticles( particles,
                         gap_space,
                         info );

    // Print particles.
    if (num_particle <= 216)
    {
        for (int i = 0; i < info.n_proc; ++i)
        {
            if (info.rank == i)
                printParticles( particles, info );
            MPI_Barrier(info.comm);
        }
    }
    // ScaFaCoS handle (contains common parameters, like box size, periodicity, etc.)
    FCS fcs = FCS_NULL;
    // ScaFaCoS result (contains information about function call, e.g. error messages when
    // the call failed)
    FCSResult result;

    // initialize the handle with the chosen method
    result = fcs_init(&fcs,method.c_str(),info.comm);
    if (!check_result(result, info.rank)) return;

    // prepare the arrays containing positions (pos), charges (q), field values (f) and potentials (pot)
    std::vector<double> pos;
    std::vector<double> q;
    std::vector<double> f(3*info.n_local_particles);
    std::vector<double> pot(info.n_local_particles);

    auto p_x = particles.slice<PositionX>();
    auto p_y = particles.slice<PositionY>();
    auto p_z = particles.slice<PositionZ>();
    auto qp = particles.slice<Charge>();

    for (int i = 0; i < info.n_local_particles; ++i)
    {
      // ScaFaCoS expects postions in a (x,y,z) AoS format
      pos.push_back(p_x(i));
      pos.push_back(p_y(i));
      pos.push_back(p_z(i));
      q.push_back(qp(i));
    }

    // parameter string to set the periodicity and disable the calculation of near field parts from the calling program
    std::string common_parameters = "periodicity,1,1,1,offset,0.0,0.0,0.0,near_field_flag,1";

    // set the box size, so that the crystal should be stable in a periodic system
    std::vector<double> box_a{(double)(crystal_size) * gap_space,0.0,0.0};
    std::vector<double> box_b{0.0,(double)(crystal_size) * gap_space,0.0};
    std::vector<double> box_c{0.0,0.0,(double)(crystal_size) * gap_space};

    result = fcs_set_box_a(fcs, box_a.data());
    if (!check_result(result,info.rank)) return;

    result = fcs_set_box_b(fcs, box_b.data());
    if (!check_result(result,info.rank)) return;

    result = fcs_set_box_c(fcs, box_c.data());
    if (!check_result(result,info.rank)) return;

    // set the other common parameters (see above)
    result = fcs_set_parameters(fcs, common_parameters.c_str(), FCS_FALSE);
    if (!check_result(result,info.rank)) return;

    // set the total number of particles
    result = fcs_set_total_particles(fcs, num_particle);
    if (!check_result(result,info.rank)) return;

    // print out the chosen parameters
    if (info.rank == 0) fcs_print_parameters(fcs);

    // call the tuning routine in order to setup the solver for the system
    result = fcs_tune(fcs,info.n_local_particles,pos.data(),q.data());
    if (!check_result(result,info.rank)) return;

    // calculate electro-static interactions
    result = fcs_run(fcs,info.n_local_particles,pos.data(),q.data(),f.data(),pot.data());
    if (!check_result(result,info.rank)) return;

    auto field = particles.slice<Field>();
    auto poten = particles.slice<Potential>();

    double check[4];
    for (int i = 0; i < 4; ++i)
        check[i] = 0.0;

    // copy results from the call to the particle structures
    for (int i = 0; i < info.n_local_particles; ++i)
    {
      for (int d = 0; d < space_dim; ++d)
      {
        field(i,d) = f.at(3*i+d);
        check[d] += f.at(3*i+d);
      }
      poten(i) = pot.at(i);
      check[3] += pot.at(i);
    }

    double total_check[4];
    // calculate total field and potential values
    MPI_Reduce(check,total_check,4,MPI_DOUBLE,MPI_SUM,0,info.comm);

    // Print particles (to check if any calculation took place and check results)
    if (num_particle <= 216)
    {
        for (int i = 0; i < info.n_proc; ++i)
        {
            if (info.rank == i)
                printParticles( particles, info );
            MPI_Barrier(info.comm);
        }
    }

    if (info.rank == 0)
        std::cout << "total potential: "
                  << total_check[3] << " "
                  << "sum of all field values: "
                  << total_check[0] << " "
                  << total_check[1] << " "
                  << total_check[2] << '\n';

}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{

    MPI_Init(&argc, &argv);

    // Initialize the kokkos runtime.
    Kokkos::initialize( argc, argv );

    // set default method for ScaFaCoS
    std::string method = "fmm";

    // setup MPI info
    parallel_info info;
    info.comm = MPI_COMM_WORLD;

    // Get MPI rank and number of ranks
    MPI_Comm_rank(info.comm, &info.rank);
    MPI_Comm_size(info.comm, &info.n_proc);

    // compute cartesian domain decomposition for the system
    // based on the number of processes
    for (int i = 0; i < space_dim; ++i)
        info.dim[i] = 0;

    MPI_Dims_create(info.n_proc,space_dim,info.dim);

    // distribute processes in x-y-z order
    info.loc[0] = info.rank % info.dim[0];
    info.loc[1] = (info.rank / info.dim[0]) % info.dim[1];
    info.loc[2] = info.rank / (info.dim[0] * info.dim[1]);

    // Run the test.
    if (argc == 4)
    {
      // to change the system size at run time, call:
      // ./ScafacosExample <crystal_dimension> <gap_space> <solver_name>
      // solver name is one of:
      // direct  direct solver (does not work for periodic systems, uses 'halo-systems' (expensive!))
      // ewald   ewald solver
      // fmm     fast multipole methode
      // p3m     particle - particle particle - mesh method
      // p2nfft  FFT based solver
      //
      // crystal dimension should be divisible by two for an un-charged system
      int c_size = atoi(argv[1]);
      double g_space = atof(argv[2]);
      method = argv[3];

      // compute the local partition of the crystal stored on local process,
      // number of particles on local domain and offset of local crystal
      // partition within the crystal
      info.n_local_particles = 1;
      for (int i = 0; i < space_dim; ++i)
      {
        info.loc_crystal[i] = c_size / info.dim[i];
        info.loc_crystal[i] += (info.loc[i] < ( c_size % info.dim[i] ))?1:0;
        info.n_local_particles *= info.loc_crystal[i];
        info.off_crystal[i] = (info.loc[i] < (c_size % info.dim[i]))
                              ? (info.loc[i] * info.loc_crystal[i])
                              : ( (c_size % info.dim[i]) * (info.loc_crystal[i]+1)
                                + (info.loc[i] - (c_size % info.dim[i])) * info.loc_crystal[i] );
      }

      exampleMain(
              c_size * c_size * c_size,
              c_size,
              method,
              info,
              g_space);
    }
    else
    {
      // compute the local partition of the crystal stored on local process,
      // number of particles on local domain and offset of local crystal
      // partition within the crystal
      info.n_local_particles = 1;
      for (int i = 0; i < space_dim; ++i)
      {
        info.loc_crystal[i] = default_crystal_size / info.dim[i];
        info.loc_crystal[i] += (info.loc[i] < ( default_crystal_size % info.dim[i] ))?1:0;
        info.n_local_particles *= info.loc_crystal[i];
        info.off_crystal[i] = (info.loc[i] <= (default_crystal_size % info.dim[i]))
                              ? (info.loc[i] * info.loc_crystal[i])
                              : ( (default_crystal_size % info.dim[i]) * (info.loc_crystal[i]+1)
                                + (info.loc[i] - (default_crystal_size % info.dim[i])) * info.loc_crystal[i] );
      }

      exampleMain(
            default_crystal_size * default_crystal_size * default_crystal_size,
            default_crystal_size,
            method,
            info,
            default_gap_space);
    }

    // Finalize.
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}

//---------------------------------------------------------------------------//
