#include "definitions.h"
#include "particles.h"

// function to initialize particles as a NaCl crystal of length crystal_size
void initializeParticles(ParticleList particles, int crystal_size)
{
  auto init_parts = KOKKOS_LAMBDA(const int idx)
  {
    // Calculate location of particle in crystal
    int idx_x = idx % crystal_size;
    int idx_y = (idx / crystal_size) % crystal_size;
    int idx_z = idx / (crystal_size * crystal_size);	

    // Initialize position.
    particles.slice<Position>()( idx, 0 ) =  ((double)idx_x * 0.5);
    particles.slice<Position>()( idx, 1 ) =  ((double)idx_y * 0.5);
    particles.slice<Position>()( idx, 2 ) =  ((double)idx_z * 0.5);

    // Initialize velocity.
    for ( int d = 0; d < SPACE_DIM; ++d )
        particles.slice<Velocity>()( idx, d ) = 0.0;

    // Initialize field
    for ( int d = 0; d < SPACE_DIM; ++d )
        particles.slice<Force>()( idx, d ) = 0.0;

    // Create alternating charge
    particles.slice<Charge>()(idx) = (((idx_x + idx_y + idx_z)%2)?1.0:-1.0)*COULOMB_PREFACTOR_INV;

    // Set potential
    particles.slice<Potential>()(idx) = 0.0;

    // Set global particle index
    particles.slice<Index>()( idx ) = idx+1l;
  };
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0,particles.size()),init_parts);
}


// function to initialize uniform cubic mesh
void initializeMesh(ParticleList mesh, int width)
{
  int ptsx = std::round(std::pow(mesh.size(),1.0/3.0));//number of points in each dimension
  auto init_mesh = KOKKOS_LAMBDA( const int idx ) 
  {
    // Calculate location of particle in crystal
    int idx_x = idx % ptsx;
    int idx_y = (idx / ptsx) % ptsx;
    int idx_z = idx / (ptsx * ptsx);	

    // Initialize position.
    mesh.slice<Position>()( idx, 0 ) = ((double)idx_x * width / (ptsx));
    mesh.slice<Position>()( idx, 1 ) = ((double)idx_y * width / (ptsx));
    mesh.slice<Position>()( idx, 2 ) = ((double)idx_z * width / (ptsx));

    // Initialize velocity.
    for ( int d = 0; d < SPACE_DIM; ++d )
        mesh.slice<Velocity>()( idx, d ) = 0.0;

    // Initialize field
    for ( int d = 0; d < SPACE_DIM; ++d )
          mesh.slice<Force>()( idx, d ) = 0.0;

    // Create charge = 0
    mesh.slice<Charge>()(idx) = 0.0;//(((idx_x + idx_y + idx_z)%2)?1.0:-1.0)*COULOMB_PREFACTOR_INV;

    // Set potential
    mesh.slice<Potential>()(idx) = 0.0;

    // Set global particle index
    mesh.slice<Index>()( idx ) = idx+1l;
  };
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0,mesh.size()),init_mesh);
}

