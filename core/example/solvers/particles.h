#ifndef TDS_PARTICLES_INCLUDED
#define TDS_PARTICLES_INCLUDED

#include "definitions.h"
#include <impl/Cabana_Index.hpp>

//initialize a NaCl crystal of given edge length
//initializeParticles(particles, width)
  //width is the number of particles in each direction, for a total of width^3 particles
void initializeParticles( ParticleList, int );

//void printParticles( ParticleList, int target_idx = -1 );

//initialize a cubic mesh for the NaCl problem
//initializeMesh(mesh, width)
   //width is the number of mesh points in each direction
void initializeMesh( ParticleList, int );

//void printMesh(ParticleList, int target_idx = -1 );
#endif
