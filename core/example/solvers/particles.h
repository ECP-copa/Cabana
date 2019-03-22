#ifndef TDS_PARTICLES_INCLUDED
#define TDS_PARTICLES_INCLUDED

#include "definitions.h"
//#include <impl/Cabana_Index.hpp>

//initialize a NaCl crystal in a cubic arrangement with width^3 particles
void initializeParticles(ParticleList particles, int width);

//void printParticles( ParticleList, int target_idx = -1 );

//initialize a cubic mesh for the NaCl problem with width^3 mesh points
void initializeMesh(ParticleList mesh, int width);

//void printMesh(ParticleList, int target_idx = -1 );
#endif
