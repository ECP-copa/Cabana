#ifndef TDS_PARTICLES_INCLUDED
#define TDS_PARTICLES_INCLUDED

#include "definitions.h"
#include <impl/Cabana_Index.hpp>

// initialize a NaCl crystal of given edge length
void initializeParticles( ParticleList, int );
//void printParticles( ParticleList, int target_idx = -1 );
void initializeMesh( ParticleList, int );
//void printMesh(ParticleList, int target_idx = -1 );
#endif
