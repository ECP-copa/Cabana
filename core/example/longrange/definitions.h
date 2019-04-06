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
#ifndef DEFS
#define DEFS
#include <Cabana_AoSoA.hpp>
#include <CabanaCore_config.hpp>

//Math definitions needed for solvers
constexpr double PI(3.141592653589793238462643);
constexpr double PI_SQRT(1.772453850905516);
constexpr double PI_SQ(PI*PI);// 9.869604401089359
constexpr double PI_DIV_SQ(1.0/PI_SQ);//0.101321183642338

constexpr float COULOMB_PREFACTOR(1.0);
constexpr float COULOMB_PREFACTOR_INV(1.0);


//#define COULOMB_PREFACTOR 1.0 / ( 4.0 * PI * 8.854187817e-12)
//#define COULOMB_PREFACTOR_INV ( 4.0 * PI * 8.854187817e-12)

// User field enumeration. These will be used to index into the data set. Must
// start at 0 and increment contiguously.

enum UserParticleFields
{
    Position,
    Velocity,
    Force,
    Charge,
    Potential,
    Index
};

// Designate the types that the particles will hold.
// Needs the SPACE_DIM
using ParticleDataTypes =
    Cabana::MemberTypes<double[SPACE_DIM],        // (0) x-position type
                        double[SPACE_DIM],        // (1) velocity type
		        double[SPACE_DIM],	  // (2) forces
                        double,		          // (3) Charge
                        double,                   // (4) potential
                        long                      // (5) global index
                        >;


// Declare the memory space.
#ifdef Cabana_ENABLE_Cuda
using MemorySpace = Cabana::CudaUVMSpace;
using ExecutionSpace = Kokkos::Cuda;
#elif Cabana_ENABLE_OpenMP
using MemorySpace = Cabana::HostSpace;
using ExecutionSpace = Kokkos::OpenMP;
#elif Cabana_ENABLE_Pthread
using MemorySpace = Cabana::HostSpace;
using ExecutionSpace = Kokkos::Threads;
#elif Cabana_ENABLE_Serial
using MemorySpace = Cabana::HostSpace;
using ExecutionSpace = Kokkos::Serial;
#endif

// Set the type for the particle AoSoA.
using ParticleList = Cabana::AoSoA<ParticleDataTypes,MemorySpace,INNER_ARRAY_SIZE>;
#endif
