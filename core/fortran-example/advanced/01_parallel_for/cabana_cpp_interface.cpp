#include <Cabana_Parallel.hpp>
#include <Cabana_ExecutionPolicy.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

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

#include <chrono>
#include <cstdlib>
#include <iostream>

#include "../Fortran_features/cabana_fortran_cpp_defs.h"

// Whether to use the GPU version
#ifndef USE_GPU
#define USE_GPU 0
#endif

// If using the CPU version, whether to use OpenMP
#ifndef USE_OMP
#define USE_OMP 1
#endif

// Declare the memory and execution spaces.
#if USE_GPU == 1
using MemorySpace = Cabana::CudaUVMSpace;
using ExecutionSpace = Kokkos::Cuda;
#else
using MemorySpace = Cabana::HostSpace;
#if USE_OMP == 1
using ExecutionSpace = Kokkos::OpenMP;
#else
using ExecutionSpace = Kokkos::Serial;
#endif
#endif

/* Most particle routines can be written as a loop over particles.
   In the GPU case, launch a parallel_for over particles
   In the CPU case, launch a parallel_for over vectors
   The vector loop is currently inclusive of the ends, so if you ask to operate over
   particles 15-33 and your vector length is 16, you will operate over particle 1-48.
*/

// Define the function to be called by Fortran main
extern "C" int parallel_for_example( int sp, int ep );

// Define the kernel that will be called inside the parallel_for
extern "C" CABANA_FUNCTION void parallel_for_example_f( int );

int parallel_for_example( int start_pt, int end_pt)
{
  auto local_lambda = KOKKOS_LAMBDA( const int idx )
  {
     parallel_for_example_f(idx);
  };
  Kokkos::RangePolicy<ExecutionSpace> range_policy_vec( start_pt-1, end_pt );
  Kokkos::parallel_for( range_policy_vec, local_lambda, "example_op" );
  return 0;
}



