#include <Cabana_Parallel.hpp>
#include <Cabana_ExecutionPolicy.hpp>
#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>

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


// Wrappers for C++ routines that will need to be called by the Fortran code.
extern "C" void cabana_initialize( void );
extern "C" void cabana_finalize( void );
extern "C" void kokkos_fence( void );

void cabana_initialize() {
  Cabana::initialize();
}

void cabana_finalize( void ) {
  Cabana::finalize();
}

void kokkos_fence() {
  Kokkos::fence();
}
