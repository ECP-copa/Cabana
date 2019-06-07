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

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <iostream>

/* Define the inner vector length of SOA */
#include "veclen.h"

//---------------------------------------------------------------------------//
// parallel for example.
//---------------------------------------------------------------------------//

/* This is the coresponding struct_of_array defined by DataTypes (see below) */
struct local_data_struct_t {     
  double d0[VECLEN];     
  double d1[VECLEN];     
};

/* Declare functions that will be mixed with Fortran */
extern "C" {
  /* written in C++; called by Fortran */
  void c_kokkos_initlize( void ); 
  void c_kokkos_finalize( void ); 
  void parallelForExample(local_data_struct_t*, int, int);
  //void parallelForExample(double *);
  
  /* wirtten in Fortran; called by C++ */
  void initialization(local_data_struct_t*,int, int); 
#if USE_GPU == 1
   __host__ __device__ 
#endif
void kernel_1(local_data_struct_t*,int,int);
#if USE_GPU == 1
   __host__ __device__ 
#endif
void kernel_2(local_data_struct_t*,int,int,int,int);      
}


void parallelForExample(local_data_struct_t * struct_p, int num_soa, int num_p)
{
    /*
      In previous examples we have demonstrated using the Slice directly with
      programming models such as OpenMP and CUDA. Now we present a more
      portable strategy for threading over slices using Kokkos as well as
      Cabana variations of Kokkos concepts.

      Just as we demonstrated in previous examples using the Slice, both 1D
      and 2D indexing schemes are available to access data within the
      slice. Depending on the kernel to be used with the slice different types
      of threading algorithms and indexing schemes will give better
      performance.

      The purpose of this example is to demonstrate how to efficiently and
      portably use these different indexing schemes in threaded parallel code
      and provide examples of which patterns to use for a given kernel. In
      general, two types of computational kernels will be developed for
      operations on AoSoA data:

          1) Kernels that allow for sequential data access and therefore readily
          vectorize/coalesce using 2D indexing

          2) Kernels that do not easily vectorize/coalesce and therefore will
          possible benefit from the maximum parallelism afforded by 1D
          indexing.

      We demonstrate both cases in this example.
    */
 
    /*
      Declare the AoSoA parameters.
    */
    using DataTypes = Cabana::MemberTypes<double,double>;
#if USE_GPU == 1
    using MemorySpace = Kokkos::CudaSpace;
    //using MemorySpace = Cabana::CudaUVMSpace;
    using ExecutionSpace = Kokkos::Cuda;
#else
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
#endif
    /*
       Create the AoSoA.
    */
    using AoSoA_t = Cabana::AoSoA<DataTypes,MemorySpace,VECLEN,Kokkos::MemoryUnmanaged>;
    // local_data_struct_t *struct_p;
    // cudaMalloc((void **)&struct_p, num_soa*sizeof(local_data_struct_t));
    // Do convert
    auto ptr = reinterpret_cast<typename AoSoA_t::soa_type*>(struct_p);
    AoSoA_t aosoa( ptr, num_soa, num_p );
    auto slice_0 = Cabana::slice<0>(aosoa); 
    auto slice_1 = Cabana::slice<1>(aosoa); 
    /*
      Create slices and assign some data. One might consider using a parallel
      for loop in this case - especially when the code being written is for an
      arbitrary memory space.

      In C++, one would do
      auto slice_0 = aosoa->slice<0>();
      auto slice_1 = aosoa->slice<1>();

      For aosoa to be usable in Fortran, we cast it to local_data_struct_t,
      which is a C struct with exactly the same memory layout as the data in 
      aosoa->ptr().
     */
    

    /* Call the Fortran subroutine */
    initialization(struct_p,num_soa,num_p);

    /*
      KERNEL 1 - VECTORIZED/COALESCED

      Start with a kernel that we know will vectorize/coalesce. In this case
      we want 2D indexing. On a CPU, this 2D indexing can be used to write a
      vector-length inner loop that will vectorize while on a GPU a 2D thread
      layout to promote coalescing will be used.

      We write a Kokkos lambda function for the kernel. Note that the kernel
      has 2 indices, s and a, corresponding to the 2D struct and array index
      of the element on which the computation will be performed. This is
      intended to be used with the `access()` function of the slice.
     */
    auto vector_kernel =
      KOKKOS_LAMBDA( const int s, const int a )
      {
        kernel_1(struct_p,s,a);
        /*
	  What is written in a Fortran kernel is the floowing C++ operations:	  
        */	
        //slice_0.access(s,a) = slice_1.access(s,a); 
       };
    

     /*
       Now we define the execution policy for the 2D indexing scheme. A
       `Cabana::SimdPolicy` is used here. Think of this policy as an additional
       execution policy complementary to existing Kokkos policies.

       The template parameters of this type include the vector length
       indicating the size of the SIMD loops to be performed, and the Kokkos
       execution space over which perform the threading. This example uses
       OpenMP for the threading but CUDA, Serial, or any other Kokkos execution
       space can similarly be used as long as it is compatible with the memory
       space of the AoSoA.

       Note: The policy is defined over a 1D index range even though
       the intention is to perform 2D parallel loops. The policy will
       automatically perform the decomposition into 2D indices and pass them to
       the functor.

       Note: The policy also accepts work tag arguments if one is using a
       functor class design to dispatch work over different operators. The
       vector length must come first in the template parameters with the
       execution space and work tag to follow.
     */

     Cabana::SimdPolicy<VECLEN,ExecutionSpace> simd_policy( 0, num_p );

     /*
       Finally, perform the parallel loop. We have added a parallel for concept
       in Cabana for this that is complementary to existing parallel for
       implementations in Kokkos.

       Note: We fence after the kernel is completed for safety but this may not
       be needed depending on the memory/execution space being used. When the
       CUDA UVM memory space is used this fence is necessary to ensure
       completion of the kernel on the device before UVM data is accessed on
       the host. Not fencing in the case of using CUDA UVM will typically
       result in a bus error.
     */
     Cabana::simd_parallel_for( simd_policy, vector_kernel, "vector_op" );
     Kokkos::fence();

     /*
       KERNEL 2 - NO VECTORIZATION/COALESCING

       Next we use a kernel that will not vectorize due to random memory
       access. In this case a 2D loop may be OK on the GPU due to the fact that
       all work elements will still receive a thread. However, on the CPU, the
       inner vector-length loop will not vectorize and therefore be computed in
       serial on each thread. Using a 1D loop with 1D indexing means each work
       element will get a thread on the CPU, thus exposing more parallelism
       than the 2D loop for code that does not vectorize.
     */
     using PoolType = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
     using RandomType = Kokkos::Random_XorShift64<ExecutionSpace>;
     PoolType pool( 342343901 );
     auto rand_kernel =
         KOKKOS_LAMBDA( const int i )
         {
             auto gen = pool.get_state();
             auto rand_idx = Kokkos::rand<RandomType,int>::draw(gen,0,num_p);
     	    /*
     	      Here we need to explicitly provide the two indexes (s,a), where s is 
     	      the outer array index, and a is the inner array index.

     	      What is written in a Fortran kernel is the floowing C++ operations:
     	      slice_1(i) = slice_0( rand_idx );

     	    */	    
     	    slice_1(i) = slice_0( rand_idx );
	     //or one can call the fortran kernel
     	    // int s0 = i/VECLEN;
     	    // int a0 = i-s0*VECLEN;
     	    // int s1 = rand_idx/VECLEN;
     	    // int a1 = rand_idx-s1*VECLEN;
     	    // kernel_2(struct_p,s0,a0,s1,a1);
             pool.free_state( gen );
         };

     /*
       Because we are using 1D indexing in this case, we can directly use
       existing Kokkos execution policies and the parallel for.
      */
     Kokkos::RangePolicy<ExecutionSpace> linear_policy( 2, num_p - 2 );
     Kokkos::parallel_for( linear_policy, rand_kernel, "rand_op" );
     Kokkos::fence();

    /*
          Note: Other Kokkos parallel concepts such as reductions and scans can be
          used with Cabana slices and 1D indexing.
    */
    //delete aosoa;
}

void c_kokkos_initlize() {

  Kokkos::initialize();

}

void c_kokkos_finalize( void ) {

  Kokkos::finalize();

}
 
