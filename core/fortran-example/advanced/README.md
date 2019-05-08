This directory contains examples of Cabana for Fortran that can be readily adopted for users wanting to build a PIC code or incorporate Cabana kernels into an existing Fortran PIC code.

Example 1: parallel_for (Example1_parallel_for.f90)  
Example 2: Push (Example2_Push.f90)  
Example 3: Scatter (Example3_Scatter.f90)


cabana_cpp_interface.cpp  
This file contains all of the Kokkos/Cabana code, which is in C++. It contains an macro enabling usage of the Kokkos parallel_for both generally and specifically to a particle AoSoA. Users must specify the particle properties here.

cabana_fortran_macros.h  
This include file contains macros used in the Fortran code:  
  ATTR_DEVICE: Adds "attributes(device)" in the GPU version to specify execution space. Other Cuda options available: ATTR_HOST, ATTR_GLOBAL, ATTR_SHARED, ATTR_CONSTANT, ATTR_PINNED  

  COPY_TO_DEVICE(err,icount,d_array,h_array): Copies from host to device if GPU version, otherwise simple copy  
  COPY_TO_HOST(err,icount,h_array,d_array): Copies from device to host if GPU version, otherwise simple copy  
  CABANA_COUNT_KIND: Variable type needed for icount in the above macros.  

  FCABANA_DEVICE: Specifies that a variable or array is on the device if GPU version  

  CABANA_DEF_THREAD_NUM: Declares OMP_GET_THREAD_NUM in OpenMP version  
  CABANA_GET_THREAD_NUM(x): Gets OMP thread number  
  CABANA_ADD(x,y): Normal addition if on CPU; AtomicAdd if on GPU.  
  CABANA_DEF_MAX_THREADS: Declates OMP_GET_MAX_THREADS in OpenMP version  
  CABANA_GET_MAX_THREADS(x): Gets OMP number of threads

  PARTICLE_OP_INTERFACE(C_FUNC): Interfaces to the corresponding C++ routine in cabana_cpp_interface.cpp or particle_ops.h  
  MISC_OP_INTERFACE(C_FUNC): Interfaces to the corresponding C++ routine in misc_ops.h

cabana_fortran_wrappers.F90  
Contains Fortran wrappers for Cabana/Kokkos C++ routines:  
  cabana_initialize: Cabana::initialize  
  cabana_finalize: Cabana::finalize  
  kokkos_fence: Kokkos::fence

ptl_module.F90  
Users must specify the particle properties here and they must correspond to those in cabana_cpp_interface.cpp. The particle initialization and some utilities are also here.

particle_ops.h  
This is a list of particle operations. In these operations, the fortran kernel receives the particle data, number of structures in the SoA, and the parallel_for index as arguments. Users can add their own operations to the list and easily create custom kernels by analogy with existing ones.

grid_module.F90  
An example grid module that a user might have.

update_phase_module.F90  
The user-specified physics of the push.

push_module.F90  
The push kernel. Currently does nothing interesting besides "subcycling" - performing the push multiple times per time step. Could be expanded to include multiple algorithms to choose from.

scatter_module.F90  
The scatter kernel. Currently decides which cell to write to arbitrarily, and deposits a "1" in the specified cell.


