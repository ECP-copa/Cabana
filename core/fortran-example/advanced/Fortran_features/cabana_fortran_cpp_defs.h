// Definitions of Fortran wrappers for C++ routines
extern "C" void kokkos_initialize( void );
extern "C" void kokkos_finalize( void );
extern "C" void kokkos_fence( void );

void kokkos_initialize() {
  Kokkos::initialize();
}

void kokkos_finalize() {
  Kokkos::finalize();
}

void kokkos_fence() {
  Kokkos::fence();
}
