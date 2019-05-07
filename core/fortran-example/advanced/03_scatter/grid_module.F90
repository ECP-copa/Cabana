#include "../Fortran_features/cabana_fortran_macros.h"
#ifndef USE_GPU
#define USE_GPU 0
#endif
#ifndef SIMD_SIZE
#define SIMD_SIZE 1
#endif

module grid_module
  use, intrinsic :: ISO_C_BINDING
  implicit none
  integer FCABANA_DEVICE, allocatable :: grid(:,:)
  integer FCABANA_DEVICE :: n_cells
#if USE_GPU == 1
  integer, allocatable :: grid_h(:,:)
#endif

  contains

  subroutine grid_setup
    INTEGER :: CABANA_DEF_MAX_THREADS
    integer :: n_threads

    CABANA_GET_MAX_THREADS(n_threads)

    allocate(grid(n_cells,n_threads))
#if USE_GPU == 1
    allocate(grid_h(n_cells,n_threads))
#endif
  end subroutine

end module
