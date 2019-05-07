#include "../Fortran_features/cabana_fortran_macros.h"
#ifndef USE_GPU
#define USE_GPU 0
#endif
#ifndef SIMD_SIZE
#define SIMD_SIZE 1
#endif

module scatter_module
  use, intrinsic :: ISO_C_BINDING
  implicit none

  PARTICLE_OP_INTERFACE(particle_scatter)

  contains

  subroutine scatter
    use ptl_module, only : N_PTL
    use grid_module, only : grid
    integer :: err

    ! Zero out array
    grid=0

    ! Do the particle-to-grid operation
    err =  particle_scatter(1,N_PTL)

#ifdef USE_GPU
    ! If on GPU, we need to bring our array back to host
    call transfer_grid_to_host
#else
    ! If using OpenMP, we need to reduce the data (reduces down to the "master" thread=1)
    call reduce_sum(grid)
#endif
  end subroutine

  subroutine transfer_grid_to_host
#if USE_GPU == 1
    use grid_module, only : grid, grid_h
    use cudafor
    integer(kind=cuda_count_kind) :: icount
    integer :: err
    icount = size(grid, kind=cuda_count_kind)
    err = cudaMemcpy( grid_h, grid, icount, cudaMemcpyDeviceToHost)
#endif
  end subroutine

  subroutine reduce_sum(array)
    integer, dimension(:,:), intent(inout) :: array
    integer :: nthreads, i
    nthreads = ubound(array,2)
    do i=nthreads,2,-1
      array(:,1)=array(:,1)+array(:,i)
    enddo
  end subroutine

  !!!!!!!!!!!! KERNEL !!!!!!!!!!!!!!

  ATTR_DEVICE
  subroutine particle_scatter_f(part, n_vectors, i_particle) BIND(C,name='particle_scatter_f')
    USE, INTRINSIC :: ISO_C_BINDING
    use ptl_module, only : ptl_type, ptl_type_single
    use ptl_module, only : AoS_indices, convert_to_single_ptl, convert_from_single_ptl

    use grid_module, only : grid
    implicit none
    INTEGER(C_INT), VALUE :: n_vectors
    INTEGER(C_INT), VALUE :: i_particle
    type(ptl_type) :: part(n_vectors)
    type(ptl_type_single) :: one_part
    integer :: CABANA_DEF_THREAD_NUM
    integer :: i_vec, i_cell, ithread, istat
    integer :: a_vec, s_vec
    ! Determine thread number (for CPU case)
    CABANA_GET_THREAD_NUM(ithread)

    ! Convert to a single structure (size SIMD_SIZE) for easier use
    call AoS_indices(i_particle,s_vec,a_vec)
    call convert_to_single_ptl(part,one_part, s_vec,a_vec)

    do i_vec=1,SIMD_SIZE
      ! Determine which cell the particle is in
      i_cell=determine_cell(one_part%gid(i_vec))

      ! CABANA_ADD: atomic if cuda (ithread=1); or using data replication if OpenMP
      CABANA_ADD(grid(i_cell,ithread), 1)
    enddo

  end subroutine particle_scatter_f

  ATTR_DEVICE
  integer function determine_cell(gid)
    use grid_module, only : n_cells
    integer, intent(in) :: gid
    ! Arbitrary choice for determining cell
    determine_cell=modulo(gid-1,n_cells)+1
  end function

end module
