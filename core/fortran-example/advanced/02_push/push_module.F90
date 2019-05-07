#include "../Fortran_features/cabana_fortran_macros.h"
#ifndef USE_GPU
#define USE_GPU 0
#endif
#ifndef SIMD_SIZE
#define SIMD_SIZE 1
#endif
#ifndef VEC_LEN
#define VEC_LEN 32
#endif
module push_module
  use, intrinsic :: ISO_C_BINDING
  implicit none

  integer FCABANA_DEVICE :: n_subcycles

  PARTICLE_OP_INTERFACE(particle_push)

  contains

  subroutine push
    use ptl_module, only : N_PTL
    integer :: err
    err =  particle_push(1,N_PTL)
  end subroutine

  !!!!!!!!!!!! KERNEL !!!!!!!!!!!!!!

  ATTR_DEVICE
  subroutine particle_push_f(part, n_vectors, i_particle) BIND(C,name='particle_push_f')
    USE, INTRINSIC :: ISO_C_BINDING
    use ptl_module, only : ptl_type, ptl_type_single
    use ptl_module, only : AoS_indices, convert_to_single_ptl, convert_from_single_ptl
    use update_phase_module, only : update_phase
    implicit none

    INTEGER(C_INT), VALUE :: i_particle
    INTEGER(C_INT), VALUE :: n_vectors

    type(ptl_type) :: part(n_vectors)
    type(ptl_type_single) :: one_part
    integer :: i_cycle
    integer :: a_vec, s_vec
    ! Convert to a single structure (size SIMD_SIZE) for easier use
    call AoS_indices(i_particle,s_vec,a_vec)
    call convert_to_single_ptl(part,one_part, s_vec,a_vec)

    do i_cycle = 1, n_subcycles
        call update_phase(one_part%ph,one_part%gid)
    end do

    ! Write back to the AoSoA
    call convert_from_single_ptl(part,one_part, s_vec,a_vec)

  end subroutine

end module push_module

