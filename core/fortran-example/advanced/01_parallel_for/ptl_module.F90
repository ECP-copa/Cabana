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
module ptl_module
  use, intrinsic :: ISO_C_BINDING
  implicit none

  ! Particle information
  integer, parameter ::  ptl_nphase=6, ptl_nconst=3

  ! Number of particles and size of array to allocate
  integer :: N_PTL, N_PTL_MAX

  ! Shape of a particle SoA. This must be the same as defined in cabana_cpp_interface.cpp
  type, BIND(C) :: ptl_type
     real (C_DOUBLE) :: ph(VEC_LEN,ptl_nphase) ! 1-r, 2-z, 3-phi, 4-rho_parallel, 5-w1, 6-w2
     real (C_DOUBLE) :: ct(VEC_LEN,ptl_nconst) ! 1-mu, 2-w0, 3-f0
     integer (C_INT) :: gid(VEC_LEN)
  end type ptl_type

  ! Shape of the particle SoA to be operated on inside the kernel. SIMD_SIZE
  ! should be 1 for GPU runs, and tuned for optimal performance on CPU runs.
  type, BIND(C) :: ptl_type_single
     real (C_DOUBLE) :: ph(SIMD_SIZE,ptl_nphase)
     real (C_DOUBLE) :: ct(SIMD_SIZE,ptl_nconst)
     integer (C_INT) :: gid(SIMD_SIZE)
  end type ptl_type_single

  ! Interface to the C++ routine that allocates the AoSoA.
  interface
    integer(C_INT) function particle_allocation(num_particle) bind(C,name='particle_allocation');
      use iso_c_binding
      integer(C_INT), intent(in), value :: num_particle
    end function
  end interface

  ! Interface for initializing particle info.
  PARTICLE_OP_INTERFACE(particle_initialization)

  contains

  ! Allocate and initialize particles
  subroutine particle_setup
    integer :: err
#if USE_GPU==1
    print *, "*** GPU VERSION ***"
#else
    print *, "*** CPU VERSION ***"
#endif
    ! Allocate an AoSoA of inner array length VEC_LEN and total particles N_PTL_MAX
    err = particle_allocation(N_PTL_MAX)

    ! Assign initial values to particle info. Acts on particles 1 through N_PTL.
    err = particle_initialization(1,N_PTL)

  end subroutine

  !!!!!!!!!!!! KERNELS !!!!!!!!!!!!!!

  ATTR_DEVICE
  subroutine particle_initialization_f(part, n_vectors, i_particle) BIND(C,name='particle_initialization_f')
    USE, INTRINSIC :: ISO_C_BINDING
    implicit none

    INTEGER(C_INT), VALUE :: n_vectors
    INTEGER(C_INT), VALUE :: i_particle

    type(ptl_type) :: part(n_vectors)
    type(ptl_type_single) :: one_part

    integer :: i, i_vec, nrot
    real (8) :: fract
    integer :: a_vec, s_vec
    ! Convert to a single structure (size SIMD_SIZE) for easier use
    call AoS_indices(i_particle,s_vec,a_vec)
    call convert_to_single_ptl(part,one_part, s_vec,a_vec)

    ! Just some arbitrary particle initialization:  A spiral, for fun

    nrot = 4 ! number of rotations in spiral
    do i_vec = 1, SIMD_SIZE
      fract = dble(i_particle)/dble(n_vectors*SIMD_SIZE)

      one_part%ph(i_vec,1) = 1 + cos(fract * nrot * 6.28) * fract * 0.5D0
      one_part%ph(i_vec,2) = 1 + cos(fract * nrot * 6.28) * fract * 0.5D0
      one_part%ph(i_vec,3) = 1 + sin(fract * nrot * 6.28) * fract * 0.5D0
      one_part%ph(i_vec,4) = 1 + cos(fract * nrot * 6.28) * fract * 0.5D0
      one_part%gid(i_vec) = i_particle+1
    end do

    ! Write back to the AoSoA
    call convert_from_single_ptl(part,one_part, s_vec,a_vec)
  end subroutine

  ATTR_DEVICE
  subroutine convert_to_single_ptl(part,one_part, s_vec,a_vec)
    USE, INTRINSIC :: ISO_C_BINDING
    implicit none

    type(ptl_type) :: part(:)
    type(ptl_type_single) :: one_part
    integer, intent(in) :: a_vec, s_vec
    integer :: iy, p_vec, i_vec
    do i_vec=1,SIMD_SIZE
      p_vec = a_vec+i_vec-1
      do iy=1,ptl_nphase
        one_part%ph(i_vec,iy) = part(s_vec)%ph(p_vec,iy)
      enddo
      do iy=1,ptl_nconst
        one_part%ct(i_vec,iy) = part(s_vec)%ct(p_vec,iy)
      enddo
      one_part%gid(i_vec) = part(s_vec)%gid(p_vec)
    enddo
  end subroutine convert_to_single_ptl

  ATTR_DEVICE
  subroutine convert_from_single_ptl(part,one_part, s_vec,a_vec)
    USE, INTRINSIC :: ISO_C_BINDING
    implicit none

    type(ptl_type) :: part(:)
    type(ptl_type_single) :: one_part
    integer, intent(in) :: a_vec, s_vec
    integer :: iy, p_vec, i_vec
    do i_vec=1,SIMD_SIZE
      p_vec = a_vec+i_vec-1
      do iy=1,ptl_nphase
        part(s_vec)%ph(p_vec,iy) = one_part%ph(i_vec,iy)
      enddo
      do iy=1,ptl_nconst
        part(s_vec)%ct(p_vec,iy) = one_part%ct(i_vec,iy)
      enddo
      part(s_vec)%gid(p_vec) = one_part%gid(i_vec)
    enddo
  end subroutine

  ATTR_DEVICE
  subroutine AoS_indices(i_item,s_vec,a_vec)
    USE, INTRINSIC :: ISO_C_BINDING
    implicit none
    integer(C_INT), intent(in) :: i_item
    integer, intent(out) :: s_vec
    integer, intent(out) :: a_vec
    integer :: i_temp, i_mod
#if USE_CAB_GPU == 1
    ! GPU: access an individual particle
    i_temp = i_item
    i_mod  = modulo(i_temp,VEC_LEN)
    a_vec  = i_mod + 1
    i_temp = i_temp - i_mod
    s_vec  = i_temp/VEC_LEN + 1
#else
    ! CPU: access an array of particles
    s_vec = 1
    a_vec = 1
#endif
  end subroutine

end module ptl_module

