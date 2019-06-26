! ****************************************************************************
! * Copyright (c) 2018-2019 by the Cabana authors                            *
! * All rights reserved.                                                     *
! *                                                                          *
! * This file is part of the Cabana library. Cabana is distributed under a   *
! * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
! * the top-level directory.                                                 *
! *                                                                          *
! * SPDX-License-Identifier: BSD-3-Clause                                    *
! ****************************************************************************

! Define the inner vector length of SOA 
#include "veclen.h"
#include "../Fortran_features/cabana_fortran_macros.h"

#ifndef USE_GPU
#define USE_GPU 0
#endif

#if USE_GPU == 1
  attributes(global) &
  SUBROUTINE set(ns,p1)
  
  implicit none
  !The Fortran derived type has the same memory layout as the C struct defined by
  ! struct local_data_struct_t {     
  !   double d0[VECLEN];     
  !   double d1[VECLEN];     
  ! };
    type, BIND(C) :: ptl_type      
       real (C_DOUBLE) :: d0(VECLEN) 
       real (C_DOUBLE) :: d1(VECLEN) 
    end type ptl_type

  type(ptl_type) :: p1(*)
  !real :: p(:,:)
  integer,value :: ns
  integer :: i, j, n(2)
  i = (blockIdx%x-1)*blockDim%x + threadIdx%x
  j = (blockIdx%y-1)*blockDim%y + threadIdx%y
  if(i<=ns .and. j<=VECLEN) then
     p1(i)%d0(j) = 1.0
     p1(i)%d1(j) = 2.0   
  end if
  END SUBROUTINE set
 
  SUBROUTINE initialization(part,n_soa, num_part) BIND(C)
  USE cudafor
  use simpleOps_m
  implicit none
  integer a,s

  !Declared as AoSoA
  type(ptl_type) FCABANA_DEVICE :: part(*)

  !The number of soa and particles  
  INTEGER(C_INT), VALUE :: n_soa, num_part
  type(dim3) :: grid, tBlock 

  !1D block, 2D grid for launching the cuda kernel
  tBlock = dim3(32,1,1)
  grid = dim3(ceiling(real(n_soa)/tBlock%x), VECLEN, 1)

!Assign data to the aosoa values
!Note that num_part<n_soa*VECLEN, we fill the multiple of VECLEN anyway
  call set<<<grid,tBlock>>>( n_soa, part)

end SUBROUTINE initialization
#else
 SUBROUTINE initialization(part,num_part) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING
  implicit none
  integer a,s
  type, BIND(C) :: ptl_type      
     real (C_DOUBLE) :: d0(VECLEN) 
     real (C_DOUBLE) :: d1(VECLEN) 
  end type ptl_type
  !Declared as AoSoA
  type(ptl_type) :: part(*)
  !The number of particles  
  INTEGER(C_INT), VALUE :: num_part
  !The number of SOA 
  INTEGER :: n_soa 

  !Find out the number of soa in the aosoa
  n_soa = (num_part-1)/VECLEN+1

!Assign data to the aosoa values
!Note that num_part<n_soa*VECLEN, we fill the multiple of VECLEN anyway
  do s = 1, n_soa
     do a = 1,VECLEN
        part(s)%d0(a) = 1.0
        part(s)%d1(a) = 2.0
     end do
  end do
 END SUBROUTINE initialization
#endif 

#if USE_GPU == 1
  attributes(host,device) &
#endif     
SUBROUTINE kernel_1(part,s,a) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING
  implicit none

  type, BIND(C) :: ptl_type      
     real (C_DOUBLE) :: d0(VECLEN) 
     real (C_DOUBLE ) :: d1(VECLEN) 
  end type ptl_type

  type(ptl_type) :: part(*)
  INTEGER(C_INT), VALUE :: s,a


!Assign data to the aosoa values 
!index is shifted by 1 for the fortran convention
  part(s+1)%d0(a+1) =  part(s+1)%d1(a+1) 
!  if(s<32.and.a<32) print *,"kernel_1:",s,a,part(s+1)%d0(a+1)
end SUBROUTINE kernel_1
 
#if USE_GPU == 1
  attributes(host,device) &
#endif     
SUBROUTINE kernel_2(part,s0,a0,s1,a1) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING
  implicit none

  type, BIND(C) :: ptl_type      
     real (C_DOUBLE) :: d0(VECLEN) 
     real (C_DOUBLE) :: d1(VECLEN) 
  end type ptl_type

  type(ptl_type) :: part(*)
  INTEGER(C_INT), VALUE :: s0,a0,s1,a1

  
!Assign data to the aosoa values
!note: index is shifted by 1 for the fortran convention
  part(s1+1)%d1(a1+1) =  part(s0+1)%d1(a0+1) 

end SUBROUTINE kernel_2


 
