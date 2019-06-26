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
#include "veclen.h"
#include "../Fortran_features/cabana_fortran_macros.h"

program example
  use, intrinsic :: iso_c_binding
  implicit none
  integer, parameter :: N_PTL = 1048576
  integer, parameter :: N_SOA = ceiling(real(N_PTL)/VECLEN)
  type, BIND(C) :: ptl_type      
     real (C_DOUBLE) :: d0(VECLEN) 
     real (C_DOUBLE) :: d1(VECLEN) 
  end type ptl_type
  !An interface is necessary for calling a function defined in C++  
  interface
     subroutine parallelForExample(part, num_soa, num_p) bind(C, name="parallelForExample")
       import::ptl_type, C_INT
       type(ptl_type) FCABANA_DEVICE  :: part(*)
       integer (C_INT), value :: num_soa, num_p
     end subroutine parallelForExample
  end interface
  
  interface
     subroutine c_kokkos_finalize() bind(C, name="c_kokkos_finalize")
       use iso_c_binding
     end subroutine c_kokkos_finalize
  end interface

  interface
     subroutine c_kokkos_initlize() bind(C, name="c_kokkos_initlize")
       use iso_c_binding
     end subroutine c_kokkos_initlize
  end interface

  type(ptl_type) FCABANA_DEVICE :: part(N_SOA)


  ! initialize Kokkos
  call c_kokkos_initlize()
  
  ! the parallelForExample is in C++, which takes the fortran allocated aosoa, and calls Fortran kernels.
  call parallelForExample(part,N_SOA,N_PTL);

  ! finalize Kokkos
  call c_kokkos_finalize()

end program example
