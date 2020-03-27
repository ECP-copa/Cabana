! ****************************************************************************
! * Copyright (c) 2018-2020 by the Cabana authors                            *
! * All rights reserved.                                                     *
! *                                                                          *
! * This file is part of the Cabana library. Cabana is distributed under a   *
! * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
! * the top-level directory.                                                 *
! *                                                                          *
! * SPDX-License-Identifier: BSD-3-Clause                                    *
! ****************************************************************************

!define the inner vector length of SOA
#include "veclen.h"

SUBROUTINE soaExample (part) BIND(C,name='soaExample')
  USE, INTRINSIC :: ISO_C_BINDING
  implicit none
  integer i,j,a
  !The Fortran derived type has the same memory layout as the C struct defined by
  ! struct local_data_struct_t {
  !   double d0[3][3][VECLEN];
  !   double d1[4][VECLEN];
  !   int    d2[VECLEN];
  ! };
  type, BIND(C) :: ptl_type
     real (C_DOUBLE) :: d0(VECLEN,3,3)
     real (C_FLOAT ) :: d1(VECLEN,4)
     integer (C_INT) :: d2(VECLEN)
  end type ptl_type

  type(ptl_type) :: part

  !An interface is necessary for calling the function defined in C++
  interface
     subroutine delete_soa() bind(C)
       use iso_c_binding
     end subroutine delete_soa
  end interface

!Assign data to the soa values
  do i = 1,3
     do j = 1,3
        do a = 1,VECLEN
           part%d0(a,i,j) = 1.0 * (a + i + j)
     end do
     end do
  end do

  do i = 1,4
     do a = 1,VECLEN
        part%d1(a,i) = 1.0 * (a + i)
     end do
  end do

  do a = 1,VECLEN
     part%d2(a) = a + 1234
  end do

! ouput
  print *
  print *, "Print from a Cabana Fortran kernel:"
  print *
  do i = 1,3
     do j = 1,3
        do a = 1,VECLEN
        print *, "Soa member 0 element (",a,",",i,",",j,"):",part%d0(a,i,j)
     end do
     end do
  end do

  do i = 1,4
        do a = 1,VECLEN
           print *, "Soa member 1 element (",a,",",i,"):",part%d1(a,i)
        end do
  end do

  do a = 1,VECLEN
     print *, "Soa member 2: ",part%d2(a)
  end do
  call delete_soa();

end SUBROUTINE soaExample
