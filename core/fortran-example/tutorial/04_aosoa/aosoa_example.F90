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

! Define the inner vector length of SOA
#include "veclen.h"

SUBROUTINE aosoaExample (part,num_part) BIND(C,name='aosoaExample')
  USE, INTRINSIC :: ISO_C_BINDING
  implicit none
  integer i,j,a,s

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
    do i = 1,3
       do j = 1,3
          do a = 1,VECLEN
             part(s)%d0(a,i,j) = 1.0 * (a + i + j)
          end do
       end do
    end do
 enddo

do s = 1, n_soa
   do i = 1,4
      do a = 1,VECLEN
         part(s)%d1(a,i) = 1.0 * (a + i)
      end do
   end do
end do

do s = 1, n_soa
   do a = 1,VECLEN
      part(s)%d2(a) = a + 1234
   end do
end do


!ouput
print *
print *, "Print from a Cabana Fortran kernel:"
print *
do s = 1,n_soa
   do i = 1,3
      do j = 1,3
         do a = 1,VECLEN
            print *, "Aosoa member 0 element (",s,",",a,"),(",i,",",j,"):",part(s)%d0(a,i,j)
         end do
      end do
   end do
end do
do s = 1, n_soa
   do i = 1,4
      do a = 1,VECLEN
         print *, "Aosoa member 1 element (",s,",",a,"),(",i,"):",part(s)%d1(a,i)
      end do
   end do
end do

do s = 1, n_soa
   do a = 1,VECLEN
      print *, "Aosoa member 2 (",a,")",part(s)%d2(a)
   end do
end do



end SUBROUTINE aosoaExample
