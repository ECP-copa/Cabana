#include "veclen.h"

SUBROUTINE soaExample (part) BIND(C,name='soaExample')
  USE, INTRINSIC :: ISO_C_BINDING
  implicit none
  integer i,j,a
  type, BIND(C) :: ptl_type      
     real (C_DOUBLE) :: d0(veclen,3,3) 
     real (C_FLOAT ) :: d1(veclen,4) 
     integer (C_INT) :: d2(veclen)
  end type ptl_type

  type(ptl_type) :: part

  interface     
     subroutine delete_soa() bind(C)
       use iso_c_binding       
     end subroutine delete_soa
  end interface

!Assign data to the soa values
  do i = 1,3
     do j = 1,3
        do a = 1,veclen
           part%d0(a,i,j) = 1.0 * (a + i + j)
     end do
     end do
  end do

  do i = 1,4
     do a = 1,veclen
        part%d1(a,i) = 1.0 * (a + i)
     end do
  end do

  do a = 1,veclen
     part%d2(a) = a + 1234
  end do

! ouput 
  print *
  print *, "Print from a Cabana Fortran kernel:"
  print *
  do i = 1,3
     do j = 1,3
        do a = 1,veclen
        print *, "Soa member 0 element (",a,",",i,",",j,"):",part%d0(a,i,j)
     end do
     end do
  end do

  do i = 1,4
        do a = 1,veclen
           print *, "Soa member 1 element (",a,",",i,"):",part%d1(a,i) 
        end do
  end do

  do a = 1,veclen
     print *, "Soa member 2: ",part%d2(a)
  end do
  call delete_soa();

end SUBROUTINE soaExample
