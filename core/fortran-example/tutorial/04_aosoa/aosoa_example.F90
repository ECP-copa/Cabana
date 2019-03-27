#include "veclen.h"

SUBROUTINE aosoaExample (part,num_part) BIND(C,name='aosoaExample')
  USE, INTRINSIC :: ISO_C_BINDING
  implicit none
  integer i,j,a,s
  type, BIND(C) :: ptl_type      
     real (C_DOUBLE) :: d0(veclen,3,3) 
     real (C_FLOAT ) :: d1(veclen,4) 
     integer (C_INT) :: d2(veclen)
  end type ptl_type

  type(ptl_type) :: part(*)
  INTEGER(C_INT), VALUE :: num_part
  INTEGER :: n_soa

!Find out the number of soa in the aosoa
  n_soa = (num_part-1)/veclen+1

!Assign data to the aosoa values
!Note that num_part<n_soa*veclen, we fill the multiple of veclen anyway
 do s = 1, n_soa
    do i = 1,3
       do j = 1,3
          do a = 1,veclen
             part(s)%d0(a,i,j) = 1.0 * (a + i + j)
          end do
       end do
    end do
 enddo

do s = 1, n_soa
   do i = 1,4
      do a = 1,veclen
         part(s)%d1(a,i) = 1.0 * (a + i)
      end do
   end do
end do

do s = 1, n_soa
   do a = 1,veclen
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
         do a = 1,veclen
            print *, "Aosoa member 0 element (",s,",",a,"),(",i,",",j,"):",part(s)%d0(a,i,j)
         end do
      end do
   end do
end do
do s = 1, n_soa
   do i = 1,4
      do a = 1,veclen
         print *, "Aosoa member 1 element (",s,",",a,"),(",i,"):",part(s)%d1(a,i) 
      end do
   end do
end do

do s = 1, n_soa
   do a = 1,veclen
      print *, "Aosoa member 2 (",a,")",part(s)%d2(a)
   end do
end do



end SUBROUTINE aosoaExample
