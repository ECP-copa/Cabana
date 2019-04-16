! Define the inner vector length of SOA 
#include "veclen.h"

#ifndef USE_GPU
#define USE_GPU 0
#endif
  
#if USE_GPU == 1
  attributes(device) &
#endif
  SUBROUTINE initialization(part,num_part) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING
  implicit none
  integer i,j,a,s

  !The Fortran derived type has the same memory layout as the C struct defined by
  ! struct local_data_struct_t {     
  !   double d0[veclen];     
  !   double d1[veclen];     
  ! };

  type, BIND(C) :: ptl_type      
     real (C_DOUBLE) :: d0(veclen) 
     real (C_FLOAT ) :: d1(veclen) 
  end type ptl_type

  !Declared as AoSoA
  type(ptl_type) :: part(*)
  !The number of particles  
  INTEGER(C_INT), VALUE :: num_part
  !The number of SOA  
  INTEGER :: n_soa 

!Find out the number of soa in the aosoa
  n_soa = (num_part-1)/veclen+1

!Assign data to the aosoa values
!Note that num_part<n_soa*veclen, we fill the multiple of veclen anyway
do s = 1, n_soa
   do a = 1,veclen
      part(s)%d0(a) = 1.0
      part(s)%d1(a) = 2.0
   end do
end do

end SUBROUTINE initialization
 

#if USE_GPU == 1
  attributes(device) &
#endif     
SUBROUTINE kernel_1(part,s,a) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING
  implicit none

  type, BIND(C) :: ptl_type      
     real (C_DOUBLE) :: d0(veclen) 
     real (C_FLOAT ) :: d1(veclen) 
  end type ptl_type

  type(ptl_type) :: part(*)
  INTEGER(C_INT), VALUE :: s,a


!Assign data to the aosoa values 
!index is shifted by 1 for the fortran convention
  part(s+1)%d0(a+1) =  part(s+1)%d1(a+1) 

end SUBROUTINE kernel_1
 
#if USE_GPU == 1
  attributes(device) &
#endif     
SUBROUTINE kernel_2(part,s0,a0,s1,a1) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING
  implicit none

  type, BIND(C) :: ptl_type      
     real (C_DOUBLE) :: d0(0:veclen-1) !Here index range is the C convention
     real (C_FLOAT ) :: d1(0:veclen-1) 
  end type ptl_type

  type(ptl_type) :: part(*)
  INTEGER(C_INT), VALUE :: s0,a0,s1,a1

  
!Assign data to the aosoa values
!note: index is not shifted by 1 as in kernel_1
  part(s1)%d1(a1) =  part(s0)%d1(a0) 

end SUBROUTINE kernel_2


 
