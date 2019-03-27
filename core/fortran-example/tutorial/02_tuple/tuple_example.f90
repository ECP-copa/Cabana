SUBROUTINE tupleExample (particle) BIND(C,name='tupleExample')
  USE, INTRINSIC :: ISO_C_BINDING
  integer i,j
  type, BIND(C) :: ptl_type      
     real (C_DOUBLE) :: d0(3,3) 
     real (C_FLOAT ) :: d1(4) 
     integer (C_INT) :: d2
  end type ptl_type

  type(ptl_type) :: part

  interface     
!!$     subroutine allocate_tuple() bind(C)
!!$       use iso_c_binding       
!!$     end subroutine allocate_tuple

     subroutine delete_tuple() bind(C)
       use iso_c_binding       
     end subroutine delete_tuple

  end interface

!!$  call allocate_tuple();

  do i = 1,3
     do j = 1,3
        part%d0(i,j) = 1.0 * (i + j)
     end do
  end do

  do i = 1,4
     part%d1(i) = 1.0 * i
  end do

  part%d2 = 1234

  print *
  print *, "Print from a Cabana Fortran kernel:"
  print *
  do i = 1,3
     do j = 1,3
        print *, "Tuple member 0 element (",i,",",j,"):",part%d0(i,j)
     end do
  end do

  do i = 1,4
     print *, "Tuple member 1 element (",i,"):",part%d1(i) 
  end do

  print *, "Tuple member 2: ",part%d2

  call delete_tuple();

end SUBROUTINE tupleExample
