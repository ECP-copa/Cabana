SUBROUTINE print_hello_world () BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING
  print *,"Hello World from Cabana (Fortran)!"
end SUBROUTINE print_hello_world
    
 
