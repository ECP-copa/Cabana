program example
  use iso_c_binding

  !An interface is necessary for calling a function defined in C++    
  interface
     subroutine c_kokkos_parallel_for() bind(C, name="c_kokkos_parallel_for")
       use iso_c_binding
     end subroutine c_kokkos_parallel_for
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

  ! initialize Kokkos  
  call c_kokkos_initlize()

  ! the c_kokkos_parallel_for is in C++, which calls a Fortran kernel.  
  call c_kokkos_parallel_for()

  ! finalize Kokkos  
  call c_kokkos_finalize()

end program example
