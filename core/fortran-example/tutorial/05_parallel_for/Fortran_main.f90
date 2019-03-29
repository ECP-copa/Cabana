program example
  use iso_c_binding
  interface
     subroutine parallelForExample() bind(C, name="parallelForExample")
       use iso_c_binding
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
  
  call c_kokkos_initlize()
  call parallelForExample();
  call c_kokkos_finalize()

end program example
