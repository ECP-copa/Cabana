module kokkos_fortran_wrapper
  interface
     subroutine kokkos_finalize() bind(C, name="kokkos_finalize")
       use iso_c_binding
     end subroutine kokkos_finalize
  end interface

  interface
     subroutine kokkos_initialize() bind(C, name="kokkos_initialize")
       use iso_c_binding
     end subroutine kokkos_initialize
  end interface

  interface
     subroutine kokkos_fence() bind(C, name="kokkos_fence")
       use iso_c_binding
     end subroutine kokkos_fence
  end interface
end module kokkos_fortran_wrapper
