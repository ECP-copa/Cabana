module cabana_wrapper_module
  interface
     subroutine cabana_finalize() bind(C, name="cabana_finalize")
       use iso_c_binding
     end subroutine cabana_finalize
  end interface

  interface
     subroutine cabana_initialize() bind(C, name="cabana_initialize")
       use iso_c_binding
     end subroutine cabana_initialize
  end interface

  interface
     subroutine kokkos_fence() bind(C, name="kokkos_fence")
       use iso_c_binding
     end subroutine kokkos_fence
  end interface
end module cabana_wrapper_module
