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

program example
  use iso_c_binding

  !An interface is necessary for calling a function defined in C++
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

  ! initialize Kokkos
  call c_kokkos_initlize()

  ! the parallelForExample is in C++, which calls Fortran kernels.
  call parallelForExample();

  ! finalize Kokkos
  call c_kokkos_finalize()

end program example
