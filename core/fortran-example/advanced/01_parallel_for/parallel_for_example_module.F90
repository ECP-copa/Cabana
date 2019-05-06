! ****************************************************************************
! * Copyright (c) 2018-2019 by the Cabana authors                            *
! * All rights reserved.                                                     *
! *                                                                          *
! * This file is part of the Cabana library. Cabana is distributed under a   *
! * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
! * the top-level directory.                                                 *
! *                                                                          *
! * SPDX-License-Identifier: BSD-3-Clause                                    *
! ****************************************************************************

#include "../Fortran_features/cabana_fortran_macros.h"

module parallel_for_example_module
  use, intrinsic :: ISO_C_BINDING
  implicit none

  ! Example array
  real(8) FCABANA_DEVICE, allocatable :: particle_position(:)

  ! Interface fortran main uses to call the C++ parallel_for
  MISC_OP_INTERFACE(parallel_for_example)

  contains

  ! Allocate array
  subroutine array_setup(N_PTL)
    integer, intent(in) :: N_PTL

    allocate(particle_position(N_PTL))

  end subroutine

  ! Deallocate array
  subroutine array_deallocation()
    deallocate(particle_position)
  end subroutine

  !!!!!!!!!!!! KERNEL !!!!!!!!!!!!!!

  ! Each parallel_for kernel must be accounted for in two places:
  !   1) In some module, e.g. this module, we must add a MISC_OP_INTERFACE, which defines the name of the C++ subroutine to be called by fortrain main
  !   2) In misc_ops.h, the routine must be defined with a MISC_OP macro, which links the C++ subroutine to the internal fortran kernel to be performed
  ATTR_DEVICE
  subroutine parallel_for_example_f(i_particle) BIND(C,name='parallel_for_example_f')
    USE, INTRINSIC :: ISO_C_BINDING
    implicit none

    INTEGER(C_INT), VALUE :: i_particle

    ! As an example: set the position equal to the index
    particle_position(i_particle+1)=i_particle+1
  end subroutine

end module

