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

program main
  use kokkos_fortran_wrapper, only : kokkos_initialize, kokkos_finalize
  use parallel_for_example_module, only : array_setup, parallel_for_example, array_deallocation
  implicit none
  ! MPI-related variables
  integer :: sml_mype, sml_totalpe, sml_comm, sml_comm_null

  integer :: N_PTL, err

  ! Inputs
  N_PTL=1048576

  ! Initialize MPI
  call parallel_initialize(sml_mype, sml_totalpe, sml_comm, sml_comm_null)

  ! Initialize kokkos
  call kokkos_initialize

  ! Allocate a fortran array
  call array_setup(N_PTL)

  ! Loop over the array performing a Fortran subroutine on each element
  err = parallel_for_example(1,N_PTL)

  ! Free the fortran array
  call array_deallocation()

  ! Finalize kokkos
  call kokkos_finalize

  ! Finalize MPI
  call parallel_finalize
end program main

subroutine parallel_initialize(sml_mype, sml_totalpe, sml_comm, sml_comm_null)
  implicit none
  include 'mpif.h'
  integer, intent(out) :: sml_mype, sml_totalpe, sml_comm, sml_comm_null
  integer :: ierr
  call mpi_init(ierr)
  call mpi_comm_rank(mpi_comm_world,sml_mype,ierr)
  call mpi_comm_size(mpi_comm_world,sml_totalpe,ierr)
  call mpi_comm_dup(mpi_comm_world,sml_comm,ierr)
  sml_comm_null = mpi_comm_null
end subroutine

subroutine parallel_finalize
  integer :: ierr
  call mpi_finalize(ierr)
end subroutine
