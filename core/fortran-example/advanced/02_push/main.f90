program main
  use kokkos_fortran_wrapper, only : kokkos_initialize, kokkos_finalize
  use ptl_module, only : N_PTL, N_PTL_MAX, particle_setup
  use push_module, only : push, n_subcycles
  implicit none
  ! MPI-related variables
  integer :: sml_mype, sml_totalpe, sml_comm, sml_comm_null

  integer :: time_step, n_time_steps

  ! Inputs
  n_time_steps=10
  n_subcycles =50 ! # of push operations per time step
  N_PTL=1048576

  ! Initialize MPI
  call parallel_initialize(sml_mype, sml_totalpe, sml_comm, sml_comm_null)

  ! Initialize kokkos
  call kokkos_initialize

  ! Create cabana data structure and initialize particle attributes
  N_PTL_MAX=N_PTL ! Add a buffer if desired
  call particle_setup

  ! Enter time loop
  do time_step = 1,n_time_steps
    if (sml_mype==0) print *, 'Time step: ', time_step
    ! Call the push
    call push
  end do

  ! Finalize kokkos
  call kokkos_finalize

  ! Finalize MPI
  call parallel_finalize
end program main

subroutine parallel_initialize(sml_mype, sml_totalpe, sml_comm, sml_comm_null)
  use ptl_module
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
