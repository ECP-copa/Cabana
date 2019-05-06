program Main
  use kokkos_fortran_wrapper, only : kokkos_initialize, kokkos_finalize
  use ptl_module, only : N_PTL, N_PTL_MAX, particle_setup
  use grid_module, only : N_CELLS, grid_setup
  use scatter_module, only : scatter
  implicit none
  ! MPI-related variables
  integer :: sml_mype, sml_totalpe, sml_comm, sml_comm_null

  integer :: i_step

  ! User input
  N_PTL=1048576
  N_CELLS = 99

  ! Initialize MPI
  call parallel_initialize(sml_mype, sml_totalpe, sml_comm, sml_comm_null)

  ! Initialize kokkos
  call kokkos_initialize

  ! Create cabana data structure and initialize particle attributes
  N_PTL_MAX=N_PTL ! Add a buffer if desired
  call particle_setup

  ! Allocate grid
  call grid_setup

  ! particle to grid test: find out what cell each particle is, and count how
  ! many particles are in each cell
  call scatter

  ! Finalize kokkos
  call kokkos_finalize

  ! Finalize MPI
  call parallel_finalize
end program Main

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
