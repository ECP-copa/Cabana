#include "../Fortran_features/cabana_fortran_macros.h"
module update_phase_module
  implicit none
  contains

  ATTR_DEVICE
  subroutine update_phase(ph,gid)
    real(8), intent(inout) :: ph(:,:)
    integer, intent(in) :: gid(:)
    integer :: i_vec
    real(8) :: dt=0.1, rmax=10.0
    do i_vec=1,SIMD_SIZE
      ph(i_vec,1) = modulo(ph(i_vec,1) + ph(i_vec,4)*dt,rmax)
    enddo
  endsubroutine
end module
