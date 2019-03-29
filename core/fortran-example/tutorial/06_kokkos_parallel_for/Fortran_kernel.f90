module printHello
contains
  attributes(host,device) subroutine print_hello (i) BIND(C,name='print_hello')
    USE, INTRINSIC :: ISO_C_BINDING, only: c_int
    integer(kind=c_int),value :: i
    print *,"Hello from (frotran) =",i
    !      write(*,'(A)',advance='no') c
  end subroutine print_hello
end module printHello
