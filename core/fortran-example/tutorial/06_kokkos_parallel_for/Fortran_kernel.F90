#ifndef USE_GPU
#define USE_GPU 0
#endif


module printHello
contains  
#if USE_GPU == 1
  attributes(device) &
#endif
  subroutine print_hello (i) BIND(C,name='print_hello')
    USE, INTRINSIC :: ISO_C_BINDING, only: c_int
    integer(kind=c_int),value :: i
    print *,"Hello from (frotran) =",i
  end subroutine print_hello
end module printHello
