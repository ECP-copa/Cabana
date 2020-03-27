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
