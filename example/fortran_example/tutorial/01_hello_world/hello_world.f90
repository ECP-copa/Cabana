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

SUBROUTINE print_hello_world () BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING
  print *,"Hello World from Cabana (Fortran)!"
end SUBROUTINE print_hello_world
