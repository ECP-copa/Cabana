############################################################################
# Copyright (c) 2018-2023 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

  add_executable(HaloExchange halo_exchange_example.cpp)
  target_link_libraries(HaloExchange Cabana::Core)
  add_test(NAME Cabana_Core_Tutorial_12 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
    ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:HaloExchange> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(Cabana_Core_Tutorial_12 PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
