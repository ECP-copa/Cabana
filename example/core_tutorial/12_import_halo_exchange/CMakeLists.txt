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

  add_executable(ImportHaloExchange import_halo_exchange_example.cpp)
  target_link_libraries(ImportHaloExchange Cabana::Core)
  add_test(NAME Cabana_Core_Tutorial_12_import_halo COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
    ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:ImportHaloExchange> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(Cabana_Core_Tutorial_12_import_halo PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
