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

add_executable(GridParallel grid_parallel_example.cpp)
target_link_libraries(GridParallel Cabana::Grid)
add_test(NAME Cabana_Grid_Tutorial_09 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:GridParallel> ${MPIEXEC_POSTFLAGS})
set_tests_properties(Cabana_Grid_Tutorial_09 PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
