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

add_executable(HeffteFFT heffte_fast_fourier_transform_example.cpp)
target_link_libraries(HeffteFFT Cabana::Grid)
add_test(NAME Grid_tutorial_10 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:HeffteFFT> ${MPIEXEC_POSTFLAGS})
set_tests_properties(Grid_tutorial_10 PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
