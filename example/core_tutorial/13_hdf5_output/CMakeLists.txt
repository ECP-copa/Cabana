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

add_executable(HDF5Output hdf5_output_example.cpp)
target_link_libraries(HDF5Output Cabana::Core)
add_test(NAME Core_tutorial_13 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
  ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:HDF5Output> ${MPIEXEC_POSTFLAGS})
set_tests_properties(Core_tutorial_13 PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
