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

  add_executable(SimdParallelFor simd_parallel_for_example.cpp)
  target_link_libraries(SimdParallelFor Cabana::Core)
  add_test(NAME Cabana_Core_Tutorial_10_simd COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:SimdParallelFor>)
