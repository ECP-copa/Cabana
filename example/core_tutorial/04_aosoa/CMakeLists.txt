############################################################################
# Copyright (c) 2018-2021 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

  add_executable(ArrayOfStructsOfArrays aosoa_example.cpp)
  target_link_libraries(ArrayOfStructsOfArrays cabanacore)
  add_test(NAME Core_tutorial_04 COMMAND ${NONMPI_PRECOMMAND} ArrayOfStructsOfArrays)
