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

add_executable(IndexSpace index_space_example.cpp)
target_link_libraries(IndexSpace Cabana::Grid)
add_test(NAME Grid_tutorial_05 COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:IndexSpace>)
