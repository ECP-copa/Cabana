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

add_executable(GlobalMesh global_mesh_example.cpp)
target_link_libraries(GlobalMesh Cabana::Grid)
add_test(NAME Grid_tutorial_02 COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:GlobalMesh>)
