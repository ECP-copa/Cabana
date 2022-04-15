############################################################################
# Copyright (c) 2018-2022 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

# the next for lines are just, so that we can use this as
# a standalone CMakeLists.txt for testing of the Target export
cmake_minimum_required(VERSION 3.16)
project(Cabana_HelloWorld)
if(NOT TARGET Cabana::cabanacore)
  find_package(Cabana)
endif()

add_executable(HelloWorld hello_world.cpp)
target_link_libraries(HelloWorld Cabana::cabanacore)
add_test(NAME Core_tutorial_01 COMMAND ${NONMPI_PRECOMMAND} HelloWorld)
