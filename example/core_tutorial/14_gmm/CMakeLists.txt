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

# the next for lines are just, so that we can use this as
# a standalone CMakeLists.txt for testing of the Target export
cmake_minimum_required(VERSION 3.16)
project(Cabana_GMM)
if(NOT TARGET Cabana::cabanacore)
  find_package(Cabana)
endif()

# Dimensions of velocity space
set(VelocityDimensions "3" CACHE STRING "Properties of the dimensions of velocity space")
set(VelocityDimensionsValues 1 2 3) # 1d Cartesian, 2d cylindrical, 3d cartesian. If we ever want 2d cartestian we have to come up with something
set_property(CACHE VelocityDimensions PROPERTY STRINGS ${VelocityDimensionsValues})
message(STATUS "VelocityDimensions='${VelocityDimensions}'")

# Floating point type for particle data
set(ParticleFloatType "double" CACHE STRING "Floating point type for particle data")
set(ParticleFloatTypeValues float double)
set_property(CACHE ParticleFloatType PROPERTY STRINGS ${ParticleFloatTypeValues})
message(STATUS "ParticleFloatType='${ParticleFloatType}'")

# Floating point type for particle data
set(GmmFloatType "double" CACHE STRING "Floating point type for particle data")
set(GmmFloatTypeValues float double)
set_property(CACHE GmmFloatType PROPERTY STRINGS ${GmmFloatTypeValues})
message(STATUS "GmmFloatType='${GmmFloatType}'")

file(GLOB HEADERS "*.h *hpp")
file(GLOB SOURCES "*.cpp")

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set (CMAKE_CXX_FLAGS "-Wall -Wextra -Wshadow -g3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVelocityDimensions=${VelocityDimensions}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DParticleFloatType=${ParticleFloatType}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGmmFloatType=${GmmFloatType}")

add_executable(gmm ./main.cpp)
target_link_libraries(gmm PUBLIC Cabana::cabanacore)
target_include_directories(gmm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(gmm PUBLIC "${KokkosKernels_CMAKE_DIR}/../../../include")
install(TARGETS gmm DESTINATION ${CMAKE_INSTALL_BINDIR})
