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

#------------------------------------------------------------------------------#
# Project settings
#------------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.16)

project(Cabana LANGUAGES CXX)
set(PROJECT_VERSION "1.0-dev")

# If the user doesn't provide a build type default to release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

# Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

# Disable verbose makefiles
option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)

# use gnu standard install directories
include(GNUInstallDirs)

# point to cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#------------------------------------------------------------------------------#
# Dependencies
#------------------------------------------------------------------------------#
# find kokkos
find_package(Kokkos 3.4 REQUIRED)

# set supported kokkos devices
set(CABANA_SUPPORTED_DEVICES SERIAL THREADS OPENMP CUDA HIP SYCL OPENMPTARGET)

# check user required kokkos device types
foreach(_device ${CABANA_SUPPORTED_DEVICES})
  option(Cabana_REQUIRE_${_device} "Build Cabana with required Kokkos ${device} support" OFF)
  if( Cabana_REQUIRE_${_device} AND NOT _device STREQUAL THREADS)
    kokkos_check( DEVICES ${_device} )
  endif()
endforeach()
# Threads workaround can go when we require a minimum Kokkos version of 3.6
if(Cabana_REQUIRE_PTHREAD)
  message(DEPRECATION "Cabana_REQUIRE_PTHREAD is deprecated, use Cabana_REQUIRE_THREADS instead!")
endif()
if(Cabana_REQUIRE_PTHREAD OR Cabana_REQUIRE_THREADS)
  kokkos_check(DEVICES PTHREAD RETURN_VALUE KOKKOS_CHECK_PTHREAD_RETURN_VALUE)
  kokkos_check(DEVICES THREADS RETURN_VALUE KOKKOS_CHECK_THREADS_RETURN_VALUE)
  if(NOT KOKKOS_CHECK_PTHREAD_RETURN_VALUE AND NOT KOKKOS_CHECK_THREADS_RETURN_VALUE)
    message(FATAL_ERROR "Kokkos does not provide the Threads backend requested")
  endif()
endif()

# ensure that we can use lambdas
if(Kokkos_ENABLE_CUDA)
  kokkos_check(OPTIONS CUDA_LAMBDA)
endif()

# standard dependency macro
macro(Cabana_add_dependency)
  cmake_parse_arguments(CABANA_DEPENDENCY "" "PACKAGE;VERSION" "" ${ARGN})
  find_package( ${CABANA_DEPENDENCY_PACKAGE} ${CABANA_DEPENDENCY_VERSION} QUIET )
  string(TOUPPER "${CABANA_DEPENDENCY_PACKAGE}" CABANA_DEPENDENCY_OPTION )
  option(
    Cabana_REQUIRE_${CABANA_DEPENDENCY_OPTION}
    "Require Cabana to build with ${CABANA_DEPENDENCY_PACKAGE} support" ${CABANA_DEPENDENCY_PACKAGE}_FOUND)
  if(Cabana_REQUIRE_${CABANA_DEPENDENCY_OPTION})
    find_package( ${CABANA_DEPENDENCY_PACKAGE} ${CABANA_DEPENDENCY_VERSION} REQUIRED )
  endif()
  set(Cabana_ENABLE_${CABANA_DEPENDENCY_OPTION} ${${CABANA_DEPENDENCY_PACKAGE}_FOUND})
endmacro()

# find MPI
Cabana_add_dependency( PACKAGE MPI )

# find ArborX
Cabana_add_dependency( PACKAGE ArborX )

# find ALL
Cabana_add_dependency( PACKAGE ALL )

# find Clang Format
find_package( CLANG_FORMAT 10 )

# find hypre
Cabana_add_dependency( PACKAGE HYPRE VERSION 2.22.1 )

# find heffte
Cabana_add_dependency( PACKAGE Heffte VERSION 2.1.0 )

# find Silo
Cabana_add_dependency( PACKAGE SILO )
if(SILO_FOUND)
  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindSILO.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cabana )
endif()

#------------------------------------------------------------------------------#
# Architecture
#------------------------------------------------------------------------------#
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(Cabana_BUILD_MARCH "" CACHE STRING "Arch to use with -march= (if empty CMake will try to use 'native') in release build and only release build")

  # Try -march first. On platforms that don't support it, GCC will issue
  # a hard error, so we'll know not to use it.
  if(Cabana_BUILD_MARCH)
    set(INTERNAL_Cabana_BUILD_MARCH ${Cabana_BUILD_MARCH})
  else()
    set(INTERNAL_Cabana_BUILD_MARCH "native")
  endif()

  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=${INTERNAL_Cabana_BUILD_MARCH}" COMPILER_SUPPORTS_MARCH)
  if(COMPILER_SUPPORTS_MARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${INTERNAL_Cabana_BUILD_MARCH}")
  elseif(Cabana_BUILD_MARCH)
    message(FATAL_ERROR "The flag -march=${INTERNAL_Cabana_BUILD_MARCH} is not supported by the compiler")
  else()
    unset(INTERNAL_Cabana_BUILD_MARCH)
  endif()
endif()

##---------------------------------------------------------------------------##
## Code coverage testing
##---------------------------------------------------------------------------##
option(Cabana_ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(Cabana_ENABLE_COVERAGE_BUILD)
  message(STATUS "Enabling coverage build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -O0")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  option(WITH_ASAN "Build with address sanitizer" OFF)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
  option(WITH_MSAN "Build with memory sanitizer (experimental; requires a memory-sanitized Python interpreter)" OFF)
endif()

if(WITH_ASAN AND WITH_MSAN)
  message( FATAL_ERROR "Address sanitizer and memory sanitizer cannot be enabled simultaneously")
endif()
if(WITH_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
endif()

if(WITH_MSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=memory")
endif()

##---------------------------------------------------------------------------##
## Print the revision number to stdout
##---------------------------------------------------------------------------##
FIND_PACKAGE(Git)
IF(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  EXECUTE_PROCESS(
    COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE   Cabana_GIT_COMMIT_HASH
    )
ELSE()
  SET(Cabana_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
MESSAGE(STATUS "Cabana Revision = '${Cabana_GIT_COMMIT_HASH}'")

#------------------------------------------------------------------------------#
# Tests and Documentation
#------------------------------------------------------------------------------#
# enable unit tests
option(Cabana_ENABLE_TESTING "Build tests" OFF)
if(Cabana_ENABLE_TESTING)
  find_package(GTest 1.10 REQUIRED)
  # Workaround for FindGTest module in CMake older than 3.20
  if(TARGET GTest::gtest)
    set(gtest_target GTest::gtest)
  elseif(TARGET GTest::GTest)
    set(gtest_target GTest::GTest)
  else()
    message(FATAL_ERROR "bug in GTest find module workaround")
  endif()
  set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/cmake/test_harness)
  include(cmake/test_harness/test_harness.cmake)
  enable_testing()
endif()

# enable doxygen
find_package(Doxygen)
if(Doxygen_FOUND)
  doxygen_add_docs(doxygen core/src cajita/src)
endif()

##---------------------------------------------------------------------------##
## Libraries and Examples
##---------------------------------------------------------------------------##

add_subdirectory(core)

option(Cabana_ENABLE_CAJITA "Build Cajita (grid and particle-grid)" ON)
if(Cabana_ENABLE_CAJITA)
  if(MPI_FOUND)
    add_subdirectory(cajita)
  else()
    message(FATAL_ERROR "Cajita requires MPI")
  endif()
endif()

option(Cabana_ENABLE_EXAMPLES "Build examples" OFF)
if(Cabana_ENABLE_EXAMPLES)
  add_subdirectory(example)
endif()

# enable performance tests
option(Cabana_ENABLE_PERFORMANCE_TESTING "Build Performance Tests" OFF)
if(Cabana_ENABLE_PERFORMANCE_TESTING)
  add_subdirectory(benchmark)
endif()

##---------------------------------------------------------------------------##
## Package Configuration
##---------------------------------------------------------------------------##
configure_file(cmake/Cabana.pc.cmakein Cabana.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cabana.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

write_basic_package_version_file("CabanaConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY ExactVersion)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CabanaConfig.cmakein
  ${CMAKE_CURRENT_BINARY_DIR}/CabanaConfig.cmake @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CabanaConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/CabanaConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cabana)

##---------------------------------------------------------------------------##
## Clang Format
##---------------------------------------------------------------------------##
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES core/*.cpp core/*.hpp cajita/*hpp cajita/*cpp example/*cpp example/*hpp cmake/*cpp cmake/*hpp benchmark/*cpp benchmark/*hpp)
  add_custom_target(cabana-format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
