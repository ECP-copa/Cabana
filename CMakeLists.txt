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

#------------------------------------------------------------------------------#
# Project settings
#------------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.16)

project(Cabana LANGUAGES C CXX)
set(PROJECT_VERSION "0.8.0-dev")

# If the user doesn't provide a build type default to release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

# Disable verbose makefiles
option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)

# use gnu standard install directories
include(GNUInstallDirs)
set(Cabana_INSTALL_PACKAGEDIR "${CMAKE_INSTALL_DATADIR}/cmake/Cabana" CACHE PATH "Install location of CMake target files")

include(FeatureSummary)

# add local cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#------------------------------------------------------------------------------#
# Dependencies
#------------------------------------------------------------------------------#
# find kokkos
find_package(Kokkos 4.1 REQUIRED)

# FIXME: remove when custom layout support with mdspan Views is added.
if(Kokkos_VERSION VERSION_GREATER_EQUAL 4.6.99 AND NOT Kokkos_ENABLE_IMPL_VIEW_LEGACY)
  message(FATAL_ERROR "Current Kokkos does not support custom View layouts needed for AoSoA unless "
          "Kokkos is built with Kokkos_ENABLE_IMPL_VIEW_LEGACY=ON.")
endif()

# set supported kokkos devices
set(CABANA_SUPPORTED_DEVICES SERIAL THREADS OPENMP CUDA HIP SYCL OPENMPTARGET)

# check user required kokkos device types
foreach(_device ${CABANA_SUPPORTED_DEVICES})
  option(Cabana_REQUIRE_${_device} "Build Cabana with required Kokkos ${device} support" OFF)
  if(Cabana_REQUIRE_${_device})
    kokkos_check( DEVICES ${_device} )
  endif()
endforeach()

# ensure that we can use lambdas
if(Kokkos_ENABLE_CUDA)
  kokkos_check(OPTIONS CUDA_LAMBDA)
endif()

# standard dependency macro
macro(Cabana_add_dependency)
  cmake_parse_arguments(CABANA_DEPENDENCY "" "PACKAGE;VERSION" "COMPONENTS" ${ARGN})
  find_package( ${CABANA_DEPENDENCY_PACKAGE} ${CABANA_DEPENDENCY_VERSION} QUIET COMPONENTS ${CABANA_DEPENDENCY_COMPONENTS} )
  string(TOUPPER "${CABANA_DEPENDENCY_PACKAGE}" CABANA_DEPENDENCY_OPTION )
  option(
    Cabana_REQUIRE_${CABANA_DEPENDENCY_OPTION}
    "Require Cabana to build with ${CABANA_DEPENDENCY_PACKAGE} support" ${CABANA_DEPENDENCY_PACKAGE}_FOUND)
  if(Cabana_REQUIRE_${CABANA_DEPENDENCY_OPTION})
    find_package( ${CABANA_DEPENDENCY_PACKAGE} ${CABANA_DEPENDENCY_VERSION} REQUIRED COMPONENTS ${CABANA_DEPENDENCY_COMPONENTS} )
  endif()
  set(Cabana_ENABLE_${CABANA_DEPENDENCY_OPTION} ${${CABANA_DEPENDENCY_PACKAGE}_FOUND})
endmacro()

# find MPI
if(Cabana_REQUIRE_HDF5)
  # Workaround for using HDF5 C-bindings
  enable_language(C)
  Cabana_add_dependency( PACKAGE MPI COMPONENTS C CXX ) 
else()
  Cabana_add_dependency( PACKAGE MPI COMPONENTS CXX )
endif()
set_package_properties(MPI PROPERTIES TYPE RECOMMENDED PURPOSE "Used for distributed parallelization")

# find ArborX
Cabana_add_dependency( PACKAGE ArborX )
set_package_properties(ArborX PROPERTIES TYPE OPTIONAL PURPOSE "Used for neighbor search")

# find ALL
Cabana_add_dependency( PACKAGE ALL )
set_package_properties(ALL PROPERTIES TYPE OPTIONAL PURPOSE "Used for load balancing")

# find Clang Format
find_package( CLANG_FORMAT 14 )

# find hypre
Cabana_add_dependency( PACKAGE HYPRE VERSION 3.0.0 )
set_package_properties(HYPRE PROPERTIES TYPE OPTIONAL PURPOSE "Used for structured solves")

# find heffte
Cabana_add_dependency( PACKAGE Heffte VERSION 2.3.0 )
set_package_properties(Heffte PROPERTIES TYPE OPTIONAL PURPOSE "Used for fft calculations")
if(Heffte_FOUND)
  # ensure at least one host backend is enabled
  if(NOT Heffte_FFTW_FOUND AND NOT Heffte_MKL_FOUND)
    message(FATAL_ERROR "Cabana heFFTe support requires at least one host backend (FFTW or MKL).")
  endif()
endif()

# find Silo
Cabana_add_dependency( PACKAGE SILO )
set_package_properties(SILO PROPERTIES TYPE OPTIONAL PURPOSE "Used for I/O")
if(SILO_FOUND)
  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindSILO.cmake
    DESTINATION ${Cabana_INSTALL_PACKAGEDIR} )
endif()

# find HDF5 (XDMF)
# Set this to always get the parallel version in case both serial and parallel are installed.
# Parallel is required (see below).
set(HDF5_PREFER_PARALLEL TRUE)
# FIXME: not automatically finding since we need to enable C to use it (which
#        would break pure CXX projects downstream) unless CMake is new enough
if(CMAKE_VERSION VERSION_LESS 3.26)
  option(Cabana_REQUIRE_HDF5 "Require Cabana to build with HDF5 support" OFF)
  if(Cabana_REQUIRE_HDF5)
    # Workaround for using HDF5 C-bindings
    enable_language(C)
    find_package( HDF5 REQUIRED COMPONENTS C )
  endif()
  set(Cabana_ENABLE_HDF5 ${Cabana_REQUIRE_HDF5})
else()
  Cabana_add_dependency( PACKAGE HDF5 COMPONENTS C )
endif()
set_package_properties(HDF5 PROPERTIES TYPE OPTIONAL PURPOSE "Used for I/O")

if(Cabana_ENABLE_HDF5)
  if(NOT Cabana_ENABLE_MPI)
    message(FATAL_ERROR "Cabana HDF5 support requires MPI.")
  endif()

  include(CheckSymbolExists)
  LIST(APPEND CMAKE_REQUIRED_INCLUDES ${HDF5_INCLUDE_DIRS})
  check_symbol_exists(H5_HAVE_PARALLEL "H5pubconf.h" HDF5_IS_PARALLEL)
  if(NOT HDF5_IS_PARALLEL)
    set(Cabana_ENABLE_HDF5 OFF)
    message(WARNING "Cabana HDF5 support requires parallel HDF5.")
  endif()
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
add_feature_info(Tests Cabana_ENABLE_TESTING "Build unit tests (requires GTest)")

# Set installing tests to off by default
option(Cabana_INSTALL_TEST_EXECUTABLES "Install Cabana test executables" OFF)

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
option(Cabana_ENABLE_DOXYGEN "Build documentation" OFF)
add_feature_info(Documentation Cabana_ENABLE_DOXYGEN "Build documentation (requires Doxygen)")
if(Cabana_ENABLE_DOXYGEN)
  find_package(Doxygen REQUIRED)
  doxygen_add_docs(doxygen core/src grid/src)
endif()

##---------------------------------------------------------------------------##
## Libraries and Examples
##---------------------------------------------------------------------------##

add_subdirectory(core)

option(Cabana_ENABLE_GRID "Build grid and particle-grid capabilities" ${Cabana_ENABLE_MPI})
add_feature_info(Grid Cabana_ENABLE_GRID "Build grid and particle-grid capabilities (needs MPI)")
if(Cabana_ENABLE_GRID)
  if(MPI_FOUND)
    add_subdirectory(grid)
  else()
    message(FATAL_ERROR "Grid subpackage requires MPI")
  endif()
endif()

option(Cabana_ENABLE_EXAMPLES "Build tutorial examples" OFF)
add_feature_info(Examples Cabana_ENABLE_EXAMPLES "Build tutorial examples")
if(Cabana_ENABLE_EXAMPLES)
  add_subdirectory(example)
endif()

# enable performance tests
option(Cabana_ENABLE_PERFORMANCE_TESTING "Build Performance Tests" OFF)
add_feature_info(PerformanceTests Cabana_ENABLE_PERFORMANCE_TESTING "Build performance benchmarks and tests")
if(Cabana_ENABLE_PERFORMANCE_TESTING)
  add_subdirectory(benchmark)
endif()

##---------------------------------------------------------------------------##
## Package Configuration
##---------------------------------------------------------------------------##
write_basic_package_version_file("CabanaConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CabanaConfig.cmakein
  ${CMAKE_CURRENT_BINARY_DIR}/CabanaConfig.cmake @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CabanaConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/CabanaConfigVersion.cmake"
  DESTINATION ${Cabana_INSTALL_PACKAGEDIR})

##---------------------------------------------------------------------------##
## Clang Format
##---------------------------------------------------------------------------##
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES core/*.cpp core/*.hpp grid/*hpp grid/*cpp example/*cpp example/*hpp cmake/*cpp cmake/*hpp benchmark/*cpp benchmark/*hpp)
  add_custom_target(cabana-format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()

feature_summary(INCLUDE_QUIET_PACKAGES WHAT ALL)
