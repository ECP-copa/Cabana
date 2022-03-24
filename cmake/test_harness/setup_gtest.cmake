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

find_package(GTest 1.10)

if(NOT GTest_FOUND)
  ##---------------------------------------------------------------------------##
  # Download and unpack googletest
  ##---------------------------------------------------------------------------##
  set(GTEST_URL "https://github.com/google/googletest/archive/release-1.10.0.tar.gz" CACHE STRING "URL for GTest tarball")
  mark_as_advanced(GTEST_URL)

  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL ${GTEST_URL}
    URL_MD5         ecd1fa65e7de707cd5c00bdac56022cd
    )

  # suppress all compiler warnings when building gtest
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # NOTE I haven't actually tested on MSVC
    string(APPEND CMAKE_CXX_FLAGS " \w")
  else()
    string(APPEND CMAKE_CXX_FLAGS " -w")
  endif()
  
  # do not install gtest
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

  FetchContent_GetProperties(googletest)
  if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  endif()

  # Prevent GoogleTest from overriding our compiler/linker options
  # when building with Visual Studio
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  add_library(GTest::gtest ALIAS gtest)

endif()
