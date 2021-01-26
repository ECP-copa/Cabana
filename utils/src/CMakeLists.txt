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

configure_file(Cabana_config.hpp.cmakein Cabana_config.hpp)

set(HEADERS_PUBLIC
  Cabana_Utils.hpp
  Cabana_ParameterPack.hpp
  Cabana_Version.hpp
  )
set(SOURCES_IMPL
  impl/Cabana_Version.cpp
  )

add_library(utils ${SOURCES_IMPL})

target_link_libraries(utils Kokkos::kokkos)

target_include_directories(utils
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS utils
  EXPORT CabanaUtilsTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT CabanaUtilsTargets
  FILE CabanaUtilsTargets.cmake
  NAMESPACE Cabana::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cabana)

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cabana_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)

add_library(Cabana::utils ALIAS utils)
