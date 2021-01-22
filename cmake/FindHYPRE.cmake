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

find_package(PkgConfig QUIET)
pkg_check_modules(PC_HYPRE QUIET hypre)

find_path(HYPRE_INCLUDE_DIR
  NAMES HYPRE.h
  PATHS ${PC_HYPRE_INCLUDE_DIRS})
find_library(HYPRE_LIBRARY
  NAMES HYPRE
  PATHS ${PC_HYPRE_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(HYPRE
  FOUND_VAR HYPRE_FOUND
  REQUIRED_VARS
  HYPRE_LIBRARY
  HYPRE_INCLUDE_DIR
  VERSION_VAR HYPRE_VERSION
  )

if(HYPRE_FOUND AND HYPRE_INCLUDE_DIR AND HYPRE_LIBRARY)
  if(NOT TARGET HYPRE::hypre)
    add_library(HYPRE::hypre UNKNOWN IMPORTED)
    set_target_properties(HYPRE::hypre PROPERTIES
      IMPORTED_LOCATION ${HYPRE_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${HYPRE_INCLUDE_DIR})
  endif()
endif()

mark_as_advanced(
  HYPRE_INCLUDE_DIR
  HYPRE_LIBRARY )
