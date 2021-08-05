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
#
# Find clang-format
#
# CLANG_FORMAT_EXECUTABLE   - Path to clang-format executable
# CLANG_FORMAT_FOUND        - True if the clang-format executable was found.
# CLANG_FORMAT_VERSION      - The version of clang-format found
#

find_program(CLANG_FORMAT_EXECUTABLE
             NAMES clang-format
                   clang-format-10
                   clang-format-9
                   clang-format-8
                   clang-format-7
                   clang-format-6.0
                   clang-format-5.0
                   clang-format-4.0
                   clang-format-3.9
                   clang-format-3.8
                   clang-format-3.7
                   clang-format-3.6
                   clang-format-3.5
                   clang-format-3.4
                   clang-format-3.3
             DOC "clang-format executable")
mark_as_advanced(CLANG_FORMAT_EXECUTABLE)

# Extract version from command "clang-format -version"
if(CLANG_FORMAT_EXECUTABLE)
  execute_process(COMMAND ${CLANG_FORMAT_EXECUTABLE} -version
                  OUTPUT_VARIABLE clang_format_version
                  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(clang_format_version MATCHES "^.*clang-format version .*")
    # clang_format_version sample: "clang-format version 3.9.1-4ubuntu3~16.04.1
    # (tags/RELEASE_391/rc2)"
    string(REGEX
           REPLACE "^.*clang-format version ([.0-9]+).*"
                   "\\1"
                   CLANG_FORMAT_VERSION
                   "${clang_format_version}")
    # CLANG_FORMAT_VERSION sample: "3.9.1"
  else()
    set(CLANG_FORMAT_VERSION 0.0)
  endif()
else()
  set(CLANG_FORMAT_VERSION 0.0)
endif()

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set CLANG_FORMAT_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(CLANG_FORMAT REQUIRED_VARS CLANG_FORMAT_EXECUTABLE VERSION_VAR CLANG_FORMAT_VERSION)
