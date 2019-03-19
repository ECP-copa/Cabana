# - Find fftw3
# Find the native FFTW3 headers and libraries.
#
#  FFTW3_INCLUDE_DIRS - where to find fftw3.h, etc.
#  FFTW3_LIBRARIES    - List of libraries when using fftw3.
#  FFTW3_FOUND        - True if fftw3 found.
#
# Copyright 2009-2011 The VOTCA Development Team (http://www.votca.org)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

find_package(PkgConfig)

pkg_check_modules(PC_FFTW3 fftw3)
find_path(FFTW3_INCLUDE_DIR fftw3.h HINTS ${PC_FFTW3_INCLUDE_DIRS})

find_library(FFTW3_LIBRARY NAMES fftw3 HINTS ${PC_FFTW3_LIBRARY_DIRS} )

set(FFTW3_LIBRARIES ${FFTW3_LIBRARY} )
set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set FFTW3_FOUND to TRUE
# if all listed variables are TRUE

find_package_handle_standard_args(FFTW3 DEFAULT_MSG FFTW3_LIBRARY FFTW3_INCLUDE_DIR )

if (FFTW3_FOUND)
  include(CheckLibraryExists)
  #adding MATH_LIBRARIES here to allow static libs, this does not harm us as we are anyway using it
  check_library_exists("${FFTW3_LIBRARIES};${MATH_LIBRARIES}" fftw_plan_r2r_1d "" FOUND_FFTW_PLAN)
  if(NOT FOUND_FFTW_PLAN)
    message(FATAL_ERROR "Could not find fftw_plan_r2r_1d in ${FFTW3_LIBRARY}, take a look at the error message in ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log to find out what was going wrong. If you are using a static lib (.a) make sure you have specified all dependencies of fftw3 in FFTW3_LIBRARY by hand (i.e. -DFFTW3_LIBRARY='/path/to/libfftw3.so;/path/to/libm.so') !")
  endif(NOT FOUND_FFTW_PLAN)
endif (FFTW3_FOUND)

mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_LIBRARY )
