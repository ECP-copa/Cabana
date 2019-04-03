############################################################################
# Copyright (c) 2018-2019 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

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
