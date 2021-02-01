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

include(${CMAKE_SOURCE_DIR}/utils/test_harness/test_harness.cmake)

set(SERIAL_TESTS
  GlobalMesh
  IndexSpace
  SparseIndexSpace
  Splines
  )

if(Kokkos_ENABLE_OPENMPTARGET) #FIXME_OPENMPTARGET
  list(REMOVE_ITEM SERIAL_TESTS Parallel)
endif()

set(MPI_TESTS
  GlobalGrid
  LocalGrid
  IndexConversion
  LocalMesh
  Array
  Halo
  SplineEvaluation
  Interpolation
  Parallel
# BovWriter
  )

if(Cabana_ENABLE_HYPRE)
  list(APPEND MPI_TESTS
    HypreStructuredSolver
    )
endif()

if(Cabana_ENABLE_HEFFTE)
    list(APPEND MPI_TESTS
      FastFourierTransform
      )
endif()

Cabana_add_tests(PACKAGE Cajita NAMES ${SERIAL_TESTS})

Cabana_add_tests(MPI PACKAGE Cajita NAMES ${MPI_TESTS})
