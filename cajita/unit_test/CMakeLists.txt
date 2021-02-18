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

include(${TEST_HARNESS_DIR}/test_harness.cmake)

set(SERIAL_TESTS
  GlobalMesh
  IndexSpace
  SparseIndexSpace
  Splines
  )

set(MPI_TESTS
  GlobalGrid
  LocalGrid
  IndexConversion
  LocalMesh3d
  LocalMesh2d
  Array3d
  Array2d
  Halo3d
  Halo2d
  SplineEvaluation3d
  SplineEvaluation2d
  Interpolation3d
  Interpolation2d
  BovWriter
  Parallel
  )

if(Kokkos_ENABLE_OPENMPTARGET) #FIXME_OPENMPTARGET
  list(REMOVE_ITEM MPI_TESTS Parallel)
endif()

if(Cabana_ENABLE_HYPRE)
  list(APPEND MPI_TESTS
    HypreStructuredSolver3d
    HypreStructuredSolver2d
    )
endif()

if(Cabana_ENABLE_HEFFTE)
  list(APPEND MPI_TESTS
    FastFourierTransform
    )
endif()

Cabana_add_tests(PACKAGE Cajita NAMES ${SERIAL_TESTS})

Cabana_add_tests(MPI PACKAGE Cajita NAMES ${MPI_TESTS})
