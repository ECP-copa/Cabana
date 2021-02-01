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

set(NOBACKEND_TESTS
  Index
  CartesianGrid
  SoA
  )

set(SERIAL_TESTS
  AoSoA
  Slice
  DeepCopy
  Tuple
  Sort
  LinkedCellList
  NeighborList
  Parallel
  )

if(Cabana_ENABLE_ARBORX)
 list(APPEND SERIAL_TESTS NeighborListArborX)
endif()

set(MPI_TESTS
  CommunicationPlan
  Distributor
  Halo
  )

Cabana_add_tests_nobackend(PACKAGE cabanacore NAME ${NOBACKEND_TESTS})

Cabana_add_tests(PACKAGE cabanacore NAMES ${SERIAL_TESTS})

if(Cabana_ENABLE_MPI)
  Cabana_add_tests(MPI PACKAGE cabanacore NAMES ${MPI_TESTS})
endif()
