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

if(Kokkos_ENABLE_SERIAL OR Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_OPENMP OR Kokkos_ENABLE_HIP)
    add_executable(SparseMapPerformance Cajita_SparseMapPerformance.cpp)
    target_link_libraries(SparseMapPerformance Cajita)

    add_executable(SparsePartitionerPerformance Cajita_SparsePartitionerPerformance.cpp)
    target_link_libraries(SparsePartitionerPerformance Cajita)

    if(Cabana_ENABLE_HEFFTE)
      add_executable(FastFourierTransformPerformance Cajita_FastFourierTransformPerformance.cpp)
      target_link_libraries(FastFourierTransformPerformance Cajita)
    endif()

  if(Cabana_ENABLE_TESTING)
    target_link_libraries(SparseMapPerformance gtest)
    add_test(NAME Cajita_SparseMapPerformance COMMAND ${NONMPI_PRECOMMAND} SparseMapPerformance sparsemap_output.txt)

    target_link_libraries(SparsePartitionerPerformance gtest)
    add_test(NAME Cajita_SparsePartitionerPerformance COMMAND ${NONMPI_PRECOMMAND} SparsePartitionerPerformance sparsepartitioner_output.txt)

    if (Cabana_ENABLE_HEFFTE)
      target_link_libraries(FastFourierTransformPerformance gtest)
      add_test(NAME Cajita_FastFourierTransformPerformance COMMAND ${NONMPI_PRECOMMAND} FastFourierTransformPerformance fastfouriertransform_output.txt)
    endif()
  endif()
endif()
