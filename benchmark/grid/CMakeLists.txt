############################################################################
# Copyright (c) 2018-2023 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

if(NOT Kokkos_ENABLE_SYCL) #FIXME_SYCL
  add_executable(SparseMapPerformance Cabana_Grid_SparseMapPerformance.cpp)
  target_link_libraries(SparseMapPerformance Cabana::Grid)

  add_executable(SparsePartitionerPerformance Cabana_Grid_SparsePartitionerPerformance.cpp)
  target_link_libraries(SparsePartitionerPerformance Cabana::Grid)
endif()

add_executable(HaloPerformance Cabana_Grid_HaloPerformance.cpp)
target_link_libraries(HaloPerformance Cabana::Grid)

add_executable(InterpolationPerformance Cabana_Grid_InterpolationPerformance.cpp)
target_link_libraries(InterpolationPerformance Cabana::Grid)

if(Cabana_ENABLE_HEFFTE)
  add_executable(FastFourierTransformPerformance Cabana_Grid_FastFourierTransformPerformance.cpp)
  target_link_libraries(FastFourierTransformPerformance Cabana::Grid)
endif()

if(Cabana_ENABLE_TESTING)
  if(NOT Kokkos_ENABLE_SYCL) #FIXME_SYCL
    add_test(NAME Cabana_Grid_Performance_SparseMap COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:SparseMapPerformance> sparsemap_output.txt)
    add_test(NAME Cabana_Grid_Performance_SparsePartitioner COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:SparsePartitionerPerformance> sparsepartitioner_output.txt)
  endif()

  add_test(NAME Cabana_Grid_Performance_Halo COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:HaloPerformance> halo_output.txt)

  add_test(NAME Cabana_Grid_Performance_Interpolation COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:InterpolationPerformance> interpolation_output.txt)

  if (Cabana_ENABLE_HEFFTE)
    add_test(NAME Cabana_Grid_Performance_FastFourierTransform COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:FastFourierTransformPerformance> fastfouriertransform_output.txt)
  endif()
endif()
