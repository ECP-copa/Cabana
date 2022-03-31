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
  add_executable(BinSortPerformance Cabana_BinSortPerformance.cpp)
  target_link_libraries(BinSortPerformance cabanacore)

  add_executable(NeighborVerletPerformance Cabana_NeighborVerletPerformance.cpp)
  target_link_libraries(NeighborVerletPerformance cabanacore)

  if(Cabana_ENABLE_ARBORX)
    add_executable(NeighborArborXPerformance Cabana_NeighborArborXPerformance.cpp)
    target_link_libraries(NeighborArborXPerformance cabanacore)
  endif()

  add_executable(LinkedCellPerformance Cabana_LinkedCellPerformance.cpp)
  target_link_libraries(LinkedCellPerformance cabanacore)

  if(Cabana_ENABLE_MPI)
    add_executable(CommPerformance Cabana_CommPerformance.cpp)
    target_link_libraries(CommPerformance cabanacore)
  endif()

  if(Cabana_ENABLE_TESTING)
    target_link_libraries(BinSortPerformance gtest)
    add_test(NAME Cabana_Performance_BinSort COMMAND ${NONMPI_PRECOMMAND} BinSortPerformance sort_output.txt)

    target_link_libraries(NeighborVerletPerformance gtest)
    add_test(NAME Cabana_Performance_NeighborVerlet COMMAND ${NONMPI_PRECOMMAND} NeighborVerletPerformance verlet_output.txt)

    if(Cabana_ENABLE_ARBORX)
      target_link_libraries(NeighborArborXPerformance gtest)
      add_test(NAME Cabana_Performance_NeighborArborX COMMAND ${NONMPI_PRECOMMAND} NeighborArborXPerformance arborx_output.txt)
    endif()

    target_link_libraries(LinkedCellPerformance gtest)
    add_test(NAME Cabana_Performance_LinkedCell COMMAND ${NONMPI_PRECOMMAND} LinkedCellPerformance lcl_output.txt)

    if(Cabana_ENABLE_MPI)
      target_link_libraries(CommPerformance gtest)
      add_test(NAME Cabana_Performance_Comm COMMAND ${NONMPI_PRECOMMAND} CommPerformance 1000 comm_output.txt)
    endif()
  endif()
endif()
