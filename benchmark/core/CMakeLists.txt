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

add_executable(BinSortPerformance Cabana_BinSortPerformance.cpp)
target_link_libraries(BinSortPerformance Cabana::Core)

add_executable(NeighborVerletPerformance Cabana_NeighborVerletPerformance.cpp)
target_link_libraries(NeighborVerletPerformance Cabana::Core)

if(Cabana_ENABLE_ARBORX)
add_executable(NeighborArborXPerformance Cabana_NeighborArborXPerformance.cpp)
target_link_libraries(NeighborArborXPerformance Cabana::Core)
endif()

add_executable(LinkedCellPerformance Cabana_LinkedCellPerformance.cpp)
target_link_libraries(LinkedCellPerformance Cabana::Core)

if(Cabana_ENABLE_MPI)
add_executable(CommPerformance Cabana_CommPerformance.cpp)
target_link_libraries(CommPerformance Cabana::Core)
endif()

if(Cabana_ENABLE_TESTING)
  add_test(NAME Cabana_Performance_BinSort COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:BinSortPerformance> sort_output.txt)

  add_test(NAME Cabana_Performance_NeighborVerlet COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:NeighborVerletPerformance> verlet_output.txt)

  if(Cabana_ENABLE_ARBORX)
    add_test(NAME Cabana_Performance_NeighborArborX COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:NeighborArborXPerformance> arborx_output.txt)
  endif()

  add_test(NAME Cabana_Performance_LinkedCell COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:LinkedCellPerformance> lcl_output.txt)

  if(Cabana_ENABLE_MPI)
    add_test(NAME Cabana_Performance_Comm COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:CommPerformance> comm_output.txt)
  endif()
endif()
