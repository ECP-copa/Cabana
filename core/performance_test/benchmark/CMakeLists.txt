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

if(Kokkos_ENABLE_SERIAL OR Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_OPENMP)
  add_executable(BinSortPerformance Cabana_BinSortPerformance.cpp)
  target_link_libraries(BinSortPerformance cabanacore)

  add_executable(NeighborPerformance Cabana_NeighborPerformance.cpp)
  target_link_libraries(NeighborPerformance cabanacore)
endif()

if(Cabana_ENABLE_MPI AND Kokkos_ENABLE_CUDA AND Kokkos_ENABLE_OPENMP)
  add_executable(CommPerformance Cabana_CommPerformance.cpp)
  target_link_libraries(CommPerformance cabanacore)
endif()
