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

add_subdirectory(01_types)
add_subdirectory(02_global_mesh)
add_subdirectory(03_partitioner)
add_subdirectory(04_global_grid)
add_subdirectory(05_index_space)
add_subdirectory(06_local_grid)
add_subdirectory(07_local_mesh)
add_subdirectory(08_array)
add_subdirectory(09_grid_parallel)
if(Cabana_ENABLE_HEFFTE)
    add_subdirectory(10_fft_heffte)
endif()
add_subdirectory(11_structured_solver)
if(Cabana_ENABLE_HYPRE AND (NOT Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_HIP AND NOT Kokkos_ENABLE_SYCL))
  add_subdirectory(11_structured_solver_hypre)
  add_subdirectory(11_semi_structured_solver_multi_variate)
endif()
add_subdirectory(12_halo)
if(Cabana_ENABLE_ALL)
  add_subdirectory(13_all_loadbalancer)
endif()
add_subdirectory(14_spline)
add_subdirectory(15_interpolation)
