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

add_subdirectory(01_hello_world)
add_subdirectory(02_tuple)
add_subdirectory(03_struct_of_arrays)
add_subdirectory(04_aosoa_advanced_unmanaged)
add_subdirectory(04_aosoa)
add_subdirectory(05_slice)
add_subdirectory(06_deep_copy)
add_subdirectory(07_sorting)
add_subdirectory(08_linked_cell_list)
add_subdirectory(09_neighbor_list)
if(Cabana_ENABLE_ARBORX)
  add_subdirectory(09_neighbor_list_arborx)
endif()
add_subdirectory(10_neighbor_parallel_for)
add_subdirectory(10_simd_parallel_for)
if(Cabana_ENABLE_MPI)
  add_subdirectory(11_migration)
  add_subdirectory(12_halo_exchange)
endif()

if(Kokkos_ENABLE_OPENMP)
  add_subdirectory(05_slice_advanced_openmp)
endif()
if(Kokkos_ENABLE_CUDA)
  add_subdirectory(05_slice_advanced_cuda)
endif()
if(Cabana_ENABLE_HDF5)
  add_subdirectory(13_hdf5_output)
endif()
add_subdirectory(14_gmm)
