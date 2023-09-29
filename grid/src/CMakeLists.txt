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

configure_file(Cabana_Grid_Config.hpp.in Cabana_Grid_Config.hpp)

set(HEADERS_PUBLIC
  Cabana_Grid.hpp
  Cajita.hpp # FIXME: remove in next release
  Cabana_Grid_Array.hpp
  Cabana_Grid_BovWriter.hpp
  Cabana_Grid_GlobalGrid.hpp
  Cabana_Grid_GlobalGrid_impl.hpp
  Cabana_Grid_GlobalMesh.hpp
  Cabana_Grid_GlobalParticleComm.hpp
  Cabana_Grid_Halo.hpp
  Cabana_Grid_IndexConversion.hpp
  Cabana_Grid_IndexSpace.hpp
  Cabana_Grid_Interpolation.hpp
  Cabana_Grid_LocalGrid.hpp
  Cabana_Grid_LocalGrid_impl.hpp
  Cabana_Grid_LocalMesh.hpp
  Cabana_Grid_MpiTraits.hpp
  Cabana_Grid_Parallel.hpp
  Cabana_Grid_ParticleGridDistributor.hpp
  Cabana_Grid_ParticleInit.hpp
  Cabana_Grid_ParticleList.hpp
  Cabana_Grid_Partitioner.hpp
  Cabana_Grid_ReferenceStructuredSolver.hpp
  Cabana_Grid_SparseArray.hpp
  Cabana_Grid_SparseDimPartitioner.hpp
  Cabana_Grid_SparseHalo.hpp
  Cabana_Grid_SparseIndexSpace.hpp
  Cabana_Grid_SparseLocalGrid.hpp
  Cabana_Grid_SparseLocalGrid_impl.hpp
  Cabana_Grid_Splines.hpp
  Cabana_Grid_Types.hpp
  )

if(Cabana_ENABLE_HYPRE)
  list(APPEND HEADERS_PUBLIC
    Cabana_Grid_Hypre.hpp
    Cabana_Grid_HypreStructuredSolver.hpp
    Cabana_Grid_HypreSemiStructuredSolver.hpp
    )
endif()

if(Cabana_ENABLE_HEFFTE)
  list(APPEND HEADERS_PUBLIC
    Cabana_Grid_FastFourierTransform.hpp
    )
endif()

if(Cabana_ENABLE_ALL)
  list(APPEND HEADERS_PUBLIC
    Cabana_Grid_LoadBalancer.hpp
    )
endif()

if(Cabana_ENABLE_SILO)
  list(APPEND HEADERS_PUBLIC Cabana_Grid_SiloParticleOutput.hpp)
endif()

add_library(Grid INTERFACE)
add_library(Cabana::Grid ALIAS Grid)

### FIXME: remove Cajita in next release.
add_library(Cajita INTERFACE)
target_link_libraries(Cajita INTERFACE Grid)
if(CMAKE_VERSION VERSION_GREATER 3.17)
  if(NOT Cabana_DISABLE_CAJITA_DEPRECATION_WARNINGS)
    set_property(TARGET Cajita PROPERTY DEPRECATION "NOTE: Cajita subpackage is now Grid. Link to Cabana::Grid. Cabana::Cajita will be removed in a future release.")
  endif()
endif()

set_target_properties(Grid PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_17)

target_link_libraries(Grid INTERFACE
  Cabana::Core
  Kokkos::kokkos
  MPI::MPI_CXX
  )

if(Cabana_ENABLE_HYPRE)
  target_link_libraries(Grid INTERFACE HYPRE::HYPRE)
endif()

if(Cabana_ENABLE_HEFFTE)
  target_link_libraries(Grid INTERFACE Heffte::Heffte)
endif()

if(Cabana_ENABLE_ALL)
  target_link_libraries(Grid INTERFACE ALL::ALL)
endif()

target_include_directories(Grid
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS Grid Cajita
  EXPORT Cabana_Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES ${HEADERS_PUBLIC}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cabana_Grid_Config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
