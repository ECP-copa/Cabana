############################################################################
# Copyright (c) 2018-2022 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

configure_file(Cajita_Config.hpp.in Cajita_Config.hpp)

set(HEADERS_PUBLIC
  Cajita.hpp
  Cajita_Array.hpp
  Cajita_BovWriter.hpp
  Cajita_GlobalGrid.hpp
  Cajita_GlobalGrid_impl.hpp
  Cajita_GlobalMesh.hpp
  Cajita_Halo.hpp
  Cajita_IndexConversion.hpp
  Cajita_IndexSpace.hpp
  Cajita_Interpolation.hpp
  Cajita_LocalGrid.hpp
  Cajita_LocalGrid_impl.hpp
  Cajita_LocalMesh.hpp
  Cajita_ManualPartitioner.hpp
  Cajita_MpiTraits.hpp
  Cajita_Parallel.hpp
  Cajita_ParticleGridDistributor.hpp
  Cajita_Partitioner.hpp
  Cajita_ReferenceStructuredSolver.hpp
  Cajita_SparseArray.hpp
  Cajita_SparseDimPartitioner.hpp
  Cajita_SparseIndexSpace.hpp
  Cajita_SparseLocalGrid.hpp
  Cajita_SparseLocalGrid_impl.hpp
  Cajita_Splines.hpp
  Cajita_Types.hpp
  Cajita_UniformDimPartitioner.hpp
  )

if(Cabana_ENABLE_HYPRE)
  list(APPEND HEADERS_PUBLIC
    Cajita_HypreStructuredSolver.hpp
    )
endif()

if(Cabana_ENABLE_HEFFTE)
  list(APPEND HEADERS_PUBLIC
    Cajita_FastFourierTransform.hpp
    )
endif()

if(Cabana_ENABLE_ALL)
  list(APPEND HEADERS_PUBLIC
    Cajita_LoadBalancer.hpp
    )
endif()

if(Cabana_ENABLE_SILO)
  list(APPEND HEADERS_PUBLIC Cajita_SiloParticleOutput.hpp)
endif()

add_library(Cajita INTERFACE)
add_library(Cabana::Cajita ALIAS Cajita)

target_link_libraries(Cajita INTERFACE
  Cabana::cabanacore
  Kokkos::kokkos
  MPI::MPI_CXX
  )

if(Cabana_ENABLE_HYPRE)
  target_link_libraries(Cajita INTERFACE HYPRE::HYPRE)
endif()

if(Cabana_ENABLE_HEFFTE)
  target_link_libraries(Cajita INTERFACE Heffte::Heffte)
endif()

if(Cabana_ENABLE_ALL)
    target_link_libraries(Cajita INTERFACE ALL::ALL)
endif()

target_include_directories(Cajita
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS Cajita
  EXPORT CajitaTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT CajitaTargets
  FILE CajitaTargets.cmake
  NAMESPACE Cabana::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cabana)

install(FILES ${HEADERS_PUBLIC}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cajita_Config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
