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

configure_file(Cabana_Core_Config.hpp.cmakein Cabana_Core_Config.hpp)

set(HEADERS_PUBLIC
  Cabana_AoSoA.hpp
  Cabana_Core.hpp
  Cabana_DeepCopy.hpp
  Cabana_Fields.hpp
  Cabana_ExecutionPolicy.hpp
  Cabana_LinkedCellList.hpp
  Cabana_MemberTypes.hpp
  Cabana_NeighborList.hpp
  Cabana_Parallel.hpp
  Cabana_ParameterPack.hpp
  Cabana_ParticleInit.hpp
  Cabana_ParticleList.hpp
  Cabana_Remove.hpp
  Cabana_Slice.hpp
  Cabana_SoA.hpp
  Cabana_Sort.hpp
  Cabana_Tuple.hpp
  Cabana_Types.hpp
  Cabana_Utils.hpp
  Cabana_VerletList.hpp
  Cabana_Version.hpp
  )

if(Cabana_ENABLE_ARBORX)
  list(APPEND HEADERS_PUBLIC
    Cabana_Experimental_NeighborList.hpp
    )
endif()

if(Cabana_ENABLE_MPI)
  list(APPEND HEADERS_PUBLIC
    Cabana_CommunicationPlan.hpp
    Cabana_Halo.hpp
    Cabana_Migrate.hpp
    )
endif()

if(Cabana_ENABLE_SILO)
  list(APPEND HEADERS_PUBLIC Cabana_SiloParticleOutput.hpp)
endif()

if(Cabana_ENABLE_HDF5)
  list(APPEND HEADERS_PUBLIC Cabana_HDF5ParticleOutput.hpp)
endif()

set(HEADERS_IMPL
  impl/Cabana_CartesianGrid.hpp
  impl/Cabana_Index.hpp
  impl/Cabana_PerformanceTraits.hpp
  impl/Cabana_TypeTraits.hpp
  )

add_library(Core INTERFACE)
add_library(Cabana::Core ALIAS Core)

# Require minimum of C++17
set_target_properties(Core PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_17)

target_link_libraries(Core INTERFACE Kokkos::kokkos)

if(Cabana_ENABLE_ARBORX)
  target_link_libraries(Core INTERFACE ArborX::ArborX)
endif()

if(Cabana_ENABLE_SILO)
  target_link_libraries(Core INTERFACE SILO::silo)
endif()

if(Cabana_ENABLE_HDF5)
  # FIXME: remove when requiring newer CMake
  if(CMAKE_VERSION VERSION_LESS 3.26)
    target_link_libraries(Core INTERFACE ${HDF5_LIBRARIES})
  else()
    target_link_libraries(Core INTERFACE HDF5::HDF5)
  endif()
endif()

if(Cabana_ENABLE_MPI)
  target_link_libraries(Core INTERFACE MPI::MPI_CXX)
endif()

target_include_directories(Core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
if(Cabana_ENABLE_HDF5)
  target_include_directories(Core INTERFACE
    ${HDF5_INCLUDE_DIRS} # FIXME: remove when requiring newer CMake
  )
endif()

install(TARGETS Core
  EXPORT Cabana_Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT Cabana_Targets
  FILE Cabana_Targets.cmake
  NAMESPACE Cabana::
  DESTINATION ${Cabana_INSTALL_PACKAGEDIR})

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${HEADERS_IMPL} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/impl/)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cabana_Core_Config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
