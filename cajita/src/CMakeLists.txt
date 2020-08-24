configure_file(Cajita_Config.hpp.in Cajita_Config.hpp)

set(HEADERS_PUBLIC
  Cajita.hpp
  Cajita_Array.hpp
  Cajita_LocalGrid.hpp
  Cajita_BovWriter.hpp
  Cajita_GlobalGrid.hpp
  Cajita_GlobalMesh.hpp
  Cajita_Halo.hpp
  Cajita_IndexConversion.hpp
  Cajita_IndexSpace.hpp
  Cajita_SparseIndexSpace.hpp
  Cajita_Interpolation.hpp
  Cajita_LocalMesh.hpp
  Cajita_ManualPartitioner.hpp
  Cajita_MpiTraits.hpp
  Cajita_Parallel.hpp
  Cajita_ParameterPack.hpp
  Cajita_Partitioner.hpp
  Cajita_ReferenceStructuredSolver.hpp
  Cajita_Splines.hpp
  Cajita_Types.hpp
  Cajita_UniformDimPartitioner.hpp
  )

set(SOURCES_PUBLIC
  Cajita_LocalGrid.cpp
  Cajita_GlobalGrid.cpp
  Cajita_GlobalMesh.cpp
  Cajita_IndexSpace.cpp
  Cajita_ManualPartitioner.cpp
  Cajita_UniformDimPartitioner.cpp
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

add_library(Cajita ${SOURCES_PUBLIC})

target_link_libraries(Cajita PUBLIC
  Kokkos::kokkos
  MPI::MPI_CXX
  )

if(Cabana_ENABLE_HYPRE)
  target_link_libraries(Cajita PUBLIC HYPRE::hypre)
endif()

if(Cabana_ENABLE_HEFFTE)
  target_link_libraries(Cajita PUBLIC Heffte::Heffte)
endif()

target_include_directories(Cajita
  PUBLIC
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
