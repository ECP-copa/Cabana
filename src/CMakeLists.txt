set(HEADERS_PUBLIC
  Cajita.hpp
  Cajita_Array.hpp
  Cajita_LocalGrid.hpp
  Cajita_BovWriter.hpp
  Cajita_GlobalGrid.hpp
  Cajita_GlobalMesh.hpp
  Cajita_Halo.hpp
  Cajita_IndexSpace.hpp
  Cajita_Interpolation.hpp
  Cajita_LocalMesh.hpp
  Cajita_ManualPartitioner.hpp
  Cajita_MpiTraits.hpp
  Cajita_Partitioner.hpp
  Cajita_PointSet.hpp
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

if(HYPRE_FOUND)
  list(APPEND HEADERS_PUBLIC
    Cajita_StructuredSolver.hpp
    )
endif()

add_library(Cajita ${SOURCES_PUBLIC})

target_link_libraries(Cajita
  Kokkos::Kokkos
  MPI::MPI_CXX
  )

if(HYPRE_FOUND)
  target_link_libraries(Cajita HYPRE::hypre)
endif()

install(FILES ${HEADERS_PUBLIC}
  DESTINATION include)

target_include_directories(Cajita
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )

install(TARGETS Cajita
  EXPORT CajitaTargets
  ARCHIVE LIBRARY PUBLIC_HEADER
  ARCHIVE DESTINATION lib
  )

install(EXPORT CajitaTargets
  NAMESPACE Cajita::
  DESTINATION lib/cmake/Cajita
  )
