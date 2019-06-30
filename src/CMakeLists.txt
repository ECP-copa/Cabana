configure_file(Cajita_config.hpp.cmakein Cajita_config.hpp)

#-----------------------------------------------------------------------------

set(HEADERS
  Cajita.hpp
  Cajita_Array.hpp
  Cajita_Block.hpp
  Cajita_BovWriter.hpp
  Cajita_Domain.hpp
  Cajita_GlobalGrid.hpp
  Cajita_Halo.hpp
  Cajita_IndexSpace.hpp
  Cajita_ManualPartitioner.hpp
  Cajita_MpiTraits.hpp
  Cajita_Partitioner.hpp
  Cajita_StructuredSolver.hpp
  Cajita_Types.hpp
  Cajita_UniformDimPartitioner.hpp
  Cajita_Version.hpp
  )

set(SOURCES
  Cajita_Block.cpp
  Cajita_Domain.cpp
  Cajita_GlobalGrid.cpp
  Cajita_IndexSpace.cpp
  Cajita_ManualPartitioner.cpp
  Cajita_UniformDimPartitioner.cpp
  Cajita_Version.cpp
  )

#-----------------------------------------------------------------------------

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cajita_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#-----------------------------------------------------------------------------

add_library(cajita ${SOURCES})
target_include_directories(cajita PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(cajita Kokkos::kokkos)
target_link_libraries(cajita MPI::MPI_CXX)
target_link_libraries(cajita HYPRE::hypre)
install(TARGETS cajita LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

#-----------------------------------------------------------------------------

add_subdirectory(unit_test)
