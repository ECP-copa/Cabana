configure_file(CabanaCore_config.hpp.cmakein CabanaCore_config.hpp)

#-----------------------------------------------------------------------------

set(HEADERS_PUBLIC
  Cabana_AoSoA.hpp
  Cabana_Core.hpp
  Cabana_DeepCopy.hpp
  Cabana_ExecutionPolicy.hpp
  Cabana_LinkedCellList.hpp
  Cabana_Macros.hpp
  Cabana_MemberTypes.hpp
  Cabana_NeighborList.hpp
  Cabana_Parallel.hpp
  Cabana_Slice.hpp
  Cabana_SoA.hpp
  Cabana_Sort.hpp
  Cabana_Tuple.hpp
  Cabana_Types.hpp
  Cabana_VerletList.hpp
  Cabana_Version.hpp
  )

#-----------------------------------------------------------------------------
# MPI-dependent code
if(Cabana_ENABLE_MPI)
  list(APPEND HEADERS_PUBLIC
    Cabana_CommunicationPlan.hpp
    Cabana_Distributor.hpp
    Cabana_Halo.hpp
    )
endif()

#-----------------------------------------------------------------------------
# implementation details
set(HEADERS_IMPL
  impl/Cabana_CartesianGrid.hpp
  impl/Cabana_Index.hpp
  impl/Cabana_IndexSequence.hpp
  impl/Cabana_PerformanceTraits.hpp
  impl/Cabana_TypeTraits.hpp
  )
set(SOURCES_IMPL
  impl/Cabana_Version.cpp
  )

#-----------------------------------------------------------------------------

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${HEADERS_IMPL} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/impl/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CabanaCore_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#-----------------------------------------------------------------------------

add_library(cabanacore ${SOURCES_IMPL})
target_include_directories(cabanacore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(cabanacore Kokkos::kokkos)
if(Cabana_ENABLE_MPI)
  target_link_libraries(cabanacore MPI::MPI_CXX)
endif()
install(TARGETS cabanacore LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
