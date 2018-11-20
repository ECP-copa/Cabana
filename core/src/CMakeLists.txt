configure_file(CabanaCore_config.hpp.cmakein CabanaCore_config.hpp)

#-----------------------------------------------------------------------------

file(GLOB HEADERS_PUBLIC *.hpp)
file(GLOB SOURCES *.cpp)

#-----------------------------------------------------------------------------

file(GLOB HEADERS_IMPL impl/*.hpp)
file(GLOB SOURCES_IMPL impl/*.cpp)

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}) 
install(FILES ${HEADERS_IMPL} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/impl/)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CabanaCore_config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}) 

#-----------------------------------------------------------------------------

add_library(cabanacore ${SOURCES} ${SOURCES} ${SOURCES_IMPL})
target_include_directories(cabanacore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(cabanacore Kokkos::kokkos)
if(Cabana_ENABLE_MPI)
  target_link_libraries(cabanacore MPI::MPI_CXX)
endif()
install(TARGETS cabanacore LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
