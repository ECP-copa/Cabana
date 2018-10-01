configure_file(${PROJECT_NAME}Core_config.hpp.cmakein ${PROJECT_NAME}Core_config.hpp)

set(CABANA_INCDIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

#-----------------------------------------------------------------------------

file(GLOB HEADERS_PUBLIC *.hpp)
file(GLOB SOURCES *.cpp)

#-----------------------------------------------------------------------------

file(GLOB HEADERS_IMPL impl/*.hpp)
file(GLOB SOURCES_IMPL impl/*.cpp)

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CABANA_INCDIR}) 
INSTALL(FILES ${HEADERS_IMPL} DESTINATION ${CABANA_INCDIR}/impl/)

#-----------------------------------------------------------------------------

add_library(cabanacore ${SOURCES} ${SOURCES} ${SOURCES_IMPL})
target_include_directories(cabanacore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(cabanacore Kokkos::kokkos ${CUDA_LIBRARIES})
