# project settings
cmake_minimum_required(VERSION 3.12)

project(Cajita LANGUAGES CXX VERSION 0.1.0)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
find_package(MPI REQUIRED)
find_package(KOKKOS REQUIRED)
find_package(HYPRE REQUIRED)

# set the configure files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/Cajita_Config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/Cajita_Config.hpp)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CajitaSettings.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CajitaSettings.cmake
  @ONLY)

# obtain the repository hash if this is a git repo
add_custom_target(
  record_hash ALL VERBATIM
  COMMAND ${CMAKE_COMMAND}
    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DCAJITA_VERSION_STRING=${CAJITA_VERSION_STRING}
    -P cmake/SetupVersion.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# also run the command when configuring
execute_process(
  COMMAND ${CMAKE_COMMAND}
    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DCAJITA_VERSION_STRING=${CAJITA_VERSION_STRING}
    -P cmake/SetupVersion.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# create the library
file(GLOB SOURCES src/*cpp)

add_library(Cajita ${SOURCES})

target_link_libraries(Cajita
  Kokkos::Kokkos
  MPI::MPI_CXX
  HYPRE::hypre)

target_include_directories(Cajita
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  )

# installation
install(TARGETS Cajita
  EXPORT CajitaTargets
  ARCHIVE LIBRARY PUBLIC_HEADER
  )

install(EXPORT CajitaTargets
  NAMESPACE Cajita::
  DESTINATION lib/cmake/Cajita
  )

include(CMakePackageConfigHelpers)

configure_package_config_file(cmake/CajitaConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CajitaConfig.cmake
  INSTALL_DESTINATION lib/cmake/Cajita
  )

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/CajitaConfig.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindKokkos.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindHYPRE.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/CajitaSettings.cmake
  DESTINATION lib/cmake/Cajita )

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/ DESTINATION include
  FILES_MATCHING PATTERN "*.hpp")

install(DIRECTORY ${PROJECT_BINARY_DIR}/include/ DESTINATION include
  FILES_MATCHING PATTERN "*.hpp")

# add tests
option(Cajita_ENABLE_TESTING "Build tests" OFF)

if(Cajita_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(unit_test)
endif()
