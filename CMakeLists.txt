# project settings
cmake_minimum_required(VERSION 3.12)

project(Cajita LANGUAGES CXX VERSION 0.1.0)

# Download and unpack googletest at configure time
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

find_package(MPI REQUIRED)

find_package(Kokkos REQUIRED)

find_package(HYPRE)
if(HYPRE_FOUND)
  set(CAJITA_HAVE_HYPRE ON)
endif()

find_package(HEFFTE)
if(HEFFTE_FOUND)
  set(CAJITA_HAVE_HEFFTE ON)
endif()

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

# add code coverage build
option(Cajita_ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(Cajita_ENABLE_COVERAGE_BUILD)
      message(STATUS "Enabling coverage build")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -O0")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

# create the library
add_subdirectory(src)

# installation configuration files
include(CMakePackageConfigHelpers)

configure_package_config_file(cmake/CajitaConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CajitaConfig.cmake
  INSTALL_DESTINATION lib/cmake/Cajita
  )

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/CajitaConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/CajitaSettings.cmake
  DESTINATION lib/cmake/Cajita )

if(HYPRE_FOUND)
  install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindHYPRE.cmake
    DESTINATION lib/cmake/Cajita )
endif()

install(DIRECTORY ${PROJECT_BINARY_DIR}/include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp")

# add tests
option(Cajita_ENABLE_TESTING "Build tests" OFF)

if(Cajita_ENABLE_TESTING)
  enable_testing()
  add_subdirectory(unit_test)
endif()

# clang format
find_package(CLANG_FORMAT)
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
