# project settings
cmake_minimum_required(VERSION 3.12)

project(Cajita LANGUAGES CXX VERSION 0.1.0)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download" )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                 "${CMAKE_BINARY_DIR}/googletest-build")

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
find_package(MPI REQUIRED)
find_package(KOKKOS REQUIRED)
find_package(HYPRE)
if(HYPRE_FOUND)
  set(CAJITA_HAVE_HYPRE ON)
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
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindKOKKOS.cmake
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
