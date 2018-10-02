##---------------------------------------------------------------------------##
## Cabana/CMakeLists.txt
##---------------------------------------------------------------------------##
cmake_minimum_required(VERSION 3.1)

##---------------------------------------------------------------------------##
# Define your project name and set up major project options
##---------------------------------------------------------------------------##
project(Cabana LANGUAGES CXX VERSION 0.0.0)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(KOKKOS REQUIRED)

option(Cabana_ENABLE_Serial "Build Cabana with Serial support" ON)
find_package(OpenMP QUIET)
option(Cabana_ENABLE_OpenMP "Build Cabana with OpenMP support" ${OpenMP_FOUND})
option(Cabana_ENABLE_Cuda "Build Cabana with Cuda support" OFF)
if(NOT (Cabana_ENABLE_Serial OR Cabana_ENABLE_OpenMP OR Cabana_ENABLE_Cuda))
  message(FATAL_ERROR "NO Cabana backend enabled")
endif()

set(ENABLE_DEVICES)
foreach(_device Serial OpenMP Cuda)
  if(Cabana_ENABLE_${_device})
    if(NOT KOKKOS_GMAKE_DEVICES MATCHES ".*${_device}.*")
      message(FATAL_ERROR "Kokkos built with devices '${KOKKOS_GMAKE_DEVICES}', but Cabana with ${_device} requested.")
    endif()
    list(APPEND ENABLE_DEVICES "${_device}")
  endif()
endforeach()
message(STATUS "Enable Devices: ${ENABLE_DEVICES}")

if( Cabana_ENABLE_Cuda )
  enable_language( CUDA )
endif()

option(Cabana_ENABLE_TESTING "Build tests" OFF)
if(Cabana_ENABLE_TESTING)
  enable_testing()
endif()
##---------------------------------------------------------------------------##
## Print the revision number to stdout
##---------------------------------------------------------------------------##

FIND_PACKAGE(Git)
IF(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    EXECUTE_PROCESS(
        COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE   Cabana_GIT_COMMIT_HASH
        )
ELSE()
    SET(Cabana_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
MESSAGE(STATUS "Cabana Revision = '${Cabana_GIT_COMMIT_HASH}'")

add_subdirectory(core)
