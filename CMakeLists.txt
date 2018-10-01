##---------------------------------------------------------------------------##
## Cabana/CMakeLists.txt
##---------------------------------------------------------------------------##
cmake_minimum_required(VERSION 3.1)

##---------------------------------------------------------------------------##
# Define your project name and set up major project options
##---------------------------------------------------------------------------##
project(Cabana LANGUAGES CXX Fortran VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(KOKKOS REQUIRED)

set(Cabana_DEVICE "OpenMP" CACHE STRING "Cabana Device backend")
set_property(CACHE Cabana_DEVICE PROPERTY STRINGS Serial OpenMP Cuda)

if(NOT KOKKOS_GMAKE_DEVICES MATCHES ".*${Cabana_DEVICE}.*")
  message(FATAL_ERROR "Kokkos with devices '${KOKKOS_GMAKE_DEVICES}', but Cabana_DEVICE is ${Cabana_DEVICE}")
endif()

set(CUDA_LIBRARIES)
if(Cabana_DEVICE STREQUAL "Serial")
  set(Cabana_ENABLE_Serial ON)
elseif(Cabana_DEVICE STREQUAL "OpenMP")
  find_package(OpenMP REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(Cabana_ENABLE_OpenMP ON)
elseif(Cabana_DEVICE STREQUAL "Cuda")
  find_package(CUDA REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda")
  set(Cabana_ENABLE_Cuda ON)
else()
  message(FATAL_ERROR "Unknown Cabana Device; ${Cabana_DEVICE}")
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
