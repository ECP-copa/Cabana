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

find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(Cabana_ENABLE_OpenMP ON)
endif()

option(Cabana_ENABLE_Serial "Build Cabana serial" ON)
option(Cabana_ENABLE_Cuda "Build Cabana with Cuda" OFF)
if(Cabana_ENABLE_Cuda)
  enable_language(CUDA)
else()
  set(CUDA_LIBRARIES)
endif()

option(ENABLE_TESTING "Build tests" OFF)
if(ENABLE_TESTING)
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
