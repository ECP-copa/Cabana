##---------------------------------------------------------------------------##
## Cabana/CMakeLists.txt
##---------------------------------------------------------------------------##
cmake_minimum_required(VERSION 3.9)

##---------------------------------------------------------------------------##
# Define your project name and set up major project options
##---------------------------------------------------------------------------##
project(Cabana LANGUAGES CXX VERSION 0.1.0)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11

option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)
include(GNUInstallDirs)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(KOKKOS REQUIRED)

option(Cabana_ENABLE_Serial "Build Cabana with Serial support" ON)

option(Cabana_ENABLE_Pthread "Build Cabana with Pthread support" OFF)
if( Cabana_ENABLE_Pthread )
  find_package(Threads)
endif()

option(Cabana_ENABLE_OpenMP "Build Cabana with OpenMP support" OFF)
if( Cabana_ENABLE_OpenMP )
  find_package(OpenMP)
endif()

option(Cabana_ENABLE_Cuda "Build Cabana with Cuda support" OFF)
if( Cabana_ENABLE_Cuda )
  enable_language( CUDA )
endif()

set(CABANA_SUPPORTED_DEVICES Serial Pthread OpenMP Cuda)
set(ENABLE_DEVICES)
foreach(_device ${CABANA_SUPPORTED_DEVICES})
  if(Cabana_ENABLE_${_device})
    if(NOT KOKKOS_GMAKE_DEVICES MATCHES ".*${_device}.*")
      message(FATAL_ERROR "Kokkos built with devices '${KOKKOS_GMAKE_DEVICES}', but Cabana with ${_device} requested.")
    endif()
    list(APPEND ENABLE_DEVICES "${_device}")
  endif()
endforeach()
if(NOT ENABLE_DEVICES)
  message(FATAL_ERROR "NO Cabana backend enabled")
endif()
message(STATUS "Enable Devices: ${ENABLE_DEVICES}")

option(Cabana_ENABLE_TESTING "Build tests" OFF)
if(Cabana_ENABLE_TESTING)
  enable_testing()
endif()

find_package(Doxygen)
if(Doxygen_FOUND)
  doxygen_add_docs(doxygen core)
endif()

#------------------------------------------------------------------------------#
# Architecture
#------------------------------------------------------------------------------#
if(BUILD_MARCH AND BUILD_MCPU)
  message(FATAL_ERROR "BUILD_MARCH and BUILD_MCPU are incompatible")
endif()

# Try -march first. On platforms that don't support it, GCC will issue
# a hard error, so we'll know not to use it.
if(BUILD_MARCH)
  set(INTERNAL_BUILD_MARCH ${BUILD_MARCH})
else()
  set(INTERNAL_BUILD_MARCH "native")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=${INTERNAL_BUILD_MARCH}" COMPILER_SUPPORTS_MARCH)
if(COMPILER_SUPPORTS_MARCH)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${INTERNAL_BUILD_MARCH}")
elseif(BUILD_MARCH)
  message(FATAL_ERROR "The flag -march=${INTERNAL_BUILD_MARCH} is not supported by the compiler")
else()
  unset(INTERNAL_BUILD_MARCH)
endif()

# Try -mcpu. We do this second because it is deprecated on x86, but
# GCC won't issue a hard error, so we can't tell if it worked or not.
if (NOT INTERNAL_BUILD_MARCH)
  if(BUILD_MCPU)
    set(INTERNAL_BUILD_MCPU ${BUILD_MCPU})
  else()
    set(INTERNAL_BUILD_MCPU "native")
  endif()

  check_cxx_compiler_flag("-mcpu=${INTERNAL_BUILD_MCPU}" COMPILER_SUPPORTS_MCPU)
  if(COMPILER_SUPPORTS_MCPU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=${INTERNAL_BUILD_MCPU}")
  elseif(BUILD_MCPU)
    message(FATAL_ERROR "The flag -mcpu=${INTERNAL_BUILD_MCPU} is not supported by the compiler")
  else()
    unset(INTERNAL_BUILD_MCPU)
  endif()
endif()

option(Cabana_ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(Cabana_ENABLE_COVERAGE_BUILD)
    message(STATUS "Enabling coverage build")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
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

configure_file(cmake/Cabana.pc.cmakein Cabana.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Cabana.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
