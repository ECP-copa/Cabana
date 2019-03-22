# setup gtest
if(NOT Cabana_ENABLE_TESTING)
  return()
endif()

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/gtest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_PTHREAD=0")

include_directories(${GTEST_SOURCE_DIR})
add_library(cabana_core_gtest ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc)
set_target_properties(cabana_core_gtest PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(gtest_args --gtest_color=yes)

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
foreach(_test Version Index CartesianGrid SoA)
  add_executable(${_test}_test tst${_test}.cpp unit_test_main.cpp)
  target_link_libraries(${_test}_test cabanacore cabana_core_gtest)
  add_test(NAME ${_test}_test COMMAND ${_test}_test ${gtest_args})
endforeach()

##--------------------------------------------------------------------------##
## On-node tests with and without MPI.
##--------------------------------------------------------------------------##
macro(Cabana_add_tests)
  cmake_parse_arguments(CABANA_UNIT_TEST "MPI" "" "NAMES" ${ARGN})
  set(CABANA_UNIT_TEST_MPIEXEC_NUMPROCS 1)
  if(MPIEXEC_MAX_NUMPROCS GREATER 1)
    list(APPEND CABANA_UNIT_TEST_MPIEXEC_NUMPROCS ${MPIEXEC_MAX_NUMPROCS})
  endif()
  set(CABANA_UNIT_TEST_NUMTHREADS 1 2)
  if(CABANA_UNIT_TEST_MPI)
    set(CABANA_UNIT_TEST_MAIN mpi_unit_test_main.cpp)
  else()
    set(CABANA_UNIT_TEST_MAIN unit_test_main.cpp)
  endif()
  foreach(_device ${CABANA_SUPPORTED_DEVICES})
    if(Cabana_ENABLE_${_device})
      set(_dir ${CMAKE_CURRENT_BINARY_DIR}/${_device})
      file(MAKE_DIRECTORY ${_dir})
      foreach(_test ${CABANA_UNIT_TEST_NAMES})
        set(_file ${_dir}/tst${_test}_${_device}.cpp)
        file(WRITE ${_file} "#include <Test${_device}_Category.hpp>\n")
        file(APPEND ${_file} "#include <tst${_test}.hpp>\n")
        set(_target ${_test}_test_${_device})
        add_executable(${_target} ${_file} ${CABANA_UNIT_TEST_MAIN})
        target_include_directories(${_target} PUBLIC ${_dir})
        target_link_libraries(${_target} cabanacore cabana_core_gtest)
        if(CABANA_UNIT_TEST_MPI)
          foreach(_np ${CABANA_UNIT_TEST_MPIEXEC_NUMPROCS})
            add_test(NAME ${_target}_${_np} COMMAND
              ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
              ${_target} ${MPIEXEC_POSTFLAGS} ${gtest_args})
          endforeach()
        else()
          if(_device STREQUAL Pthread OR _device STREQUAL OpenMP)
            foreach(_thread ${CABANA_UNIT_TEST_NUMTHREADS})
              add_test(NAME ${_target}_${_thread} COMMAND
                ${_target} ${gtest_args} --kokkos-threads=${_thread})
            endforeach()
          else()
            add_test(NAME ${_target} COMMAND ${_target} ${gtest_args})
          endif()
        endif()
      endforeach()
    endif()
  endforeach()
endmacro()

Cabana_add_tests(NAMES AoSoA Slice DeepCopy Tuple Sort LinkedCellList NeighborList Parallel)
Cabana_add_tests(MPI NAMES CommunicationPlan Distributor Halo)
