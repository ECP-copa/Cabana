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

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
foreach(_test Version Index CartesianGrid SoA)
  add_executable(${_test}_test tst${_test}.cpp unit_test_main.cpp)
  target_link_libraries(${_test}_test cabanacore cabana_core_gtest)
  add_test(NAME ${_test}_test COMMAND ${_test}_test --gtest_color=yes)
endforeach()

##--------------------------------------------------------------------------##
## On-node tests.
##--------------------------------------------------------------------------##
foreach(_device ${CABANA_SUPPORTED_DEVICES})
  if(Cabana_ENABLE_${_device})
    foreach(_test AoSoA Slice DeepCopy Tuple Sort LinkedCellList NeighborList Parallel)
      add_executable(${_test}_test_${_device} ${_device}/tst${_test}_${_device}.cpp unit_test_main.cpp)
      target_link_libraries(${_test}_test_${_device} cabanacore cabana_core_gtest)
      if(_device STREQUAL Pthread OR _device STREQUAL OpenMP)
        foreach(_thread 1 2)
          add_test(NAME ${_test}_test_${_device}_${_thread} COMMAND
            ${_test}_test_${_device} --gtest_color=yes --kokkos-threads=${_thread})
        endforeach()
      else()
        add_test(NAME ${_test}_test_${_device} COMMAND ${_test}_test_${_device} --gtest_color=yes)
      endif()
    endforeach()
  endif()
endforeach()

##--------------------------------------------------------------------------##
## MPI tests.
##--------------------------------------------------------------------------##
if(${Cabana_ENABLE_MPI})
  foreach(_device ${CABANA_SUPPORTED_DEVICES})
    if(Cabana_ENABLE_${_device})
      foreach(_test CommunicationPlan Distributor Halo)
        add_executable(${_test}_test_${_device} ${_device}/tst${_test}_${_device}.cpp mpi_unit_test_main.cpp)
        target_link_libraries(${_test}_test_${_device} cabanacore cabana_core_gtest)
        set(TEST_MPIEXEC_NUMPROCS "")
        list(APPEND TEST_MPIEXEC_NUMPROCS  1)
        if (MPIEXEC_MAX_NUMPROCS GREATER 1)
          list(APPEND TEST_MPIEXEC_NUMPROCS ${MPIEXEC_MAX_NUMPROCS})
        endif()
        foreach(_np ${TEST_MPIEXEC_NUMPROCS})
          add_test(NAME ${_test}_test_${_device}_${_np} COMMAND
            ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_np}
            ${CMAKE_CURRENT_BINARY_DIR}/${_test}_test_${_device} --gtest_color=yes)
        endforeach()
      endforeach()
    endif()
  endforeach()
endif()
