# setup gtest
if(NOT Cabana_ENABLE_TESTING)
  return()
endif()

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/gtest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_PTHREAD=0")

include_directories(${GTEST_SOURCE_DIR})
add_library(cabana_core_gtest ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc)

# Fail regular expression
set(FAIL_RE "  FAILED  ")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
foreach(_test Version Index CartesianGrid SoA)
  add_executable(${_test}_test tst${_test}.cpp unit_test_main.cpp)
  target_link_libraries(${_test}_test cabanacore cabana_core_gtest)
  add_test(NAME ${_test}_test COMMAND ${_test}_test --gtest_color=yes)
  set_tests_properties(${_test}_test PROPERTIES FAIL_REGULAR_EXPRESSION "${FAIL_RE}")
endforeach()

##--------------------------------------------------------------------------##
## Serial tests.
##--------------------------------------------------------------------------##
if(Cabana_ENABLE_Serial)
  foreach(_test AoSoA Slice DeepCopy Tuple Sort NeighborList Parallel)
    add_executable(${_test}_test_Serial Serial/tst${_test}_Serial.cpp unit_test_main.cpp)
    target_link_libraries(${_test}_test_Serial cabanacore cabana_core_gtest)
    add_test(NAME ${_test}_test_Serial COMMAND ${_test}_test_Serial --gtest_color=yes)
    set_tests_properties(${_test}_test_Serial PROPERTIES FAIL_REGULAR_EXPRESSION "${FAIL_RE}")
  endforeach()
endif()

##--------------------------------------------------------------------------##
## OpenMP tests.
##--------------------------------------------------------------------------##
if(Cabana_ENABLE_OpenMP)
  foreach(_test AoSoA Slice DeepCopy Tuple Sort NeighborList Parallel)
    add_executable(${_test}_test_OpenMP OpenMP/tst${_test}_OpenMP.cpp unit_test_main.cpp)
    target_link_libraries(${_test}_test_OpenMP cabanacore cabana_core_gtest)
    add_test(NAME ${_test}_test_OpenMP COMMAND ${_test}_test_OpenMP --gtest_color=yes)
    set_tests_properties(${_test}_test_OpenMP PROPERTIES FAIL_REGULAR_EXPRESSION "${FAIL_RE}")
  endforeach()
endif()

##--------------------------------------------------------------------------##
## Cuda tests.
##--------------------------------------------------------------------------##
if(Cabana_ENABLE_Cuda)
  foreach(_test AoSoA Slice DeepCopy Tuple Sort NeighborList Parallel)
    add_executable(${_test}_test_CudaUVM Cuda/tst${_test}_CudaUVM.cpp unit_test_main.cpp)
    target_link_libraries(${_test}_test_CudaUVM cabanacore cabana_core_gtest)
    add_test(NAME ${_test}_test_CudaUVM COMMAND ${_test}_test_CudaUVM --gtest_color=yes)
    set_tests_properties(${_test}_test_CudaUVM PROPERTIES FAIL_REGULAR_EXPRESSION "${FAIL_RE}")
  endforeach()
endif()

##--------------------------------------------------------------------------##
