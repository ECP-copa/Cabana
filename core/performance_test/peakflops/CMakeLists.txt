############################################################################
# Copyright (c) 2018-2023 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

# It only makes sense to run this if testing is enabled, as we depend on gtest
if(NOT Cabana_ENABLE_TESTING)
  message(WARNING "Cabana_ENABLE_TESTING disabled, skipping performance tests")
  return()
endif()

include(${TEST_HARNESS_DIR}/test_harness.cmake)

# For intel haswell, broadwell: set expected flops/cycle = 32, veclength=8, iterations=2e6
# For intel skylake           : set expected flops/cycle = 64, veclength=16, iterations=2e8
set(Cabana_PERFORMANCE_EXPECTED_FLOPS "32" CACHE STRING "Value for expected flops in performance tests")
set(Cabana_PERFORMANCE_ERROR_MARGIN "0.9" CACHE STRING "Error margin to use in performance tests")
set(Cabana_PERFORMANCE_VECLENGTH "8" CACHE STRING "Vector Length to use when running expected flops")
set(Cabana_PERFORMANCE_ITERATIONS "20000" CACHE STRING "Number of iterations to repeat the test for (increases flops)")
set(Cabana_PERFORMANCE_SEED "76843802738543" CACHE STRING "Seed for performance test RNG")

#TODO: Should this test be pulled out into a function
if (NOT Cabana_PERFORMANCE_EXPECTED_FLOPS MATCHES "^[0-9]+$")
  message(FATAL_ERROR "Cabana_PERFORMANCE_EXPECTED_FLOPS needs to be an integer")
endif()

if (NOT Cabana_PERFORMANCE_ERROR_MARGIN MATCHES "^[0-9.]+$")
  message(FATAL_ERROR "Cabana_PERFORMANCE_ERROR_MARGIN needs to be a float")
endif()

if (NOT Cabana_PERFORMANCE_VECLENGTH MATCHES "^[0-9]+$")
  message(FATAL_ERROR "Cabana_PERFORMANCE_VECLENGTH needs to be an integer")
endif()

if (NOT Cabana_PERFORMANCE_ITERATIONS MATCHES "^[0-9]+$")
  message(FATAL_ERROR "Cabana_PERFORMANCE_ITERATIONS needs to be an integer")
endif()

if (NOT Cabana_PERFORMANCE_SEED MATCHES "^[0-9]+$")
  message(FATAL_ERROR "Cabana_PERFORMANCE_SEED needs to be an integer")
endif()

# we only want to run the below if we have access to "#pragma omp simd"
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fopenmp -Werror")
check_cxx_source_compiles(
      "int main(void) {
      const int N = 32;
      int sum[N];
      #pragma omp simd
      for (int i = 0; i < N; i++) { sum[i] = i*2; }
      return 0;
      }
      " HAVE_OMP_SIMD)
set(CMAKE_REQUIRED_FLAGS)

if (NOT HAVE_OMP_SIMD)
    # No point running these tests without omp simd
    message(WARNING "Compiler doesn't seem to support `#pragma simd`, skipping performance tests")
    return()
endif()

foreach(_test 01_cpp_simple 02_kokkos_simple_view 03_Cabana_peakflops)
  set(test "${_test}_performance_test")
  add_executable(${test} ${_test}.cpp ${TEST_HARNESS_DIR}/unit_test_main.cpp)
  target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${test} PRIVATE
    CABANA_PERFORMANCE_EXPECTED_FLOPS=${Cabana_PERFORMANCE_EXPECTED_FLOPS}
    CABANA_PERFORMANCE_ERROR_MARGIN=${Cabana_PERFORMANCE_ERROR_MARGIN}
    CABANA_PERFORMANCE_VECLENGTH=${Cabana_PERFORMANCE_VECLENGTH}
    CABANA_PERFORMANCE_ITERATIONS=${Cabana_PERFORMANCE_ITERATIONS}
    CABANA_PERFORMANCE_SEED=${Cabana_PERFORMANCE_SEED}
    )
  target_link_libraries(${test} Core ${gtest_target})
  target_compile_options(${test} PRIVATE "-fopenmp")
  add_test(NAME ${test} COMMAND ${test})
  set_tests_properties(${test} PROPERTIES LABELS "PERFORMANCE" )
endforeach()
