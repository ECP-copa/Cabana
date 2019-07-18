# It only makes sense to run this if testing is enabled, as we depend on gtest
if(NOT Cabana_ENABLE_TESTING)
  return()
endif()

set(Cabana_PERFORMANCE_EXPECTED_FLOPS "" CACHE STRING "Value for expected flops in performance tests")
set(Cabana_PERFORMANCE_ERROR_MARGIN "" CACHE STRING "Error margin to use in performance tests")
set(Cabana_PERFORMANCE_VECLENGTH "" CACHE STRING "Vector Length to use when running expected flops")
set(Cabana_PERFORMANCE_ITERATIONS "" CACHE STRING "Number of iterations to repeat the test for (increases flops)")
set(Cabana_PERFORMANCE_SEED "" CACHE STRING "Seed for performance test RNG")

if(NOT Cabana_PERFORMANCE_EXPECTED_FLOPS STREQUAL "")
    #TODO: Should this test be pulled out into a function
    if (NOT Cabana_PERFORMANCE_EXPECTED_FLOPS MATCHES "^[0-9]+$")
      message(FATAL_ERROR "Cabana_PERFORMANCE_EXPECTED_FLOPS needs to be an integer")
    endif()
    add_definitions("-DCABANA_PERFORMANCE_EXPECTED_FLOPS=${Cabana_PERFORMANCE_EXPECTED_FLOPS}")
endif()

if(NOT Cabana_PERFORMANCE_ERROR_MARGIN STREQUAL "")
    if (NOT Cabana_PERFORMANCE_ERROR_MARGIN MATCHES "^[0-9.]+$")
      message(FATAL_ERROR "Cabana_PERFORMANCE_ERROR_MARGIN needs to be a float")
    endif()
    add_definitions("-DCABANA_PERFORMANCE_ERROR_MARGIN=${Cabana_PERFORMANCE_ERROR_MARGIN}")
endif()

if(NOT Cabana_PERFORMANCE_VECLENGTH STREQUAL "")
    if (NOT Cabana_PERFORMANCE_VECLENGTH MATCHES "^[0-9]+$")
      message(FATAL_ERROR "Cabana_PERFORMANCE_VECLENGTH needs to be an integer")
    endif()
    add_definitions("-DCABANA_PERFORMANCE_VECLENGTH=${Cabana_PERFORMANCE_VECLENGTH}")
endif()

if(NOT Cabana_PERFORMANCE_ITERATIONS STREQUAL "")
    if (NOT Cabana_PERFORMANCE_ITERATIONS MATCHES "^[0-9]+$")
      message(FATAL_ERROR "Cabana_PERFORMANCE_ITERATIONS needs to be an integer")
    endif()
    add_definitions("-DCABANA_PERFORMANCE_ITERATIONS=${Cabana_PERFORMANCE_ITERATIONS}")
endif()

if(NOT Cabana_PERFORMANCE_SEED STREQUAL "")
    if (NOT Cabana_PERFORMANCE_SEED MATCHES "^[0-9]+$")
      message(FATAL_ERROR "Cabana_PERFORMANCE_SEED needs to be an integer")
    endif()
    add_definitions("-DCABANA_PERFORMANCE_SEED=${Cabana_PERFORMANCE_SEED}")
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

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/gtest)

include_directories(${GTEST_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(_test 01_cpp_simple 02_kokkos_simple_view 03_Cabana_peakflops)
  # TODO: tidy unit_test_main.cpp path
  set(test "${_test}_performance_test")
  add_executable(${test} ${_test}.cpp ${CMAKE_SOURCE_DIR}/core/unit_test/unit_test_main.cpp)
  target_link_libraries(${test} cabanacore cabana_core_gtest)
  target_compile_options(${test} PRIVATE "-fopenmp")
  add_test(NAME ${test} COMMAND ${test})
  set_tests_properties(${test} PROPERTIES LABELS "PERFORMANCE" )
endforeach()
