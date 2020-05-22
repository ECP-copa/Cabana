# It only makes sense to run this if testing is enabled, as we depend on gtest
if(NOT Cabana_ENABLE_TESTING)
  message(WARNING "Cabana_ENABLE_TESTING disabled, skipping performance tests")
  return()
endif()

set(Cabana_PERFORMANCE_EXPECTED_FLOPS "0" CACHE STRING "Value for expected flops in performance tests")
set(Cabana_PERFORMANCE_ERROR_MARGIN "0" CACHE STRING "Error margin to use in performance tests")
set(Cabana_PERFORMANCE_VECLENGTH "8" CACHE STRING "Vector Length to use when running expected flops")
set(Cabana_PERFORMANCE_ITERATIONS "10" CACHE STRING "Number of iterations to repeat the test for (increases flops)")
set(Cabana_PERFORMANCE_SEED "0" CACHE STRING "Seed for performance test RNG")

if(NOT Cabana_PERFORMANCE_EXPECTED_FLOPS STREQUAL "")
    #TODO: Should this test be pulled out into a function
    if (NOT Cabana_PERFORMANCE_EXPECTED_FLOPS MATCHES "^[0-9]+$")
      message(FATAL_ERROR "Cabana_PERFORMANCE_EXPECTED_FLOPS needs to be an integer")
    endif()
endif()

if(NOT Cabana_PERFORMANCE_ERROR_MARGIN STREQUAL "")
    if (NOT Cabana_PERFORMANCE_ERROR_MARGIN MATCHES "^[0-9.]+$")
      message(FATAL_ERROR "Cabana_PERFORMANCE_ERROR_MARGIN needs to be a float")
    endif()
endif()

if(NOT Cabana_PERFORMANCE_VECLENGTH STREQUAL "")
    if (NOT Cabana_PERFORMANCE_VECLENGTH MATCHES "^[0-9]+$")
      message(FATAL_ERROR "Cabana_PERFORMANCE_VECLENGTH needs to be an integer")
    endif()
endif()

if(NOT Cabana_PERFORMANCE_ITERATIONS STREQUAL "")
    if (NOT Cabana_PERFORMANCE_ITERATIONS MATCHES "^[0-9]+$")
      message(FATAL_ERROR "Cabana_PERFORMANCE_ITERATIONS needs to be an integer")
    endif()
endif()

if(NOT Cabana_PERFORMANCE_SEED STREQUAL "")
    if (NOT Cabana_PERFORMANCE_SEED MATCHES "^[0-9]+$")
      message(FATAL_ERROR "Cabana_PERFORMANCE_SEED needs to be an integer")
    endif()
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

foreach(_test 01_cpp_simple 02_kokkos_simple_view 03_Cabana_peakflops)
  # TODO: tidy unit_test_main.cpp path
  set(test "${_test}_performance_test")
  add_executable(${test} ${_test}.cpp ${CMAKE_SOURCE_DIR}/core/unit_test/unit_test_main.cpp)
  target_include_directories(${test} PUBLIC ${GTEST_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(${test} PUBLIC
    CABANA_PERFORMANCE_EXPECTED_FLOPS=${Cabana_PERFORMANCE_EXPECTED_FLOPS}
    CABANA_PERFORMANCE_ERROR_MARGIN=${Cabana_PERFORMANCE_ERROR_MARGIN}
    CABANA_PERFORMANCE_VECLENGTH=${Cabana_PERFORMANCE_VECLENGTH}
    CABANA_PERFORMANCE_ITERATIONS=${Cabana_PERFORMANCE_ITERATIONS}
    CABANA_PERFORMANCE_SEED=${Cabana_PERFORMANCE_SEED}
    )
  target_link_libraries(${test} cabanacore cabana_core_gtest)
  target_compile_options(${test} PRIVATE "-fopenmp")
  add_test(NAME ${test} COMMAND ${test})
  set_tests_properties(${test} PROPERTIES LABELS "PERFORMANCE" )
endforeach()
