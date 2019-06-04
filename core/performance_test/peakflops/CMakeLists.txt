# It only makes sense to run this if testing is enabled, as we depend on gtest
if(NOT Cabana_ENABLE_TESTING)
  return()
endif()

# we only want to run the below if we have access to "#pragma omp simd"
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
      "int main(void) {
      #pragma omp simd
      return 0;
      }" HAVE_OMP_SIMD)

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
  target_compile_options(${test} PRIVATE "-fopenmp-simd")
  add_test(NAME ${test} COMMAND ${test})
  set_tests_properties(${test} PROPERTIES LABELS "PERFORMANCE" )
endforeach()
