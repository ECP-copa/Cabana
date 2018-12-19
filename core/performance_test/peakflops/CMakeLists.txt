set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/gtest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_PTHREAD=0")

include_directories(${GTEST_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(_test 01_cpp_simple 02_kokkos_simple_view 03_Cabana_peakflops)
  # TODO: tidy unit_test_main.cpp path
  add_executable(${_test}_test ${_test}.cpp ${CMAKE_SOURCE_DIR}/core/unit_test/unit_test_main.cpp)
  target_link_libraries(${_test}_test cabanacore cabana_core_gtest)
  add_test(NAME ${_test}_test COMMAND ${_test}_test --gtest_color=yes)
  set_tests_properties(${_test}_test PROPERTIES LABELS "PERFORMANCE" )
endforeach()
