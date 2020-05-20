  add_executable(SimdParallelFor simd_parallel_for_example.cpp)
  target_link_libraries(SimdParallelFor cabanacore)
  add_test(NAME Core_tutorial_10_simd COMMAND ${NONMPI_PRECOMMAND} SimdParallelFor)
