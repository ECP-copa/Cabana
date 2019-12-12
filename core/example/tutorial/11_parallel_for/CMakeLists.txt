  add_executable(ParallelFor parallel_for_example.cpp)
  target_link_libraries(ParallelFor cabanacore)
  add_test(NAME Core_tutorial_11 COMMAND ${NONMPI_PRECOMMAND} ParallelFor)
