  add_executable(NeighParallelFor neighbor_parallel_for_example.cpp)
  target_link_libraries(NeighParallelFor cabanacore)
  add_test(NAME Core_tutorial_10_neighbor COMMAND ${NONMPI_PRECOMMAND} NeighParallelFor)
