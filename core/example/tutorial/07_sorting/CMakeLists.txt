  add_executable(Sorting sorting.cpp)
  target_link_libraries(Sorting cabanacore)
  add_test(NAME Core_tutorial_07 COMMAND ${NONMPI_PRECOMMAND} Sorting)
