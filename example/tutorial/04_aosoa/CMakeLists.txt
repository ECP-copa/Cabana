  add_executable(ArrayOfStructsOfArrays aosoa_example.cpp)
  target_link_libraries(ArrayOfStructsOfArrays cabanacore)
  add_test(NAME Core_tutorial_04 COMMAND ${NONMPI_PRECOMMAND} ArrayOfStructsOfArrays)
