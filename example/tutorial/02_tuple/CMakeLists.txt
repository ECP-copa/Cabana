  add_executable(Tuple tuple_example.cpp)
  target_link_libraries(Tuple cabanacore)
  add_test(NAME Core_tutorial_02 COMMAND ${NONMPI_PRECOMMAND} Tuple)
