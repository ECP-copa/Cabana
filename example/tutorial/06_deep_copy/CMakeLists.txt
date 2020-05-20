  add_executable(DeepCopy deep_copy_example.cpp)
  target_link_libraries(DeepCopy cabanacore)
  add_test(NAME Core_tutorial_06 COMMAND ${NONMPI_PRECOMMAND} DeepCopy)
