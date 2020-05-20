  add_executable(StructOfArrays soa_example.cpp)
  target_link_libraries(StructOfArrays cabanacore)
  add_test(NAME Core_tutorial_03 COMMAND ${NONMPI_PRECOMMAND} StructOfArrays)
