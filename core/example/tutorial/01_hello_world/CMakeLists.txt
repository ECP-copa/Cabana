  add_executable(HelloWorld hello_world.cpp)
  target_link_libraries(HelloWorld cabanacore)
  add_test(NAME Core_tutorial_01 COMMAND ${NONMPI_PRECOMMAND} HelloWorld)
