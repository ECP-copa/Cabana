  add_executable(VerletList verlet_list.cpp)
  target_link_libraries(VerletList cabanacore)
  add_test(NAME Core_tutorial_09 COMMAND VerletList)