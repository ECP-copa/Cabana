  add_executable(Migration migration_example.cpp)
  target_link_libraries(Migration cabanacore)
  add_test(NAME Core_tutorial_12 COMMAND Migration)