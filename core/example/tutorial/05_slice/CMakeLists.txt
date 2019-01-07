  add_executable(Slice slice_example.cpp)
  target_link_libraries(Slice cabanacore)
  add_test(NAME Core_tutorial_05 COMMAND Slice)