  add_executable(ForParallelFor parallel_for_example.F90 main.cpp)
  target_link_libraries(ForParallelFor cabanacore)
  add_test(NAME Fortran_tutorial_11 COMMAND ForParallelFor)
