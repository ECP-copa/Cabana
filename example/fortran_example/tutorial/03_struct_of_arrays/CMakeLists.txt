  add_executable(ForStructOfArrays main.cpp soa_example.F90)
  target_link_libraries(ForStructOfArrays cabanacore)
  add_test(NAME Fortran_tutorial_03 COMMAND ForStructOfArrays)
