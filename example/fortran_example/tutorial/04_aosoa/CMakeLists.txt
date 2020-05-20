  add_executable(ForArrayOfStructsOfArrays main.cpp aosoa_example.F90)
  target_link_libraries(ForArrayOfStructsOfArrays cabanacore)
  add_test(NAME Fortran_tutorial_04 COMMAND ForArrayOfStructsOfArrays)
