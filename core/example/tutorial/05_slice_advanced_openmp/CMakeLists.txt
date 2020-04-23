  add_executable(AdvancedOpenMPSlice advanced_slice_openmp.cpp)
  target_link_libraries(AdvancedOpenMPSlice cabanacore)
  add_test(NAME Core_tutorial_05_openmp COMMAND ${NONMPI_PRECOMMAND} AdvancedOpenMPSlice)
