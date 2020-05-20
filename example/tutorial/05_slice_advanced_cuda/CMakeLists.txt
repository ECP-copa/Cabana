  add_executable(AdvancedCudaSlice advanced_slice_cuda.cpp)
  target_link_libraries(AdvancedCudaSlice cabanacore)
  add_test(NAME Core_tutorial_05_cuda COMMAND ${NONMPI_PRECOMMAND} AdvancedCudaSlice)
