option(Cabana_ENABLE_EXAMPLES "Build examples" OFF)

if(Cabana_ENABLE_EXAMPLES)
  add_subdirectory(tutorial)
  add_subdirectory(fortran_example)
endif()
