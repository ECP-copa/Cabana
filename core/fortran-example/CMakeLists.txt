option(Cabana_ENABLE_FORTRAN_EXAMPLES "Build Fortran examples" OFF)
if(NOT Cabana_ENABLE_FORTRAN_EXAMPLES)
  return()
endif()

add_subdirectory(tutorial)

