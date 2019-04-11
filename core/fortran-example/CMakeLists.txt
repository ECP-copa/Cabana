option(Cabana_ENABLE_FORTRAN_EXAMPLES "Build Fortran examples" OFF)
if(NOT Cabana_ENABLE_FORTRAN_EXAMPLES)
  return()
else()
  enable_language(Fortran)
  add_subdirectory(tutorial)
endif()



