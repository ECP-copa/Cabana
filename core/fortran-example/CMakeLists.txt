option(Cabana_ENABLE_FORTRAN_EXAMPLES "Build Fortran examples" OFF)
if(Cabana_ENABLE_FORTRAN_EXAMPLES)
  enable_language(Fortran)
  add_subdirectory(advanced)
endif()

