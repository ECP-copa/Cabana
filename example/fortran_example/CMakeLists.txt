option(Cabana_ENABLE_FORTRAN_EXAMPLES "Build Fortran examples" OFF)
if(Cabana_ENABLE_FORTRAN_EXAMPLES)
  enable_language(Fortran)
  if(Cabana_ENABLE_COVERAGE_BUILD)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} --coverage -O0")
  endif()
  add_subdirectory(tutorial)
endif()



