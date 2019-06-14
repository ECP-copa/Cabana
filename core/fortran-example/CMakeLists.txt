option(Cabana_ENABLE_FORTRAN_EXAMPLES "Build Fortran examples" OFF)
if(Cabana_ENABLE_FORTRAN_EXAMPLES)
  enable_language(Fortran)
  if(Cabana_ENABLE_COVERAGE_BUILD)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} --coverage -O0")
  endif()
  # workaround until we have a real solution for #140
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL CMAKE_Fortran_COMPILER_ID)
    message(FATAL_ERROR "Mixing different compiler families is currently not supported")
  endif()
  add_subdirectory(tutorial)
  add_subdirectory(advanced)
endif()



