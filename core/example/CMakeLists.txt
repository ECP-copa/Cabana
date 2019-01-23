option(Cabana_ENABLE_EXAMPLES "Build examples" OFF)
if(NOT Cabana_ENABLE_EXAMPLES)
  return()
endif()

add_subdirectory(benchmark)
add_subdirectory(tutorial)
add_subdirectory(scafacos)
