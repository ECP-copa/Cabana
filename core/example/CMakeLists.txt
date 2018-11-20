option(Cabana_ENABLE_EXAMPLES "Build examples" OFF)
if(NOT Cabana_ENABLE_EXAMPLES)
  return()
endif()

find_package(PkgConfig)
pkg_check_modules(SCAFACOS scafacos IMPORTED_TARGET)
if(SCAFACOS_FOUND AND Cabana_ENABLE_MPI)
  add_executable(ScafacosExample scafacos_example.cpp)
  target_link_libraries(ScafacosExample cabanacore PkgConfig::SCAFACOS)
endif()

add_subdirectory(benchmark)
add_subdirectory(tutorial)
