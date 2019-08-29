# switch to find_package(scafacos) once scafacos-1.1 got released
find_package(PkgConfig)
pkg_check_modules(SCAFACOS scafacos IMPORTED_TARGET)
if(SCAFACOS_FOUND AND Cabana_ENABLE_MPI)
  add_executable(ScafacosExample scafacos_example.cpp)
  target_link_libraries(ScafacosExample cabanacore PkgConfig::SCAFACOS)
endif()
