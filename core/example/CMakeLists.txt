option(Cabana_ENABLE_EXAMPLES "Build examples" OFF)
if(NOT Cabana_ENABLE_EXAMPLES)
  return()
endif()

FIND_PACKAGE(PkgConfig)
FIND_PACKAGE(MPI)
PKG_CHECK_MODULES(SCAFACOS scafacos IMPORTED_TARGET)
IF(PKG_CONFIG_FOUND AND SCAFACOS_FOUND)
  TRIBITS_ADD_EXECUTABLE(
    ScafacosExample
    SOURCES scafacos_example.cpp
    COMM serial mpi
  )
  TARGET_LINK_LIBRARIES(Core_ScafacosExample PkgConfig::SCAFACOS ${MPI_C_LIBRARIES})
  TARGET_INCLUDE_DIRECTORIES(Core_ScafacosExample PRIVATE ${MPI_C_INCLUDE_PATH})
ENDIF()

add_subdirectory(benchmark)
add_subdirectory(tutorial)
