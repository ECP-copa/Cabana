##--------------------------------------------------------------------------##
## Cuda examples.
##--------------------------------------------------------------------------##
IF(Cabana_ENABLE_Cuda)

  TRIBITS_ADD_EXECUTABLE(
    CudaPerformanceTest
    SOURCES cuda_perf_test.cpp
    COMM serial mpi
    )

ENDIF()

IF(Cabana_ENABLE_Serial)

  TRIBITS_ADD_EXECUTABLE(
    SerialExample
    SOURCES serial_example.cpp
    COMM serial mpi
    )

ENDIF()

FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(SCAFACOS scafacos)
IF(PkgConfig_FOUND AND SCAFACOS_FOUND)
  TRIBITS_ADD_EXECUTABLE(
    ScafacosExample
    SOURCES scafacos_example.cpp
    COMM serial mpi
  ) 
  TARGET_LINK_LIBRARIES(ScafacosExample ${SCAFACOS_LIBRARIES})
  TARGET_INCLUDE_DIRECTORIES(ScafacosExample PRIVATE ${SCAFACOS_INCLUDE_DIRS})
  TARGET_COMPILE_OPTIONS(ScafacosExample PRIVATE ${SCAFACOS_CFLAGS_OTHER})

ENDIF()
