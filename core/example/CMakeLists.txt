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

  TRIBITS_ADD_EXECUTABLE(
    PeakFlops
    SOURCES Cabana_peakflops.cpp
    COMM serial mpi
    )

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
  
  TRIBITS_ADD_EXECUTABLE(
    PeakFlops
    SOURCES Cabana_peakflops.cpp
    COMM serial mpi
    )
