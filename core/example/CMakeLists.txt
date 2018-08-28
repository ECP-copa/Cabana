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

TRIBITS_ADD_EXECUTABLE(
  NeighborListMDPerfTest
  SOURCES md_neighbor_perf_test.cpp
  COMM serial mpi
  )
