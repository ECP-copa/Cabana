option(ENABLE_EXAMPLES "Build examples" OFF)
if(NOT ENABLE_EXAMPLES)
  return()
endif()
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

add_executable(PeakFlops Cabana_peakflops.cpp)
target_link_libraries(PeakFlops cabanacore)
add_executable(NeighborListMDPerfTest md_neighbor_perf_test.cpp)
target_link_libraries(NeighborListMDPerfTest cabanacore)
