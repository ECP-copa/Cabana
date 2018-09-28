option(ENABLE_EXAMPLES "Build examples" OFF)
if(NOT ENABLE_EXAMPLES)
  return()
endif()
##--------------------------------------------------------------------------##
## Cuda examples.
##--------------------------------------------------------------------------##
if(Cabana_ENABLE_Cuda)
  add_executable(CudaPerformanceTest cuda_perf_test.cpp)
  target_link_libraries(CudaPerformanceTest cabanacore)
endif()

if(Cabana_ENABLE_Serial)
  add_executable(SerialExample serial_example.cpp)
  target_link_libraries(SerialExample cabanacore)
endif()

add_executable(PeakFlops Cabana_peakflops.cpp)
target_link_libraries(PeakFlops cabanacore)
add_executable(NeighborListMDPerfTest md_neighbor_perf_test.cpp)
target_link_libraries(NeighborListMDPerfTest cabanacore)
