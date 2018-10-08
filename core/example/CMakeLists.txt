option(Cabana_ENABLE_EXAMPLES "Build examples" OFF)
if(NOT Cabana_ENABLE_EXAMPLES)
  return()
endif()

add_subdirectory(tutorial)

if(Cabana_ENABLE_Cuda)
  add_executable(CudaPerformanceTest cuda_perf_test.cpp)
  target_link_libraries(CudaPerformanceTest cabanacore)
endif()

if(Cabana_ENABLE_Serial)
  add_executable(NeighborListMDPerfTest md_neighbor_perf_test.cpp)
  target_link_libraries(NeighborListMDPerfTest cabanacore)
endif()

add_executable(PeakFlops Cabana_peakflops.cpp)
target_link_libraries(PeakFlops cabanacore)
