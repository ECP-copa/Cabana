option(Cabana_ENABLE_EXAMPLES "Build examples" OFF)
if(NOT Cabana_ENABLE_EXAMPLES)
  return()
endif()

add_subdirectory(01_hello_world)
add_subdirectory(02_tuple)
add_subdirectory(03_struct_of_arrays)

if(Cabana_ENABLE_Serial)
  add_subdirectory(04_aosoa_serial)
endif()

if(Cabana_ENABLE_OpenMP)
  add_subdirectory(04_aosoa_openmp)
endif()

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
