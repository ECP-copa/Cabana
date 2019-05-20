if(Cabana_ENABLE_Serial)
  add_executable(NeighborListMDPerfTest md_neighbor_perf_test.cpp)
  target_link_libraries(NeighborListMDPerfTest cabanacore)
endif()

# PeakFlops uses x86 asm volatile
if(CMAKE_SYSTEM_PROCESSOR MATCHES "([xX]86|[aA][mM][dD]64)")
  add_executable(PeakFlops Cabana_peakflops.cpp)
  target_link_libraries(PeakFlops cabanacore)
endif()

if(Cabana_ENABLE_MPI AND Cabana_ENABLE_Cuda AND (Cabana_ENABLE_Serial OR Cabana_ENABLE_OpenMP OR Cabana_ENABLE_Pthread) )
  add_executable(CommPerformance Cabana_CommPerformance.cpp)
  target_link_libraries(CommPerformance cabanacore)

endif()
