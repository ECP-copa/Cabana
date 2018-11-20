if(Cabana_ENABLE_Serial)
  add_executable(NeighborListMDPerfTest md_neighbor_perf_test.cpp)
  target_link_libraries(NeighborListMDPerfTest cabanacore)
endif()

add_executable(PeakFlops Cabana_peakflops.cpp)
target_link_libraries(PeakFlops cabanacore)
