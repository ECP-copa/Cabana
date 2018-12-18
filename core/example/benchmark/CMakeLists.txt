option(Cabana_ENABLE_EXAMPLES "Build examples" OFF)
if(NOT Cabana_ENABLE_EXAMPLES)
  return()
endif()


if(Cabana_ENABLE_Serial)
  add_executable(NeighborListMDPerfTest md_neighbor_perf_test.cpp)
  target_link_libraries(NeighborListMDPerfTest cabanacore)
endif()

add_subdirectory(peakflops)
