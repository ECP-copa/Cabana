if(Cabana_ENABLE_Serial)
  add_executable(NeighborListMDPerfTest md_neighbor_perf_test.cpp)
  target_link_libraries(NeighborListMDPerfTest cabanacore)
endif()

if(Cabana_ENABLE_MPI AND Cabana_ENABLE_Cuda AND Cabana_ENABLE_OpenMP)
  add_executable(CommPerformance Cabana_CommPerformance.cpp)
  target_link_libraries(CommPerformance cabanacore)
endif()

if(Cabana_ENABLE_Serial OR Cabana_ENABLE_Cuda OR Cabana_ENABLE_OpenMP)
  add_executable(BinSortPerformance Cabana_BinSortPerformance.cpp)
  target_link_libraries(BinSortPerformance cabanacore)
endif()
