if(Kokkos_ENABLE_SERIAL)
  add_executable(NeighborListMDPerfTest md_neighbor_perf_test.cpp)
  target_link_libraries(NeighborListMDPerfTest cabanacore)
endif()

if(MPI_FOUND AND Kokkos_ENABLE_CUDA AND Kokkos_ENABLE_OPENMP)
  add_executable(CommPerformance Cabana_CommPerformance.cpp)
  target_link_libraries(CommPerformance cabanacore)
endif()

if(Kokkos_ENABLE_SERIAL OR Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_OPENMP)
  add_executable(BinSortPerformance Cabana_BinSortPerformance.cpp)
  target_link_libraries(BinSortPerformance cabanacore)
endif()
