option(Cabana_ENABLE_PERFORMANCE_TESTING "Build Performance Tests" OFF)

if(Cabana_ENABLE_PERFORMANCE_TESTING)
  add_subdirectory(peakflops)
endif()

