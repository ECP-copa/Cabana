option(Cabana_ENABLE_PERFORMANCE_TESTING "Build Performance Tests" OFF)

if(NOT Cabana_ENABLE_PERFORMANCE_TESTING)
    return()
endif()

add_subdirectory(peakflops)
