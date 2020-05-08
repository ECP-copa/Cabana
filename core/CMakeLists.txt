add_subdirectory(src)

if(Cabana_ENABLE_TESTING)
  add_subdirectory(unit_test)
endif()

if(Cabana_ENABLE_PERFORMANCE_TESTING)
  add_subdirectory(performance_test)
endif()
