##---------------------------------------------------------------------------##
## File  : core/CMakeLists.txt
##---------------------------------------------------------------------------##
add_subdirectory(src)
add_subdirectory(unit_test)

if(Cabana_ENABLE_Performance_Testing)
    add_subdirectory(performance_test)
endif()

add_subdirectory(example)
##---------------------------------------------------------------------------##
## end of core/CMakeLists.txt
##---------------------------------------------------------------------------##
