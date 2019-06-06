  add_executable(Migration migration_example.cpp)
  target_link_libraries(Migration cabanacore)
  add_test(NAME Core_tutorial_12 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
    ${MPIEXEC_MAX_NUMPROCS} ${CMAKE_CURRENT_BINARY_DIR}/Migration)
  set_tests_properties(Core_tutorial_12 PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
