  add_executable(Migration migration_example.cpp)
  target_link_libraries(Migration cabanacore)
  add_test(NAME Core_tutorial_11 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
    ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/Migration ${MPIEXEC_POSTFLAGS})
  set_tests_properties(Core_tutorial_11 PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
