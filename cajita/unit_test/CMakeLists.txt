set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/gtest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_PTHREAD=0")

include_directories(${GTEST_SOURCE_DIR})
add_library(cabana_cajita_gtest ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc)
set_target_properties(cabana_cajita_gtest PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(gtest_args --gtest_color=yes)

set(CAJITA_TEST_DEVICES)
foreach(_device ${CABANA_SUPPORTED_DEVICES})
  if(Kokkos_ENABLE_${_device})
    list(APPEND CAJITA_TEST_DEVICES ${_device})
    if(_device STREQUAL CUDA)
      list(APPEND CAJITA_TEST_DEVICES CUDA_UVM)
    endif()
  endif()
endforeach()

macro(Cajita_add_tests)
  cmake_parse_arguments(CAJITA_UNIT_TEST "MPI" "" "NAMES" ${ARGN})
  set(CAJITA_UNIT_TEST_MPIEXEC_NUMPROCS 1)
  foreach( _np 2 4 8 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_np})
      list(APPEND CAJITA_UNIT_TEST_MPIEXEC_NUMPROCS ${_np})
    endif()
  endforeach()
  set(CAJITA_UNIT_TEST_NUMTHREADS 1)
  foreach( _nt 2 4 8 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_nt})
      list(APPEND CAJITA_UNIT_TEST_NUMTHREADS ${_nt})
    endif()
  endforeach()
  set(CAJITA_UNIT_TEST_MAIN unit_test_main.cpp)
  foreach(_device ${CAJITA_TEST_DEVICES})
    set(_dir ${CMAKE_CURRENT_BINARY_DIR}/${_device})
    file(MAKE_DIRECTORY ${_dir})
    foreach(_test ${CAJITA_UNIT_TEST_NAMES})
      set(_file ${_dir}/tst${_test}_${_device}.cpp)
      file(WRITE ${_file} "#include <Test${_device}_Category.hpp>\n")
      file(APPEND ${_file} "#include <tst${_test}.hpp>\n")
      set(_target Cajita_${_test}_test_${_device})
      add_executable(${_target} ${_file} ${CAJITA_UNIT_TEST_MAIN})
      target_include_directories(${_target} PRIVATE ${_dir} ${CMAKE_CURRENT_SOURCE_DIR})
      target_link_libraries(${_target} PRIVATE Cajita cabana_cajita_gtest)
      if(CAJITA_UNIT_TEST_MPI)
        foreach(_np ${CAJITA_UNIT_TEST_MPIEXEC_NUMPROCS})
          add_test(NAME ${_target}_${_np} COMMAND
            ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
            ${_target} ${MPIEXEC_POSTFLAGS} ${gtest_args})
        endforeach()
      else()
        if(_device STREQUAL OpenMP)
          foreach(_thread ${CAJITA_UNIT_TEST_NUMTHREADS})
            add_test(NAME ${_target}_${_thread} COMMAND
              ${_target} ${gtest_args} --kokkos-threads=${_thread})
          endforeach()
        else()
          add_test(NAME ${_target} COMMAND ${_target} ${gtest_args})
        endif()
      endif()
    endforeach()
  endforeach()
endmacro()

set(SERIAL_TESTS
  GlobalMesh
  IndexSpace
  Splines
  )

set(MPI_TESTS
  GlobalGrid
  LocalGrid
  LocalMesh
  Array
  Halo
  SplineEvaluation
  Interpolation
  BovWriter
  )

if(Cabana_ENABLE_HYPRE)
  list(APPEND MPI_TESTS
    HypreStructuredSolver
    )
endif()

if(Cabana_ENABLE_HEFFTE)
  list(APPEND MPI_TESTS
    FastFourierTransform
    )
endif()

Cajita_add_tests(NAMES ${SERIAL_TESTS})

Cajita_add_tests(MPI NAMES ${MPI_TESTS})
