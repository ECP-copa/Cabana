set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/gtest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_PTHREAD=0")

include_directories(${GTEST_SOURCE_DIR})
add_library(cabana_core_gtest ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc)
set_target_properties(cabana_core_gtest PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    )
set(gtest_args --gtest_color=yes)

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
foreach(_test Version Index CartesianGrid SoA)
  set(_target Cabana_${_test}_test)
  add_executable(${_target} tst${_test}.cpp unit_test_main.cpp)
  target_link_libraries(${_target} cabanacore cabana_core_gtest)
  add_test(NAME ${_target} COMMAND ${NONMPI_PRECOMMAND} ${_target} ${gtest_args})
endforeach()

##--------------------------------------------------------------------------##
## On-node tests with and without MPI.
##--------------------------------------------------------------------------##
set(CABANA_TEST_DEVICES)
foreach(_device ${CABANA_SUPPORTED_DEVICES})
  if(Kokkos_ENABLE_${_device})
    list(APPEND CABANA_TEST_DEVICES ${_device})
    if(_device STREQUAL CUDA)
      list(APPEND CABANA_TEST_DEVICES CUDA_UVM)
    endif()
  endif()
endforeach()

macro(Cabana_add_tests)
  cmake_parse_arguments(CABANA_UNIT_TEST "MPI" "" "NAMES" ${ARGN})
  set(CABANA_UNIT_TEST_MPIEXEC_NUMPROCS 1)
  foreach( _np 2 4 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_np})
      list(APPEND CABANA_UNIT_TEST_MPIEXEC_NUMPROCS ${_np})
    endif()
  endforeach()
  if(MPIEXEC_MAX_NUMPROCS GREATER 4)
    list(APPEND CABANA_UNIT_TEST_MPIEXEC_NUMPROCS ${MPIEXEC_MAX_NUMPROCS})
  endif()
  set(CABANA_UNIT_TEST_NUMTHREADS 1)
  foreach( _nt 2 4 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_nt})
      list(APPEND CABANA_UNIT_TEST_NUMTHREADS ${_nt})
    endif()
  endforeach()
  if(CABANA_UNIT_TEST_MPI)
    set(CABANA_UNIT_TEST_MAIN mpi_unit_test_main.cpp)
  else()
    set(CABANA_UNIT_TEST_MAIN unit_test_main.cpp)
  endif()
  foreach(_device ${CABANA_TEST_DEVICES})
    set(_dir ${CMAKE_CURRENT_BINARY_DIR}/${_device})
    file(MAKE_DIRECTORY ${_dir})
    foreach(_test ${CABANA_UNIT_TEST_NAMES})
      set(_file ${_dir}/tst${_test}_${_device}.cpp)
      file(WRITE ${_file}
        "#include <Test${_device}_Category.hpp>\n"
        "#include <tst${_test}.hpp>\n"
      )
      set(_target Cabana_${_test}_test_${_device})
      add_executable(${_target} ${_file} ${CABANA_UNIT_TEST_MAIN})
      target_include_directories(${_target} PRIVATE ${_dir}
        ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
      target_link_libraries(${_target} cabanacore cabana_core_gtest)
      if(CABANA_UNIT_TEST_MPI)
        foreach(_np ${CABANA_UNIT_TEST_MPIEXEC_NUMPROCS})
          # NOTE: When moving to CMake 3.10+ make sure to use MPIEXEC_EXECUTABLE instead
          add_test(NAME ${_target}_np_${_np} COMMAND
            ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
            ${_target} ${MPIEXEC_POSTFLAGS} ${gtest_args})
        endforeach()
      else()
        if(_device STREQUAL PTHREAD OR _device STREQUAL OPENMP)
          foreach(_thread ${CABANA_UNIT_TEST_NUMTHREADS})
            add_test(NAME ${_target}_nt_${_thread} COMMAND
              ${NONMPI_PRECOMMAND} ${_target} ${gtest_args} --kokkos-threads=${_thread})
          endforeach()
        else()
          add_test(NAME ${NONMPI_PRECOMMAND} ${_target} COMMAND ${NONMPI_PRECOMMAND} ${_target} ${gtest_args})
        endif()
      endif()
    endforeach()
  endforeach()
endmacro()

Cabana_add_tests(NAMES AoSoA Slice DeepCopy Tuple Sort LinkedCellList NeighborList Parallel)
if(Cabana_ENABLE_ARBORX)
  Cabana_add_tests(NAMES NeighborListArborX)
endif()
if(Cabana_ENABLE_MPI)
  Cabana_add_tests(MPI NAMES CommunicationPlan Distributor Halo)
  if(Cabana_ENABLE_CAJITA)
    Cabana_add_tests(MPI NAMES ParticleGridCommunication)
  endif()
endif()
