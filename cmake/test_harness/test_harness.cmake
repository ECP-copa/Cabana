############################################################################
# Copyright (c) 2018-2023 by the Cabana authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Cabana library. Cabana is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

include(FindPackageHandleStandardArgs)
find_program(VALGRIND_EXECUTABLE valgrind)
find_package_handle_standard_args(VALGRIND REQUIRED_VARS VALGRIND_EXECUTABLE)
if(VALGRIND_FOUND)
  set(VALGRIND_ARGS --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes --error-exitcode=1)
endif()

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
macro(Cabana_add_tests_nobackend)
  cmake_parse_arguments(CABANA_UNIT_TEST "" "PACKAGE" "NAMES" ${ARGN})
  foreach(_test ${CABANA_UNIT_TEST_NAMES})
    set(_target Cabana_${CABANA_UNIT_TEST_PACKAGE}_Test_${_test}_test)
    add_executable(${_target} tst${_test}.cpp ${TEST_HARNESS_DIR}/unit_test_main.cpp)
    target_link_libraries(${_target} PRIVATE ${CABANA_UNIT_TEST_PACKAGE} ${gtest_target})
    add_test(NAME ${_target} COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:${_target}> ${gtest_args})
    set_property(TEST ${_target} PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
    if(VALGRIND_FOUND)
      add_test(NAME ${_target}_valgrind COMMAND ${NONMPI_PRECOMMAND} ${VALGRIND_EXECUTABLE} ${VALGRIND_ARGS} $<TARGET_FILE:${_target}> ${gtest_args})
      set_property(TEST ${_target}_valgrind PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
    endif()
    if(Cabana_INSTALL_TEST_EXECUTABLES)
      install(TARGETS ${_target}
              RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/Cabana/tests)
    endif()
  endforeach()
endmacro()

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
  cmake_parse_arguments(CABANA_UNIT_TEST "MPI" "PACKAGE" "NAMES" ${ARGN})
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
    set(CABANA_UNIT_TEST_MAIN ${TEST_HARNESS_DIR}/mpi_unit_test_main.cpp)
  else()
    set(CABANA_UNIT_TEST_MAIN ${TEST_HARNESS_DIR}/unit_test_main.cpp)
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
      if(CABANA_UNIT_TEST_MPI)
        set(_target Cabana_${CABANA_UNIT_TEST_PACKAGE}_Test_${_test}_MPI_${_device})
      else()
        set(_target Cabana_${CABANA_UNIT_TEST_PACKAGE}_Test_${_test}_${_device})
      endif()
      add_executable(${_target} ${_file} ${CABANA_UNIT_TEST_MAIN})
      target_include_directories(${_target} PRIVATE ${_dir}
        ${TEST_HARNESS_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
      target_link_libraries(${_target} PRIVATE ${CABANA_UNIT_TEST_PACKAGE} ${gtest_target})
      if(CABANA_UNIT_TEST_MPI)
        foreach(_np ${CABANA_UNIT_TEST_MPIEXEC_NUMPROCS})
          add_test(NAME ${_target}_np_${_np} COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${_target}> ${MPIEXEC_POSTFLAGS} ${gtest_args})
          set_property(TEST ${_target}_np_${_np} PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
        endforeach()
      else()
        if(_device STREQUAL THREADS OR _device STREQUAL OPENMP)
          foreach(_thread ${CABANA_UNIT_TEST_NUMTHREADS})
            add_test(NAME ${_target}_nt_${_thread} COMMAND
                    ${NONMPI_PRECOMMAND} $<TARGET_FILE:${_target}> ${gtest_args} --kokkos-num-threads=${_thread})
            if(_device STREQUAL OPENMP)
              set_property(TEST ${_target}_nt_${_thread} PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_thread})
            endif()
            if(VALGRIND_FOUND)
              add_test(NAME ${_target}_nt_${_thread}_valgrind COMMAND
                ${NONMPI_PRECOMMAND} ${VALGRIND_EXECUTABLE} ${VALGRIND_ARGS} $<TARGET_FILE:${_target}> ${gtest_args} --kokkos-num-threads=${_thread})
              if(_device STREQUAL OPENMP)
                set_property(TEST ${_target}_nt_${_thread}_valgrind PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_thread})
              endif()
            endif()
          endforeach()
        else()
          add_test(NAME ${_target} COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:${_target}> ${gtest_args})
          set_property(TEST ${_target} PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
          if(VALGRIND_FOUND)
            add_test(NAME ${_target}_valgrind COMMAND ${NONMPI_PRECOMMAND} ${VALGRIND_EXECUTABLE} ${VALGRIND_ARGS} $<TARGET_FILE:${_target}> ${gtest_args})
            set_property(TEST ${_target}_valgrind PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
          endif()
        endif()
      endif()
      if(Cabana_INSTALL_TEST_EXECUTABLES)
        install(TARGETS ${_target}
                RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/Cabana/tests)
      endif()
    endforeach()
  endforeach()
endmacro()
