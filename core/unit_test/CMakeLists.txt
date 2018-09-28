# setup gtest
if(NOT ENABLE_TESTING)
  return()
endif()

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/gtest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_PTHREAD=0")

include_directories(${GTEST_SOURCE_DIR})
add_library(cabana_core_gtest ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc)

# Fail regular expression
set(FAIL_RE "  FAILED  ")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
add_executable(Version_test tstVersion.cpp unit_test_main.cpp)
target_link_libraries(Version_test cabanacore cabana_core_gtest)
add_test(NAME Version_test COMMAND Version_test --gtest_color=yes)
set_tests_properties(Version_test PROPERTIES FAIL_REGULAR_EXPRESSION "${FAIL_RE}")

#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  Version_test
#  SOURCES tstVersion.cpp unit_test_main.cpp
#  COMM serial mpi
#  NUM_MPI_PROCS 1
#  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
#  TESTONLYLIBS cabana_core_gtest
#  ARGS "--gtest_color=yes"
#  )

#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  Index_test
#  SOURCES tstIndex.cpp unit_test_main.cpp
#  COMM serial mpi
#  NUM_MPI_PROCS 1
#  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
#  TESTONLYLIBS cabana_core_gtest
#  ARGS "--gtest_color=yes"
#  )
#
#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  CartesianGrid_test
#  SOURCES tstCartesianGrid.cpp unit_test_main.cpp
#  COMM serial mpi
#  NUM_MPI_PROCS 1
#  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
#  TESTONLYLIBS cabana_core_gtest
#  ARGS "--gtest_color=yes"
#  )
#
#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  SoA_test
#  SOURCES tstSoA.cpp unit_test_main.cpp
#  COMM serial mpi
#  NUM_MPI_PROCS 1
#  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
#  TESTONLYLIBS cabana_core_gtest
#  ARGS "--gtest_color=yes"
#  )

##--------------------------------------------------------------------------##
## Serial tests.
##--------------------------------------------------------------------------##
IF(Cabana_ENABLE_Serial)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    AoSoA_test_Serial
    SOURCES Serial/tstAoSoA_Serial.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Slice_test_Serial
    SOURCES Serial/tstSlice_Serial.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    DeepCopy_test_Serial
    SOURCES Serial/tstDeepCopy_Serial.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tuple_test_Serial
    SOURCES Serial/tstTuple_Serial.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Sort_test_Serial
    SOURCES Serial/tstSort_Serial.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

    TRIBITS_ADD_EXECUTABLE_AND_TEST(
    NeighborList_test_Serial
    SOURCES Serial/tstNeighborList_Serial.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Parallel_test_Serial
    SOURCES Serial/tstParallel_Serial.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

ENDIF()

##--------------------------------------------------------------------------##
## OpenMP tests.
##--------------------------------------------------------------------------##
IF(Cabana_ENABLE_OpenMP)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    AoSoA_test_OpenMP
    SOURCES OpenMP/tstAoSoA_OpenMP.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Slice_test_OpenMP
    SOURCES OpenMP/tstSlice_OpenMP.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    DeepCopy_test_OpenMP
    SOURCES OpenMP/tstDeepCopy_OpenMP.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tuple_test_OpenMP
    SOURCES OpenMP/tstTuple_OpenMP.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Sort_test_OpenMP
    SOURCES OpenMP/tstSort_OpenMP.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    NeighborList_test_OpenMP
    SOURCES OpenMP/tstNeighborList_OpenMP.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Parallel_test_OpenMP
    SOURCES OpenMP/tstParallel_OpenMP.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

ENDIF()

##--------------------------------------------------------------------------##
## Cuda tests.
##--------------------------------------------------------------------------##
IF(Cabana_ENABLE_Cuda)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    AoSoA_test_CudaUVM
    SOURCES Cuda/tstAoSoA_CudaUVM.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Slice_test_CudaUVM
    SOURCES Cuda/tstSlice_CudaUVM.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    DeepCopy_test_CudaUVM
    SOURCES Cuda/tstDeepCopy_CudaUVM.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Tuple_test_CudaUVM
    SOURCES Cuda/tstTuple_CudaUVM.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Sort_test_CudaUVM
    SOURCES Cuda/tstSort_CudaUVM.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    NeighborList_test_CudaUVM
    SOURCES Cuda/tstNeighborList_CudaUVM.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Parallel_test_CudaUVM
    SOURCES Cuda/tstParallel_CudaUVM.cpp unit_test_main.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    TESTONLYLIBS cabana_core_gtest
    ARGS "--gtest_color=yes"
    )

ENDIF()

##--------------------------------------------------------------------------##
