# setup gtest
SET(GTEST_SOURCE_DIR ${${PROJECT_NAME}_SOURCE_DIR}/gtest)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_PTHREAD=0")

INCLUDE_DIRECTORIES(${GTEST_SOURCE_DIR})
TRIBITS_ADD_LIBRARY(
  cabana_core_gtest
  HEADERS ${GTEST_SOURCE_DIR}/gtest/gtest.h
  SOURCES ${GTEST_SOURCE_DIR}/gtest/gtest-all.cc
  TESTONLY
  )

# Fail regular expression
SET(FAIL_RE "  FAILED  ")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Version_test
  SOURCES tstVersion.cpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  TESTONLYLIBS cabana_core_gtest
  ARGS "--gtest_color=yes"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Index_test
  SOURCES tstIndex.cpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  TESTONLYLIBS cabana_core_gtest
  ARGS "--gtest_color=yes"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  CartesianGrid_test
  SOURCES tstCartesianGrid.cpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  TESTONLYLIBS cabana_core_gtest
  ARGS "--gtest_color=yes"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  SoA_test
  SOURCES tstSoA.cpp unit_test_main.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  TESTONLYLIBS cabana_core_gtest
  ARGS "--gtest_color=yes"
  )

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
    MemberSlice_test_Serial
    SOURCES Serial/tstMemberSlice_Serial.cpp unit_test_main.cpp
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
    Particle_test_Serial
    SOURCES Serial/tstParticle_Serial.cpp unit_test_main.cpp
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
    MemberSlice_test_OpenMP
    SOURCES OpenMP/tstMemberSlice_OpenMP.cpp unit_test_main.cpp
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
    Particle_test_OpenMP
    SOURCES OpenMP/tstParticle_OpenMP.cpp unit_test_main.cpp
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
    MemberSlice_test_CudaUVM
    SOURCES Cuda/tstMemberSlice_CudaUVM.cpp unit_test_main.cpp
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
    Particle_test_CudaUVM
    SOURCES Cuda/tstParticle_CudaUVM.cpp unit_test_main.cpp
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