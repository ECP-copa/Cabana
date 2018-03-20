SET(PASS_RE "No errors detected")
SET(FAIL_RE "detected in the test module")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})

##--------------------------------------------------------------------------##
## General tests.
##--------------------------------------------------------------------------##
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Version_test
  SOURCES tstVersion.cpp unit_test_main.cpp
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "${PASS_RE}"
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Index_test
  SOURCES tstIndex.cpp unit_test_main.cpp
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "${PASS_RE}"
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  SoA_test
  SOURCES tstSoA.cpp unit_test_main.cpp
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "${PASS_RE}"
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  )

##--------------------------------------------------------------------------##
## Serial tests.
##--------------------------------------------------------------------------##
IF(Cabana_ENABLE_Serial)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    AoSoA_test_Serial
    SOURCES serial/tstAoSoA_Serial.cpp unit_test_main.cpp
    COMM serial mpi
    PASS_REGULAR_EXPRESSION "${PASS_RE}"
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    MemberSlice_test_Serial
    SOURCES serial/tstMemberSlice_Serial.cpp unit_test_main.cpp
    COMM serial mpi
    PASS_REGULAR_EXPRESSION "${PASS_RE}"
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Parallel_test_Serial
    SOURCES serial/tstParallel_Serial.cpp unit_test_main.cpp
    COMM serial mpi
    PASS_REGULAR_EXPRESSION "${PASS_RE}"
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    )

ENDIF()

##--------------------------------------------------------------------------##
## OpenMP tests.
##--------------------------------------------------------------------------##
IF(Cabana_ENABLE_OpenMP)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    AoSoA_test_OpenMP
    SOURCES openmp/tstAoSoA_OpenMP.cpp unit_test_main.cpp
    COMM serial mpi
    PASS_REGULAR_EXPRESSION "${PASS_RE}"
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    MemberSlice_test_OpenMP
    SOURCES openmp/tstMemberSlice_OpenMP.cpp unit_test_main.cpp
    COMM serial mpi
    PASS_REGULAR_EXPRESSION "${PASS_RE}"
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Parallel_test_OpenMP
    SOURCES openmp/tstParallel_OpenMP.cpp unit_test_main.cpp
    COMM serial mpi
    PASS_REGULAR_EXPRESSION "${PASS_RE}"
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    )

ENDIF()

##--------------------------------------------------------------------------##
## Cuda tests.
##--------------------------------------------------------------------------##
IF(Cabana_ENABLE_Cuda)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    AoSoA_test_CudaUVM
    SOURCES cuda/tstAoSoA_CudaUVM.cpp unit_test_main.cpp
    COMM serial mpi
    PASS_REGULAR_EXPRESSION "${PASS_RE}"
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    MemberSlice_test_CudaUVM
    SOURCES cuda/tstMemberSlice_CudaUVM.cpp unit_test_main.cpp
    COMM serial mpi
    PASS_REGULAR_EXPRESSION "${PASS_RE}"
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    )

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    Parallel_test_CudaUVM
    SOURCES cuda/tstParallel_CudaUVM.cpp unit_test_main.cpp
    COMM serial mpi
    PASS_REGULAR_EXPRESSION "${PASS_RE}"
    FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
    )

ENDIF()

##--------------------------------------------------------------------------##