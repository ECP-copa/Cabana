SET(PASS_RE "No errors detected")
SET(FAIL_RE "detected in the test module")

##--------------------------------------------------------------------------##
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Version_test
  SOURCES tstVersion.cpp unit_test_main.cpp
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "${PASS_RE}"
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  )

##--------------------------------------------------------------------------##
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  SoA_test
  SOURCES tstSoA.cpp unit_test_main.cpp
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "${PASS_RE}"
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  )

##--------------------------------------------------------------------------##
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  AoSoA_test
  SOURCES tstAoSoA.cpp unit_test_main.cpp
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "${PASS_RE}"
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  )

##--------------------------------------------------------------------------##
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  MemberSlice_test
  SOURCES tstMemberSlice.cpp unit_test_main.cpp
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "${PASS_RE}"
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  )

##--------------------------------------------------------------------------##
TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Parallel_test
  SOURCES tstParallel.cpp unit_test_main.cpp
  COMM serial mpi
  PASS_REGULAR_EXPRESSION "${PASS_RE}"
  FAIL_REGULAR_EXPRESSION "${FAIL_RE}"
  )

##--------------------------------------------------------------------------##