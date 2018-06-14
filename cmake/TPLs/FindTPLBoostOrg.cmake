GLOBAL_SET(BoostOrg_INCLUDE_DIRS "${Boost_INCLUDE_DIRS}")
GLOBAL_SET(BoostOrg_LIBRARY_DIRS "${Boost_LIBRARY_DIRS}")

TRIBITS_TPL_FIND_INCLUDE_DIRS_AND_LIBRARIES(
  BoostOrg
  REQUIRED_HEADERS boost/version.hpp boost/mpl/at.hpp
  REQUIRED_LIBS_NAMES boost_unit_test_framework
  )

# Use CMake FindBoost module to check version is sufficient
SET(BOOST_INCLUDEDIR "${BoostOrg_INCLUDE_DIRS}")
SET(Boost_NO_SYSTEM_PATHS ON)
FIND_PACKAGE(Boost 1.61.0 REQUIRED)
