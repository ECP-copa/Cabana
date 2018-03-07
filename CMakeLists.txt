##---------------------------------------------------------------------------##
## Cabana/CMakeLists.txt
##---------------------------------------------------------------------------##

##---------------------------------------------------------------------------##
# Define your project name and set up major project options
##---------------------------------------------------------------------------##

INCLUDE(${CMAKE_SOURCE_DIR}/ProjectName.cmake)

# CMake requires that you declare the CMake project in the top-level file and
# not in an include file
PROJECT(${PROJECT_NAME} NONE)

# Add support for extra repositories; everything handled by native above
SET(${PROJECT_NAME}_SUPPORT_EXTRA_REPOS FALSE)

# Disable Tribits generating HTML dependencies webpage and xml files
SET(${PROJECT_NAME}_DEPS_XML_OUTPUT_FILE OFF CACHE BOOL "" )

# Disable Tribits export system to save time configuring
SET(${PROJECT_NAME}_ENABLE_INSTALL_CMAKE_CONFIG_FILES OFF CACHE BOOL "")

# Disable Tribits export makefiles system to save time configuring
SET(${PROJECT_NAME}_ENABLE_EXPORT_MAKEFILES OFF CACHE BOOL "")

# Turn on C++11
SET(${PROJECT_NAME}_ENABLE_CXX11 ON CACHE BOOL "")

# Turn on the core package.
SET(${PROJECT_NAME}_ENABLE_Core ON CACHE BOOL "")

##---------------------------------------------------------------------------##
# B) Pull in the TriBITS system and execute
##---------------------------------------------------------------------------##

# Point to tribits. Tribits is managed via a git submodule.
SET(${PROJECT_NAME}_TRIBITS_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tribits/tribits" CACHE PATH "" )
MESSAGE("${PROJECT_NAME}_TRIBITS_DIR = '${${PROJECT_NAME}_TRIBITS_DIR}'")
INCLUDE(${${PROJECT_NAME}_TRIBITS_DIR}/TriBITS.cmake)

# CMake requires this be in the top file and not in an include file
CMAKE_MINIMUM_REQUIRED(VERSION ${TRIBITS_CMAKE_MINIMUM_REQUIRED})

##---------------------------------------------------------------------------##
## Print the revision number to stdout
##---------------------------------------------------------------------------##

FIND_PACKAGE(Git)
IF(GIT_FOUND AND EXISTS ${${PACKAGE_NAME}_SOURCE_DIR}/.git)
    EXECUTE_PROCESS(
        COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE   Cabana_GIT_COMMIT_HASH
        )
ELSE()
    SET(Cabana_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
MESSAGE("Cabana Revision = '${Cabana_GIT_COMMIT_HASH}'")

##---------------------------------------------------------------------------##
## Do standard postprocessing
##---------------------------------------------------------------------------##

# Do all of the processing for this Tribits project
TRIBITS_PROJECT()

##---------------------------------------------------------------------------##
## end of CMakeLists.txt
##---------------------------------------------------------------------------##
