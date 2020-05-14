SET(CAJITA_GIT_COMMIT_HASH "No hash available")

IF(EXISTS ${SOURCE_DIR}/.git)
  FIND_PACKAGE(Git QUIET)
  IF(GIT_FOUND)
    EXECUTE_PROCESS(
      COMMAND          ${GIT_EXECUTABLE} log --pretty=format:%h -n 1
      OUTPUT_VARIABLE  CAJITA_GIT_COMMIT_HASH)
    ENDIF()
ENDIF()
MESSAGE("Cajita hash = '${CAJITA_GIT_COMMIT_HASH}'")

configure_file(${SOURCE_DIR}/src/Cajita_Version.hpp.in
               ${BINARY_DIR}/include/Cajita_Version.hpp)
