# - Find kokkos
# Creates a Kokkos::kokkos imported target
#
#  KOKKOS_INCLUDE_DIRS  - where to find kokkos.h, etc.
#  KOKKOS_LIBRARIES     - List of libraries when using kokkos.
#  KOKKOS_FOUND         - True if kokkos found.
#  KOKKOS_SETTINGS_DIR - path to kokkos_generated_settings.cmake
#

find_package(PkgConfig QUIET)
pkg_check_modules(PC_KOKKOS kokkos QUIET)

find_path(KOKKOS_SETTINGS_DIR kokkos_generated_settings.cmake HINTS ${PC_KOKKOS_PREFIX})

find_path(KOKKOS_INCLUDE_DIR Kokkos_Core.hpp HINTS ${PC_KOKKOS_INCLUDE_DIRS})
find_library(KOKKOS_LIBRARY NAMES kokkos kokkoscore HINTS ${PC_KOKKOS_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set KOKKOS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(KOKKOS DEFAULT_MSG KOKKOS_SETTINGS_DIR KOKKOS_INCLUDE_DIR KOKKOS_LIBRARY)

mark_as_advanced(KOKKOS_SETTINGS_DIR KOKKOS_INCLUDE_DIR KOKKOS_LIBRARY)

if(KOKKOS_SETTINGS_DIR AND KOKKOS_INCLUDE_DIR AND KOKKOS_LIBRARY)
  include(${KOKKOS_SETTINGS_DIR}/kokkos_generated_settings.cmake)
  # https://github.com/kokkos/kokkos/issues/1838
  set(KOKKOS_CXX_FLAGS_WITHOUT_INCLUDES_STRING)
  set(KOKKOS_CXX_FLAGS_WITHOUT_INCLUDES)
  foreach(_f ${KOKKOS_CXX_FLAGS})
    if(NOT _f MATCHES "-I.*")
      set(KOKKOS_CXX_FLAGS_WITHOUT_INCLUDES_STRING "${KOKKOS_CXX_FLAGS_WITHOUT_INCLUDES_STRING} ${_f}")
      list(APPEND KOKKOS_CXX_FLAGS_WITHOUT_INCLUDES "${_f}")
    endif()
  endforeach()
  add_library(Kokkos::kokkos UNKNOWN IMPORTED)
  set_target_properties(Kokkos::kokkos PROPERTIES
    IMPORTED_LOCATION ${KOKKOS_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${KOKKOS_INCLUDE_DIR}
    INTERFACE_COMPILE_OPTIONS "${KOKKOS_CXX_FLAGS_WITHOUT_INCLUDES}"
    INTERFACE_LINK_LIBRARIES "${KOKKOS_EXTRA_LIBS}")
  # check for an empty link flags string to fix a trailing whitespace error when
  # the link flags are empty (e.g. the serial only case)
  if(KOKKOS_LINK_FLAGS)
    set_property(TARGET Kokkos::kokkos APPEND_STRING PROPERTY INTERFACE_LINK_LIBRARIES " ${KOKKOS_LINK_FLAGS}")
  endif()
endif()
