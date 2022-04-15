find_package(PkgConfig QUIET)
pkg_check_modules(PC_SILO silo QUIET)

find_path(SILO_INCLUDE_DIR silo.h pmpio.h HINTS ${PC_SILO_INCLUDE_DIRS})
find_library(SILO_LIBRARY NAMES silo siloh5 HINTS ${PC_SILO_LIBRARY_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SILO DEFAULT_MSG SILO_INCLUDE_DIR SILO_LIBRARY)

mark_as_advanced(SILO_INCLUDE_DIR SILO_LIBRARY)

if(SILO_INCLUDE_DIR AND SILO_LIBRARY AND NOT TARGET SILO::silo)
  add_library(SILO::silo UNKNOWN IMPORTED)
  set_target_properties(SILO::silo PROPERTIES
    IMPORTED_LOCATION ${SILO_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${SILO_INCLUDE_DIR})
  if(SILO_LINK_FLAGS)
    set_property(TARGET SILO::silo APPEND_STRING PROPERTY INTERFACE_LINK_LIBRARIES " ${SILO_LINK_FLAGS}")
  endif()
endif()
