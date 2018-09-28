# - Find kokkos
# Find the native KOKKOS headers and libraries.
#
#  KOKKOS_INCLUDE_DIRS - where to find kokkos.h, etc.
#  KOKKOS_LIBRARIES    - List of libraries when using kokkos.
#  KOKKOS_FOUND        - True if kokkos found.
#

find_path(KOKKOS_INCLUDE_DIR Kokkos_Core.hpp)

find_library(KOKKOS_LIBRARY NAMES kokkos)

set(KOKKOS_LIBRARIES ${KOKKOS_LIBRARY})
set(KOKKOS_INCLUDE_DIRS ${KOKKOS_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set KOKKOS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(KOKKOS DEFAULT_MSG KOKKOS_LIBRARY KOKKOS_INCLUDE_DIR)

mark_as_advanced(KOKKOS_INCLUDE_DIR KOKKOS_LIBRARY)
