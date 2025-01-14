# search prefix path
set(MPI_Advance_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE STRING "Help cmake to find MPI_Advance")

# check include
find_path(MPI_Advance_INCLUDE_DIR NAMES mpi_advance.h HINTS ${MPI_Advance_PREFIX}/include)

# check lib
find_library(MPI_Advance_LIBRARY NAMES mpi_advance
	HINTS ${MPI_Advance_PREFIX}/lib)

# setup found
if (MPI_Advance_INCLUDE_DIR AND MPI_Advance_LIBRARY)
	set(MPI_Advance_FOUND ON)
endif()

# handle QUIET/REQUIRED
include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set MPI_Advance_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(MPI_Advance DEFAULT_MSG MPI_Advance_INCLUDE_DIR MPI_Advance_LIBRARY)

# Hide internal variables
mark_as_advanced(MPI_Advance_INCLUDE_DIR MPI_Advance_FOUND MPI_Advance_LIBRARY MPI_Advance_PREFIX)
