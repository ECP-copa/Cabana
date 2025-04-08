# Set MPI_ADVANCE_DIR if not already defined
if(NOT DEFINED MPI_ADVANCE_DIR)
    execute_process(
        COMMAND spack location -i mpi-advance
        OUTPUT_VARIABLE MPI_ADVANCE_install
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(MPI_ADVANCE_DIR ${MPI_ADVANCE_install})
endif()

# Find the headers
find_path(
    MPI_ADVANCE_INCLUDE_DIR
    NAMES mpi_advance.h
    PATHS ${MPI_ADVANCE_DIR}/include
          ENV CPATH
    NO_DEFAULT_PATH
)

# Find the library
find_library(
    MPI_ADVANCE_LIBRARY
    NAMES libmpi_advance.a
    PATHS ${MPI_ADVANCE_DIR}/lib
          ENV LIBRARY_PATH
    NO_DEFAULT_PATH
)

# Check and handle errors properly
if (MPI_ADVANCE_LIBRARY)
    message(STATUS "Found mpi-advance library: ${MPI_Advance_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find mpi-advance library!")
endif()

if (MPI_ADVANCE_INCLUDE_DIR)
    message(STATUS "Found mpi-advance INCLUDE_DIR: ${MPI_ADVANCE_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Could not find mpi-advance INCLUDE_DIR!")
endif()

# Setup 'FOUND' flag
if (MPI_ADVANCE_INCLUDE_DIR AND MPI_ADVANCE_LIBRARY)
    set(MPI_ADVANCE_FOUND ON)
endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mpi-advance DEFAULT_MSG MPI_ADVANCE_INCLUDE_DIR)

# Hide internal variables
mark_as_advanced(MPI_ADVANCE_INCLUDE_DIR MPI_ADVANCE_DIR)
