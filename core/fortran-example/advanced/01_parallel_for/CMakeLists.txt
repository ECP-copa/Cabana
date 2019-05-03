  if( Cabana_ENABLE_Cuda )
    add_definitions(-DSIMD_SIZE=1  -DVEC_LEN=32 -DUSE_GPU=1)
  else()
    add_definitions(-DSIMD_SIZE=32 -DVEC_LEN=32 -DUSE_GPU=0)
  endif()

  if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
    set(MPI_Fortran_COMPILE_FLAGS "-r -ta=tesla:cuda9.0 -Minfo=accel -Mcuda=cuda9.0")
    set(CMAKE_CXX_FLAGS "--relocatable-device-code=true")
#    set(MPI_CXX_COMPILE_FLAGS "--relocatable-device-code=true")
    set(CMAKE_EXE_LINKER_FLAGS "-pgc++libs -ta=tesla:cuda9.0 -Minfo=accel -Mcuda=cuda9.0")
  else()
    message("Only PGI is supported for compiling fortran cuda code.\n Skip.")
    return()
  endif()


add_definitions(${MPI_Fortran_COMPILE_FLAGS})
include_directories(${MPI_Fortran_INCLUDE_PATH})
link_directories(${MPI_Fortran_LIBRARIES})
 
  add_executable(ForParallelFor2 parallel_for_example_module.F90 kokkos_fortran_wrappers.F90 cabana_cpp_interface.cpp main.f90 ptl_module.F90)
  set_target_properties(ForParallelFor2 PROPERTIES COMPILE_FLAGS "${MPI_Fortran_COMPILE_FLAGS}")
  set_target_properties(ForParallelFor2 PROPERTIES LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")

  target_link_libraries(ForParallelFor2 cabanacore ${MPI_Fortran_LIBRARIES})
  add_test(NAME Fortran_tutorial_12 COMMAND ForParallelFor2)
  set_property(TARGET ForParallelFor2 PROPERTY LINKER_LANGUAGE Fortran)
