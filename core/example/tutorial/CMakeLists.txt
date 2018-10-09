add_subdirectory(01_hello_world)
add_subdirectory(02_tuple)
add_subdirectory(03_struct_of_arrays)

if(Cabana_ENABLE_Serial)
  add_subdirectory(04_aosoa_serial)
  add_subdirectory(05_slice_serial)
endif()

if(Cabana_ENABLE_OpenMP)
  add_subdirectory(04_aosoa_openmp)
  add_subdirectory(05_slice_openmp)
endif()

if(Cabana_ENABLE_Cuda)
  add_subdirectory(04_aosoa_cuda)
  add_subdirectory(05_slice_cuda)
endif()

if(Cabana_ENABLE_Cuda AND (Cabana_ENABLE_Serial OR Cabana_ENABLE_OpenMP OR Cabana_ENABLE_Pthread))
  add_subdirectory(06_deep_copy)
endif()