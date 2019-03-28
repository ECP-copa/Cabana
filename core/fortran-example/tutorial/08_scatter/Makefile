# Current options:
# EX1-cpu
# EX1-gpu
# EX2-cpu
# EX2-gpu

# Paths and flags the User must replace:
export CABANA_DIR=/ccs/home/scheinberg/summit/Cabana
export KOKKOS_SRC_DIR=/ccs/home/scheinberg/summit/kokkos
export KOKKOS_INSTALL_DIR=/ccs/home/scheinberg/summit/kokkos/install
export NVCC_WRAPPER_DEFAULT_COMPILER=mpiCC

F90_GPU_FLAGS= -ta=tesla:cc70 -Minfo=accel -Mcuda=cuda9.2,cc70
CPP_GPU_FLAGS= --relocatable-device-code=true -arch=sm_70

# The rest should be platform independent:

CABANA_FLAGS= -I$(CABANA_DIR)/build -I$(CABANA_DIR)/build/core/src -I$(CABANA_DIR)/core/src
KOKKOS_FLAGS= -I$(KOKKOS_INSTALL_DIR)/include
KOKKOS_LINK_FLAGS= -L$(KOKKOS_INSTALL_DIR)/lib -lkokkos
USER_FLAGS= -DSIMD_SIZE=$(SIMD_SIZE) -DVEC_LEN=$(VEC_LEN) -DUSE_GPU=$(USE_GPU)

F90=mpifort

CXX=$(KOKKOS_SRC_DIR)/bin/nvcc_wrapper

F90FLAGS= -r $(USER_FLAGS) $(F90_GPU_FLAGS) 
CPPFLAGS= -pedantic -O3 -mp -std=c++11 --expt-extended-lambda $(USER_FLAGS) $(KOKKOS_FLAGS) $(CABANA_FLAGS) $(CPP_GPU_FLAGS)
F90LINKFLAGS= -O3 -mp -pgc++libs $(CABANA_FLAGS) $(KOKKOS_FLAGS) $(KOKKOS_LINK_FLAGS) $(F90_GPU_FLAGS)

DEPS = cabana_fortran_macros.h
OBJ_CABANA = cabana_cpp_interface.o
OBJ_KERNEL = cabana_fortran_wrappers.o ptl_module.o grid_module.o scatter_module.o update_phase_module.o push_module.o parallel_for_example_module.o
OBJ_ALL = $(OBJ_KERNEL) $(OBJ_CABANA) $(OBJ_MAIN)

%.o: %.f90 $(DEPS)
	$(F90) $(F90FLAGS) -c $<

%.o: %.F90 $(DEPS)
	$(F90) $(F90FLAGS) -c $<

%.o: %.cpp $(DEPS)
	$(CXX) $(CPPFLAGS) -c $<

set_ex1:
	@echo ' '
	@echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	@echo !!! Building Example 1 - parallel_for
	$(eval OBJ_MAIN := Example1_parallel_for.o)

set_ex2:
	@echo ' '
	@echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	@echo !!! Building Example 2 - Push
	$(eval OBJ_MAIN := Example2_Push.o)

set_ex3:
	@echo ' '
	@echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	@echo !!! Building Example 3 - Scatter
	$(eval OBJ_MAIN := Example3_Scatter.o)


set_gpu:
	@echo !!! GPU version
	@echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	@echo ' '
	$(eval USE_GPU := 1)
	$(eval SIMD_SIZE := 1)
	$(eval VEC_LEN := 32)

set_no_gpu:
	@echo !!! CPU version
	@echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	@echo ' '
	$(eval USE_GPU := 0)
	$(eval SIMD_SIZE := 32)
	$(eval VEC_LEN := 32)

EX1-gpu: set_ex1 set_gpu $(OBJ_KERNEL) $(OBJ_CABANA) Example1_parallel_for.o
	$(F90) -o $@ $(OBJ_ALL) $(F90LINKFLAGS)

EX1-cpu: set_ex1 set_no_gpu $(OBJ_KERNEL) $(OBJ_CABANA) Example1_parallel_for.o
	$(F90) -o $@ $(OBJ_ALL) $(F90LINKFLAGS)

EX2-gpu: set_ex2 set_gpu $(OBJ_KERNEL) $(OBJ_CABANA) Example2_Push.o
	$(F90) -o $@ $(OBJ_ALL) $(F90LINKFLAGS)

EX2-cpu: set_ex2 set_no_gpu $(OBJ_KERNEL) $(OBJ_CABANA) Example2_Push.o
	$(F90) -o $@ $(OBJ_ALL) $(F90LINKFLAGS)

EX3-gpu: set_ex3 set_gpu $(OBJ_KERNEL) $(OBJ_CABANA) Example3_Scatter.o
	$(F90) -o $@ $(OBJ_ALL) $(F90LINKFLAGS)

EX3-cpu: set_ex3 set_no_gpu $(OBJ_KERNEL) $(OBJ_CABANA) Example3_Scatter.o
	$(F90) -o $@ $(OBJ_ALL) $(F90LINKFLAGS)


clean:
	rm *.o *.mod
