#ifndef USE_GPU
#define USE_GPU 0
#endif

#if USE_GPU == 1
#define ATTR_DEVICE attributes(device) &
#define ATTR_HOST attributes(host) &
#define ATTR_GLOBAL attributes(global) &
#define ATTR_SHARED attributes(shared) &
#define ATTR_CONSTANT attributes(constant) &
#define ATTR_PINNED attributes(pinned) &

#define COPY_TO_DEVICE(err,icount,d_array,h_array) icount = size(h_array, kind=cuda_count_kind); \
  err = cudaMemcpy( d_array, h_array, icount, cudaMemcpyHostToDevice)

#define COPY_TO_HOST(err,icount,h_array,d_array) icount = size(d_array, kind=cuda_count_kind); \
      err = cudaMemcpy( h_array, d_array, icount, cudaMemcpyDeviceToHost)

#define FCABANA_DEVICE , device

#define CABANA_COUNT_KIND (kind=cuda_count_kind)


#define CABANA_DEF_THREAD_NUM cabana_istat
#define CABANA_GET_THREAD_NUM(x) x=1
#define CABANA_ADD(x,y) cabana_istat=atomicAdd(x,y)
#define CABANA_DEF_MAX_THREADS omp_get_max_threads
#define CABANA_GET_MAX_THREADS(x) x=1


#else
#define ATTR_DEVICE
#define ATTR_HOST
#define ATTR_GLOBAL
#define ATTR_SHARED
#define ATTR_CONSTANT
#define ATTR_PINNED

#define COPY_TO_DEVICE(err,icount,d_array,h_array) d_array = h_array
#define COPY_TO_HOST(err,icount,h_array,d_array) h_array = d_array

#define FCABANA_DEVICE

#define CABANA_COUNT_KIND


#define CABANA_DEF_THREAD_NUM omp_get_thread_num
#define CABANA_GET_THREAD_NUM(x) x=1+OMP_GET_THREAD_NUM()
#define CABANA_ADD(x,y) x=x+y
#define CABANA_DEF_MAX_THREADS omp_get_max_threads
#define CABANA_GET_MAX_THREADS(x) x=OMP_GET_MAX_THREADS()


#endif


#define PARTICLE_OP_INTERFACE(C_FUNC) \
  interface; \
     integer(C_INT) function C_FUNC(sp, num_particle) bind(C, name=#C_FUNC); \
       use iso_c_binding; \
       integer(C_INT), intent(in), value :: sp; \
       integer(C_INT), intent(in), value :: num_particle; \
     end function; \
  end interface

#define MISC_OP_INTERFACE(C_FUNC) \
  interface; \
     integer(C_INT) function C_FUNC(sp, ep) bind(C, name=#C_FUNC); \
       use iso_c_binding; \
       integer(C_INT), intent(in), value :: sp; \
       integer(C_INT), intent(in), value :: ep; \
     end function; \
  end interface
